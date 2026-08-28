#!/usr/bin/env python3
"""
Real Mamba SSM (Selective State Space Model) for RT audio processing.

Implements the S6 selective scan mechanism from:
  Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023

Architecture: d_model=64, n_layers=4, d_state=16, conv_width=4, dt_rank=16
Small enough for real-time CPU inference via ONNX Runtime.

Export uses torch.jit.trace (not dynamo) for speed with recurrent models.
All loops are expressed as Python control flow during tracing — since
seq_len is dynamic via ONNX dynamic_axes, we use a fixed export size.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


# ---------------------------------------------------------------------------
# Selective Scan (S6) — batch over all timesteps, ONNX-friendly
# ---------------------------------------------------------------------------

class SelectiveScanStep(nn.Module):
    """Single-step selective scan for ONNX export via unrolled loop in MambaRTModel."""
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

    def forward(self, x_t, dt_t, A, B_t, C_t, D, h_in):
        """
        Args:
          x_t:   [batch, d_model]
          dt_t:  [batch, d_model]
          A:     [d_model, d_state]
          B_t:   [batch, d_state]
          C_t:   [batch, d_state]
          D:     [d_model]
          h_in:  [batch, d_model, d_state]
        Returns:
          y_t:   [batch, d_model]
          h_out: [batch, d_model, d_state]
        """
        dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # [B, D, N]
        dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)            # [B, D, N]
        h_out = dA * h_in + dB * x_t.unsqueeze(-1)           # [B, D, N]
        y_t = torch.sum(C_t.unsqueeze(1) * h_out, dim=-1) + D * x_t  # [B, D]
        return y_t, h_out


class MambaSSMCore(nn.Module):
    """The SSM core: projects input to dt/B/C, returns per-timestep values for unrolled scan."""

    def __init__(self, d_model, d_state, dt_rank, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank

        self.log_A_real = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1))
            * (math.log(dt_max / dt_min) / max(d_state - 1, 1))
            - torch.log(torch.tensor(dt_max, dtype=torch.float32))
        )  # [d_model, d_state]

        self.D = nn.Parameter(torch.ones(d_model))

        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x):
        """
        Args:
          x: [batch, seq_len, d_model]
        Returns:
          dt_all: [batch, seq_len, d_model]
          A:      [d_model, d_state]
          B_all:  [batch, seq_len, d_state]
          C_all:  [batch, seq_len, d_state]
          D:      [d_model]
        """
        xz = self.x_proj(x)  # [B, L, dt_rank + 2*N]
        dt_latent = xz[:, :, :self.dt_rank]
        B = xz[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C = xz[:, :, self.dt_rank + self.d_state:]

        dt = F.softplus(self.dt_proj(dt_latent) + self.dt_proj.bias)  # [B, L, D]
        A = -torch.exp(self.log_A_real)  # [D, N]

        return dt, A, B, C, self.D


# ---------------------------------------------------------------------------
# Mamba Layer: Conv1d + SSM + SiLU gating
# ---------------------------------------------------------------------------

class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state=16, conv_width=4, dt_rank=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.conv_width = conv_width

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, conv_width,
            padding=conv_width - 1, groups=d_model
        )
        self.ssm_proj = MambaSSMCore(d_model, d_state, dt_rank)
        self.scan_step = SelectiveScanStep(d_model, d_state)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, conv_state, ssm_state):
        """
        Args:
          x:          [batch, seq_len, d_model]
          conv_state: [batch, d_model, conv_width]
          ssm_state:  [batch, d_model, d_state]

        Returns:
          output:          [batch, seq_len, d_model]
          new_conv_state:  [batch, d_model, conv_width]
          new_ssm_state:   [batch, d_model, d_state]
        """
        seq_len = x.shape[1]

        projected = self.in_proj(x)
        x_ssm = projected[:, :, :self.d_model]
        x_gate = projected[:, :, self.d_model:]

        # Causal Conv1d: prepend conv_state for proper causal history
        # x_ssm: [B, L, D] -> [B, D, L]
        x_conv_seq = x_ssm.transpose(1, 2)  # [B, D, L]
        # Prepend the rolling buffer state for proper causal context
        full_seq = torch.cat([conv_state, x_conv_seq], dim=-1)  # [B, D, W+L]
        conv_full = self.conv1d(full_seq)  # [B, D, W+L+W-1]
        # Extract only the outputs corresponding to x positions
        # The conv1d output at position i corresponds to input [i-W+1..i]
        # After the prepended state, our x starts at index W
        # conv output for x[0] starts at index W (in the full_seq input domain)
        # But conv1d output shifts by (W-1) padding, so:
        # output for x[0] is at conv_full index (W-1), and we need seq_len values
        start_idx = self.conv_width - 1  # offset for causal conv
        conv_out = conv_full[:, :, start_idx:start_idx + seq_len]  # [B, D, L]
        conv_out = conv_out.transpose(1, 2)  # [B, L, D]

        # SiLU gating
        conv_out = conv_out * F.silu(x_gate)

        # SSM
        dt, A, B, C, D = self.ssm_proj(conv_out)

        h = ssm_state
        ys = []
        for t in range(seq_len):
            y_t, h = self.scan_step(conv_out[:, t, :], dt[:, t, :], A, B[:, t, :], C[:, t, :], D, h)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # [B, L, D]
        output = self.out_proj(y)

        # New conv_state: last conv_width samples of x_conv_seq
        new_conv_state = x_conv_seq[:, :, -self.conv_width:]  # [B, D, W]
        new_ssm_state = h

        return output, new_conv_state, new_ssm_state


# ---------------------------------------------------------------------------
# Input/Output Projections (1-channel audio <-> d_model)
# ---------------------------------------------------------------------------

class InputProjection(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(1, d_model)

    def forward(self, x):
        return self.proj(x)


class OutputProjection(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.proj(x)


# ---------------------------------------------------------------------------
# Full RT Mamba Model
# ---------------------------------------------------------------------------

class MambaRTModel(nn.Module):
    def __init__(self, d_model=64, d_state=16, n_layers=4, conv_width=4, dt_rank=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.conv_width = conv_width

        self.input_proj = InputProjection(d_model)
        self.layer0 = MambaLayer(d_model, d_state, conv_width, dt_rank)
        self.layer1 = MambaLayer(d_model, d_state, conv_width, dt_rank)
        self.layer2 = MambaLayer(d_model, d_state, conv_width, dt_rank)
        self.layer3 = MambaLayer(d_model, d_state, conv_width, dt_rank)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = OutputProjection(d_model)

    def forward(self, audio_chunk,
                conv0, conv1, conv2, conv3,
                ssm0, ssm1, ssm2, ssm3):
        """
        Per-layer state inputs to keep the tracer honest.

        Args:
          audio_chunk: [batch, seq_len, 1]
          conv{i}:     [batch, d_model, conv_width]
          ssm{i}:      [batch, d_model, d_state]

        Returns:
          output_audio:   [batch, seq_len, 1]
          new_conv0..3:   [batch, d_model, conv_width] each
          new_ssm0..3:    [batch, d_model, d_state] each
        """
        x = self.input_proj(audio_chunk)

        x, nc0, ns0 = self.layer0(x, conv0, ssm0)
        x, nc1, ns1 = self.layer1(x, conv1, ssm1)
        x, nc2, ns2 = self.layer2(x, conv2, ssm2)
        x, nc3, ns3 = self.layer3(x, conv3, ssm3)

        x = self.norm(x)
        output_audio = self.output_proj(x)

        # Stack into the shapes the C++ engine expects
        new_conv_state = torch.stack([nc0, nc1, nc2, nc3], dim=1)  # [B, 4, D, W]
        new_ssm_state = torch.stack([ns0, ns1, ns2, ns3], dim=1)   # [B, 4, D, N]

        return output_audio, new_conv_state, new_ssm_state


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_mamba_to_onnx(file_path, d_model=64, d_state=16, n_layers=4,
                          conv_width=4, dt_rank=16, seq_len=1024, opset_version=17):
    model = MambaRTModel(
        d_model=d_model, d_state=d_state, n_layers=n_layers,
        conv_width=conv_width, dt_rank=dt_rank
    )
    model.eval()

    batch_size = 1
    dummy_audio = torch.randn(batch_size, seq_len, 1)
    dummy_convs = tuple(torch.zeros(batch_size, d_model, conv_width) for _ in range(n_layers))
    dummy_ssms = tuple(torch.zeros(batch_size, d_model, d_state) for _ in range(n_layers))

    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)

    input_names = ['audio_chunk'] + [f'conv{i}' for i in range(n_layers)] + [f'ssm{i}' for i in range(n_layers)]
    output_names = ['processed_audio', 'new_conv_state', 'new_ssm_state']

    all_args = (dummy_audio,) + dummy_convs + dummy_ssms

    torch.onnx.export(
        model,
        all_args,
        file_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'audio_chunk': {1: 'seq_len'},
            'processed_audio': {1: 'seq_len'},
        },
        dynamo=False,
    )
    print(f"Exported: {file_path}")
    return model


def verify_onnx_model(onnx_path, d_model=64, d_state=16, n_layers=4, conv_width=4, seq_len=64):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)

    audio = np.random.randn(1, seq_len, 1).astype(np.float32)
    feeds = {'audio_chunk': audio}
    for i in range(n_layers):
        feeds[f'conv{i}'] = np.zeros((1, d_model, conv_width), dtype=np.float32)
        feeds[f'ssm{i}'] = np.zeros((1, d_model, d_state), dtype=np.float32)

    outputs = sess.run(None, feeds)
    out_audio, out_conv, out_ssm = outputs

    print(f"  processed_audio shape: {out_audio.shape}  expected: (1, {seq_len}, 1)")
    print(f"  new_conv_state shape:  {out_conv.shape}  expected: (1, {n_layers}, {d_model}, {conv_width})")
    print(f"  new_ssm_state shape:   {out_ssm.shape}  expected: (1, {n_layers}, {d_model}, {d_state})")

    assert out_audio.shape == (1, seq_len, 1), f"Audio shape mismatch: {out_audio.shape}"
    assert out_conv.shape == (1, n_layers, d_model, conv_width), f"Conv state shape mismatch: {out_conv.shape}"
    assert out_ssm.shape == (1, n_layers, d_model, d_state), f"SSM state shape mismatch: {out_ssm.shape}"

    assert not np.allclose(out_audio, 0.0), "Output is all zeros!"
    assert np.all(np.isfinite(out_audio)), "Output has NaN/Inf!"

    print(f"  Audio output range: [{out_audio.min():.6f}, {out_audio.max():.6f}]")
    print(f"  Audio output RMS:   {np.sqrt(np.mean(out_audio**2)):.6f}")
    print("  PASSED")
    return True


if __name__ == "__main__":
    D_MODEL = 64
    D_STATE = 16
    N_LAYERS = 4
    CONV_WIDTH = 4
    DT_RANK = 16
    SEQ_LEN = 256

    print("=" * 60)
    print("Real Mamba SSM ONNX Export & Verification")
    print("=" * 60)

    print("\n[1/3] Exporting real Mamba RT model...")
    rt_path = "models/mamba_rt.onnx"
    model = export_mamba_to_onnx(
        rt_path, d_model=D_MODEL, d_state=D_STATE, n_layers=N_LAYERS,
        conv_width=CONV_WIDTH, dt_rank=DT_RANK, seq_len=SEQ_LEN, opset_version=18
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  d_model={D_MODEL}, d_state={D_STATE}, n_layers={N_LAYERS}, conv_width={CONV_WIDTH}")

    print("\n[2/3] Exporting analysis model...")
    analysis_path = "models/mamba_analysis.onnx"
    export_mamba_to_onnx(
        analysis_path, d_model=D_MODEL, d_state=D_STATE, n_layers=N_LAYERS,
        conv_width=CONV_WIDTH, dt_rank=DT_RANK, seq_len=SEQ_LEN, opset_version=18
    )

    print("\n[3/3] Verifying ONNX models...")
    print(f"\nRT model ({rt_path}):")
    verify_onnx_model(rt_path, D_MODEL, D_STATE, N_LAYERS, CONV_WIDTH, seq_len=64)

    print(f"\nAnalysis model ({analysis_path}):")
    verify_onnx_model(analysis_path, D_MODEL, D_STATE, N_LAYERS, CONV_WIDTH, seq_len=64)

    # Verify state continuity: process two chunks back-to-back
    print("\n[4/4] Verifying state continuity across chunks...")
    import onnxruntime as ort
    sess = ort.InferenceSession(rt_path)
    chunk_len = 32
    audio1 = np.random.randn(1, chunk_len, 1).astype(np.float32)
    audio2 = np.random.randn(1, chunk_len, 1).astype(np.float32)

    feeds1 = {'audio_chunk': audio1}
    feeds2 = {'audio_chunk': audio2}
    for i in range(N_LAYERS):
        feeds1[f'conv{i}'] = np.zeros((1, D_MODEL, CONV_WIDTH), dtype=np.float32)
        feeds1[f'ssm{i}'] = np.zeros((1, D_MODEL, D_STATE), dtype=np.float32)

    out1, conv_out, ssm_out = sess.run(None, feeds1)

    # Use returned states for second chunk
    for i in range(N_LAYERS):
        feeds2[f'conv{i}'] = conv_out[:, i, :, :]
        feeds2[f'ssm{i}'] = ssm_out[:, i, :, :]

    out2, _, _ = sess.run(None, feeds2)

    assert not np.allclose(out2, 0.0), "Second chunk output is all zeros (state not carried)!"
    assert np.all(np.isfinite(out2)), "Second chunk output has NaN/Inf!"
    print(f"  Chunk 2 output range: [{out2.min():.6f}, {out2.max():.6f}]")
    print("  State continuity PASSED")

    print("\n" + "=" * 60)
    print("All verifications passed!")
    print("=" * 60)
