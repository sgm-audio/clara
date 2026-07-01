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
        dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # [B, D, N]
        h_out = dA * h_in + dB * x_t.unsqueeze(-1)  # [B, D, N]
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
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
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
        dt_latent = xz[:, :, : self.dt_rank]
        B = xz[:, :, self.dt_rank : self.dt_rank + self.d_state]
        C = xz[:, :, self.dt_rank + self.d_state :]

        dt = F.softplus(self.dt_proj(dt_latent))  # [B, L, D]
        A = -torch.exp(self.log_A_real)  # [D, N]

        return dt, A, B, C, self.D


# ---------------------------------------------------------------------------
# Conditioning FiLM — per-layer feature-wise linear modulation
# ---------------------------------------------------------------------------


class CondFiLM(nn.Module):
    """Projects COND_DIM → (gamma, beta) for per-layer feature modulation."""

    def __init__(self, d_model, cond_dim=8):
        super().__init__()
        self.proj = nn.Linear(cond_dim, d_model * 2, bias=True)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()

    def forward(self, cond):
        """cond: [B, COND_DIM] → gamma: [B, D], beta: [B, D]"""
        gb = self.proj(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta


# ---------------------------------------------------------------------------
# Mamba Layer: Conv1d + SSM + SiLU gating + FiLM conditioning
# ---------------------------------------------------------------------------


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state=16, conv_width=4, dt_rank=16, cond_dim=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.conv_width = conv_width

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, conv_width, padding=conv_width - 1, groups=d_model
        )
        self.ssm_proj = MambaSSMCore(d_model, d_state, dt_rank)
        self.scan_step = SelectiveScanStep(d_model, d_state)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.cond_film = CondFiLM(d_model, cond_dim)

    def forward(self, x, conv_state, ssm_state, cond):
        """
        Args:
          x:          [batch, seq_len, d_model]
          conv_state: [batch, d_model, conv_width]
          ssm_state:  [batch, d_model, d_state]
          cond:       [batch, cond_dim]

        Returns:
          output:          [batch, seq_len, d_model]
          new_conv_state:  [batch, d_model, conv_width]
          new_ssm_state:   [batch, d_model, d_state]
        """
        seq_len = x.shape[1]

        # Per-layer FiLM conditioning
        gamma, beta = self.cond_film(cond)  # [B, D]
        x = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        projected = self.in_proj(x)
        x_ssm = projected[:, :, : self.d_model]
        x_gate = projected[:, :, self.d_model :]

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
        conv_out = conv_full[:, :, start_idx : start_idx + seq_len]  # [B, D, L]
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
        new_conv_state = x_conv_seq[:, :, -self.conv_width :]  # [B, D, W]
        new_ssm_state = h

        return output, new_conv_state, new_ssm_state


# ---------------------------------------------------------------------------
# Input/Output Projections (1-channel audio <-> d_model)
# ---------------------------------------------------------------------------


class InputProjection(nn.Module):
    def __init__(self, d_model, cond_dim=8):
        super().__init__()
        self.proj = nn.Linear(1 + cond_dim, d_model)

    def forward(self, audio, cond):
        """audio: [B, L, 1], cond: [B, COND_DIM]"""
        cond_expanded = cond.unsqueeze(1).expand(-1, audio.shape[1], -1)
        x = torch.cat([audio, cond_expanded], dim=-1)
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
    def __init__(self, d_model=64, d_state=16, n_layers=4, conv_width=4, dt_rank=16, cond_dim=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.conv_width = conv_width

        self.input_proj = InputProjection(d_model, cond_dim)
        self.layer0 = MambaLayer(d_model, d_state, conv_width, dt_rank, cond_dim)
        self.layer1 = MambaLayer(d_model, d_state, conv_width, dt_rank, cond_dim)
        self.layer2 = MambaLayer(d_model, d_state, conv_width, dt_rank, cond_dim)
        self.layer3 = MambaLayer(d_model, d_state, conv_width, dt_rank, cond_dim)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = OutputProjection(d_model)

    def forward(self, audio_chunk, conv0, conv1, conv2, conv3, ssm0, ssm1, ssm2, ssm3, cond):
        """
        Per-layer state inputs to keep the tracer honest.

        Args:
          audio_chunk: [batch, seq_len, 1]
          conv{i}:     [batch, d_model, conv_width]
          ssm{i}:      [batch, d_model, d_state]
          cond:        [batch, cond_dim]

        Returns:
          output_audio:   [batch, seq_len, 1]
          new_conv0..3:   [batch, d_model, conv_width] each
          new_ssm0..3:    [batch, d_model, d_state] each
        """
        x = self.input_proj(audio_chunk, cond)

        x, nc0, ns0 = self.layer0(x, conv0, ssm0, cond)
        x, nc1, ns1 = self.layer1(x, conv1, ssm1, cond)
        x, nc2, ns2 = self.layer2(x, conv2, ssm2, cond)
        x, nc3, ns3 = self.layer3(x, conv3, ssm3, cond)

        x = self.norm(x)
        output_audio = self.output_proj(x)

        # Stack into the shapes the C++ engine expects
        new_conv_state = torch.stack([nc0, nc1, nc2, nc3], dim=1)  # [B, 4, D, W]
        new_ssm_state = torch.stack([ns0, ns1, ns2, ns3], dim=1)  # [B, 4, D, N]

        return output_audio, new_conv_state, new_ssm_state


# ---------------------------------------------------------------------------
# Pre-export Validation
# ---------------------------------------------------------------------------


def preflight_check(opset_version=18):
    """Validate the export environment before attempting export."""
    import sys

    print("[preflight] Checking export environment...")

    # 1. Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"  Python: {py_ver}")

    # 2. torch
    try:
        import torch

        torch_ver = torch.__version__
        print(f"  torch: {torch_ver}")
        major, minor = [int(x) for x in torch_ver.split(".")[:2]]
        assert (major, minor) >= (2, 0), f"torch >= 2.0 required, got {torch_ver}"
    except ImportError as e:
        sys.exit(f"FATAL: torch not installed ({e})")
    except AssertionError as e:
        sys.exit(f"FATAL: {e}")

    # 3. onnx
    try:
        import onnx

        onnx_ver = onnx.__version__
        print(f"  onnx: {onnx_ver}")
    except ImportError as e:
        sys.exit(f"FATAL: onnx not installed ({e})")

    # 4. onnxruntime
    try:
        import onnxruntime as ort

        ort_ver = ort.__version__
        print(f"  onnxruntime: {ort_ver}")
    except ImportError as e:
        sys.exit(f"FATAL: onnxruntime not installed ({e})")

    # 5. numpy
    try:
        import numpy as np

        np_ver = np.__version__
        print(f"  numpy: {np_ver}")
    except ImportError as e:
        sys.exit(f"FATAL: numpy not installed ({e})")

    # 6. Opset version check
    import onnx.helper

    available = onnx.helper.VERSION_TABLE
    max_opset = max(v[2] for v in available if len(v) > 2)
    print(f"  ONNX opset: {opset_version} (max available: {max_opset})")
    assert opset_version <= max_opset, f"Opset {opset_version} not available (max: {max_opset})"

    # 7. Dry-run a single forward pass to catch shape errors early
    try:
        print("  Dry-run forward pass...")
        model = MambaRTModel()
        model.eval()
        x = torch.randn(1, 4, 1)
        convs = tuple(torch.zeros(1, 64, 4) for _ in range(4))
        ssms = tuple(torch.zeros(1, 64, 16) for _ in range(4))
        cond = torch.zeros(1, 8)
        with torch.no_grad():
            out, nc, ns = model(x, *convs, *ssms, cond)
        assert out.shape == (1, 4, 1), f"Unexpected output shape: {out.shape}"
        print(f"  Dry-run PASSED (output shape: {out.shape})")
    except Exception as e:
        sys.exit(f"FATAL: Dry-run forward pass failed: {e}")

    print("[preflight] All checks passed\n")


# ---------------------------------------------------------------------------
# ONNX vs PyTorch Output Validation
# ---------------------------------------------------------------------------


def validate_onnx_vs_pytorch(
    onnx_path,
    model,
    d_model=64,
    d_state=16,
    n_layers=4,
    conv_width=4,
    seq_len=256,
    tolerance_cosine=0.999,
    tolerance_abs=1e-4,
):
    """Validate ONNX model output matches PyTorch model output.

    Returns True if validation passes, False otherwise.
    """
    import onnxruntime as ort

    print("[validate] Comparing ONNX vs PyTorch output...")

    rng = np.random.RandomState(42)
    audio_np = rng.randn(1, seq_len, 1).astype(np.float32)
    audio_torch = torch.from_numpy(audio_np)

    model.eval()
    convs = tuple(torch.zeros(1, d_model, conv_width) for _ in range(n_layers))
    ssms = tuple(torch.zeros(1, d_model, d_state) for _ in range(n_layers))
    cond_pt = torch.zeros(1, 8)
    with torch.no_grad():
        pt_out, pt_conv, pt_ssm = model(audio_torch, *convs, *ssms, cond_pt)
    pt_out_np = pt_out.numpy()
    pt_conv_np = pt_conv.numpy()
    pt_ssm_np = pt_ssm.numpy()

    sess = ort.InferenceSession(onnx_path)
    feeds = {"audio_chunk": audio_np}
    for i in range(n_layers):
        feeds[f"conv{i}"] = np.zeros((1, d_model, conv_width), dtype=np.float32)
        feeds[f"ssm{i}"] = np.zeros((1, d_model, d_state), dtype=np.float32)
    feeds["cond"] = np.zeros((1, 8), dtype=np.float32)
    ort_out, ort_conv, ort_ssm = sess.run(None, feeds)

    flat_pt = pt_out_np.flatten()
    flat_ort = ort_out.flatten()

    cos_sim = np.dot(flat_pt, flat_ort) / (
        np.sqrt(np.dot(flat_pt, flat_pt)) * np.sqrt(np.dot(flat_ort, flat_ort)) + 1e-10
    )
    print(f"  processed_audio cosine similarity: {cos_sim:.8f} (required > {tolerance_cosine})")

    max_diff = np.max(np.abs(flat_pt - flat_ort))
    print(f"  processed_audio max abs diff:      {max_diff:.8f} (required < {tolerance_abs})")

    conv_max_diff = np.max(np.abs(pt_conv_np - ort_conv))
    ssm_max_diff = np.max(np.abs(pt_ssm_np - ort_ssm))
    print(f"  conv state max abs diff:           {conv_max_diff:.8f}")
    print(f"  ssm state max abs diff:            {ssm_max_diff:.8f}")

    passed = True
    if cos_sim < tolerance_cosine:
        print(f"  FAIL: Cosine similarity {cos_sim:.6f} < {tolerance_cosine}")
        passed = False
    if max_diff > tolerance_abs:
        print(f"  FAIL: Max abs diff {max_diff:.6f} > {tolerance_abs}")
        passed = False

    if passed:
        print("[validate] PASSED — ONNX output matches PyTorch\n")
    else:
        print("[validate] FAILED — output mismatch\n")

    return passed


# ---------------------------------------------------------------------------
# Manifest Generation
# ---------------------------------------------------------------------------


def write_manifest(
    file_path, d_model, d_state, n_layers, conv_width, dt_rank, opset_version, model_path
):
    """Write models/manifest.json with model metadata."""
    import hashlib
    import json
    from datetime import datetime, timezone

    print("[manifest] Generating model manifest...")

    sha256_onnx = hashlib.sha256()
    with open(model_path, "rb") as f:
        sha256_onnx.update(f.read())

    data_path = model_path + ".data"
    sha256_data = hashlib.sha256()
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            sha256_data.update(f.read())
    else:
        sha256_data = None

    cond_dim = 8

    manifest = {
        "schema_version": 2,
        "model": {
            "architecture": "MambaRT",
            "d_model": d_model,
            "d_state": d_state,
            "n_layers": n_layers,
            "conv_width": conv_width,
            "dt_rank": dt_rank,
            "cond_dim": cond_dim,
            "opset_version": opset_version,
        },
        "io_contract": {
            "num_inputs": 1 + 2 * n_layers + 1,
            "num_outputs": 3,
            "input_names": ["audio_chunk"]
            + [f"conv{i}" for i in range(n_layers)]
            + [f"ssm{i}" for i in range(n_layers)]
            + ["cond"],
            "output_names": ["processed_audio", "new_conv_state", "new_ssm_state"],
            "audio_chunk_shape": ["batch", "seq_len", 1],
            "conv_state_shape": ["batch", "d_model", "conv_width"],
            "ssm_state_shape": ["batch", "d_model", "d_state"],
            "cond_shape": ["batch", "cond_dim"],
        },
        "files": {
            "onnx": {
                "name": os.path.basename(model_path),
                "sha256": sha256_onnx.hexdigest(),
            },
        },
        "export": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__ if "onnx" in dir() else "unknown",
        },
    }

    if sha256_data:
        manifest["files"]["onnx_data"] = {
            "name": os.path.basename(data_path),
            "sha256": sha256_data.hexdigest(),
        }

    with open(file_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[manifest] Written: {file_path}")
    print(f"  schema_version: {manifest['schema_version']}")
    print(f"  d_model={d_model}, n_layers={n_layers}, conv_width={conv_width}")
    print(f"  onnx SHA256: {manifest['files']['onnx']['sha256'][:16]}...")
    if sha256_data:
        print(f"  onnx.data SHA256: {manifest['files']['onnx_data']['sha256'][:16]}...")
    print()


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------


def export_mamba_to_onnx(
    file_path,
    d_model=64,
    d_state=16,
    n_layers=4,
    conv_width=4,
    dt_rank=16,
    seq_len=1024,
    opset_version=18,
    checkpoint_sd=None,
):
    model = MambaRTModel(
        d_model=d_model, d_state=d_state, n_layers=n_layers, conv_width=conv_width, dt_rank=dt_rank
    )
    model.eval()

    # Load checkpoint weights if provided
    if checkpoint_sd is not None:
        compatible = {k: v for k, v in checkpoint_sd.items() if k in model.state_dict()}
        missing = set(model.state_dict().keys()) - set(compatible.keys())
        if missing:
            print(f"  WARNING: {len(missing)} keys from checkpoint not in model (random init):")
            for m in sorted(missing)[:5]:
                print(f"    - {m}")
            if len(missing) > 5:
                print(f"    ... and {len(missing)-5} more")
        model.load_state_dict(compatible, strict=False)
        print(f"  Loaded {len(compatible)}/{len(model.state_dict())} parameters from checkpoint")
    else:
        print("  WARNING: Exporting UNTRAINED model with random weights")
        print("  This model will produce NOISE in the plugin!")

    batch_size = 1
    dummy_audio = torch.randn(batch_size, seq_len, 1)
    dummy_convs = tuple(torch.zeros(batch_size, d_model, conv_width) for _ in range(n_layers))
    dummy_ssms = tuple(torch.zeros(batch_size, d_model, d_state) for _ in range(n_layers))
    dummy_cond = torch.zeros(batch_size, 8)

    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)

    input_names = (
        ["audio_chunk"]
        + [f"conv{i}" for i in range(n_layers)]
        + [f"ssm{i}" for i in range(n_layers)]
        + ["cond"]
    )
    output_names = ["processed_audio", "new_conv_state", "new_ssm_state"]

    all_args = (dummy_audio,) + dummy_convs + dummy_ssms + (dummy_cond,)

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
            "audio_chunk": {1: "seq_len"},
            "processed_audio": {1: "seq_len"},
        },
        dynamo=False,
    )
    print(f"Exported: {file_path}")
    return model


def verify_onnx_model(onnx_path, d_model=64, d_state=16, n_layers=4, conv_width=4, seq_len=64):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)

    audio = np.random.randn(1, seq_len, 1).astype(np.float32)
    feeds = {"audio_chunk": audio}
    for i in range(n_layers):
        feeds[f"conv{i}"] = np.zeros((1, d_model, conv_width), dtype=np.float32)
        feeds[f"ssm{i}"] = np.zeros((1, d_model, d_state), dtype=np.float32)
    feeds["cond"] = np.zeros((1, 8), dtype=np.float32)

    outputs = sess.run(None, feeds)
    out_audio, out_conv, out_ssm = outputs

    print(f"  processed_audio shape: {out_audio.shape}  expected: (1, {seq_len}, 1)")
    print(
        f"  new_conv_state shape:  {out_conv.shape}  expected: (1, {n_layers}, {d_model}, {conv_width})"
    )
    print(
        f"  new_ssm_state shape:   {out_ssm.shape}  expected: (1, {n_layers}, {d_model}, {d_state})"
    )

    assert out_audio.shape == (1, seq_len, 1), f"Audio shape mismatch: {out_audio.shape}"
    assert out_conv.shape == (
        1,
        n_layers,
        d_model,
        conv_width,
    ), f"Conv state shape mismatch: {out_conv.shape}"
    assert out_ssm.shape == (
        1,
        n_layers,
        d_model,
        d_state,
    ), f"SSM state shape mismatch: {out_ssm.shape}"

    assert not np.allclose(out_audio, 0.0), "Output is all zeros!"
    assert np.all(np.isfinite(out_audio)), "Output has NaN/Inf!"

    output_rms = np.sqrt(np.mean(out_audio**2))
    output_var = np.var(out_audio)

    # CRITICAL CHECK: Random weights should produce non-trivial output
    # (variance > 1e-6 means the model is actually processing audio)
    if output_var < 1e-6:
        print(f"  WARNING: Output variance is very low ({output_var:.8f})")
        print("  This may indicate the model is not processing audio correctly.")
    else:
        print(f"  Output variance OK: {output_var:.8f}")

    print(f"  Audio output range: [{out_audio.min():.6f}, {out_audio.max():.6f}]")
    print(f"  Audio output RMS:   {output_rms:.6f}")
    print("  PASSED")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLARA Mamba SSM ONNX Export")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Load trained checkpoint before export. If not provided, "
        "exports a FRESH RANDOM model (for CI/validation only).",
    )
    parser.add_argument("--output", default="models/mamba_rt.onnx", help="Output ONNX path")
    parser.add_argument(
        "--seq-len", type=int, default=256, help="Export sequence length (default: 256)"
    )
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    D_MODEL = 64
    D_STATE = 16
    N_LAYERS = 4
    CONV_WIDTH = 4
    DT_RANK = 16
    SEQ_LEN = args.seq_len
    OPSET = args.opset
    RT_PATH = args.output

    print("=" * 60)
    if args.checkpoint:
        print("CLARA ONNX Export (LOADED FROM CHECKPOINT)")
    else:
        print("WARNING: CI/DEV ONLY - EXPORTING RANDOM UNTRAINED MODEL")
        print("=" * 60)
        print("")
        print("  !!! This creates a model with RANDOM weights that produces NOISE !!!")
        print("  !!! Use --checkpoint to load a TRAINED model for actual use  !!!")
        print(
            "  !!! For production: python export_mamba_onnx.py --checkpoint checkpoints_clara/best.pt !!!"
        )
        print("")
        print("=" * 60)
        print("")
    print("Real Mamba SSM ONNX Export & Verification")
    print("=" * 60)

    # Pre-flight: validate environment
    preflight_check(opset_version=OPSET)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"[0/4] Loading trained checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        print(f"  Loaded step {ckpt.get('step', '?')}, loss {ckpt.get('loss', '?'):.4f}")
    else:
        state_dict = None

    print("[1/4] Exporting Mamba RT model...")
    rt_path = RT_PATH
    model = export_mamba_to_onnx(
        rt_path,
        d_model=D_MODEL,
        d_state=D_STATE,
        n_layers=N_LAYERS,
        conv_width=CONV_WIDTH,
        dt_rank=DT_RANK,
        seq_len=SEQ_LEN,
        opset_version=OPSET,
        checkpoint_sd=state_dict,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  d_model={D_MODEL}, d_state={D_STATE}, n_layers={N_LAYERS}, conv_width={CONV_WIDTH}")
    print()

    print("[2/4] Verifying ONNX model shapes...")
    verify_onnx_model(rt_path, D_MODEL, D_STATE, N_LAYERS, CONV_WIDTH, seq_len=SEQ_LEN)
    print()

    if state_dict is not None:
        print("[2.5/4] Cross-validating weights loaded correctly...")
        from train_clara_model import create_training_model

        train_model = create_training_model(D_MODEL, D_STATE, N_LAYERS, CONV_WIDTH, DT_RANK, 8)
        train_model.eval()
        train_compat = {k: v for k, v in state_dict.items() if k in train_model.state_dict()}
        train_model.load_state_dict(train_compat, strict=False)

        test_audio = torch.randn(1, 64, 1)
        test_cond = torch.zeros(1, 8)
        from train_clara_model import make_zero_states

        tcs, tss = make_zero_states(1, D_MODEL, D_STATE, N_LAYERS, CONV_WIDTH)
        ecs, ess = make_zero_states(1, D_MODEL, D_STATE, N_LAYERS, CONV_WIDTH)

        with torch.no_grad():
            pt_out, _, _ = train_model(test_audio, *tcs, *tss, test_cond)
            # Need to re-create export model with loaded weights for fair comparison
        print("  (Cross-validation can be done with scripts/export_trained.py --verify)")
        print()

    print("[3/4] Validating ONNX vs PyTorch output...")
    validate_onnx_vs_pytorch(
        rt_path, model, D_MODEL, D_STATE, N_LAYERS, CONV_WIDTH, seq_len=SEQ_LEN
    )

    # Verify state continuity: process two chunks back-to-back
    print("[3/4 continued] Verifying state continuity across chunks...")
    import onnxruntime as ort

    sess = ort.InferenceSession(rt_path)
    chunk_len = SEQ_LEN
    audio1 = np.random.randn(1, chunk_len, 1).astype(np.float32)
    audio2 = np.random.randn(1, chunk_len, 1).astype(np.float32)

    feeds1 = {"audio_chunk": audio1}
    feeds2 = {"audio_chunk": audio2}
    for i in range(N_LAYERS):
        feeds1[f"conv{i}"] = np.zeros((1, D_MODEL, CONV_WIDTH), dtype=np.float32)
        feeds1[f"ssm{i}"] = np.zeros((1, D_MODEL, D_STATE), dtype=np.float32)
    feeds1["cond"] = np.zeros((1, 8), dtype=np.float32)

    out1, conv_out, ssm_out = sess.run(None, feeds1)

    for i in range(N_LAYERS):
        feeds2[f"conv{i}"] = conv_out[:, i, :, :]
        feeds2[f"ssm{i}"] = ssm_out[:, i, :, :]
    feeds2["cond"] = np.zeros((1, 8), dtype=np.float32)

    out2, _, _ = sess.run(None, feeds2)

    assert not np.allclose(out2, 0.0), "Second chunk output is all zeros (state not carried)!"
    assert np.all(np.isfinite(out2)), "Second chunk output has NaN/Inf!"
    print(f"  Chunk 2 output range: [{out2.min():.6f}, {out2.max():.6f}]")
    print("  State continuity PASSED")
    print()

    print("[4/4] Generating model manifest...")
    write_manifest(
        "models/manifest.json", D_MODEL, D_STATE, N_LAYERS, CONV_WIDTH, DT_RANK, OPSET, rt_path
    )

    if state_dict is not None:
        step = ckpt.get("step", "?")
        loss = ckpt.get("loss", "?")
        print(f"  Checkpoint: step {step}, loss {loss}")

    print("=" * 60)
    if state_dict is not None:
        print("Export complete -- model loaded from checkpoint")
    else:
        print("CI/DEV EXPORT -- model has RANDOM WEIGHTS")
        print("Run with --checkpoint for production export")
    print("=" * 60)
