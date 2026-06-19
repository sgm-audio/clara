"""
CLARA Mamba RT Model — Parallel (training) and Sequential (export) variants.
Reconstructed from verify_training_pipeline.py interface contract.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Selective Scan Step ──

class SelectiveScanStep(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

    def forward(self, x_t, dt_t, A, B_t, C_t, D, h_in):
        dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
        dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
        h_out = dA * h_in + dB * x_t.unsqueeze(-1)
        y_t = torch.sum(C_t.unsqueeze(1) * h_out, dim=-1) + D * x_t
        return y_t, h_out


# ── SSM Core ──

class MambaSSMCore(nn.Module):
    def __init__(self, d_model, d_state, dt_rank, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.log_A_real = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1))
            * (math.log(dt_max / dt_min) / max(d_state - 1, 1))
            - torch.log(torch.tensor(dt_max, dtype=torch.float32))
        )
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
        xz = self.x_proj(x)
        dt_latent = xz[:, :, :self.dt_rank]
        B = xz[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C = xz[:, :, self.dt_rank + self.d_state:]
        dt = F.softplus(self.dt_proj(dt_latent))
        A = -torch.exp(self.log_A_real)
        return dt, A, B, C, self.D


# ── FiLM Conditioning ──

class CondFiLM(nn.Module):
    def __init__(self, d_model, cond_dim=8):
        super().__init__()
        self.proj = nn.Linear(cond_dim, d_model * 2, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond):
        gb = self.proj(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta


# ── Single Mamba Layer ──

class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state=16, conv_width=4, dt_rank=16, cond_dim=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.conv_width = conv_width
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, conv_width, padding=conv_width-1, groups=d_model)
        self.ssm_proj = MambaSSMCore(d_model, d_state, dt_rank)
        self.scan_step = SelectiveScanStep(d_model, d_state)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.cond_film = CondFiLM(d_model, cond_dim)

    def forward(self, x, conv_state, ssm_state, cond):
        """x: (B, L, D) or (B, 1, D) for single step"""
        seq_len = x.shape[1]
        gamma, beta = self.cond_film(cond)
        x = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        projected = self.in_proj(x)
        x_ssm = projected[:, :, :self.d_model]
        x_gate = projected[:, :, self.d_model:]
        x_conv_seq = x_ssm.transpose(1, 2)
        full_seq = torch.cat([conv_state, x_conv_seq], dim=-1)
        conv_full = self.conv1d(full_seq)
        start_idx = self.conv_width - 1
        conv_out = conv_full[:, :, start_idx:start_idx + seq_len]
        conv_out = conv_out.transpose(1, 2)
        conv_out = conv_out * F.silu(x_gate)
        dt, A, B, C, D = self.ssm_proj(conv_out)
        h = ssm_state
        ys = []
        for t in range(seq_len):
            y_t, h = self.scan_step(conv_out[:, t, :], dt[:, t, :], A, B[:, t, :], C[:, t, :], D, h)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        output = self.out_proj(y)
        new_conv_state = x_conv_seq[:, :, -self.conv_width:]
        new_ssm_state = h
        return output, new_conv_state, new_ssm_state


# ── Input / Output Projections ──

class InputProjection(nn.Module):
    def __init__(self, d_model, cond_dim=8):
        super().__init__()
        self.proj = nn.Linear(1 + cond_dim, d_model)

    def forward(self, audio, cond):
        cond_expanded = cond.unsqueeze(1).expand(-1, audio.shape[1], -1)
        x = torch.cat([audio, cond_expanded], dim=-1)
        return self.proj(x)


class OutputProjection(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.proj(x)


# ────────────────────────────────────────────────────────────────────────────
# MambaRTModelParallel — for training, processes all timesteps at once
#   forward(audio, conv0-3, ssm0-3, cond) -> (out, new_conv, new_ssm)
#   Can optionally chunk into 64-sample sub-blocks matching C++ engine
# ────────────────────────────────────────────────────────────────────────────

class MambaRTModelParallel(nn.Module):
    """Parallel-scan Mamba RT model for training. All timesteps at once.

    When chunk_size > 0, processes audio in fixed-size sub-blocks with
    state carryover — matching exactly what the C++ inference engine does.
    This improves gradient stability and enables longer training clips.
    """

    def __init__(self, d_model=64, d_state=16, n_layers=4, conv_width=4, dt_rank=16, cond_dim=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.conv_width = conv_width
        self.input_proj = InputProjection(d_model, cond_dim)
        self.layers = nn.ModuleList([
            MambaLayer(d_model, d_state, conv_width, dt_rank, cond_dim)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = OutputProjection(d_model)

    def _forward_full(self, audio_chunk, convs, ssms, cond):
        """Process full sequence in one pass (fast for short clips)."""
        x = self.input_proj(audio_chunk, cond)
        new_convs = []
        new_ssms = []
        for i, (layer, cs, ss) in enumerate(zip(self.layers, convs, ssms)):
            # cond passed per-layer for FiLM
            x, nc, ns = layer(x, cs, ss, cond)
            new_convs.append(nc)
            new_ssms.append(ns)
        x = self.norm(x)
        output_audio = self.output_proj(x)
        new_conv_state = torch.stack(new_convs, dim=1)
        new_ssm_state = torch.stack(new_ssms, dim=1)
        return output_audio, new_conv_state, new_ssm_state

    def _forward_chunked(self, audio_chunk, convs, ssms, cond, chunk_size=64):
        """Process in fixed-size chunks matching C++ engine's kSubBlockSize.

        Each chunk of up to chunk_size samples is fed through the full model,
        and conv/ssm state is carried between chunks. Partial final chunks
        are zero-padded and valid output is extracted.
        """
        B, total_len, _ = audio_chunk.shape
        device = audio_chunk.device

        # Convert per-layer state tuples to lists for mutation
        conv_list = list(convs)
        ssm_list = list(ssms)
        output_chunks = []

        for offset in range(0, total_len, chunk_size):
            end = min(offset + chunk_size, total_len)
            chunk = audio_chunk[:, offset:end, :]  # (B, chunk_len, 1)

            # Zero-pad final chunk if needed
            pad_len = chunk_size - chunk.shape[1]
            if pad_len > 0:
                pad = torch.zeros(B, pad_len, 1, device=device)
                chunk = torch.cat([chunk, pad], dim=1)

            # Forward through the model
            x = self.input_proj(chunk, cond)
            for i, (layer, cs, ss) in enumerate(zip(self.layers, conv_list, ssm_list)):
                x, nc, ns = layer(x, cs, ss, cond)
                conv_list[i] = nc
                ssm_list[i] = ns
            x = self.norm(x)
            out_chunk = self.output_proj(x)  # (B, chunk_size, 1)

            # Trim padding
            if pad_len > 0:
                out_chunk = out_chunk[:, :chunk.shape[1] - pad_len, :]
            output_chunks.append(out_chunk)

        output_audio = torch.cat(output_chunks, dim=1)
        new_conv_state = torch.stack(conv_list, dim=1)
        new_ssm_state = torch.stack(ssm_list, dim=1)
        return output_audio, new_conv_state, new_ssm_state

    def forward(self, audio_chunk, conv0, conv1, conv2, conv3,
                ssm0, ssm1, ssm2, ssm3, cond, chunk_size=0):
        """Forward pass with optional sub-block chunking.

        Args:
            audio_chunk: (B, seq_len, 1)
            conv0..conv3: (B, d_model, conv_width) per-layer conv state
            ssm0..ssm3: (B, d_model, d_state) per-layer SSM state
            cond: (B, cond_dim) conditioning vector
            chunk_size: 0 = process full sequence, 64 = match C++ kSubBlockSize
        """
        convs = [conv0, conv1, conv2, conv3]
        ssms = [ssm0, ssm1, ssm2, ssm3]

        if chunk_size > 0:
            return self._forward_chunked(audio_chunk, convs, ssms, cond, chunk_size)
        else:
            return self._forward_full(audio_chunk, convs, ssms, cond)


# ────────────────────────────────────────────────────────────────────────────
# MambaRTModelSequential — for ONNX export, single step at a time
#   forward(audio_chunk, conv0-3, ssm0-3, cond) -> (out, new_conv, new_ssm)
#   audio_chunk: (B, 1, 1) — one sample at a time
# ────────────────────────────────────────────────────────────────────────────

class MambaRTModelSequential(nn.Module):
    """Single-step Mamba RT model for ONNX export with state carryover."""

    def __init__(self, d_model=64, d_state=16, n_layers=4, conv_width=4, dt_rank=16, cond_dim=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.conv_width = conv_width
        self.input_proj = InputProjection(d_model, cond_dim)
        self.layers = nn.ModuleList([
            MambaLayer(d_model, d_state, conv_width, dt_rank, cond_dim)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = OutputProjection(d_model)

    def forward(self, audio_chunk, conv0, conv1, conv2, conv3,
                ssm0, ssm1, ssm2, ssm3, cond):
        """Single step: audio_chunk (B, 1, 1), returns (B, 1, 1) + states."""
        x = self.input_proj(audio_chunk, cond)
        convs = [conv0, conv1, conv2, conv3]
        ssms = [ssm0, ssm1, ssm2, ssm3]
        new_convs = []
        new_ssms = []
        for layer, cs, ss in zip(self.layers, convs, ssms):
            x, nc, ns = layer(x, cs, ss, cond)
            new_convs.append(nc)
            new_ssms.append(ns)
        x = self.norm(x)
        output_audio = self.output_proj(x)
        new_conv_state = torch.stack(new_convs, dim=1)
        new_ssm_state = torch.stack(new_ssms, dim=1)
        return output_audio, new_conv_state, new_ssm_state


# ── Helpers ──

def make_zero_states(batch_size, d_model, d_state, n_layers, conv_width=4):
    """Returns (convs_tuple, ssms_tuple) of zero initial states."""
    convs = tuple(torch.zeros(batch_size, d_model, conv_width) for _ in range(n_layers))
    ssms = tuple(torch.zeros(batch_size, d_model, d_state) for _ in range(n_layers))
    return convs, ssms


def make_zero_cond(batch_size, cond_dim=8):
    """Returns zero conditioning vector."""
    return torch.zeros(batch_size, cond_dim)


def create_training_model(d_model=64, d_state=16, n_layers=4, conv_width=4, dt_rank=16, cond_dim=8):
    """Factory function for the training (parallel) model."""
    return MambaRTModelParallel(d_model, d_state, n_layers, conv_width, dt_rank, cond_dim)


def create_export_model(d_model=64, d_state=16, n_layers=4, conv_width=4, dt_rank=16, cond_dim=8):
    """Factory function for the export (sequential) model."""
    return MambaRTModelSequential(d_model, d_state, n_layers, conv_width, dt_rank, cond_dim)
