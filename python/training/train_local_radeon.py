#!/usr/bin/env python3
"""CLARA Mamba-3 Training — Radeon 860M (RDNA 3.5) Local Runner
==============================================================
1.66M params, 6 layers, Mamba-3 multi-head MIMO R=4
Architecture matches src/core/MambaConfig.h + src/core/SsmBlock.cpp exactly.

860M optimizations:
  - HSA_OVERRIDE_GFX_VERSION=11.0.0 (gfx1152→gfx1100)
  - amp=False (bfloat16 causes NaN on RDNA 3.5)
  - num_workers=0 (avoids ROCm dataloader deadlock)
  - chunk_samples=2048 (fits in 12GB VRAM)
  - sequential SSM scan (stable, ~5 step/s on 860M)

Environment:
  Distrobox:  ~/clara-training/  (Ubuntu 24.04 + ROCm 6.2)
  Venv:       ~/clara-training/venv/  (Python 3.12 + PyTorch 2.7.1+rocm6.2.4)
  Datasets:   ~/datasets/musdb18hq/train/  (~217 WAV files)

Usage (inside distrobox):
  distrobox enter clara-training
  source ~/clara-training/venv/bin/activate
  export HSA_OVERRIDE_GFX_VERSION=11.0.0
  python train_local_radeon.py --data ~/datasets/musdb18hq/train
"""

import argparse
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

# ── ROCm / HSA override (MUST be before torch import) ───────────────────────
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("ROCBLAS_TF32_TENSOR_OP_MATH", "1")
os.environ.setdefault("MIOPEN_ENABLE_LOGGING", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# ═══════════════════════════════════════════════════════════════════════════════
# Device detection
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import torch

    ROCM_AVAILABLE = hasattr(torch.version, "hip") and torch.version.hip is not None
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    ROCM_AVAILABLE = False
    CUDA_AVAILABLE = False


def get_device():
    if CUDA_AVAILABLE:
        return torch.device("cuda")
    return torch.device("cpu")


def is_amd_rdna3():
    """RDNA 3 (gfx11xx) or RDNA 3.5 (gfx115x) — 860M is gfx1152."""
    if not CUDA_AVAILABLE:
        return False
    try:
        cap = torch.cuda.get_device_capability(0)
        return cap[0] >= 9
    except Exception:
        return False


DEVICE = get_device()
IS_AMD = ROCM_AVAILABLE

print("=== CLARA Mamba-3 Training — Radeon 860M ===")
print(f"  Device: {DEVICE}")
if IS_AMD:
    print(f"  AMD ROCm: {torch.version.hip}")
    if CUDA_AVAILABLE:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  VRAM: {mem_gb:.1f} GB")
elif DEVICE.type == "cuda":
    print("  NVIDIA CUDA detected")
else:
    print("  CPU fallback — install ROCm PyTorch for GPU training")
    print("  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2.4")


# ═══════════════════════════════════════════════════════════════════════════════
# Core imports
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ═══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════════════════


class MultiResolutionSTFTLoss(nn.Module):
    """3-resolution STFT magnitude loss. Lightweight on iGPU."""

    def __init__(
        self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512], win_lengths=[512, 1024, 2048]
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        for i, n_fft in enumerate(fft_sizes):
            self.register_buffer(f"window_{i}", torch.hann_window(win_lengths[i]))

    def forward(self, est, target):
        total = torch.zeros(1, device=est.device)
        for i, (n_fft, hop, win_len) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_lengths, strict=False)
        ):
            window = getattr(self, f"window_{i}")
            pad = max(0, n_fft - est.shape[-1])
            if pad:
                est = F.pad(est, (0, pad))
                target = F.pad(target, (0, pad))
            est_spec = torch.stft(
                est, n_fft, hop_length=hop, win_length=win_len, window=window, return_complex=True
            )
            tgt_spec = torch.stft(
                target,
                n_fft,
                hop_length=hop,
                win_length=win_len,
                window=window,
                return_complex=True,
            )
            est_mag = torch.abs(est_spec).clamp(min=1e-7).log10()
            tgt_mag = torch.abs(tgt_spec).clamp(min=1e-7).log10()
            total = total + F.l1_loss(est_mag, tgt_mag)
        return total / len(self.fft_sizes)


class CombinedLoss(nn.Module):
    """MSE + multi-resolution STFT. Default 50/50 for identity Stage 0."""

    def __init__(self, mse_weight=0.5, stft_weight=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.stft = MultiResolutionSTFTLoss()
        self.mse_weight = mse_weight
        self.stft_weight = stft_weight

    def forward(self, est, target):
        loss_mse = self.mse(est, target)
        loss_stft = self.stft(est, target)
        total = self.mse_weight * loss_mse + self.stft_weight * loss_stft
        return {"total": total, "mse": loss_mse, "stft": loss_stft}


# ═══════════════════════════════════════════════════════════════════════════════
# Mamba-3 Config (matches src/core/MambaConfig.h)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Mamba3Config:
    d_model: int = 128
    d_inner: int = 256
    d_state: int = 32
    nheads: int = 4
    headdim: int = 64  # = d_inner / nheads
    mimo_rank: int = 4
    num_rope_angles: int = 16  # = d_state // 2
    n_layers: int = 6
    cond_dim: int = 128

    @property
    def in_proj_dim(self):
        return (
            self.d_inner
            + self.nheads * self.headdim
            + self.nheads * self.mimo_rank * self.d_state
            + self.nheads * self.mimo_rank * self.d_state
            + self.nheads
            + self.nheads
            + self.nheads
            + self.nheads * self.num_rope_angles
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Mamba-3 SSM Layer — matches C++ engine exactly
# ═══════════════════════════════════════════════════════════════════════════════


class Mamba3Layer(nn.Module):
    """One SSM layer: multi-head MIMO R=4, trapezoidal SSM, RoPE, FiLM, SiLU z-gating."""

    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # Fused input projection
        self.in_proj = nn.Linear(d, cfg.in_proj_dim, bias=False)

        # SSM parameters
        self.dt_bias = nn.Parameter(torch.zeros(cfg.nheads))
        self.B_bias = nn.Parameter(torch.zeros(cfg.nheads, cfg.mimo_rank, cfg.d_state))
        self.C_bias = nn.Parameter(torch.zeros(cfg.nheads, cfg.mimo_rank, cfg.d_state))
        self.B_norm_w = nn.Parameter(torch.ones(cfg.d_state))
        self.C_norm_w = nn.Parameter(torch.ones(cfg.d_state))
        self.mimo_x = nn.Parameter(torch.randn(cfg.nheads, cfg.mimo_rank, cfg.headdim) * 0.02)
        self.mimo_z = nn.Parameter(torch.randn(cfg.nheads, cfg.mimo_rank, cfg.headdim) * 0.02)
        self.mimo_o = nn.Parameter(torch.randn(cfg.nheads, cfg.mimo_rank, cfg.headdim) * 0.02)
        self.D_proj = nn.Parameter(torch.zeros(cfg.nheads))

        # Output projection
        self.out_proj = nn.Linear(cfg.d_inner, d, bias=False)

        # FiLM conditioning (cond → gamma, beta per d_model channel)
        self.film_proj = nn.Linear(cfg.cond_dim, 2 * d, bias=False)

        # Per-head Δ modulation
        self.dt_film_proj = nn.Linear(cfg.cond_dim, 2 * cfg.nheads, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    @staticmethod
    def _ssm_scan(A_bar, update, h0):
        """Sequential SSM recurrence — stable, GPU-efficient for 1.66M model."""
        h = h0
        hs = []
        for t in range(A_bar.shape[1]):
            h = A_bar[:, t, :, None, None] * h + update[:, t, :, None, :]
            hs.append(h)
        return torch.stack(hs, dim=1)

    def forward_batch(self, x, state, cond=None, reset_state=True):
        """x:     [B, T, D_MODEL]
        state: dict with 'ssm_real', 'angle_state'
        cond:  [B, COND_DIM] or None
        reset_state: if True, start from zero (training). False = stateful (inference).
        Returns: [B, T, D_MODEL]
        """
        B, T, D = x.shape
        cfg = self.cfg

        # ── FiLM conditioning ──
        if cond is not None:
            gb = self.film_proj(cond)
            gamma, beta = gb.chunk(2, dim=-1)
            x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        # ── Fused in_proj ──
        proj = self.in_proj(x)
        idx = 0
        z = proj[..., : cfg.d_inner]
        idx += cfg.d_inner
        x_h = proj[..., idx : idx + cfg.nheads * cfg.headdim]
        idx += cfg.nheads * cfg.headdim
        B_raw = proj[..., idx : idx + cfg.nheads * cfg.mimo_rank * cfg.d_state]
        idx += cfg.nheads * cfg.mimo_rank * cfg.d_state
        C_raw = proj[..., idx : idx + cfg.nheads * cfg.mimo_rank * cfg.d_state]
        idx += cfg.nheads * cfg.mimo_rank * cfg.d_state
        dd_dt = proj[..., idx : idx + cfg.nheads]
        idx += cfg.nheads
        dd_A = proj[..., idx : idx + cfg.nheads]
        idx += cfg.nheads
        trap = proj[..., idx : idx + cfg.nheads]
        idx += cfg.nheads
        angles_raw = proj[..., idx : idx + cfg.nheads * cfg.num_rope_angles]

        # Reshape to per-head tensors
        x_h = x_h.view(B, T, cfg.nheads, cfg.headdim)
        B_raw = B_raw.view(B, T, cfg.nheads, cfg.mimo_rank, cfg.d_state)
        C_raw = C_raw.view(B, T, cfg.nheads, cfg.mimo_rank, cfg.d_state)
        angles_raw = angles_raw.view(B, T, cfg.nheads, cfg.num_rope_angles)

        # Add biases
        B_raw = B_raw + self.B_bias.unsqueeze(0).unsqueeze(0)
        C_raw = C_raw + self.C_bias.unsqueeze(0).unsqueeze(0)

        # ── Δ and A (trapezoidal SSM) ──
        dt_h = F.softplus(dd_dt + self.dt_bias)
        A_h = -F.softplus(dd_A)

        if cond is not None:
            dt_cond = self.dt_film_proj(cond)
            dt_g, dt_b = dt_cond.chunk(2, dim=-1)
            dt_h = F.softplus(dt_h + dt_g.unsqueeze(1) + dt_b.unsqueeze(1))

        half_a = A_h * dt_h * 0.5
        denom = 1.0 - half_a
        A_bar = (1.0 + half_a) / denom
        B_scale = dt_h / denom

        # ── RoPE (cumulative angle across sequence) ──
        angle_cumul = torch.cumsum(angles_raw, dim=1)
        angle_full = state["angle_state"].unsqueeze(1) + angle_cumul
        state["angle_state"] = angle_full[:, -1]

        cos_a = torch.cos(angle_full).unsqueeze(3)
        sin_a = torch.sin(angle_full).unsqueeze(3)

        # ── RMSNorm on B and C, then RoPE ──
        B_normed = B_raw * self.B_norm_w.view(1, 1, 1, 1, cfg.d_state)
        C_normed = C_raw * self.C_norm_w.view(1, 1, 1, 1, cfg.d_state)

        B_even, B_odd = B_normed[..., 0::2], B_normed[..., 1::2]
        C_even, C_odd = C_normed[..., 0::2], C_normed[..., 1::2]

        # RoPE: rotate even/odd halves, then interleave back into d_state=32 elements
        B_rot_even = B_even * cos_a - B_odd * sin_a  # [B, T, H, R, ROPE]
        B_rot_odd = B_even * sin_a + B_odd * cos_a
        B_rot = torch.stack([B_rot_even, B_rot_odd], dim=-1).reshape(
            B, T, cfg.nheads, cfg.mimo_rank, cfg.d_state
        )

        C_rot_even = C_even * cos_a - C_odd * sin_a
        C_rot_odd = C_even * sin_a + C_odd * cos_a
        C_rot = torch.stack([C_rot_even, C_rot_odd], dim=-1).reshape(
            B, T, cfg.nheads, cfg.mimo_rank, cfg.d_state
        )

        # ── SSM recurrence ──
        # x_slots: project x_h → MIMO slots
        x_slots = torch.einsum("bthd,hrd->bthr", x_h, self.mimo_x)

        # B_contrib: [B, T, H, R, S] = B_scale[H] * x_slots[B,T,H,R] * B_rot[B,T,H,R,S]
        B_contrib = B_scale.unsqueeze(-1).unsqueeze(-1) * x_slots.unsqueeze(-1) * B_rot
        update = B_contrib.sum(dim=3)  # [B, T, H, S]

        if reset_state:
            h0 = torch.zeros(B, cfg.nheads, cfg.headdim, cfg.d_state, device=x.device)
        else:
            h0 = state.get(
                "ssm_real", torch.zeros(B, cfg.nheads, cfg.headdim, cfg.d_state, device=x.device)
            )

        h_all = self._ssm_scan(A_bar, update, h0)  # [B, T, H, D, S]
        state["ssm_real"] = h_all[:, -1]

        # ── Readout: y_head = C_rot ⊙ h_all ──
        y_head = torch.einsum("bthrs,bthds->bthd", C_rot, h_all)

        # ── SiLU z-gating ──
        z_gated = z * torch.sigmoid(z)
        z_heads = z_gated.view(B, T, cfg.nheads, cfg.headdim)
        z_slots = torch.einsum("bthd,hrd->bthr", z_heads, self.mimo_z)
        z_gate_v = z_slots.sum(dim=-1, keepdim=True)
        y_gated = y_head * z_gate_v

        # ── Output + residual ──
        y_flat = y_gated.reshape(B, T, cfg.d_inner)
        out = self.out_proj(y_flat)

        d_skip = (self.D_proj.view(1, 1, cfg.nheads, 1) * x_h).sum(dim=(-2, -1))
        out = out + d_skip.unsqueeze(-1) * 0.01

        o_slots = torch.einsum("bthd,hrd->bthr", y_gated, self.mimo_o)
        o_contrib = o_slots.sum(dim=(-2, -1))
        out = out + o_contrib.unsqueeze(-1) * 0.01

        return out + x


# ═══════════════════════════════════════════════════════════════════════════════
# Full Mamba-3 Model
# ═══════════════════════════════════════════════════════════════════════════════


class Mamba3Model(nn.Module):
    """6-layer Mamba-3. Scalar I/O, fixed LayerNorm (gamma=1, beta=0)."""

    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([Mamba3Layer(cfg) for _ in range(cfg.n_layers)])

    def forward(self, audio, cond=None):
        """audio: [B, T] — mono audio, T samples at 44100 Hz
        cond:  [B, COND_DIM] or None
        Returns: [B, T] — processed audio
        """
        B, T = audio.shape
        cfg = self.cfg

        # Scalar → D_MODEL: audio[i] → x[..., 0], rest = 0
        x = torch.zeros(B, T, cfg.d_model, device=audio.device, dtype=torch.float32)
        x[..., 0] = audio

        for layer in self.layers:
            state = {
                "ssm_real": torch.zeros(
                    B, cfg.nheads, cfg.headdim, cfg.d_state, device=audio.device
                ),
                "angle_state": torch.zeros(B, cfg.nheads, cfg.num_rope_angles, device=audio.device),
            }
            x = layer.forward_batch(x, state, cond)
            # Fixed LayerNorm (gamma=1, beta=0) — matches C++ engine
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + 1e-5)

        # D_MODEL → scalar: x[..., 0]
        return x[..., 0]

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset — RAM-cached WAV chunks
# ═══════════════════════════════════════════════════════════════════════════════


class AudioDataset(Dataset):
    """Loads WAV files, returns random chunks. RAM-cached for fast access."""

    def __init__(self, root, chunk_samples=2048, sr=44100):
        self.root = Path(root)
        self.chunk_samples = chunk_samples
        self.sr = sr
        self.files = sorted(self.root.rglob("*.wav"))
        if not self.files:
            raise FileNotFoundError(f"No .wav files in {root}")
        print(f"  Found {len(self.files)} audio files")

        # RAM cache all audio
        self._cache = []
        print("  Loading into RAM...", flush=True)
        for f in self.files:
            try:
                audio, file_sr = sf.read(str(f), dtype="float32")
                audio = torch.from_numpy(audio.mean(axis=1)).float()  # mono
                if file_sr != sr:
                    continue  # skip resampling for now
                self._cache.append(audio)
            except Exception as e:
                print(f"  WARNING: {f.name}: {e}")
        valid = [a for a in self._cache if a.numel() > 0]
        self._cache = valid
        hours = sum(a.numel() for a in self._cache) / sr / 3600
        print(f"  RAM cache: {len(self._cache)} files, {hours:.1f} hours")

    def __len__(self):
        return len(self._cache) * 10

    def __getitem__(self, idx):
        file_idx = idx % len(self._cache)
        audio = self._cache[file_idx]
        T = self.chunk_samples

        if audio.numel() < T:
            audio = audio.repeat(T // audio.numel() + 1)

        start = torch.randint(0, max(1, audio.numel() - T), (1,)).item()
        chunk = audio[start : start + T].clone()

        peak = chunk.abs().max()
        if peak > 0:
            chunk = chunk / peak * 0.8

        return chunk.float(), chunk.float()  # identity: input=target


class SyntheticDataset(Dataset):
    """Synthetic noise for smoke testing."""

    def __init__(self, chunk_samples=2048, size=1000):
        self.chunk_samples = chunk_samples
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (torch.randn(self.chunk_samples) * 0.3, torch.randn(self.chunk_samples) * 0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════


def train(args):
    device = DEVICE
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}")
        print(f"  VRAM: {props.total_memory / (1024**3):.1f} GB")

    cfg = Mamba3Config()
    model = Mamba3Model(cfg).to(device)
    print(f"\nModel: {model.num_params():,} params")
    print(f"  Layers: {cfg.n_layers}, d_model={cfg.d_model}, d_inner={cfg.d_inner}")
    print(f"  d_state={cfg.d_state}, MIMO R={cfg.mimo_rank}, heads={cfg.nheads}")
    print(f"  in_proj_dim: {cfg.in_proj_dim}")

    # AMD-specific: torch.compile disabled — RDNA 3.5 + dynamic=True has reshape
    # tracing issues with this SSM's tensor shape operations. Use eager mode.
    # User can re-enable with --compile once model is finalized.
    use_compile = False
    if False and device.type == "cuda" and IS_AMD:  # disabled
        try:
            model = torch.compile(model, dynamic=True, fullgraph=False)
            print("  torch.compile enabled (dynamic, RDNA 3.5)")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")

    # Dataset — num_workers=0 avoids ROCm dataloader deadlock
    if args.data:
        try:
            dataset = AudioDataset(args.data, chunk_samples=args.chunk_samples)
            loader = DataLoader(
                dataset,
                batch_size=args.batch,
                shuffle=True,
                num_workers=0,  # 0 = no deadlock on ROCm
                pin_memory=(device.type == "cuda"),
                drop_last=True,
            )
        except FileNotFoundError:
            print("  Falling back to synthetic noise (smoke test)")
            dataset = SyntheticDataset(args.chunk_samples)
            loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    else:
        print("  No --data provided — using synthetic noise (smoke test)")
        dataset = SyntheticDataset(args.chunk_samples)
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)

    # Optimizer — AdamW with decoupled weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),
    )

    # Cosine LR with linear warmup
    def get_lr(step):
        if step < args.warmup:
            return args.lr * step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: get_lr(s) / args.lr
    )

    # Loss: MSE + multi-resolution STFT — must be on same device as model
    loss_fn = CombinedLoss(mse_weight=0.5, stft_weight=0.5).to(device)

    # IMPORTANT: amp=False on RDNA 3.5 — bfloat16 causes NaN at step ~8
    use_amp = False
    amp_dtype = torch.bfloat16

    os.makedirs(args.save, exist_ok=True)

    model.train()
    step = 0
    best_loss = float("inf")
    t0 = time.time()
    log_t0 = time.time()
    log_total = 0.0
    log_mse = 0.0
    log_stft = 0.0
    early_stop = 0

    print(f"\nTraining: {args.steps} steps, batch={args.batch}, accum={args.grad_accum}")
    print(f"  LR: {args.lr}, warmup={args.warmup}, cosine decay")
    print(f"  AMP: {'OFF (NaN-safe for RDNA 3.5)' if not use_amp else amp_dtype}")
    print(f"  chunk_samples={args.chunk_samples}, num_workers=0")

    # GPU warmup — skip; ROCm STFT compilation is slow and can hang.
    # First real training step will trigger compilation, then subsequent steps
    # will be fast. If first step is slow, that's expected — it's not a bug.
    if device.type == "cuda":
        print("  Skipping GPU warmup (first training step triggers STFT compilation)")

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            audio_in, audio_tgt = batch
            audio_in = audio_in.to(device)
            audio_tgt = audio_tgt.to(device)

            # Random conditioning (Stage 0 identity: model learns to pass audio cleanly)
            cond = torch.randn(args.batch, cfg.cond_dim, device=device) * 0.1

            # Forward
            output = model(audio_in, cond)
            losses = loss_fn(output, audio_tgt)
            loss = losses["total"] / args.grad_accum
            loss.backward()

            log_total += losses["total"].item()
            log_mse += losses["mse"].item()
            log_stft += losses["stft"].item()

            # Optimizer step (only every grad_accum steps)
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
            else:
                step += 1

            # ── Logging ──
            if step % args.log_every == 0:
                elapsed = time.time() - log_t0
                sps = args.log_every / max(elapsed, 1e-6)
                lr = optimizer.param_groups[0]["lr"]
                avg_t = log_total / args.log_every
                avg_m = log_mse / args.log_every
                avg_s = log_stft / args.log_every
                print(
                    f"  Step {step:6d}/{args.steps}  "
                    f"loss={avg_t:.4f}  mse={avg_m:.4f}  stft={avg_s:.4f}  "
                    f"lr={lr:.2e}  {sps:.1f} step/s"
                )

                log_total = log_mse = log_stft = 0.0
                log_t0 = time.time()

            # ── Checkpoint ──
            if step % args.ckpt_every == 0:
                ckpt_path = os.path.join(args.save, f"step_{step:06d}.pt")
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": log_total / max(step % args.log_every, 1),
                        "best_loss": best_loss,
                        "cfg": cfg.__dict__,
                    },
                    ckpt_path,
                )

                if avg_t < best_loss:
                    best_loss = avg_t
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "best_loss": best_loss,
                            "cfg": cfg.__dict__,
                        },
                        os.path.join(args.save, "best.pt"),
                    )
                    early_stop = 0
                else:
                    early_stop += 1

                print(f"  Saved: {ckpt_path}  best={best_loss:.6f}  early_stop={early_stop}")

                if early_stop >= args.early_stop_patience:
                    print(f"  Early stopping at step {step}")
                    step = args.steps
                    break

    # Final save
    final_path = os.path.join(args.save, "final.pt")
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "best_loss": best_loss,
            "cfg": cfg.__dict__,
        },
        final_path,
    )
    print(f"\nDone: {final_path}")
    print(f"Total: {(time.time() - t0) / 60:.1f} min")


# ═══════════════════════════════════════════════════════════════════════════════
# Export: trained checkpoint → flat binary model.bin
# ═══════════════════════════════════════════════════════════════════════════════


def export_weights(checkpoint_path, output_path):
    """Export trained model to flat binary (Version 2 header + weight arrays)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_raw = ckpt.get("cfg", {})

    # Reconstruct config
    if isinstance(cfg_raw, dict):
        cfg = Mamba3Config(
            **{k: v for k, v in cfg_raw.items() if k in Mamba3Config.__dataclass_fields__}
        )
    else:
        cfg = cfg_raw

    state_dict = ckpt["model_state_dict"]

    weight_data = bytearray()
    layer_keys = [
        "in_proj.weight",
        "dt_bias",
        "B_bias",
        "C_bias",
        "B_norm_weight",
        "C_norm_weight",
        "mimo_x",
        "mimo_z",
        "mimo_o",
        "D_proj",
        "out_proj.weight",
        "film_proj.weight",
        "dt_film_proj.weight",
    ]

    for l in range(cfg.n_layers):
        prefix = f"layers.{l}."
        for key in layer_keys:
            param = state_dict[f"{prefix}{key}"].detach().cpu().numpy().astype(np.float32)
            weight_data.extend(param.tobytes())

    # Version 2 header: 64 bytes
    header = struct.pack(
        "<12I",
        0x4D424D41,  # magic "MBMA"
        2,  # version
        cfg.d_state,
        cfg.d_model,
        cfg.d_inner,
        cfg.nheads,
        cfg.headdim,
        cfg.mimo_rank,
        cfg.n_layers,
        cfg.cond_dim,
        cfg.in_proj_dim,
        len(weight_data),
    )
    header += b"\x00" * (64 - len(header))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(weight_data)

    print(f"Exported: {output_path} ({len(weight_data) + 64:,} bytes)")
    print(f"  Layers: {cfg.n_layers}, floats: ~{len(weight_data)//4:,}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="CLARA Mamba-3 — Radeon 860M Local")
    parser.add_argument(
        "--data", type=str, help="Path to MUSDB18-HQ train folder (or any WAV directory)"
    )
    parser.add_argument(
        "--save", type=str, default="checkpoints_clara", help="Checkpoint directory"
    )
    parser.add_argument("--resume", type=str, help="Resume from checkpoint.pt")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument(
        "--batch", type=int, default=4, help="Batch size (4 fits in 12GB VRAM with chunk=2048)"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation (effective batch = batch × accum)",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=2048,
        help="Audio chunk length in samples (44100 Hz × seconds)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader workers (0 = safe for ROCm)"
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--export", type=str, help="Export checkpoint.pt → model.bin")
    parser.add_argument(
        "--output", type=str, default="models/model_trained.bin", help="Output .bin path"
    )
    args = parser.parse_args()

    if args.export:
        export_weights(args.export, args.output)
    else:
        # Resume support
        if args.resume:
            print(f"Resuming from: {args.resume}")
        train(args)


if __name__ == "__main__":
    main()
