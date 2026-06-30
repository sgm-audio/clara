#!/usr/bin/env python3
"""
CLARA Mamba-3 Training — Radeon 860M (Ryzen AI 7 350) Local Runner
======================================================================
Optimized for AMD RDNA 3.5 iGPU with shared system memory (30GB).

ROCm 6.x required. Run the ROCm setup block first if not installed.
Falls back to CPU if ROCm is not available.

Usage:
    python train_local_radeon.py --data /path/to/musdb18hq/train
"""

import os
import sys
import math
import time
import struct
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# ── ROCm environment setup (must be before torch import) ──────────────────────
# AMD RDNA 3.5 (gfx1151) — set before torch loads
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')
os.environ.setdefault('ROCBLAS_TF32_TENSOR_OP_MATH', '1')
os.environ.setdefault('MIOPEN_ENABLE_LOGGING', '0')
os.environ.setdefault('MIOPEN_ENABLE_LOGGING_MASK', '0')
os.environ.setdefault('HSA_ENABLE_SDMA', '0')       # iGPU: disable SDMA for less overhead
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')

# ═══════════════════════════════════════════════════════════════════════════════
# Detect device
# ═══════════════════════════════════════════════════════════════════════════════

def get_device():
    """Auto-detect best available device: HIP (AMD), CUDA (NVIDIA), CPU."""
    # Check for ROCm/HIP
    try:
        import torch
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            if torch.cuda.is_available():
                return torch.device('cuda')
    except (ImportError, AttributeError):
        pass

    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device('cuda')
    except ImportError:
        pass

    return torch.device('cpu')

def is_amd_rocm():
    try:
        import torch
        return hasattr(torch.version, 'hip') and torch.version.hip is not None
    except:
        return False

def is_amd_rdna3():
    """Check if AMD GPU is RDNA 3 or newer (gfx11xx or gfx12xx)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        # HIP device props don't expose arch name directly; check compute capability
        # RDNA 3 = sm_90 equivalent, RDNA 3.5 (gfx1151) = sm_90+
        cap = torch.cuda.get_device_capability(0)
        # AMD gmx can't be mapped directly; use torch.hip if available
        return cap[0] >= 9  # conservative: RDNA3+ maps to sm_90+
    except:
        return False

def auto_batch_size():
    """Auto-tune batch size based on available system RAM."""
    try:
        import torch
        if torch.cuda.is_available():
            # Use 75% of available GPU memory, minimum 4
            props = torch.cuda.get_device_properties(0)
            mem_gb = props.total_memory / (1024**3)
            # ~1.6M params, d_model=128, 6 layers, 16K state per layer
            # Rough estimate: 4GB per batch slot for activations
            batch = max(4, min(16, int(mem_gb * 0.3)))
            return batch
    except:
        pass
    # CPU: use more workers, moderate batch (fits in RAM)
    import psutil
    mem_gb = psutil.virtual_memory().available / (1024**3)
    return max(4, min(8, int(mem_gb * 0.15)))

# ═══════════════════════════════════════════════════════════════════════════════
# Imports (after ROCm env)
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# ── Verify AMD ROCm ───────────────────────────────────────────────────────────
DEVICE = get_device()
IS_AMD = is_amd_rocm()

print(f"=== CLARA Mamba-3 Local Training (Radeon 860M) ===")
print(f"Device: {DEVICE}")
if IS_AMD:
    print(f"  AMD ROCm detected — RDNA 3.5 optimized")
    try:
        import torch
        print(f"  torch version: {torch.__version__} (HIP)")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except:
        pass
elif DEVICE.type == 'cuda':
    print(f"  NVIDIA CUDA detected")
    try:
        cap = torch.cuda.get_device_capability(0)
        print(f"  Compute: sm_{cap[0]}{cap[1]}")
    except:
        pass
else:
    print(f"  CPU fallback — ROCm not detected")
    print(f"  Install ROCm 6.x for AMD GPU acceleration:")
    print(f"    wget https://repo.radeon.com/rocm/rocm.gpg -O /tmp/rocm.gpg && "
          f"sudo alien --to-deb /tmp/rocm.gpg 2>/dev/null || true")
    print(f"    # Then: pip install torch --index-url https://download.pytorch.org/whl/rocm6.1")
    print(f"    # Or: pip install torch --index-url https://download.pytorch.org/whl/rocm6.2")

# ═══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════════════════

class MultiResolutionSTFTLoss(nn.Module):
    """3-resolution STFT magnitude loss. Lightweight — runs on iGPU without issue."""
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512],
                 win_lengths=[512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        for i, n_fft in enumerate(fft_sizes):
            self.register_buffer(f'window_{i}', torch.hann_window(win_lengths[i]))

    def forward(self, est, target):
        total = torch.zeros(1, device=est.device)
        for i, (n_fft, hop, win_len) in enumerate(
                zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            window = getattr(self, f'window_{i}')
            pad = max(0, n_fft - est.shape[-1])
            if pad:
                est = F.pad(est, (0, pad))
                target = F.pad(target, (0, pad))
            est_spec = torch.stft(est, n_fft, hop_length=hop, win_length=win_len,
                                  window=window, return_complex=True)
            tgt_spec = torch.stft(target, n_fft, hop_length=hop, win_length=win_len,
                                  window=window, return_complex=True)
            est_mag = torch.abs(est_spec).clamp(min=1e-7).log10()
            tgt_mag = torch.abs(tgt_spec).clamp(min=1e-7).log10()
            total = total + F.l1_loss(est_mag, tgt_mag)
        return total / len(self.fft_sizes)


class CombinedLoss(nn.Module):
    """MSE + multi-resolution STFT. Default 50/50 for Stage 0 identity training."""
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
        return {'total': total, 'mse': loss_mse, 'stft': loss_stft}


# ═══════════════════════════════════════════════════════════════════════════════
# Model (same Mamba-3 as C++ engine)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Mamba3Config:
    d_model:       int = 128
    d_inner:       int = 256
    d_state:       int = 32
    nheads:        int = 4
    headdim:       int = 64
    mimo_rank:     int = 4
    num_rope_angles: int = 16
    n_layers:      int = 6
    cond_dim:      int = 128

    @property
    def in_proj_dim(self):
        return (self.d_inner
                + self.nheads * self.headdim
                + self.nheads * self.mimo_rank * self.d_state
                + self.nheads * self.mimo_rank * self.d_state
                + self.nheads
                + self.nheads
                + self.nheads
                + self.nheads * self.num_rope_angles)


class Mamba3Layer(nn.Module):
    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_model, cfg.in_proj_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(cfg.nheads))
        self.B_bias = nn.Parameter(torch.zeros(cfg.nheads, cfg.mimo_rank, cfg.d_state))
        self.C_bias = nn.Parameter(torch.zeros(cfg.nheads, cfg.mimo_rank, cfg.d_state))
        self.B_norm_weight = nn.Parameter(torch.ones(cfg.d_state))
        self.C_norm_weight = nn.Parameter(torch.ones(cfg.d_state))
        self.mimo_x = nn.Parameter(torch.randn(cfg.nheads, cfg.mimo_rank, cfg.headdim) * 0.02)
        self.mimo_z = nn.Parameter(torch.randn(cfg.nheads, cfg.mimo_rank, cfg.headdim) * 0.02)
        self.mimo_o = nn.Parameter(torch.randn(cfg.nheads, cfg.mimo_rank, cfg.headdim) * 0.02)
        self.D_proj = nn.Parameter(torch.zeros(cfg.nheads))
        self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=False)
        self.film_proj = nn.Linear(cfg.cond_dim, 2 * cfg.d_model, bias=False)
        self.dt_film_proj = nn.Linear(cfg.cond_dim, 2 * cfg.nheads, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def apply_rope(self, x, angles):
        D = x.shape[-1]
        half = D // 2
        x_even, x_odd = x[..., :half], x[..., half:]
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        while cos_a.dim() < x_even.dim():
            cos_a = cos_a.unsqueeze(1)
            sin_a = sin_a.unsqueeze(1)
        rot_even = x_even * cos_a - x_odd * sin_a
        rot_odd  = x_even * sin_a + x_odd * cos_a
        return torch.cat([rot_even, rot_odd], dim=-1)

    @staticmethod
    def _ssm_scan(A_bar, update, h0):
        h = h0
        hs = []
        for t in range(A_bar.shape[1]):
            h = A_bar[:, t, :, None, None] * h + update[:, t, :, None, :]
            hs.append(h)
        return torch.stack(hs, dim=1)

    def forward_batch(self, x, state, cond=None, reset_state=True):
        B, T, D = x.shape
        cfg = self.cfg

        if cond is not None:
            gb = self.film_proj(cond)
            gamma, beta = gb.chunk(2, dim=-1)
            x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        proj = self.in_proj(x)
        idx = 0
        z         = proj[..., :idx + cfg.d_inner];                                  idx += cfg.d_inner
        x_h       = proj[..., idx:idx + cfg.nheads * cfg.headdim];                  idx += cfg.nheads * cfg.headdim
        B_raw     = proj[..., idx:idx + cfg.nheads * cfg.mimo_rank * cfg.d_state]; idx += cfg.nheads * cfg.mimo_rank * cfg.d_state
        C_raw     = proj[..., idx:idx + cfg.nheads * cfg.mimo_rank * cfg.d_state]; idx += cfg.nheads * cfg.mimo_rank * cfg.d_state
        dd_dt     = proj[..., idx:idx + cfg.nheads];                                idx += cfg.nheads
        dd_A      = proj[..., idx:idx + cfg.nheads];                                idx += cfg.nheads
        trap      = proj[..., idx:idx + cfg.nheads];                                idx += cfg.nheads
        angles_raw = proj[..., idx:idx + cfg.nheads * cfg.num_rope_angles]

        x_h = x_h.view(B, T, cfg.nheads, cfg.headdim)
        B_raw = B_raw.view(B, T, cfg.nheads, cfg.mimo_rank, cfg.d_state)
        C_raw = C_raw.view(B, T, cfg.nheads, cfg.mimo_rank, cfg.d_state)
        angles_raw = angles_raw.view(B, T, cfg.nheads, cfg.num_rope_angles)

        B_raw = B_raw + self.B_bias.unsqueeze(0).unsqueeze(0)
        C_raw = C_raw + self.C_bias.unsqueeze(0).unsqueeze(0)

        dt_h = F.softplus(dd_dt + self.dt_bias)
        A_h  = -F.softplus(dd_A)
        if cond is not None:
            dt_cond = self.dt_film_proj(cond)
            dt_g, dt_b = dt_cond.chunk(2, dim=-1)
            dt_h = F.softplus(dt_h + dt_g.unsqueeze(1) + dt_b.unsqueeze(1))

        half_a = A_h * dt_h * 0.5
        denom  = 1.0 - half_a
        A_bar_all  = (1.0 + half_a) / denom
        B_scale_all = dt_h / denom

        angle_cumul = torch.cumsum(angles_raw, dim=1)
        angle_full = state['angle_state'].unsqueeze(1) + angle_cumul
        state['angle_state'] = angle_full[:, -1]

        cos_a = torch.cos(angle_full).unsqueeze(3)
        sin_a = torch.sin(angle_full).unsqueeze(3)

        B_normed = B_raw * self.B_norm_weight.view(1, 1, 1, 1, cfg.d_state)
        C_normed = C_raw * self.C_norm_weight.view(1, 1, 1, 1, cfg.d_state)

        B_even = B_normed[..., 0::2]
        B_odd  = B_normed[..., 1::2]
        C_even = C_normed[..., 0::2]
        C_odd  = C_normed[..., 1::2]

        B_rot_even = B_even * cos_a - B_odd * sin_a
        B_rot_odd  = B_even * sin_a + B_odd * cos_a
        B_rot = torch.stack([B_rot_even, B_rot_odd], dim=-1)
        B_rot = B_rot.reshape(B, T, cfg.nheads, cfg.mimo_rank, cfg.d_state)

        C_rot_even = C_even * cos_a - C_odd * sin_a
        C_rot_odd  = C_even * sin_a + C_odd * cos_a
        C_rot = torch.stack([C_rot_even, C_rot_odd], dim=-1)
        C_rot = C_rot.reshape(B, T, cfg.nheads, cfg.mimo_rank, cfg.d_state)

        x_slots = torch.einsum('bthd,hrd->bthr', x_h, self.mimo_x)
        B_contrib = (B_scale_all.unsqueeze(-1).unsqueeze(-1)
                     * x_slots.unsqueeze(-1) * B_rot)
        update = B_contrib.sum(dim=3)

        if reset_state:
            h0 = torch.zeros(B, cfg.nheads, cfg.headdim, cfg.d_state, device=x.device)
        else:
            h0 = state.get('ssm_real', torch.zeros(B, cfg.nheads, cfg.headdim, cfg.d_state, device=x.device))
        h_all = self._ssm_scan(A_bar_all, update, h0)
        state['ssm_real'] = h_all[:, -1]

        y_head = torch.einsum('bthrs,bthds->bthd', C_rot, h_all)

        z_gated = z * torch.sigmoid(z)
        z_heads = z_gated.view(B, T, cfg.nheads, cfg.headdim)
        z_slots = torch.einsum('bthd,hrd->bthr', z_heads, self.mimo_z)
        z_gate_val = z_slots.sum(dim=-1, keepdim=True)
        y_gated = y_head * z_gate_val

        y_flat = y_gated.reshape(B, T, cfg.d_inner)
        out = self.out_proj(y_flat)

        d_contrib = (self.D_proj.view(1, 1, cfg.nheads, 1) * x_h).sum(dim=(-2, -1))
        out = out + d_contrib.unsqueeze(-1) * 0.01

        o_slots = torch.einsum('bthd,hrd->bthr', y_gated, self.mimo_o)
        o_contrib = o_slots.sum(dim=(-2, -1))
        out = out + o_contrib.unsqueeze(-1) * 0.01

        return out + x


class Mamba3Model(nn.Module):
    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([Mamba3Layer(cfg) for _ in range(cfg.n_layers)])

    def forward(self, audio, cond=None):
        B, T = audio.shape
        cfg = self.cfg
        x = torch.zeros(B, T, cfg.d_model, device=audio.device, dtype=torch.float32)
        x[..., 0] = audio
        for layer in self.layers:
            state = {
                'ssm_real': torch.zeros(B, cfg.nheads, cfg.headdim, cfg.d_state, device=audio.device),
                'angle_state': torch.zeros(B, cfg.nheads, cfg.num_rope_angles, device=audio.device),
            }
            x = layer.forward_batch(x, state, cond)
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + 1e-5)
        return x[..., 0]

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class AudioDataset(Dataset):
    def __init__(self, root, chunk_samples=2048, sr=44100):
        self.root = Path(root)
        self.chunk_samples = chunk_samples
        self.sr = sr
        self.files = sorted(self.root.rglob("*.wav"))
        if not self.files:
            raise FileNotFoundError(f"No .wav files found in {root}")
        print(f"  Found {len(self.files)} audio files")

    def __len__(self):
        return len(self.files) * 10

    def __getitem__(self, idx):
        file_idx = idx % len(self.files)
        path = self.files[file_idx]
        T = self.chunk_samples
        try:
            info = torchaudio.info(str(path))
            n_frames = info.num_frames
            file_sr = info.sample_rate
        except Exception:
            return torch.zeros(T), torch.zeros(T)

        if file_sr != self.sr:
            return torch.zeros(T), torch.zeros(T)

        if n_frames < T:
            try:
                audio, _ = torchaudio.load(str(path))
                audio = audio.mean(0)
            except:
                return torch.zeros(T), torch.zeros(T)
            repeats = T // audio.numel() + 1
            audio = audio.repeat(repeats)
            start = int(torch.randint(0, max(1, audio.numel() - T), (1,)).item())
            chunk = audio[start:start + T]
        else:
            start = int(torch.randint(0, max(1, n_frames - T), (1,)).item())
            try:
                audio, _ = torchaudio.load(str(path), frame_offset=start, num_frames=T)
                chunk = audio.mean(0)
            except:
                return torch.zeros(T), torch.zeros(T)

        peak = chunk.abs().max()
        if peak > 0:
            chunk = chunk / peak * 0.8
        return chunk.float(), chunk.float()


class SyntheticDataset(Dataset):
    def __init__(self, chunk_samples=2048, size=1000):
        self.chunk_samples = chunk_samples
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return (torch.randn(self.chunk_samples) * 0.3,
                torch.randn(self.chunk_samples) * 0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = DEVICE
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU memory: {mem_gb:.1f} GB")

    # System RAM info
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"  System RAM: {mem.total / (1024**3):.0f} GB total, "
              f"{mem.available / (1024**3):.1f} GB available")
    except ImportError:
        pass

    if args.data:
        data_dir = Path(args.data)
        if data_dir.exists():
            n_files = len(sorted(data_dir.rglob("*.wav")))
            print(f"Dataset: {data_dir} ({n_files} WAV files)")
        else:
            print(f"WARNING: dataset path does not exist: {data_dir}")

    cfg = Mamba3Config()
    model = Mamba3Model(cfg).to(device)
    print(f"\nModel: {model.num_params():,} params")
    print(f"  Layers: {cfg.n_layers}, d_model={cfg.d_model}, d_state={cfg.d_state}")
    print(f"  MIMO: R={cfg.mimo_rank}, heads={cfg.nheads}, headdim={cfg.headdim}")

    # AMD-specific: compile for RDNA 3.5
    if IS_AMD and device.type == 'cuda':
        print(f"\n[AMD ROCm] Applying RDNA 3.5 optimizations...")
        try:
            model = torch.compile(model, dynamic=True, fullgraph=False)
            print(f"  torch.compile enabled (dynamic, RDNA 3.5)")
        except Exception as e:
            print(f"  torch.compile not available: {e}")
    elif device.type == 'cuda' and not IS_AMD:
        # NVIDIA: try torch.compile too
        try:
            model = torch.compile(model, dynamic=True, fullgraph=False)
            print(f"\ntorch.compile enabled (dynamic)")
        except Exception as e:
            print(f"torch.compile not available: {e}")

    # Dataset
    try:
        dataset = AudioDataset(args.data, chunk_samples=args.chunk_samples)
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                            num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    except FileNotFoundError:
        print("  Falling back to synthetic noise (Stage 0 smoke test)")
        dataset = SyntheticDataset(args.chunk_samples)
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                            num_workers=args.num_workers)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def get_lr(step):
        if step < args.warmup:
            return args.lr * step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: get_lr(s) / args.lr)

    # Loss
    loss_fn = CombinedLoss(mse_weight=0.5, stft_weight=0.5)

    # Mixed precision (AMD RDNA 3.5 supports bf16 natively)
    use_amp = (device.type == 'cuda')
    amp_dtype = torch.bfloat16 if IS_AMD else torch.float16
    # ponytail: GradScaler placement — created only on cuda, None on cpu
    from torch.amp import GradScaler
    scaler = GradScaler('cuda', enabled=use_amp) if device.type == 'cuda' else None

    os.makedirs(args.save, exist_ok=True)

    model.train()
    step = 0
    t0 = time.time()
    log_t0 = time.time()
    log_loss_total, log_loss_mse, log_loss_stft = 0.0, 0.0, 0.0

    print(f"\nTraining: {args.steps} steps, batch={args.batch}, accum={args.grad_accum}")
    print(f"  LR: {args.lr}, warmup={args.warmup}, cosine decay")
    if use_amp:
        print(f"  AMP: {amp_dtype} (AMD RDNA 3.5)" if IS_AMD else f"  AMP: {amp_dtype}")

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            audio_in, audio_tgt = batch
            audio_in = audio_in.to(device)
            audio_tgt = audio_tgt.to(device)

            cond = torch.randn(args.batch, cfg.cond_dim, device=device) * 0.1

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    output = model(audio_in, cond)
                    losses = loss_fn(output, audio_tgt)
                    loss = losses['total'] / args.grad_accum
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(audio_in, cond)
                losses = loss_fn(output, audio_tgt)
                loss = losses['total'] / args.grad_accum
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

            scheduler.step()
            log_loss_total += losses['total'].item()
            log_loss_mse += losses['mse'].item()
            log_loss_stft += losses['stft'].item()
            step += 1

            # Logging
            if step % args.log_every == 0:
                elapsed = time.time() - log_t0
                sps = args.log_every / max(elapsed, 1e-6)
                current_lr = optimizer.param_groups[0]['lr']
                avg_total = log_loss_total / args.log_every
                avg_mse = log_loss_mse / args.log_every
                avg_stft = log_loss_stft / args.log_every
                print(f"  Step {step:6d}/{args.steps}  "
                      f"loss={avg_total:.4f}  mse={avg_mse:.4f}  stft={avg_stft:.4f}  "
                      f"lr={current_lr:.2e}  {sps:.1f} step/s")
                log_loss_total, log_loss_mse, log_loss_stft = 0.0, 0.0, 0.0
                log_t0 = time.time()

            # Checkpoint
            if step % args.ckpt_every == 0:
                ckpt_path = os.path.join(args.save, f"step_{step:06d}.pt")
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'cfg': cfg.__dict__,
                }, ckpt_path)
                print(f"  Saved: {ckpt_path}")

    # Final save
    final_path = os.path.join(args.save, "final.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'best_loss': log_loss_total / max(step % args.log_every, 1),
        'cfg': cfg.__dict__,
    }, final_path)
    print(f"\nFinal: {final_path}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")


# ═══════════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════════

def export_weights(checkpoint_path, output_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg_raw = ckpt.get('cfg', {})
    if isinstance(cfg_raw, dict):
        cfg = Mamba3Config(**{k: v for k, v in cfg_raw.items()
                              if k in Mamba3Config.__dataclass_fields__})
    else:
        cfg = cfg_raw
    state_dict = ckpt['model_state_dict']

    weight_data = bytearray()
    layer_keys = ['in_proj.weight', 'dt_bias', 'B_bias', 'C_bias',
                  'B_norm_weight', 'C_norm_weight', 'mimo_x', 'mimo_z',
                  'mimo_o', 'D_proj', 'out_proj.weight',
                  'film_proj.weight', 'dt_film_proj.weight']

    for l in range(cfg.n_layers):
        prefix = f'layers.{l}.'
        for key in layer_keys:
            param = state_dict[f'{prefix}{key}'].detach().cpu().numpy().astype(np.float32)
            weight_data.extend(param.tobytes())

    header = struct.pack('<12I',
        0x4D424D41, 2, cfg.d_state, cfg.d_model, cfg.d_inner,
        cfg.nheads, cfg.headdim, cfg.mimo_rank, cfg.n_layers,
        cfg.cond_dim, cfg.in_proj_dim, len(weight_data))
    header += b'\x00' * (64 - len(header))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(weight_data)
    print(f"Exported: {output_path} ({len(weight_data) + 64:,} bytes)")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CLARA Mamba-3 — Radeon 860M Local")
    parser.add_argument('--data', type=str, required=True,
                        help='Path to MUSDB18-HQ train folder')
    parser.add_argument('--save', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--batch', type=int, default=None,
                        help='Batch size (default: auto-detect based on available memory)')
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--clip-grad-norm', type=float, default=1.0)
    parser.add_argument('--chunk-samples', type=int, default=2048)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--ckpt-every', type=int, default=5000)
    parser.add_argument('--warmup', type=int, default=2000)
    parser.add_argument('--export', type=str, help='Export checkpoint to model.bin')
    parser.add_argument('--output', type=str, default='models/model_trained.bin')
    args = parser.parse_args()

    # Auto batch size if not specified
    if args.batch is None:
        args.batch = auto_batch_size()
        print(f"[Auto] Batch size set to: {args.batch}")

    if args.export:
        export_weights(args.export, args.output)
    else:
        train(args)


if __name__ == '__main__':
    main()
