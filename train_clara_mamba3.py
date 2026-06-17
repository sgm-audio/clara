#!/usr/bin/env python3
"""
CLARA Mamba-3 Training Pipeline — train_clara_mamba3.py

Phase 8: Trains the 757K-param Mamba-3 model for real-time audio processing.
Matches reference_forward.py architecture bit-for-bit.

Architecture: d_model=128, d_inner=256, nheads=4, headdim=64,
               d_state=32, mimo_rank=4, n_layers=6, cond_dim=128

Stage 0 (SSL): Identity mapping — model learns to pass audio cleanly.
Stage 1 (per-effect): Supervised training for saturation/compression/denoising.

Usage:
    # Stage 0: identity pretraining
    python train_clara_mamba3.py --data ~/datasets/musdb18hq/train --device cuda --steps 100000

    # Export weights after training
    python train_clara_mamba3.py --export checkpoints/final.pt --output models/model_trained.bin

    # Resume from checkpoint
    python train_clara_mamba3.py --resume checkpoints/step_050000.pt --steps 100000
"""

import argparse
import math
import os
import struct
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════

class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT magnitude loss.

    Computes L1 loss on log-magnitude spectrograms at multiple
    FFT resolutions. Helps produce perceptually cleaner output
    than raw MSE alone.

    Ponytail: 3 resolutions, no complex loss, no window overlap tuning.
    """

    def __init__(self, fft_sizes: list = [512, 1024, 2048],
                 hop_sizes: list = [128, 256, 512],
                 win_lengths: list = [512, 1024, 2048],
                 window: str = 'hann_window'):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        # Pre-register window buffers
        for i, (n_fft, win_len) in enumerate(zip(fft_sizes, win_lengths)):
            window_tensor = torch.hann_window(win_len)
            self.register_buffer(f'window_{i}', window_tensor)

    def forward(self, est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """est, target: [B, T] mono audio."""
        total_loss = torch.zeros(1, device=est.device)
        for i, (n_fft, hop, win_len) in enumerate(
                zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            window = getattr(self, f'window_{i}')
            # Pad if too short for STFT
            min_len = n_fft
            pad = max(0, min_len - est.shape[-1])
            if pad > 0:
                est_pad = torch.nn.functional.pad(est, (0, pad))
                tgt_pad = torch.nn.functional.pad(target, (0, pad))
            else:
                est_pad, tgt_pad = est, target
            # STFT
            est_spec = torch.stft(est_pad, n_fft, hop_length=hop,
                                  win_length=win_len, window=window,
                                  return_complex=True)  # [B, F, T']
            tgt_spec = torch.stft(tgt_pad, n_fft, hop_length=hop,
                                  win_length=win_len, window=window,
                                  return_complex=True)
            # Log magnitude
            est_mag = torch.abs(est_spec).clamp(min=1e-7).log10()
            tgt_mag = torch.abs(tgt_spec).clamp(min=1e-7).log10()
            total_loss = total_loss + F.l1_loss(est_mag, tgt_mag)
        return total_loss / len(self.fft_sizes)


class CombinedLoss(nn.Module):
    """MSE + multi-resolution STFT loss.

    Default weights favor MSE for identity Stage 0.
    Adjust weights for later stages.
    """

    def __init__(self, mse_weight: float = 0.5, stft_weight: float = 0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.stft = MultiResolutionSTFTLoss()
        self.mse_weight = mse_weight
        self.stft_weight = stft_weight

    def forward(self, est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.mse_weight * self.mse(est, target)
                + self.stft_weight * self.stft(est, target))


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Mamba3Config:
    # Model dimensions (must match src/core/MambaConfig.h)
    d_model:       int = 128
    d_inner:       int = 256
    d_state:       int = 32
    nheads:        int = 4
    headdim:       int = 64     # = d_inner / nheads
    mimo_rank:     int = 4
    num_rope_angles: int = 16  # = d_state / 2
    n_layers:      int = 6
    cond_dim:      int = 128

    # in_proj output dimension
    @property
    def in_proj_dim(self) -> int:
        return (self.d_inner
                + self.nheads * self.headdim                     # x per head
                + self.nheads * self.mimo_rank * self.d_state   # B
                + self.nheads * self.mimo_rank * self.d_state   # C
                + self.nheads                                    # dd_dt
                + self.nheads                                    # dd_A
                + self.nheads                                    # trap
                + self.nheads * self.num_rope_angles)           # angles


# ═══════════════════════════════════════════════════════════════════
# Mamba-3 SSM Layer
# ═══════════════════════════════════════════════════════════════════

class Mamba3Layer(nn.Module):
    """One SSM layer with multi-head MIMO R=4, RoPE, FiLM, SiLU z-gating."""

    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.cfg = cfg
        d_model, d_inner, nheads, headdim = cfg.d_model, cfg.d_inner, cfg.nheads, cfg.headdim
        d_state, rank, n_rope = cfg.d_state, cfg.mimo_rank, cfg.num_rope_angles

        # 0. in_proj (fused): maps d_model → in_proj_dim
        self.in_proj = nn.Linear(d_model, cfg.in_proj_dim, bias=False)

        # 1-12: other weight params (no biases — all fused into in_proj)
        self.dt_bias = nn.Parameter(torch.zeros(nheads))
        self.B_bias  = nn.Parameter(torch.zeros(nheads, rank, d_state))
        self.C_bias  = nn.Parameter(torch.zeros(nheads, rank, d_state))
        self.B_norm_weight = nn.Parameter(torch.ones(d_state))
        self.C_norm_weight = nn.Parameter(torch.ones(d_state))
        self.mimo_x  = nn.Parameter(torch.randn(nheads, rank, headdim) * 0.02)
        self.mimo_z  = nn.Parameter(torch.randn(nheads, rank, headdim) * 0.02)
        self.mimo_o  = nn.Parameter(torch.randn(nheads, rank, headdim) * 0.02)
        self.D_proj  = nn.Parameter(torch.zeros(nheads))

        # 10. out_proj: maps d_inner → d_model
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # 11. film_proj: maps cond → 2 * d_model (gamma, beta)
        self.film_proj = nn.Linear(cfg.cond_dim, 2 * d_model, bias=False)

        # 12. dt_film_proj: maps cond → 2 * nheads (per-head dt modulation)
        self.dt_film_proj = nn.Linear(cfg.cond_dim, 2 * nheads, bias=False)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Xavier-uniform init for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def apply_rope(self, x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """RoPE rotation. x: [..., D], angles: [..., D/2]."""
        D = x.shape[-1]
        half = D // 2
        x_even, x_odd = x[..., :half], x[..., half:]
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        # Broadcast angles to match x's rank: add dims in the middle
        while cos_a.dim() < x_even.dim():
            cos_a = cos_a.unsqueeze(1)
            sin_a = sin_a.unsqueeze(1)
        rot_even = x_even * cos_a - x_odd * sin_a
        rot_odd  = x_even * sin_a + x_odd * cos_a
        return torch.cat([rot_even, rot_odd], dim=-1)

    @staticmethod
    def _ssm_scan(A_bar: torch.Tensor, update: torch.Tensor,
                  h0: torch.Tensor) -> torch.Tensor:
        """Associative scan for SSM recurrence h[t+1] = A[t]*h[t] + B[t].

        A_bar: [B, T, H]        — state transition per head
        update: [B, T, H, S]    — input contribution per head, state dim
        h0: [B, H, D, S]        — initial state (broadcast over D)
        Returns: [B, T, H, D, S] — h[t] for t=1..T (state after each step)
        """
        h = h0
        hs = []
        for t in range(A_bar.shape[1]):
            h = A_bar[:, t, :, None, None] * h + update[:, t, :, None, :]
            hs.append(h)
        return torch.stack(hs, dim=1)

    def forward_batch(self, x: torch.Tensor, state: dict,
                       cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process all T samples with fully vectorized ops (no Python sample loop).

        x:     [B, T, D_MODEL] — full sequence of embeddings
        state: dict with 'ssm_real', 'angle_state' — mutated in-place
        cond:  [B, COND_DIM] or None
        Returns: [B, T, D_MODEL]
        """
        B, T, D = x.shape
        cfg = self.cfg

        # ── Step 0: FiLM (batched over T) ──
        if cond is not None:
            gb = self.film_proj(cond)                              # [B, 2*D]
            gamma, beta = gb.chunk(2, dim=-1)                      # each [B, D]
            x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1) # [B, T, D]

        # ── Step 1: in_proj (batched over T) ──
        proj = self.in_proj(x)                                     # [B, T, IN_PROJ]
        idx = 0
        z     = proj[..., :idx + cfg.d_inner];                           idx += cfg.d_inner
        x_h   = proj[..., idx:idx + cfg.nheads * cfg.headdim];          idx += cfg.nheads * cfg.headdim
        B_raw = proj[..., idx:idx + cfg.nheads * cfg.mimo_rank * cfg.d_state]; idx += cfg.nheads * cfg.mimo_rank * cfg.d_state
        C_raw = proj[..., idx:idx + cfg.nheads * cfg.mimo_rank * cfg.d_state]; idx += cfg.nheads * cfg.mimo_rank * cfg.d_state
        dd_dt = proj[..., idx:idx + cfg.nheads];                        idx += cfg.nheads
        dd_A  = proj[..., idx:idx + cfg.nheads];                        idx += cfg.nheads
        trap  = proj[..., idx:idx + cfg.nheads];                        idx += cfg.nheads
        angles_raw = proj[..., idx:idx + cfg.nheads * cfg.num_rope_angles]

        # Reshape batched components: [B, T, ...]
        x_h = x_h.view(B, T, cfg.nheads, cfg.headdim)                  # [B, T, H, D]
        B_raw = B_raw.view(B, T, cfg.nheads, cfg.mimo_rank, cfg.d_state)
        C_raw = C_raw.view(B, T, cfg.nheads, cfg.mimo_rank, cfg.d_state)
        angles_raw = angles_raw.view(B, T, cfg.nheads, cfg.num_rope_angles)

        # Add biases
        B_raw = B_raw + self.B_bias.unsqueeze(0).unsqueeze(0)          # [B, T, H, R, S]
        C_raw = C_raw + self.C_bias.unsqueeze(0).unsqueeze(0)

        # ── Step 2: A and Δ (batched over T) ──
        dt_h = F.softplus(dd_dt + self.dt_bias)                       # [B, T, H]
        A_h  = -F.softplus(dd_A)                                       # [B, T, H]
        if cond is not None:
            dt_cond = self.dt_film_proj(cond)                          # [B, 2*H]
            dt_g, dt_b = dt_cond.chunk(2, dim=-1)
            dt_h = F.softplus(dt_h + dt_g.unsqueeze(1) + dt_b.unsqueeze(1))

        # Trapezoidal (batched over T)
        half_a = A_h * dt_h * 0.5
        denom  = 1.0 - half_a
        A_bar_all  = (1.0 + half_a) / denom                           # [B, T, H]
        B_scale_all = dt_h / denom                                    # [B, T, H]

        # ── Steps 3+4: Angles, Norm + RoPE (fully vectorized over T) ──

        # Cumulative angles: angle[t] = angle[0] + cumsum(angles_raw[:t])
        angle_cumul = torch.cumsum(angles_raw, dim=1)                  # [B, T, H, ROPE]
        # Full angle at each step (with initial state)
        angle_full = state['angle_state'].unsqueeze(1) + angle_cumul   # [B, T, H, ROPE]
        # Update final state (last angle)
        state['angle_state'] = angle_full[:, -1]

        # Precompute cos/sin for RoPE
        cos_a = torch.cos(angle_full).unsqueeze(3)                     # [B, T, H, 1, ROPE]
        sin_a = torch.sin(angle_full).unsqueeze(3)

        # Norm B/C (batched over T)
        B_normed = B_raw * self.B_norm_weight.view(1, 1, 1, 1, cfg.d_state)
        C_normed = C_raw * self.C_norm_weight.view(1, 1, 1, 1, cfg.d_state)

        # RoPE: split even/odd, rotate, interleave
        # B_normed: [B, T, H, R, S], split into even/odd along last dim
        B_even = B_normed[..., 0::2]  # [B, T, H, R, ROPE]
        B_odd  = B_normed[..., 1::2]
        C_even = C_normed[..., 0::2]
        C_odd  = C_normed[..., 1::2]

        B_rot_even = B_even * cos_a - B_odd * sin_a
        B_rot_odd  = B_even * sin_a + B_odd * cos_a
        # Interleave: [even, odd, even, odd, ...]
        B_rot = torch.stack([B_rot_even, B_rot_odd], dim=-1)           # [B, T, H, R, ROPE, 2]
        B_rot = B_rot.reshape(B, T, cfg.nheads, cfg.mimo_rank, cfg.d_state)

        C_rot_even = C_even * cos_a - C_odd * sin_a
        C_rot_odd  = C_even * sin_a + C_odd * cos_a
        C_rot = torch.stack([C_rot_even, C_rot_odd], dim=-1)
        C_rot = C_rot.reshape(B, T, cfg.nheads, cfg.mimo_rank, cfg.d_state)

        # ── Step 5: SSM recurrence (vectorized associative scan) ──

        # x_slots[t,h,r] = sum_d x_h[t,h,d] * mimo_x[h,r,d]
        x_slots = torch.einsum('bthd,hrd->bthr', x_h, self.mimo_x)    # [B, T, H, R]

        # B_contrib[t,h,r,s] = B_scale[t,h] * x_slots[t,h,r] * B_rot[t,h,r,s]
        B_contrib = (B_scale_all.unsqueeze(-1).unsqueeze(-1)
                     * x_slots.unsqueeze(-1)
                     * B_rot)                                         # [B, T, H, R, S]
        update = B_contrib.sum(dim=3)                                  # [B, T, H, S]

        # SSM recurrence h[t+1] = A[t] * h[t] + update[t]
        h0 = state['ssm_real'].detach().clone() if 'ssm_real' in state else None
        # For training, start from zero state each time (state is reset per forward)
        h0 = torch.zeros(B, cfg.nheads, cfg.headdim, cfg.d_state, device=x.device)
        h_all = self._ssm_scan(A_bar_all, update, h0)                  # [B, T, H, D, S]

        # Update state with final value (for stateful processing)
        state['ssm_real'] = h_all[:, -1]

        # Readout: y_head[t,h,d] = sum_r,s C_rot[t,h,r,s] * h[t+1,h,d,s]
        # h_all[t] = h[t+1] (state after sample t)
        y_head = torch.einsum('bthrs,bthds->bthd', C_rot, h_all)      # [B, T, H, D]

        # ── Step 6: z-gating (batched over T) ──
        z_gated = z * torch.sigmoid(z)                                 # [B, T, D_INNER]
        z_heads = z_gated.view(B, T, cfg.nheads, cfg.headdim)          # [B, T, H, D]
        z_slots = torch.einsum('bthd,hrd->bthr', z_heads, self.mimo_z) # [B, T, H, R]
        z_gate_val = z_slots.sum(dim=-1, keepdim=True)                 # [B, T, H, 1]
        y_gated = y_head * z_gate_val                                  # [B, T, H, D]

        # ── Step 7: Output + residual (batched over T) ──
        y_flat = y_gated.reshape(B, T, cfg.d_inner)                    # [B, T, D_INNER]
        out = self.out_proj(y_flat)                                     # [B, T, D_MODEL]

        # D skip (broadcast over T)
        d_contrib = (self.D_proj.view(1, 1, cfg.nheads, 1)
                     * x_h).sum(dim=(-2, -1))                          # [B, T]
        out = out + d_contrib.unsqueeze(-1) * 0.01

        # MIMO o-coupling
        o_slots = torch.einsum('bthd,hrd->bthr', y_gated, self.mimo_o) # [B, T, H, R]
        o_contrib = o_slots.sum(dim=(-2, -1))                          # [B, T]
        out = out + o_contrib.unsqueeze(-1) * 0.01

        # Residual
        out = out + x                                                    # [B, T, D]

        return out


# ═══════════════════════════════════════════════════════════════════
# Full Mamba-3 Model
# ═══════════════════════════════════════════════════════════════════

class Mamba3Model(nn.Module):
    """6-layer Mamba-3 model with per-layer LayerNorm."""

    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([Mamba3Layer(cfg) for _ in range(cfg.n_layers)])
        self.ln_gamma = nn.ParameterList([nn.Parameter(torch.ones(cfg.d_model)) for _ in range(cfg.n_layers)])
        self.ln_beta  = nn.ParameterList([nn.Parameter(torch.zeros(cfg.d_model)) for _ in range(cfg.n_layers)])
        self.input_proj = nn.Linear(1, cfg.d_model, bias=False)
        self.output_proj = nn.Linear(cfg.d_model, 1, bias=False)

    def forward(self, audio: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        audio: [B, T] — mono audio, T samples
        cond:  [B, COND_DIM] or None
        Returns: [B, T]

        Batched over T for all per-sample-independent ops (FiLM, in_proj, dt/A).
        Sequential loop over T for SSM recurrence only.
        """
        B, T = audio.shape
        cfg = self.cfg

        # Embed scalar → D_MODEL
        x = self.input_proj(audio.unsqueeze(-1))  # [B, T, D_MODEL]

        for layer_idx, layer in enumerate(self.layers):
            # Per-layer state (batch-aware, reset per layer)
            state = {
                'ssm_real': torch.zeros(B, cfg.nheads, cfg.headdim, cfg.d_state, device=audio.device),
                'angle_state': torch.zeros(B, cfg.nheads, cfg.num_rope_angles, device=audio.device),
            }
            x = layer.forward_batch(x, state, cond)  # [B, T, D_MODEL]
            # LayerNorm (batched)
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x = self.ln_gamma[layer_idx] * (x - mean) / torch.sqrt(var + 1e-5) + self.ln_beta[layer_idx]

        # Output projection: [B, T, D] → [B, T]
        return self.output_proj(x).squeeze(-1)

    def reset_state(self):
        """Not needed — state is created fresh per forward pass."""
        pass

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════

class AudioDataset(Dataset):
    """Loads WAV files, returns random chunks for identity training."""

    def __init__(self, root: str, chunk_samples: int = 2048, sr: int = 44100):
        self.root = Path(root)
        self.chunk_samples = chunk_samples
        self.sr = sr
        self.files = sorted(self.root.rglob("*.wav"))
        if not self.files:
            raise FileNotFoundError(f"No .wav files found in {root}")
        print(f"  Found {len(self.files)} audio files")
        # Pre-load all audio into RAM for fast access
        self._cache = []
        print(f"  Pre-loading {len(self.files)} files into RAM...")
        for f in self.files:
            try:
                audio, file_sr = sf.read(str(f))
                audio = torch.from_numpy(audio.T).float()  # [channels, samples]
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                if file_sr != sr:
                    # ponytail: reject mismatched SR files, user's data is all 44.1k
                    print(f"  WARNING: {f.name} is {file_sr}Hz (expected {sr}Hz), skipping")
                    continue
                self._cache.append(audio.mean(dim=0))  # mono
            except Exception as e:
                print(f"  WARNING: skipping {f.name}: {e}")
        self.files = [f for f, a in zip(self.files, self._cache) if a.numel() > 0]
        self._cache = [a for a in self._cache if a.numel() > 0]
        print(f"  Loaded {len(self._cache)} files, "
              f"{sum(a.numel() for a in self._cache) / sr / 3600:.1f} hours of audio")

    def __len__(self):
        return len(self.files) * 10  # 10 chunks per file per epoch

    def __getitem__(self, idx):
        file_idx = idx % len(self._cache)
        audio = self._cache[file_idx]
        T = self.chunk_samples
        if audio.numel() < T:
            # pad if too short
            repeats = T // audio.numel() + 1
            audio = audio.repeat(repeats)
        start = torch.randint(0, max(1, audio.numel() - T), (1,)).item()
        chunk = audio[start:start + T]
        # Normalize to prevent clipping
        peak = chunk.abs().max()
        if peak > 0:
            chunk = chunk / peak * 0.8
        return chunk.float(), chunk.float()  # identity mapping


class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic random noise dataset for smoke testing."""
    def __init__(self, chunk_samples: int = 2048, size: int = 1000):
        self.chunk_samples = chunk_samples
        self.size = size
    def __len__(self): return self.size
    def __getitem__(self, idx):
        return (torch.randn(self.chunk_samples) * 0.3,
                torch.randn(self.chunk_samples) * 0.3)


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    cfg = Mamba3Config()
    model = Mamba3Model(cfg).to(device)
    print(f"Model: {model.num_params():,} params")
    print(f"  Layers: {cfg.n_layers}, d_model={cfg.d_model}, d_state={cfg.d_state}")
    print(f"  MIMO: R={cfg.mimo_rank}, heads={cfg.nheads}, headdim={cfg.headdim}")
    print(f"  in_proj_dim: {cfg.in_proj_dim}")

    # Resume
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_step = ckpt.get('step', 0)
        print(f"Resumed from step {start_step}")

    # Dataset
    if args.data:
        dataset = AudioDataset(args.data, chunk_samples=args.chunk_samples)
    else:
        print("  No data dir — using synthetic noise (Stage 0 smoke test)")
        dataset = SyntheticDataset(args.chunk_samples)
        print("  No data dir — using synthetic noise (Stage 0 smoke test)")

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # Loss: MSE + multi-resolution STFT for perceptual quality
    loss_fn = CombinedLoss(mse_weight=0.5, stft_weight=0.5)

    # Training loop
    model.train()
    step = start_step
    total_loss = 0.0
    best_loss = float('inf')
    os.makedirs(args.save, exist_ok=True)

    print(f"\nTraining: {args.steps} steps, batch={args.batch}, accum={args.grad_accum}")
    t0 = time.time()

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            audio_in, audio_tgt = batch
            audio_in = audio_in.to(device)
            audio_tgt = audio_tgt.to(device)

            # Generate conditioning vector (Stage 0: random, for identity training)
            cond = torch.randn(args.batch, cfg.cond_dim, device=device) * 0.1

            # Forward with conditioning
            output = model(audio_in, cond)

            # Loss
            loss = loss_fn(output, audio_tgt) / args.grad_accum
            loss.backward()
            total_loss += loss.item()

            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Logging
            if step % args.log_every == 0:
                avg_loss = total_loss / args.log_every
                elapsed = time.time() - t0
                sps = (step - start_step) / max(elapsed, 0.01)
                print(f"  Step {step:6d}/{args.steps} | loss={avg_loss:.6f} | {sps:.1f} step/s")
                total_loss = 0.0

            # Checkpoint
            if step % args.ckpt_every == 0:
                avg_loss = total_loss / max(step % args.log_every, 1)
                ckpt_path = os.path.join(args.save, f"step_{step:06d}.pt")
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'cfg': cfg,
                }, ckpt_path)
                print(f"  Saved: {ckpt_path}")

    # Final save
    final_path = os.path.join(args.save, "final.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'cfg': cfg,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")


# ═══════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════

def export_weights(checkpoint_path: str, output_path: str):
    """Export trained model to flat binary model.bin (Version 2)."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('cfg', Mamba3Config())
    state_dict = ckpt['model_state_dict']

    weight_data = bytearray()
    layer_keys_template = [
        'in_proj.weight',
        'dt_bias',
        'B_bias',
        'C_bias',
        'B_norm_weight',
        'C_norm_weight',
        'mimo_x',
        'mimo_z',
        'mimo_o',
        'D_proj',
        'out_proj.weight',
        'film_proj.weight',
        'dt_film_proj.weight',
    ]

    for l in range(cfg.n_layers):
        prefix = f'layers.{l}.'
        for key in layer_keys_template:
            full_key = f'{prefix}{key}'
            param = state_dict[full_key].detach().cpu().numpy().astype(np.float32)
            weight_data.extend(param.tobytes())

    # Header (64 bytes)
    header = struct.pack('<12I',
        0x4D424D41,             # magic "MBMA"
        2,                       # version
        cfg.d_state,
        cfg.d_model,
        cfg.d_inner,
        cfg.nheads,
        cfg.headdim,
        cfg.mimo_rank,
        cfg.n_layers,
        cfg.cond_dim,
        cfg.in_proj_dim,
        len(weight_data),       # weight_data_size (bytes)
    )
    # Pad to 64 bytes
    header += b'\x00' * (64 - len(header))

    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(weight_data)

    print(f"Exported: {output_path} ({len(weight_data) + 64:,} bytes)")
    print(f"  Layers: {cfg.n_layers}, params: ~{len(weight_data)//4:,} floats")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CLARA Mamba-3 Training Pipeline")
    parser.add_argument('--data', type=str, help='Path to training dataset (WAV files)')
    parser.add_argument('--save', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--steps', type=int, default=100000, help='Total training steps')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--grad-accum', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--clip-grad-norm', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--chunk-samples', type=int, default=2048, help='Audio chunk length in samples')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--log-every', type=int, default=100, help='Log interval (steps)')
    parser.add_argument('--ckpt-every', type=int, default=5000, help='Checkpoint interval')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    # Export mode
    parser.add_argument('--export', type=str, help='Export checkpoint to model.bin')
    parser.add_argument('--output', type=str, default='models/model_trained.bin', help='Output .bin path')

    args = parser.parse_args()

    if args.export:
        export_weights(args.export, args.output)
    else:
        train(args)


if __name__ == '__main__':
    main()
