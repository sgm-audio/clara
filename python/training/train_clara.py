#!/usr/bin/env python3
"""CLARA Mamba-3 Training Pipeline — train_clara.py
=======================================================

Mamba-3 MIMO R=8 architecture with complex poles, spectral domain processing,
CREPE f0 conditioning, and cross-track attention.

Architecture: PerTrackProcessor (mamba3 reference)
  - R=8 structured spectral features (6 bands + transient + centroid)
  - Complex trapezoidal SSM (resonant bandpass heads)
  - FiLM conditioning from f0 + semantic + cross-track + loudness
  - Spectral decoder: d_model → complex STFT → ISTFT

Phase strategy (mamba3 reference):
  Phase 1 (0-20%):  Contrastive semantic encoder pretraining (NT-Xent)
  Phase 2 (20-50%): Single-track reconstruction (no cross-track)
  Phase 3 (50-100%): Cross-track conditioned reconstruction

Training: ROCm AMD 860M, bf16 AMP + fp32 SSM, gradient checkpointing.

Export: Flat binary model.bin (Version 3 — complex pole weight format).

Usage:
  python train_clara.py --data ~/datasets/musdb18hq/train --device cuda --amp
  python train_clara.py --export checkpoints/best.pt --output models/model.bin
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════
# ROCm Environment (must be set before any torch import)
# ═══════════════════════════════════════════════════════════════════
import os

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1100")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("ROC_VISIBLE_DEVICES", "0")
os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb=256"
)
os.environ.setdefault("MIOPEN_FIND_MODE", "5")

import math
import random
import signal
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class Mamba3Config:
    """Model dimensions — must match src/core/MambaConfig.h."""

    # Audio
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512  # ~11.6 ms @ 44.1 kHz
    n_mels: int = 128

    # MIMO spectral features
    R: int = 8  # [sub_bass, bass, low_mid, mid, high_mid, air, transient, centroid]
    mimo_rank: int = 8  # same as R in this architecture

    # Mamba-3 core
    d_model: int = 128
    d_inner: int = 256
    d_state: int = 32
    nheads: int = 4
    headdim: int = 64  # = d_inner / nheads
    n_layers: int = 6
    dt_min: float = 0.001
    dt_max: float = 0.1

    # Conditioning
    d_semantic: int = 128  # semantic encoder output
    d_f0: int = 32  # CREPE projection
    d_cross: int = 64  # cross-track attention output
    cond_dim: int = 128  # final FiLM conditioning dim

    @property
    def in_proj_dim(self) -> int:
        """Fused in_proj output dimension (no RoPE angles)."""
        return (
            self.d_inner  # z
            + self.nheads * self.headdim  # x per head
            + self.nheads * self.mimo_rank * self.d_state  # B
            + self.nheads * self.mimo_rank * self.d_state  # C
            + self.nheads * 3  # dd_dt + dd_A_real + dd_A_imag
            + self.nheads
        )  # trap

    @property
    def F_bins(self) -> int:
        return self.n_fft // 2 + 1


# ═══════════════════════════════════════════════════════════════════
# R=8 Structured Spectral Feature Extractor
# ═══════════════════════════════════════════════════════════════════


class R8FeatureExtractor(nn.Module):
    """Extracts R=8 structured spectral features per frame:
      [0] sub_bass   20-80 Hz       — deep low end
      [1] bass       80-250 Hz      — bass fundamentals
      [2] low_mid    250-800 Hz     — low harmonics, body
      [3] mid        800-2500 Hz    — vocals, instrument fundamentals
      [4] high_mid   2500-8000 Hz   — presence, attack
      [5] air        8000-20000 Hz  — brilliance
      [6] transient                 — onset density
      [7] centroid                  — spectral balance

    Band boundaries are learnable (soft triangular filterbank).
    """

    # Six musical frequency band pairs
    BAND_PAIRS: list[tuple[float, float]] = [
        (20, 80),
        (80, 250),
        (250, 800),
        (800, 2500),
        (2500, 8000),
        (8000, 20000),
    ]

    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.cfg = cfg
        sr = cfg.sample_rate
        n_fft = cfg.n_fft

        # Frequency axis (positive freqs only)
        self.register_buffer("freqs", torch.linspace(0, sr / 2, n_fft // 2 + 1))

        # Learnable band centers (Hz), init at geometric mean of boundaries
        centers_hz = [math.sqrt(lo * hi) for (lo, hi) in self.BAND_PAIRS]
        self.log_centers = nn.Parameter(torch.log(torch.tensor(centers_hz, dtype=torch.float32)))
        # Learnable log-bandwidths (~1/3 octave init)
        self.log_bandwidths = nn.Parameter(torch.full((6,), math.log(0.5)))

        # Transient detector: tiny 1-D conv on waveform envelope
        self.transient_conv = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=5, padding=2, bias=False
        )

        # Centroid normalization
        self.log_max_freq = math.log(sr / 2)

        # Project R=8 → d_model
        self.input_proj = nn.Linear(cfg.R, cfg.d_model)

    def _soft_band_energy(self, mag_sq: Tensor) -> Tensor:
        """mag_sq: (B, T, F) — squared magnitude spectrogram
        returns: (B, T, 6) — log-compressed band energies
        """
        freqs = self.freqs.view(1, 1, -1, 1)  # (1, 1, F, 1)
        centers = torch.exp(self.log_centers).view(1, 1, 1, -1)  # (1, 1, 1, 6)
        bws = torch.exp(self.log_bandwidths).view(1, 1, 1, -1)  # (1, 1, 1, 6)

        log_ratio = torch.log(freqs.clamp(min=1.0) / centers)  # (1, 1, F, 6)
        weights = F.relu(1.0 - log_ratio.abs() / bws)  # (1, 1, F, 6)
        weights = weights / (weights.sum(dim=2, keepdim=True).clamp(min=1e-8))

        energies = (mag_sq.unsqueeze(-1) * weights).sum(dim=2)  # (B, T, 6)
        return torch.log1p(energies)  # log-compress

    def _transient_density(self, waveform: Tensor, T: int) -> Tensor:
        """waveform: (B, L)
        T: target number of frames (must match STFT output T)
        returns: (B, T, 1) — per-frame transient density
        """
        envelope = waveform.abs().unsqueeze(1)  # (B, 1, L)
        diff = F.relu(self.transient_conv(envelope))  # (B, 1, L)
        pooled = F.adaptive_avg_pool1d(diff, T)  # (B, 1, T)
        return pooled.transpose(1, 2)  # (B, T, 1)

    def _spectral_centroid(self, mag_sq: Tensor) -> Tensor:
        """mag_sq: (B, T, F)
        returns: (B, T, 1) — normalized log spectral centroid
        """
        freqs = self.freqs.clamp(min=1.0)
        weights = mag_sq / (mag_sq.sum(dim=-1, keepdim=True).clamp(min=1e-8))
        centroid_hz = (weights * freqs).sum(dim=-1, keepdim=True)  # (B, T, 1)
        return torch.log(centroid_hz.clamp(min=1.0)) / self.log_max_freq

    def forward(self, waveform: Tensor, mag_sq: Tensor) -> Tensor:
        """waveform: (B, L)
        mag_sq: (B, T, F) — |STFT|^2
        Returns: (B, T, d_model) — embedded features
        """
        band_energy = self._soft_band_energy(mag_sq)  # (B, T, 6)
        T = mag_sq.shape[1]
        transient = self._transient_density(waveform, T)  # (B, T, 1)
        centroid = self._spectral_centroid(mag_sq)  # (B, T, 1)
        r8 = torch.cat([band_energy, transient, centroid], dim=-1)  # (B, T, 8)
        return self.input_proj(r8)  # (B, T, d_model)

    @torch.no_grad()
    def extract_r8_raw(self, waveform: Tensor, mag_sq: Tensor) -> Tensor:
        """Raw R=8 features (before d_model projection) for MIMO input."""
        band_energy = self._soft_band_energy(mag_sq)
        T = mag_sq.shape[1]
        transient = self._transient_density(waveform, T)
        centroid = self._spectral_centroid(mag_sq)
        return torch.cat([band_energy, transient, centroid], dim=-1)  # (B, T, 8)


# ═══════════════════════════════════════════════════════════════════
# Mamba-3 Complex Pole MIMO Layer
# ═══════════════════════════════════════════════════════════════════


class Mamba3MIMOLayer(nn.Module):
    """Complex-pole Mamba-3 MIMO layer with trapezoidal discretization.

    Each of H heads acts as a resonant bandpass filter:
      - Real part controls decay (bandwidth)
      - Imag part controls center frequency
      - MIMO B/C project R-dimensional input per head

    No RoPE, no causal conv1d (Mamba-3 drops both).
    fp32 SSM core even under bf16 AMP for complex arithmetic stability.
    """

    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.cfg = cfg
        self.H = cfg.nheads
        self.N = cfg.d_state  # complex state dim per head
        self.d = cfg.d_model
        self.R = cfg.mimo_rank
        self.head_dim = cfg.headdim

        # Fused in_proj: d_model → in_proj_dim (no RoPE angles)
        self.in_proj = nn.Linear(cfg.d_model, cfg.in_proj_dim, bias=False)

        # Complex pole parameters (per head)
        self.log_a_real = nn.Parameter(torch.zeros(cfg.nheads))  # decay (exp → negative)
        self.a_imag = nn.Parameter(torch.linspace(0, math.pi, cfg.nheads))  # frequency spread 0..π

        # Δ parameters
        self.dt_bias = nn.Parameter(torch.full((cfg.nheads,), -4.0))  # softplus(-4)≈0.018
        self.log_dt_min = math.log(cfg.dt_min)
        self.log_dt_max = math.log(cfg.dt_max)

        # MIMO B: R → N*2 (real+imag per state dim)
        self.B_proj = nn.Linear(cfg.R, cfg.d_state * 2, bias=True)

        # MIMO C: N*2 → head_dim (readout from complex state)
        self.C_proj = nn.Linear(cfg.d_state * 2, self.head_dim, bias=False)

        # Output projection & norm
        self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=False)
        self.norm = nn.LayerNorm(cfg.d_model)

        # MIMO coupling weights
        self.mimo_x = nn.Parameter(torch.randn(cfg.nheads, cfg.R, cfg.headdim) * 0.02)
        self.mimo_z = nn.Parameter(torch.randn(cfg.nheads, cfg.R, cfg.headdim) * 0.02)
        self.mimo_o = nn.Parameter(torch.randn(cfg.nheads, cfg.R, cfg.headdim) * 0.02)
        self.D_proj = nn.Parameter(torch.zeros(cfg.nheads))

        # FiLM conditioning
        self.film_proj = nn.Linear(cfg.cond_dim, cfg.d_model * 2, bias=True)
        self.dt_film_proj = nn.Linear(cfg.cond_dim, cfg.nheads * 2, bias=True)

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

        # Init
        self._init_weights()

    def _init_weights(self):
        """Xavier-uniform for linears. Zero-init FiLM for identity start."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
        with torch.no_grad():
            self.film_proj.weight.zero_()
            self.film_proj.bias.zero_()
            self.dt_film_proj.weight.zero_()
            self.dt_film_proj.bias.zero_()

    # ── Trapezoidal discretization (complex) ─────────────────────

    def _discretize_trapezoidal(
        self, a_real: Tensor, a_imag: Tensor, dt: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Complex trapezoidal discretization (Mamba-3 innovation).

        A_d = (1 + dt·(α+jω)/2) / (1 - dt·(α+jω)/2)  — |A_d| ≤ 1 for α < 0
        B_d = dt / (1 - dt·(α+jω)/2)

        a_real: (H,)
        a_imag: (H,)
        dt: (B, T, H)
        Returns: A_d (B, T, H), B_d_scale (B, T, H) both complex64
        """
        half_dt_a = torch.complex(
            dt * (-torch.exp(a_real)) / 2, dt * a_imag / 2
        )  # (B, T, H) complex

        numerator = 1 + half_dt_a
        denominator = 1 - half_dt_a

        A_d = numerator / denominator  # |A_d| ≤ 1
        B_d_scale = dt / denominator  # complex

        return A_d, B_d_scale

    # ── Parallel associative scan ─────────────────────────────────

    @staticmethod
    def _parallel_associative_scan(a: Tensor, u: Tensor) -> Tensor:
        """Blelloch parallel prefix scan: O(log T) steps.
        a: (B, T, H, N) complex
        u: (B, T, H, N) complex
        Returns: h (B, T, H, N) complex where h[t] = a[t]*h[t-1] + u[t]
        """
        B, T, H, N = u.shape
        step = 1
        while step < T:
            a_slice = a[:, step:]  # (B, T-step, H, N)
            u_slice = u[:, step:]
            a_prev = a[:, : T - step]
            u_prev = u[:, : T - step]

            new_u = a_slice * u_prev + u_slice
            new_a = a_slice * a_prev

            u = torch.cat([u[:, :step], new_u], dim=1)
            a = torch.cat([a[:, :step], new_a], dim=1)
            step *= 2
        return u

    # ── SSM scans ────────────────────────────────────────────────

    def _ssm_scan_parallel(self, x_r8: Tensor, A_d: Tensor, B_d: Tensor) -> Tensor:
        """x_r8: (B, T, R) — MIMO input
        A_d: (B, T, H) complex
        B_d: (B, T, H) complex
        Returns: (B, T, H, head_dim)
        """
        B, T, R = x_r8.shape
        H, N = self.H, self.N

        # MIMO B: R → N complex per head
        bc = self.B_proj(x_r8)  # (B, T, N*2)
        B_real, B_imag = bc.chunk(2, dim=-1)
        B_vec = torch.complex(B_real, B_imag)  # (B, T, N)

        # u_t = B_d_t * B_vec_t
        a = A_d.unsqueeze(-1)  # (B, T, H, 1)
        b_d = B_d.unsqueeze(-1)  # (B, T, H, 1)
        b_vec = B_vec.unsqueeze(2)  # (B, T, 1, N)
        u = b_d * b_vec  # (B, T, H, N)

        # Parallel associative scan
        h = self._parallel_associative_scan(a, u)  # (B, T, H, N) complex

        # Readout: cat real+imag → C projection
        h_cat = torch.cat([h.real, h.imag], dim=-1)  # (B, T, H, N*2)
        out = self.C_proj(h_cat)  # (B, T, H, head_dim)
        return out

    def _ssm_scan_sequential(self, x_r8: Tensor, A_d: Tensor, B_d: Tensor) -> Tensor:
        """O(T) sequential scan — bit-exact with C++ engine."""
        B, T, R = x_r8.shape
        H, N = self.H, self.N

        bc = self.B_proj(x_r8)
        B_real, B_imag = bc.chunk(2, dim=-1)
        B_vec = torch.complex(B_real, B_imag)

        h = torch.zeros(B, H, N, dtype=torch.complex64, device=x_r8.device)
        outputs = []
        for t in range(T):
            A_d_t = A_d[:, t, :]  # (B, H)
            B_d_t = B_d[:, t, :]  # (B, H)
            B_input = B_d_t.unsqueeze(-1) * B_vec[:, t, :].unsqueeze(1)
            h = A_d_t.unsqueeze(-1) * h + B_input
            h_cat = torch.cat([h.real, h.imag], dim=-1)
            y_t = self.C_proj(h_cat)
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)

    # ── Forward ──────────────────────────────────────────────────

    def _forward_impl(self, x: Tensor, x_r8: Tensor, cond: Tensor | None = None) -> Tensor:
        """x: (B, T, d_model)
        x_r8: (B, T, R)
        cond: (B, T, cond_dim) or None
        Returns: (B, T, d_model)
        """
        B, T, D = x.shape
        residual = x

        # 1. FiLM on input
        if cond is not None:
            gb = self.film_proj(cond)
            gamma, beta = gb.chunk(2, dim=-1)
            x = x * (1 + gamma) + beta

        # 2. Fused in_proj
        proj = self.in_proj(x)  # (B, T, in_proj_dim)
        idx = 0
        z = proj[..., : idx + self.cfg.d_inner]
        idx += self.cfg.d_inner
        x_h = proj[..., idx : idx + self.H * self.head_dim]
        idx += self.H * self.head_dim
        B_raw = proj[..., idx : idx + self.H * self.R * self.N]
        idx += self.H * self.R * self.N
        C_raw = proj[..., idx : idx + self.H * self.R * self.N]
        idx += self.H * self.R * self.N
        dd_dt = proj[..., idx : idx + self.H]
        idx += self.H
        _dd_A_real = proj[..., idx : idx + self.H]
        idx += self.H
        _dd_A_imag = proj[..., idx : idx + self.H]
        idx += self.H
        trap_val = proj[..., idx : idx + self.H]  # unused in complex mode

        # 3. Δ projection + clamping
        dt = F.softplus(dd_dt + self.dt_bias)
        dt = dt.clamp(min=self.cfg.dt_min, max=self.cfg.dt_max)

        # FiLM on Δ
        if cond is not None:
            dt_gb = self.dt_film_proj(cond)
            dt_g, dt_b = dt_gb.chunk(2, dim=-1)
            dt = F.softplus(dt + dt_g + dt_b)

        # 4. Complex trapezoidal discretization + SSM scan (fp32 forced)
        with torch.autocast(device_type=x.device.type, enabled=False):
            dt_fp32 = dt.float()
            x_r8_fp32 = x_r8.float()
            A_d, B_d_scale = self._discretize_trapezoidal(
                self.log_a_real.float(), self.a_imag.float(), dt_fp32
            )
            out = self._ssm_scan_parallel(x_r8_fp32, A_d, B_d_scale)

        out = out.to(x.dtype)  # (B, T, H, head_dim)
        out = out.reshape(B, T, self.cfg.d_inner)  # (B, T, d_inner)

        # 5. z-gating
        z_gated = z * torch.sigmoid(z)
        z_heads = z_gated.view(B, T, self.H, self.head_dim)
        z_slots = torch.einsum("bthd,hrd->bthr", z_heads, self.mimo_z)
        z_gate_val = z_slots.sum(dim=-1, keepdim=True)
        y_gated = out.view(B, T, self.H, self.head_dim) * z_gate_val
        y_flat = y_gated.reshape(B, T, self.cfg.d_inner)

        # 6. Output projection + D skip + residual
        out = self.out_proj(y_flat)

        # D skip
        x_h_heads = x_h.view(B, T, self.H, self.head_dim)
        d_contrib = (self.D_proj.view(1, 1, self.H, 1) * x_h_heads).sum(dim=(-2, -1))
        out = out + d_contrib.unsqueeze(-1) * 0.01

        # MIMO o-coupling
        y_gated_h = y_gated.reshape(B, T, self.H, self.head_dim)
        o_slots = torch.einsum("bthd,hrd->bthr", y_gated_h, self.mimo_o)
        o_contrib = o_slots.sum(dim=(-2, -1))
        out = out + o_contrib.unsqueeze(-1) * 0.01

        out = self.norm(out + residual)
        return out

    def forward(self, x: Tensor, x_r8: Tensor, cond: Tensor | None = None) -> Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(
                self._forward_impl,
                x,
                x_r8,
                cond,
                use_reentrant=False,
            )
        return self._forward_impl(x, x_r8, cond)


# ═══════════════════════════════════════════════════════════════════
# Semantic Encoder (Phase 1 contrastive pretraining)
# ═══════════════════════════════════════════════════════════════════


class SemanticEncoder(nn.Module):
    """Lightweight contrastive semantic encoder.
    Conv stem → 2 Mamba layers → mean pool → projection head.
    Trained with NT-Xent in Phase 1.
    """

    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv1d(cfg.n_mels, cfg.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=3, padding=1),
            nn.LayerNorm(cfg.d_model),
        )
        # 2 Mamba layers with layer-local config override
        enc_cfg = Mamba3Config(
            d_model=cfg.d_model,
            d_inner=cfg.d_inner,
            d_state=cfg.d_state,
            nheads=cfg.nheads,
            headdim=cfg.headdim,
            n_layers=1,
            R=cfg.R,
            mimo_rank=cfg.mimo_rank,
            cond_dim=cfg.cond_dim,
        )
        self.mamba_layers = nn.ModuleList([Mamba3MIMOLayer(enc_cfg) for _ in range(2)])
        self.proj_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, mel: Tensor, r8: Tensor, return_sequence: bool = False) -> Tensor:
        """mel: (B, T, n_mels) — log-mel spectrogram
        r8: (B, T, R) — raw MIMO features
        Returns: (B, d_model) if not return_sequence, else (B, T, d_model)
        """
        x = mel.transpose(1, 2)  # (B, n_mels, T)
        x = self.conv_stem[0](x)  # Conv1d
        x = self.conv_stem[1](x)  # GELU
        x = self.conv_stem[2](x)  # Conv1d
        x = x.transpose(1, 2)  # (B, T, d_model)
        x = self.conv_stem[3](x)  # LayerNorm

        for layer in self.mamba_layers:
            x = layer(x, r8)  # No conditioning in encoder

        if return_sequence:
            return x

        pooled = x.mean(dim=1)
        return F.normalize(self.proj_head(pooled), dim=-1)


# ═══════════════════════════════════════════════════════════════════
# F0 Conditioner (CREPE projection)
# ═══════════════════════════════════════════════════════════════════


class F0Conditioner(nn.Module):
    """Projects CREPE 360-D f0 probability vector → d_f0."""

    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(360, 128),
            nn.GELU(),
            nn.Linear(128, cfg.d_f0),
            nn.LayerNorm(cfg.d_f0),
        )

    def forward(self, crepe_probs: Tensor) -> Tensor:
        """crepe_probs: (B, T, 360) → (B, T, d_f0)"""
        return self.proj(crepe_probs)


# ═══════════════════════════════════════════════════════════════════
# Cross-Track Aggregator
# ═══════════════════════════════════════════════════════════════════


class CrossTrackAggregator(nn.Module):
    """Causal multi-head attention over track dimension.
    Each timestep independently attends over K other tracks.
    """

    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_semantic,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
        )
        self.proj = nn.Linear(cfg.d_semantic, cfg.d_cross)
        self.norm = nn.LayerNorm(cfg.d_cross)

    def forward(self, target_emb: Tensor, other_embs: Tensor) -> Tensor:
        """target_emb: (B, T, d_semantic)
        other_embs: (B, K, T, d_semantic) — K other tracks
        Returns: (B, T, d_cross)
        """
        B, T, D = target_emb.shape
        K = other_embs.shape[1]

        q = target_emb.reshape(B * T, 1, D)
        kv = other_embs.permute(0, 2, 1, 3).reshape(B * T, K, D)

        ctx, _ = self.attn(q, kv, kv)
        ctx = ctx.squeeze(1).reshape(B, T, D)
        return self.norm(self.proj(ctx))


# ═══════════════════════════════════════════════════════════════════
# Full Per-Track Processor
# ═══════════════════════════════════════════════════════════════════


class PerTrackProcessor(nn.Module):
    """Complete per-track processor.

    Pipeline: STFT → R=8 features → semantic enc → f0 cond →
              Mamba stack → decoder → ISTFT

    Cross-track conditioning optional (graceful degradation to single-track).
    """

    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.cfg = cfg

        # Feature extraction
        self.r8_extractor = R8FeatureExtractor(cfg)
        self.semantic_enc = SemanticEncoder(cfg)
        self.f0_cond = F0Conditioner(cfg)
        self.cross_track = CrossTrackAggregator(cfg)

        # Conditioning MLP
        cond_in = cfg.d_semantic + cfg.d_f0 + cfg.d_cross + 4  # loudness
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, cfg.cond_dim),
            nn.GELU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
            nn.LayerNorm(cfg.cond_dim),
        )

        # Mamba stack (n_layers)
        self.mamba_layers = nn.ModuleList([Mamba3MIMOLayer(cfg) for _ in range(cfg.n_layers)])

        # Decoder: d_model → 2 × F_bins (real + imag STFT coefficients)
        F_bins = cfg.F_bins
        self.decoder = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Linear(cfg.d_model * 2, 2 * F_bins),
        )

        # Mel projection (learnable non-negative)
        self.mel_proj = nn.Linear(F_bins, cfg.n_mels, bias=False)
        with torch.no_grad():
            self.mel_proj.weight.abs_()

        # Pre-register STFT window
        self.register_buffer("stft_window", torch.hann_window(cfg.n_fft), persistent=False)

    # ── STFT helpers ─────────────────────────────────────────────

    def _compute_stft(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        """waveform: (B, L)

        Returns:
          mag_sq: (B, T, F) — squared magnitude
          stft:   (B, T, F) complex — for reconstruction

        """
        cfg = self.cfg
        s = torch.stft(
            waveform,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            window=self.stft_window,
            return_complex=True,
            normalized=False,
        )  # (B, F, T) complex
        s = s.transpose(1, 2)  # (B, T, F)
        return s.abs().pow(2), s

    def _compute_mel(self, mag_sq: Tensor) -> Tensor:
        """Learnable mel projection (non-negative weights)."""
        projected = self.mel_proj(mag_sq).clamp(min=0.0)
        return torch.log1p(projected)

    def _loudness_features(self, waveform: Tensor, T: int) -> Tensor:
        """4-dim loudness per frame: [rms, peak, crest_factor, lufs_approx].
        waveform: (B, L)
        T: target frames
        Returns: (B, T, 4)
        """
        B = waveform.shape[0]
        L = waveform.shape[-1]
        frame_size = (L + T - 1) // T
        pad_amount = T * frame_size - L
        if pad_amount > 0:
            wav = F.pad(waveform, (0, pad_amount))
        else:
            wav = waveform
        frames = wav.reshape(B, T, frame_size)

        rms = frames.pow(2).mean(-1, keepdim=True).sqrt()
        peak = frames.abs().max(-1, keepdim=True).values
        crest = (peak / (rms + 1e-8)).clamp(min=1e-8).log()
        lufs = 10 * torch.log10(rms.pow(2) + 1e-8)

        return torch.cat([rms, peak, crest, lufs], dim=-1)

    def waveform_from_stft(self, stft: Tensor) -> Tensor:
        """Reconstruct waveform from predicted complex STFT.
        stft: (B, T, F) complex
        Returns: (B, L) waveform
        """
        cfg = self.cfg
        return torch.istft(
            stft.transpose(1, 2),
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            window=self.stft_window,
        )

    # ── Forward ──────────────────────────────────────────────────

    def forward(
        self, waveform: Tensor, crepe_probs: Tensor, other_embs: Tensor | None = None
    ) -> dict:
        """waveform: (B, L)
        crepe_probs: (B, T, 360)
        other_embs: (B, K, T, d_semantic) or None for Phase 1/2

        Returns dict with:
          'output_stft': (B, T, F) complex — predicted STFT
          'track_emb':   (B, T, d_semantic) — for use as other_embs in parallel tracks
          'r8_raw':      (B, T, R) — raw structured features (debug)
        """
        cfg = self.cfg

        # ── Feature extraction ──
        mag_sq, _ = self._compute_stft(waveform)
        T = mag_sq.shape[1]

        x = self.r8_extractor(waveform, mag_sq)  # (B, T, d_model)
        r8_raw = self.r8_extractor.extract_r8_raw(waveform, mag_sq)  # (B, T, 8)
        mel = self._compute_mel(mag_sq)  # (B, T, n_mels)

        # ── Semantic encoding (per-frame) ──
        sem_emb = self.semantic_enc(mel, r8_raw, return_sequence=True)  # (B, T, d_semantic)

        # ── F0 conditioning ──
        f0_emb = self.f0_cond(crepe_probs)  # (B, T, d_f0)

        # ── Cross-track aggregation ──
        if other_embs is not None:
            cross_ctx = self.cross_track(sem_emb, other_embs)  # (B, T, d_cross)
        else:
            cross_ctx = torch.zeros(waveform.shape[0], T, cfg.d_cross, device=waveform.device)

        # ── Build conditioning vector ──
        loudness = self._loudness_features(waveform, T)  # (B, T, 4)
        cond_in = torch.cat([sem_emb, f0_emb, cross_ctx, loudness], dim=-1)
        cond = self.cond_mlp(cond_in)  # (B, T, cond_dim)

        # ── Mamba stack ──
        for layer in self.mamba_layers:
            x = layer(x, r8_raw, cond=cond)

        # ── Decode to STFT ──
        stft_pred = self.decoder(x)  # (B, T, 2*F_bins)
        F_bins = cfg.F_bins
        stft_real = stft_pred[..., :F_bins].float()
        stft_imag = stft_pred[..., F_bins:].float()
        stft_out = torch.complex(stft_real, stft_imag)  # (B, T, F)

        return {
            "output_stft": stft_out,
            "track_emb": sem_emb,
            "r8_raw": r8_raw,
        }

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def mamba_params(self) -> int:
        return sum(p.numel() for l in self.mamba_layers for p in l.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════
# Contrastive Encoding Helpers
# ═══════════════════════════════════════════════════════════════════


def _encode_for_contrastive(model: PerTrackProcessor, wav: Tensor, crepe: Tensor) -> Tensor:
    """Run semantic encoder on waveform for contrastive embedding.
    Returns: (B, d_model) L2-normalized
    """
    mag_sq, _ = model._compute_stft(wav)
    mel = model._compute_mel(mag_sq)
    r8 = model.r8_extractor.extract_r8_raw(wav, mag_sq)
    return model.semantic_enc(mel, r8, return_sequence=False)


def _augment_for_contrastive(wav: Tensor, crepe: Tensor, sr: int = 44100) -> tuple[Tensor, Tensor]:
    """Augment waveform+CREPE for contrastive pair generation.
    ALWAYS runs pitch shift on CPU to avoid GPU OOM from sinc resample kernels.
    """
    orig_device = wav.device

    # Pitch shift — ALWAYS run on CPU (GPU sinc kernel = 6GB+ OOM)
    if random.random() < 0.5:
        semitones = random.uniform(-2.0, 2.0)
        factor = 2 ** (semitones / 12.0)
        orig_len = wav.shape[-1]
        new_freq = int(sr * factor)
        # Force CPU for resample
        wav_cpu = wav.cpu()
        wav_cpu = torchaudio.functional.resample(wav_cpu.unsqueeze(0), sr, new_freq).squeeze(0)
        if wav_cpu.shape[-1] > orig_len:
            wav_cpu = wav_cpu[..., :orig_len]
        else:
            wav_cpu = F.pad(wav_cpu, (0, orig_len - wav_cpu.shape[-1]))
        wav = wav_cpu.to(orig_device)
        # Roll CREPE
        bins_shift = int(round(5 * semitones))
        if abs(bins_shift) > 0 and abs(bins_shift) < 360:
            rolled = torch.zeros_like(crepe)
            if bins_shift > 0:
                rolled[..., bins_shift:] = crepe[..., :-bins_shift]
            else:
                rolled[..., :bins_shift] = crepe[..., -bins_shift:]
            crepe = rolled

    # Already on correct device

    # Gain ±4 dB
    gain_db = random.uniform(-4, 4)
    wav = wav * (10 ** (gain_db / 20))

    # Polarity
    if random.random() < 0.5:
        wav = -wav

    # Circular time shift
    shift = random.randint(0, wav.shape[-1] // 4)
    wav = torch.roll(wav, shift, dims=-1)
    # Shift CREPE to match
    frame_shift = shift // 512  # hop_length
    if frame_shift > 0:
        crepe = torch.roll(crepe, frame_shift, dims=0)

    # Additive noise
    noise_level = random.uniform(0, 0.005)
    wav = wav + noise_level * torch.randn_like(wav)

    return wav.clamp(-1.0, 1.0), crepe


# ═══════════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════════

from loss import NTXentLoss, TrackProcessorLoss

# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════


def train(args):
    if not args.data:
        print("ERROR: --data is required for training")
        sys.exit(1)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        if args.amp:
            print("  AMP: bfloat16 (fp32 SSM core)")

    amp = args.amp and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if amp else torch.float32

    # ── Model ──
    cfg = Mamba3Config()
    model = PerTrackProcessor(cfg).to(device)
    print(f"Model: {model.num_params():,} params")
    print(f"  Mamba core: {model.mamba_params:,} params")
    print(f"  Layers: {cfg.n_layers}, d_model={cfg.d_model}, d_state={cfg.d_state}")
    print(f"  R={cfg.R} spectral features, nheads={cfg.nheads}")
    print(f"  in_proj_dim: {cfg.in_proj_dim}")
    print(f"  STFT: n_fft={cfg.n_fft}, hop={cfg.hop_length}")

    # Gradient checkpointing disabled.
    # ponytail: torch.checkpoint with use_reentrant=False leaks memory
    # (25GB RSS in 10 steps on real data). Model is 81 MB peak without it.
    # Also incompatible with AMP on ROCm (CheckpointError).
    print("  Gradient checkpointing: OFF (disabled — memory leak)")

    # ── Losses ──
    task_loss = TrackProcessorLoss(cfg.sample_rate).to(device)
    contrastive = NTXentLoss(temperature=0.1).to(device)

    # ── Optimizer + scheduler ──
    decay_params = [
        p
        for n, p in model.named_parameters()
        if "bias" not in n and "norm" not in n and p.requires_grad
    ]
    nodecay_params = [
        p for n, p in model.named_parameters() if ("bias" in n or "norm" in n) and p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    def lr_lambda(step: int) -> float:
        warmup = args.warmup
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ──
    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from step {start_step}, best_val_loss={best_val_loss:.4f}")

    # ── Dataset ──
    print(f"Data: {args.data}")
    from dataset import make_dataloaders

    train_loader, val_loader, n_train_tracks = make_dataloaders(
        data_root=args.data,
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        clip_seconds=args.clip_seconds,
        batch_size=args.batch,
        val_split=args.val_split,
        num_workers=args.num_workers,
        cache_limit=args.cache_limit,
    )

    # Phase boundaries
    if args.skip_contrastive:
        PHASE_1_END = 0  # Skip contrastive phase entirely
        PHASE_2_END = int(args.steps * 0.50)
    else:
        PHASE_1_END = int(args.steps * 0.20)
        PHASE_2_END = int(args.steps * 0.50)

    # ── Signal handler ──
    _state = {"step": start_step, "interrupted": False}

    def _signal_handler(signum, frame):
        if _state["interrupted"]:
            print("\n! Second interrupt — exiting hard.", flush=True)
            sys.exit(130)
        print(f"\n! Caught signal {signum} — saving interrupt checkpoint...", flush=True)
        _state["interrupted"] = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ── Atomic checkpoint ──
    save_path = Path(args.save)
    save_path.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(step, loss_val, is_best=False, name=None):
        ckpt = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cfg": cfg.__dict__,
            "loss": loss_val,
            "best_val_loss": best_val_loss,
        }
        fname = name if name else f"checkpoint_{step:06d}.pt"
        tmp = save_path / f".{fname}.tmp"
        final = save_path / fname
        torch.save(ckpt, tmp)
        tmp.replace(final)
        if is_best:
            best_tmp = save_path / ".best.pt.tmp"
            torch.save(ckpt, best_tmp)
            best_tmp.replace(save_path / "best.pt")

    # ── Phase-aware validation ──
    @torch.no_grad()
    def run_validation(current_phase: int) -> float:
        model.eval()
        total_loss = 0.0
        n_batches = 0
        for batch in val_loader:
            if batch is None:
                continue
            v_target = batch["target"].to(device)
            v_crepe = batch["target_crepe"].to(device)[
                ..., : 1 + v_target.shape[-1] // cfg.hop_length, :
            ]

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp):
                if current_phase == 1:
                    # Phase 1: contrastive validation
                    z1 = _encode_for_contrastive(model, v_target, v_crepe)
                    aug_wav, aug_crepe = _augment_for_contrastive(
                        v_target, v_crepe, cfg.sample_rate
                    )
                    z2 = _encode_for_contrastive(model, aug_wav, aug_crepe)
                    vl = contrastive(z1, z2)
                else:
                    # Phase 2/3: reconstruction validation
                    if current_phase == 3:
                        v_other = batch["other_stems"].to(device)
                        v_other_crepe = batch["other_crepe"].to(device)
                        K = v_other.shape[1]
                        embs = []
                        for k in range(K):
                            ck = v_other_crepe[:, k, :, :]
                            out_k = model(v_other[:, k, :], ck, other_embs=None)
                            embs.append(out_k["track_emb"])
                        other_embs = torch.stack(embs, dim=1)
                        out = model(v_target, v_crepe, other_embs=other_embs)
                    else:
                        out = model(v_target, v_crepe, other_embs=None)
                    vl_dict = task_loss(out["output_stft"], v_target, model)
                    vl = vl_dict["total"]

            if not (torch.isnan(vl) or torch.isinf(vl)):
                total_loss += vl.item()
                n_batches += 1

        model.train()
        return total_loss / max(1, n_batches)

    # ── GPU flags ──
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    # ── Training loop ──
    step = start_step
    plateau_count = 0
    last_log_time = time.time()
    last_log_step = step
    last_phase = None

    print(
        f"\nStarting training: {args.steps} steps, batch={args.batch}, accum={args.grad_accum}, device={device}"
    )
    print(f"  Phase 1 (contrastive):  0 → {PHASE_1_END}")
    print(f"  Phase 2 (single-track): {PHASE_1_END} → {PHASE_2_END}")
    print(f"  Phase 3 (cross-track):  {PHASE_2_END} → {args.steps}")
    if args.precomputed:
        print("  [Note] --precomputed is deprecated; StemDataset loads .crepe.npy automatically")
    if start_step > 0:
        print(f"  Resuming at step {start_step}")
    print()

    while step < args.steps:
        for batch in train_loader:
            if batch is None:
                continue
            if step >= args.steps:
                break

            target = batch["target"].to(device)
            target_crepe = batch["target_crepe"].to(device)
            # Trim CREPE to match waveform length's frame count
            T_frames = 1 + target.shape[-1] // cfg.hop_length
            target_crepe = target_crepe[:, :T_frames, :]
            other_stems = batch["other_stems"].to(device)
            other_crepe = batch["other_crepe"].to(device)

            current_phase = 1 if step < PHASE_1_END else 2 if step < PHASE_2_END else 3

            if current_phase != last_phase:
                print(f"\n{'─'*60}")
                print(f"  ENTERING PHASE {current_phase} at step {step}")
                print(f"{'─'*60}\n", flush=True)
                last_phase = current_phase
                task_loss.global_step = 0  # Reset SI-SDR ramp for new phase

            # ── Gradient accumulation ──
            # Scale loss by 1/accum so that optimizer.step() every N batches
            # gives the same effective gradient as accum× larger batch.
            accum = args.grad_accum
            do_step = ((step + 1) % accum == 0) or (step + 1 >= args.steps)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp):
                if current_phase == 1:
                    # Phase 1: Contrastive — only semantic encoder is trained
                    aug1_wav, aug1_crepe = _augment_for_contrastive(
                        target, target_crepe, cfg.sample_rate
                    )
                    aug2_wav, aug2_crepe = _augment_for_contrastive(
                        target, target_crepe, cfg.sample_rate
                    )
                    z1 = _encode_for_contrastive(model, aug1_wav, aug1_crepe)
                    z2 = _encode_for_contrastive(model, aug2_wav, aug2_crepe)
                    loss = contrastive(z1, z2)
                    loss_dict = {"total": loss, "contrastive": loss}

                elif current_phase == 2:
                    # Phase 2: Single-track reconstruction
                    out = model(target, target_crepe, other_embs=None)
                    loss_dict = task_loss(out["output_stft"], target, model)
                    loss = loss_dict["total"]

                else:
                    # Phase 3: Cross-track conditioned
                    # ponytail: no_grad for other-stem emb extraction — autograd
                    # would track 4× model passes simultaneously (OOM). We only
                    # backprop through the target pass.
                    K = other_stems.shape[1]
                    other_embs_list = []
                    with torch.no_grad():
                        for k in range(K):
                            stem_k = other_stems[:, k, :]
                            crepe_k = other_crepe[:, k, :T_frames, :]
                            out_k = model(stem_k, crepe_k, other_embs=None)
                            other_embs_list.append(out_k["track_emb"].detach())
                    other_embs = torch.stack(other_embs_list, dim=1)
                    out = model(target, target_crepe, other_embs=other_embs)
                    loss_dict = task_loss(out["output_stft"], target, model)
                    loss = loss_dict["total"]

                # Scale by 1/accum so backward() accumulates grads internally
                loss_scaled = loss / accum

            # ── NaN/Inf skip ──
            if torch.isnan(loss_scaled) or torch.isinf(loss_scaled):
                print(f"[step {step}] NaN/Inf loss — skipping batch", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue

            # ── Backward ──
            loss_scaled.backward()

            if do_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # ── Logging ──
            if step % args.log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                now = time.time()
                sps = (step - last_log_step) / max(1e-6, now - last_log_time)
                last_log_time = now
                last_log_step = step

                log_str = (
                    f"[step {step:6d}] p{current_phase} loss={loss.item():.4f} "
                    f"lr={lr_now:.2e} {sps:.2f} step/s"
                )
                if "mr_stft" in loss_dict:
                    log_str += f" stft={loss_dict['mr_stft'].item():.4f}"
                if "mel_l1" in loss_dict:
                    log_str += f" mel={loss_dict['mel_l1'].item():.4f}"
                if "si_sdr" in loss_dict and loss_dict["si_sdr"].item() != 0:
                    log_str += f" si_sdr={loss_dict['si_sdr'].item():.4f}"
                print(log_str, flush=True)

            # ── Validation (decoupled) ──
            if step > 0 and step % args.val_every == 0:
                print(f"  Running validation (phase {current_phase})...", flush=True)
                val_loss = run_validation(current_phase)
                print(f"  val_loss = {val_loss:.4f}  (best={best_val_loss:.4f})", flush=True)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_checkpoint(step, loss.item(), is_best=True)
                    print("  ✓ new best val loss — saved best.pt", flush=True)
                    plateau_count = 0
                else:
                    plateau_count += 1
                    if plateau_count >= args.early_stop_patience:
                        print(
                            f"  ! Early stopping — no improvement for "
                            f"{args.early_stop_patience} val checks",
                            flush=True,
                        )
                        _save_checkpoint(step, loss.item(), name="final_earlystop.pt")
                        print(f"\nTotal time: {(time.time() - last_log_time) / 60:.1f} min")
                        return model

            # ── Checkpoint (decoupled) ──
            if step > 0 and step % args.ckpt_every == 0:
                _save_checkpoint(step, loss.item())
                if step % (args.ckpt_every * 5) == 0:
                    print(f"  ✓ checkpoint at step {step}", flush=True)

            # ── Signal interrupt ──
            if _state["interrupted"]:
                print(f"\n! Saving interrupt checkpoint at step {step}...", flush=True)
                _save_checkpoint(step, loss.item(), name=f"interrupt_{step:06d}.pt")
                print(
                    f"  ✓ saved. Resume with: --resume {save_path}/interrupt_{step:06d}.pt",
                    flush=True,
                )
                return model

            step += 1
            if step >= args.steps:
                break

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    _save_checkpoint(step, loss.item(), name="final.pt")
    print(f"Total time: {(time.time() - last_log_time) / 60:.1f} min")
    return model


# ═══════════════════════════════════════════════════════════════════
# Export (Version 3 — complex pole weight format)
# ═══════════════════════════════════════════════════════════════════


def export_weights(checkpoint_path: str, output_path: str):
    """Export trained model to flat binary model.bin (Version 3).

    15 arrays per layer (v3 adds log_a_real, a_imag):
    [0]  in_proj_w        [in_proj_dim, d_model]
    [1]  dt_bias          [nheads]
    [2]  B_bias           [nheads, R, d_state]
    [3]  C_bias           [nheads, R, d_state]
    [4]  B_norm_weight    [d_state]
    [5]  C_norm_weight    [d_state]
    [6]  mimo_x           [nheads, R, headdim]
    [7]  mimo_z           [nheads, R, headdim]
    [8]  mimo_o           [nheads, R, headdim]
    [9]  D_proj           [nheads]
    [10] out_proj_w       [d_model, d_inner]
    [11] film_proj_w      [2*d_model, cond_dim]
    [12] dt_film_proj_w   [2*nheads, cond_dim]
    [13] log_a_real       [nheads]          ← NEW in v3
    [14] a_imag           [nheads]          ← NEW in v3
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_raw = ckpt.get("cfg", {})
    if isinstance(cfg_raw, dict):
        cfg = Mamba3Config(
            **{k: v for k, v in cfg_raw.items() if k in Mamba3Config.__dataclass_fields__}
        )
    else:
        cfg = cfg_raw
    state = ckpt["model"]

    state_dict = ckpt["model"]

    weight_data = bytearray()

    # Write R8 extractor weights (shared frontend)
    r8_keys = [k for k in state_dict if k.startswith("r8_extractor.") and "weight" in k]
    for k in sorted(r8_keys):
        arr = state_dict[k].detach().cpu().numpy().astype(np.float32)
        weight_data.extend(arr.tobytes())

    # Per-layer Mamba weights (17 arrays per layer)
    mamba_keys = [
        "in_proj.weight",  # [0]  [in_proj_dim, d_model]
        "dt_bias",  # [1]  [nheads]
        "B_proj.weight",  # [2]  [N*2, R]
        "B_proj.bias",  # [3]  [N*2]
        "C_proj.weight",  # [4]  [head_dim, N*2]
        "mimo_x",  # [5]  [nheads, R, headdim]
        "mimo_z",  # [6]  [nheads, R, headdim]
        "mimo_o",  # [7]  [nheads, R, headdim]
        "D_proj",  # [8]  [nheads]
        "out_proj.weight",  # [9]  [d_model, d_inner]
        "film_proj.weight",  # [10] [2*d_model, cond_dim]
        "film_proj.bias",  # [11] [2*d_model]
        "dt_film_proj.weight",  # [12] [2*nheads, cond_dim]
        "dt_film_proj.bias",  # [13] [2*nheads]
        "log_a_real",  # [14] [nheads]
        "a_imag",  # [15] [nheads]
        "norm.weight",  # [16] [d_model]
        "norm.bias",  # [17] [d_model]
    ]

    for l in range(cfg.n_layers):
        prefix = f"mamba_layers.{l}."
        for key in mamba_keys:
            full_key = f"{prefix}{key}"
            param = state_dict[full_key].detach().cpu().numpy().astype(np.float32)
            weight_data.extend(param.tobytes())

    # Semantic encoder, decoder, cond_mlp, f0_cond, cross_track are
    # Python-only for training. C++ engine uses frozen Mamba stack + R8.

    # Header (64 bytes)

    header = struct.pack(
        "<12I",
        0x4D424D41,  # magic "MBMA"
        3,  # version (3 = complex pole format)
        cfg.d_state,
        cfg.d_model,
        cfg.d_inner,
        cfg.nheads,
        cfg.headdim,
        cfg.mimo_rank,  # = R = 8
        cfg.n_layers,
        cfg.cond_dim,
        cfg.in_proj_dim,  # 1552
        len(weight_data),  # weight_data_size (bytes)
    )
    header += b"\x00" * (64 - len(header))

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(weight_data)

    print(f"Exported: {output_path} ({len(weight_data) + 64:,} bytes)")
    print(f"  Version: 3 (complex poles), R={cfg.mimo_rank}")
    print(f"  Layers: {cfg.n_layers}, arrays/layer: {len(mamba_keys)}")
    total_floats = len(weight_data) // 4
    print(f"  Total floats: {total_floats:,}")


# ═══════════════════════════════════════════════════════════════════
# Main / CLI
# ═══════════════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CLARA Mamba-3 Training Pipeline")

    # Data + I/O
    parser.add_argument("--data", help="Path to MUSDB18-HQ train/ root (not needed for --export)")
    parser.add_argument("--save", default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", help="Resume from checkpoint path")

    # Training schedule
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective_batch=batch*accum)",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument(
        "--clip-seconds", type=float, default=4.0, help="Audio clip duration in seconds"
    )
    parser.add_argument("--amp", action="store_true", help="Enable bf16 AMP (fp32 SSM core)")
    parser.add_argument(
        "--skip-contrastive",
        action="store_true",
        default=True,
        help="Skip Phase 1 contrastive pretraining (default: True)",
    )

    # DataLoader
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-limit", type=int, default=16)
    parser.add_argument(
        "--precomputed",
        type=str,
        default=None,
        help="[Deprecated] StemDataset loads .crepe.npy automatically; "
        "this flag is a passthrough for backward compat.",
    )

    # Validation + checkpointing
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--device", default="cuda")

    # Export mode
    parser.add_argument("--export", help="Export checkpoint to model.bin")
    parser.add_argument("--output", default="models/model_trained.bin")

    args = parser.parse_args()

    if args.export:
        export_weights(args.export, args.output)
    else:
        train(args)


if __name__ == "__main__":
    main()
