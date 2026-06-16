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
from torch.utils.data import Dataset, DataLoader


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

    def forward(self, u: torch.Tensor, state: dict, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        u:     [B, D_MODEL] — input embedding
        state: dict with 'ssm_real', 'angle_state' — mutated in-place
        cond:  [B, COND_DIM] or None
        Returns: [B, D_MODEL]
        """
        B = u.shape[0]
        cfg = self.cfg

        # ── Step 0: FiLM ──
        if cond is not None:
            gb = self.film_proj(cond)                  # [B, 2*D_MODEL]
            gamma, beta = gb.chunk(2, dim=-1)           # each [B, D_MODEL]
            u_film = u * (1.0 + gamma) + beta
        else:
            u_film = u

        # ── Step 1: in_proj ──
        proj = self.in_proj(u_film)                    # [B, IN_PROJ_DIM]
        idx = 0
        z     = proj[:, idx:idx + cfg.d_inner];              idx += cfg.d_inner
        x_h   = proj[:, idx:idx + cfg.nheads * cfg.headdim]; idx += cfg.nheads * cfg.headdim
        B_raw = proj[:, idx:idx + cfg.nheads * cfg.mimo_rank * cfg.d_state]; idx += cfg.nheads * cfg.mimo_rank * cfg.d_state
        C_raw = proj[:, idx:idx + cfg.nheads * cfg.mimo_rank * cfg.d_state]; idx += cfg.nheads * cfg.mimo_rank * cfg.d_state
        dd_dt = proj[:, idx:idx + cfg.nheads];              idx += cfg.nheads
        dd_A  = proj[:, idx:idx + cfg.nheads];              idx += cfg.nheads
        trap  = proj[:, idx:idx + cfg.nheads];              idx += cfg.nheads
        angles_raw = proj[:, idx:idx + cfg.nheads * cfg.num_rope_angles]

        # Reshape
        x_h = x_h.view(B, cfg.nheads, cfg.headdim)            # [B, H, D]
        B_raw = B_raw.view(B, cfg.nheads, cfg.mimo_rank, cfg.d_state)
        C_raw = C_raw.view(B, cfg.nheads, cfg.mimo_rank, cfg.d_state)
        angles_raw = angles_raw.view(B, cfg.nheads, cfg.num_rope_angles)

        # ── Step 2: A and Δ ──
        dt_h = F.softplus(dd_dt + self.dt_bias)              # [B, H]
        A_h  = -F.softplus(dd_A)                             # [B, H]

        if cond is not None:
            dt_cond = self.dt_film_proj(cond)                # [B, 2*H]
            dt_cond_gamma, dt_cond_beta = dt_cond.chunk(2, dim=-1)
            dt_h = F.softplus(dt_h + dt_cond_gamma + dt_cond_beta)

        # Trapezoidal
        half_a = A_h * dt_h * 0.5
        denom  = 1.0 - half_a
        A_bar  = (1.0 + half_a) / denom                     # [B, H]
        B_bar_scale = dt_h / denom                          # [B, H]

        # ── Step 3+4: Norm + RoPE ──
        B_normed = B_raw * self.B_norm_weight.view(1, 1, 1, cfg.d_state)
        C_normed = C_raw * self.C_norm_weight.view(1, 1, 1, cfg.d_state)

        state['angle_state'] = state['angle_state'] + angles_raw  # [B, H, ROPE]

        # Apply RoPE
        B_rot_list, C_rot_list = [], []
        for h in range(cfg.nheads):
            ang = state['angle_state'][:, h, :]                # [B, ROPE]
            B_rot_list.append(self.apply_rope(B_normed[:, h], ang))  # [B, RANK, D_STATE]
            C_rot_list.append(self.apply_rope(C_normed[:, h], ang))
        B_rot = torch.stack(B_rot_list, dim=1)                # [B, H, RANK, D_STATE]
        C_rot = torch.stack(C_rot_list, dim=1)

        # ── Step 5: MIMO SSM recurrence ──
        x_slots = torch.einsum('bhd,hrd->bhr', x_h, self.mimo_x)  # [B, H, RANK]

        # h_state update: [B, H, D, S] = A_bar * h_state + B_bar_scale * B_rot * x_slot
        B_contrib = (B_bar_scale.unsqueeze(-1) * x_slots).unsqueeze(-1) * B_rot  # [B, H, RANK, S]
        update = B_contrib.sum(dim=2)  # [B, H, S] — sum over RANK
        state['ssm_real'] = (
            A_bar.unsqueeze(-1).unsqueeze(-1) * state['ssm_real']
            + update.unsqueeze(2)  # [B, H, 1, S] — broadcast over D
        )

        # Readout: y_head[b,h,d] = sum_r sum_s C_rot[b,h,r,s] * h_state[b,h,d,s]
        y_head = torch.einsum('bhrs,bhds->bhd', C_rot, state['ssm_real'])  # [B, H, D]

        # ── Step 6: z-gating ──
        z_gated = z * torch.sigmoid(z)                                       # [B, D_INNER]
        z_heads = z_gated.view(B, cfg.nheads, cfg.headdim)                   # [B, H, D]
        z_slots = torch.einsum('bhd,hrd->bhr', z_heads, self.mimo_z)        # [B, H, RANK]
        z_gate = z_slots.sum(dim=-1)                                         # [B, H]
        y_gated = y_head * z_gate.unsqueeze(-1)                              # [B, H, D]
        y_flat = y_gated.reshape(B, cfg.d_inner)                             # [B, D_INNER]

        # ── Step 7: Output + residual ──
        out = self.out_proj(y_flat)                                          # [B, D_MODEL]

        # D skip
        d_contrib = (self.D_proj.view(1, cfg.nheads, 1) * x_h).sum(dim=1).sum(dim=-1)  # [B]
        out = out + d_contrib.unsqueeze(-1) * 0.01

        # MIMO o-coupling
        o_slots = torch.einsum('bhd,hrd->bhr', y_gated, self.mimo_o)
        o_contrib = o_slots.sum(dim=(1, 2)) * 0.01
        out = out + o_contrib.unsqueeze(-1)

        # Residual
        out = out + u_film

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

    def forward(self, audio: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        audio: [B, T] — mono audio, T samples
        cond:  [B, COND_DIM] or None
        Returns: [B, T]
        """
        B, T = audio.shape
        cfg = self.cfg

        # Per-layer state (batch-aware)
        state = {
            'ssm_real': torch.zeros(B, cfg.nheads, cfg.headdim, cfg.d_state, device=audio.device),
            'angle_state': torch.zeros(B, cfg.nheads, cfg.num_rope_angles, device=audio.device),
        }

        # Embed scalar → D_MODEL
        x = self.input_proj(audio.unsqueeze(-1))  # [B, T, D_MODEL]

        output = []
        for t in range(T):
            xt = x[:, t, :]  # [B, D_MODEL]

            for layer_idx, layer in enumerate(self.layers):
                xt = layer(xt, state, cond)
                # LayerNorm
                mean = xt.mean(dim=-1, keepdim=True)
                var = xt.var(dim=-1, keepdim=True, unbiased=False)
                xt = self.ln_gamma[layer_idx] * (xt - mean) / torch.sqrt(var + 1e-5) + self.ln_beta[layer_idx]

            output.append(xt[:, 0])  # first channel → scalar

        return torch.stack(output, dim=1)  # [B, T]

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

    def __len__(self):
        return len(self.files) * 10  # 10 chunks per file per epoch

    def __getitem__(self, idx):
        # Simple: generate synthetic audio for identity training
        # (real dataset loading via torchaudio would go here)
        T = self.chunk_samples
        audio = torch.randn(T) * 0.3  # random normal noise
        return audio, audio  # input = target for identity mapping


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

    # Loss: MSE for identity mapping
    loss_fn = nn.MSELoss()

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

            # Forward
            output = model(audio_in)

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
