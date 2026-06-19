#!/usr/bin/env python3
"""
CLARA Mamba RT Parallel Training — trains the 76K-param exportable model.
===========================================================================
Model: MambaRTModelParallel (from train_clara_model.py)
  - Time-domain (NO STFT) — matches ONNX export architecture
  - 10-input / 3-output I/O contract
  - Directly exportable to ONNX via train_clara_export.py

Phase 1: Identity autoencoder — model learns to reconstruct input.
  - cond vector = zeros (no conditioning)
  - Loss: STFT magnitude L1 + Mel-L1 + SI-SDR (ramped in)

Usage:
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    python train_clara_parallel.py --data ~/datasets/musdb18hq/train

Resume:
    python train_clara_parallel.py --data ~/datasets/musdb18hq/train \\
        --resume checkpoints_clara/checkpoint_005000.pt
===========================================================================
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchaudio

warnings.filterwarnings("ignore", message="At least one mel filterbank has all zero values")

if sys.platform != 'win32':
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# Import the correct model — the one that CAN be exported to ONNX
sys.path.insert(0, os.path.dirname(__file__))
from train_clara_model import MambaRTModelParallel, MambaRTModelSequential, make_zero_states, make_zero_cond


# ══════════════════════════════════════════════════════════════════════════
# Dataset — flat WAV files (reused from train_clara_now.py)
# ══════════════════════════════════════════════════════════════════════════

class FlatWavDataset(Dataset):
    """Dataset that loads WAV files, crops to clip_seconds, augments."""
    def __init__(self, root: str, sample_rate: int = 44100,
                 clip_seconds: float = 0.5, augment: bool = True,
                 cache_in_ram: bool = False):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.clip_len = int(clip_seconds * sample_rate)
        self.augment = augment
        self.cache_in_ram = cache_in_ram
        self._ram_cache: dict[str, Tensor] = {}

        wav_files  = list(self.root.rglob("*.wav"))  + list(self.root.rglob("*.WAV"))
        flac_files = list(self.root.rglob("*.flac")) + list(self.root.rglob("*.FLAC"))
        self.audio_files = sorted([f for f in wav_files + flac_files if f.is_file()])
        assert len(self.audio_files) > 0, f"No audio files (*.wav, *.flac) found in {root}"

        if self.cache_in_ram:
            self._preload()

    def _preload(self):
        n = len(self.audio_files)
        total_size_bytes = 0
        estimate_gb = os.path.getsize(self.audio_files[0]) if n > 0 else 0
        if estimate_gb > 0:
            est_total_gb = (estimate_gb / 1e9) * n
            print(f"Pre-loading {n} files into RAM (est. {est_total_gb:.1f} GB raw WAV)...")
            # Warn if this seems tight — but don't abort; let the user decide
            try:
                with open('/proc/meminfo') as f:
                    for line in f:
                        if 'MemAvailable' in line:
                            avail_kb = int(line.split()[1])
                            avail_gb = avail_kb / 1024 / 1024
                            if avail_gb < est_total_gb * 1.5:
                                print(f"  ⚠ WARNING: only {avail_gb:.1f} GB free RAM for ~{est_total_gb:.1f} GB cache!")
                                print(f"  If training dies silently, re-run without --cache or use --num-workers 0 --no-cache")
                            break
            except Exception:
                pass
        print(f"  Loading [{n} files]:", end=' ', flush=True)
        for i, path in enumerate(self.audio_files):
            if i % max(1, n // 20) == 0:
                print(f"{i}/{n}", end=' ', flush=True)
            wav = self._load_file_uncached(path)
            if wav is not None:
                self._ram_cache[str(path)] = wav
                total_size_bytes += wav.numel() * 4
            else:
                print(f"\n  WARNING: Failed to load {path.name}", flush=True)
        total_gb = total_size_bytes / 1e9
        print(f"\n  RAM cache: {len(self._ram_cache)}/{n} files, {total_gb:.2f} GB", flush=True)

    def _load_file_uncached(self, path: Path) -> Optional[Tensor]:
        try:
            data, sr = sf.read(str(path), dtype='float32', always_2d=True)
        except Exception as e:
            print(f"\n  ERROR loading {path.name}: {e}", flush=True)
            return None
        if data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data[:, 0]
        wav = torch.from_numpy(data.copy()).float()
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, self.sample_rate).squeeze(0)
        return wav

    def _load_file(self, path: Path) -> Optional[Tensor]:
        if self.cache_in_ram:
            return self._ram_cache.get(str(path))
        return self._load_file_uncached(path)

    def __len__(self) -> int:
        return len(self.audio_files)

    def _crop(self, wav: Tensor) -> Tensor:
        L = wav.shape[-1]
        if L <= self.clip_len:
            return F.pad(wav, (0, self.clip_len - L))
        start = random.randint(0, L - self.clip_len)
        return wav[start:start + self.clip_len]

    def _augment(self, wav: Tensor) -> Tensor:
        gain_db = random.uniform(-6, 6)
        wav = wav * (10 ** (gain_db / 20))
        if random.random() < 0.5:
            wav = -wav
        return wav.clamp(-1.0, 1.0)

    def __getitem__(self, idx: int) -> dict:
        file_idx = idx % len(self.audio_files)
        path = self.audio_files[file_idx]
        wav = self._load_file(path)
        if wav is None:
            return {'audio': torch.zeros(self.clip_len), 'valid': False}
        wav = self._crop(wav)
        if self.augment:
            wav = self._augment(wav)
        return {'audio': wav, 'valid': True}


def collate_valid(batch: list[dict]) -> Optional[dict]:
    """Filter out invalid samples, stack valid ones."""
    batch = [b for b in batch if b['valid']]
    if not batch:
        return None
    return {'audio': torch.stack([b['audio'] for b in batch])}


# ══════════════════════════════════════════════════════════════════════════
# Loss — STFT magnitude L1 + Mel-L1 + SI-SDR (matches LeanClaraLoss)
# ══════════════════════════════════════════════════════════════════════════

class ClaraLoss(nn.Module):
    """Identity training loss: STFT-L1 + Mel-L1 + SI-SDR (ramped)."""
    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.sr = sample_rate
        self.n_fft = 1024
        self.hop = 256
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=64)
        self.global_step = 0

    def _si_sdr(self, pred: Tensor, target: Tensor) -> Tensor:
        eps = 1e-8
        target_energy = (target ** 2).sum(dim=-1, keepdim=True)
        dot = (pred * target).sum(dim=-1, keepdim=True)
        proj = dot / (target_energy + eps) * target
        noise = pred - proj
        si_sdr = 10 * torch.log10((proj ** 2).sum(-1) / ((noise ** 2).sum(-1) + eps) + eps)
        return -si_sdr.mean()

    def _mel_l1(self, pred_wave: Tensor, target_wave: Tensor) -> Tensor:
        mel = self.mel_transform.to(pred_wave.device)
        p = torch.log1p(mel(pred_wave))
        t = torch.log1p(mel(target_wave))
        return F.l1_loss(p, t)

    def _si_sdr_weight(self) -> float:
        if self.global_step < 5000:
            return 0.0
        elif self.global_step > 15000:
            return 0.1
        else:
            return 0.1 * (self.global_step - 5000) / 10000

    def forward(self, pred_wave: Tensor, target_wave: Tensor) -> dict[str, Tensor]:
        # Reshape: model outputs (B, T, 1), loss expects (B, T)
        if pred_wave.dim() == 3 and pred_wave.shape[-1] == 1:
            pred_wave = pred_wave.squeeze(-1)
        if target_wave.dim() == 3 and target_wave.shape[-1] == 1:
            target_wave = target_wave.squeeze(-1)

        L = min(pred_wave.shape[-1], target_wave.shape[-1])
        pred_wave = pred_wave[..., :L]
        target_wave = target_wave[..., :L]

        losses = {}

        # STFT magnitude L1
        window = torch.hann_window(self.n_fft, device=pred_wave.device)
        p_stft = torch.stft(pred_wave, self.n_fft, self.hop, window=window, return_complex=True)
        t_stft = torch.stft(target_wave, self.n_fft, self.hop, window=window, return_complex=True)
        losses['stft'] = F.l1_loss(p_stft.abs(), t_stft.abs())

        # Mel-L1
        losses['mel_l1'] = self._mel_l1(pred_wave, target_wave)

        # SI-SDR (ramped in)
        si_w = self._si_sdr_weight()
        if si_w > 0:
            losses['si_sdr'] = self._si_sdr(pred_wave, target_wave) * si_w
        else:
            losses['si_sdr'] = torch.tensor(0.0, device=pred_wave.device)

        losses['total'] = 1.0 * losses['stft'] + 0.5 * losses['mel_l1'] + losses['si_sdr']

        self.global_step += 1
        return losses


# ══════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════

def create_model(d_model=64, d_state=16, n_layers=4, conv_width=4, dt_rank=16, cond_dim=8):
    """Create the parallel-scan training model (exportable to ONNX)."""
    model = MambaRTModelParallel(
        d_model=d_model, d_state=d_state, n_layers=n_layers,
        conv_width=conv_width, dt_rank=dt_rank, cond_dim=cond_dim,
    )
    return model


def get_lr(step: int, warmup: int = 2000, max_lr: float = 3e-4) -> float:
    """Cosine schedule with linear warmup."""
    if step < warmup:
        return max_lr * step / warmup
    progress = (step - warmup) / max(1, 50000 - warmup)
    return max_lr * 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))


def save_checkpoint(model, optimizer, scheduler, step, loss, best_val_loss,
                    arch, save_dir, is_best=False):
    """Save checkpoint in format compatible with train_clara_export.py."""
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    ckpt = {
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else {},
        'loss': loss,
        'best_val_loss': best_val_loss,
        'architecture': arch,
    }

    # Regular checkpoint
    ckpt_path = path / f'checkpoint_{step:06d}.pt'
    torch.save(ckpt, ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    # Best validation checkpoint
    if is_best:
        best_path = path / 'best.pt'
        torch.save(ckpt, best_path)
        print(f"  Best model updated: {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load checkpoint. Handles architecture mismatches gracefully."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} (expected for fresh start)")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)} (ignored)")
    return ckpt


def validate(model, val_loader, loss_fn, device, cond_dim=8):
    """Run validation pass, return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            audio = batch['audio'].to(device)  # (B, T)
            B = audio.shape[0]

            # Prepare model inputs: audio (B, T, 1), cond (B, 8), zero states
            audio_in = audio.unsqueeze(-1)  # (B, T, 1)
            cond = make_zero_cond(B, cond_dim).to(device)
            convs, ssms = make_zero_states(B, 64, 16, 4)
            convs = tuple(c.to(device) for c in convs)
            ssms = tuple(s.to(device) for s in ssms)

            out, _, _ = model(audio_in, *convs, *ssms, cond, chunk_size=64)

            losses = loss_fn(out, audio)
            total_loss += losses['total'].item()
            n_batches += 1

    model.train()
    return total_loss / max(1, n_batches)


def train(args):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Create model
    model = create_model(
        d_model=args.d_model, d_state=args.d_state,
        n_layers=args.n_layers, conv_width=args.conv_width,
        dt_rank=args.dt_rank, cond_dim=args.cond_dim,
    )
    model.to(device)
    arch_name = f"MambaRTModelParallel_d{args.d_model}_l{args.n_layers}"
    print(f"Model: {arch_name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer with weight decay filter
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'log_A_real' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': args.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=args.lr)

    scheduler = None
    start_step = 0
    best_val_loss = float('inf')
    loss_fn = ClaraLoss(sample_rate=args.sample_rate)
    loss_fn = loss_fn.to(device)

    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = load_checkpoint(args.resume, model)
        start_step = ckpt.get('step', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        if 'optimizer' in ckpt and args.resume_optimizer:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
                print("  Optimizer state loaded")
                for group in optimizer.param_groups:
                    group['lr'] = args.lr
            except Exception as e:
                print(f"  Optimizer load failed (starting fresh): {e}")
        loss_fn.global_step = start_step
        print(f"  Resumed at step {start_step}, best_val_loss={best_val_loss:.6f}")

    # Dataset
    data_root = Path(args.data)
    train_root = data_root / 'train' if (data_root / 'train').exists() else data_root
    val_root = data_root / 'test' if (data_root / 'test').exists() else None

    if val_root:
        train_ds = FlatWavDataset(
            root=str(train_root), sample_rate=args.sample_rate,
            clip_seconds=args.clip_seconds, augment=True,
            cache_in_ram=args.cache,
        )
        val_ds = FlatWavDataset(
            root=str(val_root), sample_rate=args.sample_rate,
            clip_seconds=args.clip_seconds, augment=False,
            cache_in_ram=False,
        )
    else:
        # No test/ subdirectory — auto-split files 90/10 (seed 42)
        train_ds = FlatWavDataset(
            root=str(train_root), sample_rate=args.sample_rate,
            clip_seconds=args.clip_seconds, augment=True,
            cache_in_ram=args.cache,
        )
        all_files = list(train_ds.audio_files)
        rng = random.Random(42)
        rng.shuffle(all_files)
        split = max(1, int(len(all_files) * 0.9))
        train_ds.audio_files = all_files[:split]
        val_files = all_files[split:]
        print(f"  Auto-split: {len(train_ds.audio_files)} train, {len(val_files)} val "
              f"(seed=42, 90/10)")

        val_ds = FlatWavDataset(
            root=str(train_root), sample_rate=args.sample_rate,
            clip_seconds=args.clip_seconds, augment=False,
            cache_in_ram=False,
        )
        val_ds.audio_files = val_files

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_valid, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=0, pin_memory=True,
        collate_fn=collate_valid, drop_last=True,
    ) if val_ds.audio_files else None

    print(f"Train: {len(train_ds):,} chunks from {len(train_ds.audio_files)} files")
    if val_ds:
        print(f"Val:   {len(val_ds):,} chunks from {len(val_ds.audio_files)} files")
    print(f"Batch: {args.batch}, Accum: {args.grad_accum}, Eff batch: {args.batch * args.grad_accum}")

    # Training loop
    model.train()
    step = start_step
    current_lr = args.lr
    t0 = time.time()
    best_loss_sofar = best_val_loss
    early_stop_counter = 0

    # GPU warmup — single chunk to avoid MIOpen compilation hang on ROCm.
    # Runs BEFORE iter(train_loader) to avoid 'spawn' multiprocess conflict.
    print("\nGPU warmup...", end=' ', flush=True)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model.eval()
    try:
        with torch.no_grad():
            warmup_len = min(64, int(args.clip_seconds * args.sample_rate))
            dummy = torch.randn(1, warmup_len, 1, device=device)
            cd = make_zero_cond(1, args.cond_dim).to(device)
            cs, ss = make_zero_states(1, args.d_model, args.d_state, args.n_layers, args.conv_width)
            cs = tuple(c.to(device) for c in cs)
            ss = tuple(s.to(device) for s in ss)
            model(dummy, *cs, *ss, cd, chunk_size=64)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        print("done")
    except Exception as e:
        print(f"\nWARNING: GPU warmup failed ({e}) — continuing with cold GPU")
    model.train()

    train_iter = iter(train_loader)

    print(f"\n{'='*60}")
    print(f"Starting Phase 1 Identity Training")
    print(f"Steps: {step} → {args.steps}")
    print(f"{'='*60}\n")

    while step < args.steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if batch is None:
            continue

        audio = batch['audio'].to(device)
        B = audio.shape[0]
        T = audio.shape[-1]

        # Prepare model inputs
        audio_in = audio.unsqueeze(-1)  # (B, T, 1)
        cond = make_zero_cond(B, args.cond_dim).to(device)
        convs, ssms = make_zero_states(B, args.d_model, args.d_state, args.n_layers, args.conv_width)
        convs = tuple(c.to(device) for c in convs)
        ssms = tuple(s.to(device) for s in ssms)

        # Forward (with chunked sub-block processing matching C++ engine)
        out, _, _ = model(audio_in, *convs, *ssms, cond, chunk_size=64)

        # Loss
        losses = loss_fn(out, audio)
        loss_total = losses['total']

        # Update LR every step (for logging accuracy)
        current_lr = get_lr(step, warmup=args.warmup, max_lr=args.lr)

        # Backward (with gradient accumulation)
        (loss_total / args.grad_accum).backward()

        # Optimizer step
        if (step - start_step + 1) % args.grad_accum == 0:
            # Gradient clipping
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            # Apply LR to optimizer
            for group in optimizer.param_groups:
                group['lr'] = current_lr

            optimizer.step()
            optimizer.zero_grad()

        # Logging
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            steps_since = min(args.log_every, step - start_step + 1)
            rate = steps_since / max(elapsed, 1e-6)
            # Get current LR from optimizer (safe fallback if warmup logic not hit)
            lr_val = current_lr if 'current_lr' in dir() else optimizer.param_groups[0]['lr']
            print(
                f"[step {step:6d}] "
                f"loss={losses['total'].item():.4f} "
                f"stft={losses['stft'].item():.4f} "
                f"mel={losses['mel_l1'].item():.4f} "
                f"lr={lr_val:.2e} "
                f"{rate:.1f} step/s"
            )
            t0 = time.time()

        # Validation
        if val_loader and step > 0 and step % args.val_every == 0:
            val_loss = validate(model, val_loader, loss_fn, device, args.cond_dim)
            print(f"  [VAL] step {step:6d}: val_loss={val_loss:.6f} "
                  f"(best={best_val_loss:.6f})")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, step, loss_total.item(),
                                best_val_loss, arch_name, args.save, is_best=True)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= args.early_stop_patience:
                    print(f"  Early stopping at step {step}")
                    break

        # Checkpoint
        if step > 0 and step % args.ckpt_every == 0:
            save_checkpoint(model, optimizer, scheduler, step, loss_total.item(),
                            best_val_loss, arch_name, args.save)

        step += 1

    # Final save — loss_total may not exist if loop never ran
    final_loss = locals().get('loss_total', torch.tensor(0.0)).item()
    save_checkpoint(model, optimizer, scheduler, step, final_loss,
                    best_val_loss, arch_name, args.save)

    # Make final checkpoint compatible with train_clara_export.py
    final_path = Path(args.save) / 'final.pt'
    try:
        ckpt = torch.load(Path(args.save) / f'checkpoint_{step:06d}.pt',
                           map_location='cpu', weights_only=False)
        ckpt['model_state_dict'] = ckpt.pop('model')
        torch.save(ckpt, final_path)
        print(f"\nFinal model saved: {final_path}")
    except Exception as e:
        print(f"\nCould not create final.pt (last ckpt may be missing): {e}")
        # Fallback: save current model directly
        final_ckpt = {
            'step': step, 'model_state_dict': model.state_dict(),
            'architecture': arch_name,
        }
        torch.save(final_ckpt, final_path)
        print(f"Fallback final model saved: {final_path}")

    print(f"\n{'='*60}")
    print(f"Training Complete! {step} steps, best_val_loss={best_val_loss:.6f}")
    print(f"Next: python train_clara_export.py --checkpoint {final_path} --output models/mamba_rt.onnx")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CLARA Mamba RT Parallel Training")
    parser.add_argument('--data', required=True, help='Path to MUSDB18-HQ dataset (train dir)')
    parser.add_argument('--save', default='checkpoints_clara', help='Checkpoint save directory')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint path')
    parser.add_argument('--resume-optimizer', action='store_true',
                        help='Load optimizer state on resume')
    parser.add_argument('--steps', type=int, default=50000, help='Total training steps')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--grad-accum', type=int, default=4, help='Grad accumulation steps')
    parser.add_argument('--lr', type=float, default=3e-4, help='Max learning rate')
    parser.add_argument('--warmup', type=int, default=2000, help='Warmup steps')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--clip-grad-norm', type=float, default=1.0, help='Gradient clipping norm')

    # Model architecture
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--d-state', type=int, default=16)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--conv-width', type=int, default=4)
    parser.add_argument('--dt-rank', type=int, default=16)
    parser.add_argument('--cond-dim', type=int, default=8)

    # Data
    parser.add_argument('--sample-rate', type=int, default=44100)
    parser.add_argument('--clip-seconds', type=float, default=0.5,
                        help='Clip duration. 0.5=fast, 2.0=better quality, 0.25=smoke test')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers (0=safe for ROCm/APU, 2-4=faster)')
    parser.add_argument('--cache', action='store_true', help='RAM pre-cache dataset')

    # Logging
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--ckpt-every', type=int, default=500)
    parser.add_argument('--val-every', type=int, default=500)
    parser.add_argument('--early-stop-patience', type=int, default=10)

    # Hardware
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Override device selection')

    args = parser.parse_args()
    print(f"CLARA Mamba RT Parallel Training")
    print(f"{'='*50}")
    print(f"PyTorch: {torch.__version__}")
    print(f"ROCm/HIP: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
    print(f"CUDA avail: {torch.cuda.is_available()}")

    train(args)


if __name__ == '__main__':
    main()
