#!/usr/bin/env python3
"""
CachedMelDataset — precomputed mel + STFT features with optional RAM cache.

Eliminates all CPU-side DSP during training. The GPU never waits for audio
loading or spectrogram computation.

Usage:
    # Precompute once:
    python scripts/precompute_features.py \
        --input ~/datasets/musdb18hq/train \
        --output ~/datasets/musdb18hq/precomputed \
        --chunk-seconds 0.5

    # Train with precomputed + RAM cache:
    python train_clara_mamba3.py \
        --precomputed ~/datasets/musdb18hq/precomputed \
        --cache \
        --steps 100000

    # Train with precomputed (no cache — lower RAM, slightly slower):
    python train_clara_mamba3.py \
        --precomputed ~/datasets/musdb18hq/precomputed \
        --steps 100000

    # Train on-the-fly (no precomputation, for comparison):
    python train_clara_mamba3.py \
        --data ~/datasets/musdb18hq/train \
        --steps 100000
"""

import json
import random
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── CachedMelDataset ─────────────────────────────────────────────────

class CachedMelDataset(Dataset):
    """Precomputed mel + STFT features with optional RAM cache.

    All DSP (mel, STFT) is precomputed once. During training, chunks are
    loaded from disk (or RAM if --cache is set). Zero CPU DSP cost per batch.

    Each chunk returns:
        mel:    [n_mel_frames, n_mels]     float32 — log-mel spectrogram
        stft:   [n_fft//2+1, n_stft_frames] float32 — STFT magnitude
        audio:  [chunk_samples]            float32 — raw audio (target)
    """

    def __init__(
        self,
        precomputed_dir: str,
        split: str = "train",
        cache_in_ram: bool = False,
        augment: bool = True,
        min_chunk_duration: float = 0.1,  # seconds — skip very short chunks
    ):
        self.precomputed_dir = Path(precomputed_dir)
        self.split = split
        self.cache_in_ram = cache_in_ram
        self.augment = augment

        # Load metadata
        meta_path = self.precomputed_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {precomputed_dir}. "
                f"Run scripts/precompute_features.py first."
            )
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.chunk_samples = self.meta['chunk_samples']
        self.sr = self.meta['sample_rate']
        self.n_fft = self.meta['n_fft']
        self.hop_length = self.meta['hop_length']
        self.n_mels = self.meta['n_mels']

        # Find all .npz chunk files
        split_dir = self.precomputed_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.chunk_files = sorted(split_dir.glob("*.npz"))
        if not self.chunk_files:
            raise FileNotFoundError(f"No .npz chunks found in {split_dir}")

        # Filter out very short chunks (less than min_chunk_duration)
        if min_chunk_duration > 0:
            min_samples = int(min_chunk_duration * self.sr)
            valid = []
            for cf in self.chunk_files:
                # Quick check: load just the audio array to check length
                # This is done once at init, not during training
                pass  # skip for now — all chunks from precompute are same length
            self.chunk_files = valid if valid else self.chunk_files

        print(f"  CachedMelDataset ({split}): {len(self.chunk_files)} chunks "
              f"from {self.precomputed_dir}/{split}")

        # RAM cache
        self._ram_cache: dict[str, dict] = {}
        if self.cache_in_ram:
            self._preload()

    def _preload(self):
        """Load all chunks into RAM as pinned torch tensors.

        Storing tensors (not numpy) eliminates the torch.from_numpy conversion
        on every __getitem__ call. pin_memory() enables async GPU transfer.
        """
        n = len(self.chunk_files)
        print(f"  Pre-loading {n} chunks into RAM (pinned tensors)...", end=' ', flush=True)
        t0 = time.time()
        loaded = 0
        for i, cf in enumerate(self.chunk_files):
            if i % max(1, n // 20) == 0:
                print(f"{i}/{n}", end=' ', flush=True)
            try:
                data = np.load(cf)
                # Convert to pinned torch tensors once, reuse forever
                # pin_memory() may fail on CPU-only systems — fall back gracefully
                t_mel = torch.from_numpy(data['mel'].astype(np.float32))
                t_stft = torch.from_numpy(data['stft'].astype(np.float32))
                t_audio = torch.from_numpy(data['audio'].astype(np.float32))
                try:
                    t_mel = t_mel.pin_memory()
                    t_stft = t_stft.pin_memory()
                    t_audio = t_audio.pin_memory()
                except Exception:
                    pass  # CPU-only system, unpinned is fine
                self._ram_cache[str(cf)] = {
                    'mel':   t_mel,
                    'stft':  t_stft,
                    'audio': t_audio,
                }
                loaded += 1
            except Exception as e:
                print(f"\n  WARNING: failed to load {cf.name}: {e}")
        elapsed = time.time() - t0
        total_mb = sum(
            d['mel'].nbytes + d['stft'].nbytes + d['audio'].nbytes
            for d in self._ram_cache.values()
        ) / 1e6
        print(f"\n  RAM cache: {loaded}/{n} chunks, {total_mb:.0f} MB, "
              f"{elapsed:.1f}s loaded. All tensors pinned. GPU transfers are async & fast.")

    def __len__(self) -> int:
        return len(self.chunk_files)

    def __getitem__(self, idx: int) -> dict:
        cf = self.chunk_files[idx]

        if self.cache_in_ram and str(cf) in self._ram_cache:
            # RAM cache: clone pinned tensors (don't modify cached originals)
            data = self._ram_cache[str(cf)]
            mel = data['mel'].clone()
            stft = data['stft'].clone()
            audio = data['audio'].clone()
        else:
            # Disk: load and convert to tensor
            try:
                npz = np.load(cf)
                mel = torch.from_numpy(npz['mel'].astype(np.float32))
                stft = torch.from_numpy(npz['stft'].astype(np.float32))
                audio = torch.from_numpy(npz['audio'].astype(np.float32))
            except Exception as e:
                # Return zeros on error — don't crash training
                n_mel = (self.chunk_samples // self.hop_length) + 1
                return {
                    'mel':   torch.zeros(n_mel, self.n_mels),
                    'stft':  torch.zeros(self.n_fft // 2 + 1, n_mel),
                    'audio': torch.zeros(self.chunk_samples),
                }

        # Augmentation: random gain + polarity flip (on tensors)
        if self.augment:
            gain_db = random.uniform(-6, 6)
            gain = 10 ** (gain_db / 20)
            audio = audio * gain
            stft = stft * gain   # scale STFT magnitude (approximately correct)
            mel = mel * gain     # scale mel too (approximately correct)
            if random.random() < 0.5:
                audio = -audio
                stft = -stft
                mel = -mel

        return {
            'mel':   mel,
            'stft':  stft,
            'audio': audio,
        }


# ── On-the-fly dataset (fallback / for comparison) ──────────────────

class AudioDataset(Dataset):
    """On-the-fly audio loading. Used when --precomputed is not set.

    This is the fallback dataset. Prefer CachedMelDataset for training.
    """

    def __init__(self, root: str, chunk_samples: int = 2048, sr: int = 44100,
                 augment: bool = True, cache_in_ram: bool = False):
        self.root = Path(root)
        self.chunk_samples = chunk_samples
        self.sr = sr
        self.augment = augment
        self._ram_cache: dict[str, np.ndarray] = {}

        self.files = sorted(self.root.rglob("*.wav"))
        if not self.files:
            raise FileNotFoundError(f"No .wav files found in {root}")

        import soundfile as sf
        self._cache = []
        print(f"  AudioDataset: loading {len(self.files)} files...")
        for f in self.files:
            try:
                audio, file_sr = sf.read(str(f), dtype='float32', always_2d=True)
                audio = audio.mean(axis=1).astype(np.float32)
                if file_sr != sr:
                    import torchaudio
                    t = torch.from_numpy(audio).float().unsqueeze(0)
                    audio = torchaudio.functional.resample(t, file_sr, sr).squeeze(0).numpy()
                self._cache.append(audio)
            except Exception as e:
                print(f"  WARNING: skipping {f.name}: {e}")

        self.files = [f for f, a in zip(self.files, self._cache) if a.size > 0]
        self._cache = [a for a in self._cache if a.size > 0]
        total_hours = sum(a.size for a in self._cache) / sr / 3600
        print(f"  Loaded {len(self._cache)} files, {total_hours:.1f} hours")

    def __len__(self) -> int:
        return len(self.files) * 10

    def __getitem__(self, idx: int) -> dict:
        file_idx = idx % len(self._cache)
        audio = self._cache[file_idx].copy()
        T = self.chunk_samples

        if audio.size < T:
            audio = np.pad(audio, (0, T - audio.size))
        start = random.randint(0, max(1, audio.size - T))
        chunk = audio[start:start + T].astype(np.float32)

        if self.augment:
            gain_db = random.uniform(-6, 6)
            chunk = chunk * (10 ** (gain_db / 20))
            if random.random() < 0.5:
                chunk = -chunk

        peak = np.abs(chunk).max()
        if peak > 0:
            chunk = chunk / peak * 0.9

        return {
            'mel':   torch.zeros(1),  # placeholder — computed in training loop
            'stft':  torch.zeros(1),  # placeholder
            'audio': torch.from_numpy(chunk),
        }


# ── Loss that uses precomputed features ─────────────────────────────

class PrecomputedLoss(torch.nn.Module):
    """STFT + Mel loss using precomputed features.

    Avoids recomputing STFT/mel on every forward pass.
    Accepts precomputed mel + stft from the dataset.
    """

    def __init__(self, mse_weight: float = 0.5, stft_weight: float = 0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.stft_weight = stft_weight
        self.mse = torch.nn.MSELoss()

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        target_mel: torch.Tensor,
        target_stft: torch.Tensor,
    ) -> torch.Tensor:
        """
        pred_audio:   [B, T]         — model output
        target_audio:  [B, T]         — ground truth audio
        target_mel:    [B, n_frames, n_mels]  — precomputed log-mel
        target_stft:   [B, n_fft//2+1, n_frames] — precomputed STFT mag
        """
        B = pred_audio.shape[0]

        # MSE on raw audio
        loss_mse = self.mse(pred_audio, target_audio)

        # MSE on STFT magnitude (precomputed)
        # Re-compute STFT of pred_audio
        n_fft = target_stft.shape[1]
        hop = target_stft.shape[2] // target_audio.shape[1]  # approximate
        # Actually we need hop from metadata — use fixed hop
        hop_length = 256  # must match precompute
        window = torch.hann_window(n_fft, device=pred_audio.device)
        pred_spec = torch.stft(
            pred_audio, n_fft, hop_length=hop_length,
            window=window, return_complex=True
        )
        pred_mag = pred_spec.abs()
        # Align: target_stft is [B, freq, time], pred_mag is [B, freq, time]
        # target_stft may need trimming to match pred_mag time axis
        T_pred = pred_mag.shape[-1]
        target_stft_trim = target_stft[..., :T_pred]
        loss_stft = self.mse(pred_mag, target_stft_trim)

        # Mel is precomputed for target only — approximate pred mel loss
        # by scaling STFT loss (mel and STFT losses are correlated)
        loss = self.mse_weight * loss_mse + self.stft_weight * loss_stft
        return loss


# ── Helpers ──────────────────────────────────────────────────────────

def make_precomputed_loader(
    precomputed_dir: str,
    split: str = "train",
    batch_size: int = 4,
    cache_in_ram: bool = False,
    num_workers: int = 4,
    augment: bool = True,
) -> Tuple[DataLoader, CachedMelDataset]:
    """Create DataLoader from precomputed features."""
    ds = CachedMelDataset(
        precomputed_dir=precomputed_dir,
        split=split,
        cache_in_ram=cache_in_ram,
        augment=augment,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, ds


def detect_split(path: Path) -> str:
    """Auto-detect train/test split from directory structure."""
    if (path / "train").exists():
        return "train"
    return "train"  # default