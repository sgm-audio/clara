#!/usr/bin/env python3
"""
CLARA Mamba-3 Dataset — dataset.py

Multi-stem MUSDB18-HQ dataset with pre-extracted CREPE f0.

Architecture reference: mamba3 PerTrackProcessor StemDataset.
Supports: pitch-shift augmentation, LRU caching, collate_valid.

Directory layout expected:
  root/
    Artist Name - Track Title/
      bass.wav + bass.crepe.npy + bass.pitched.npy
      drums.wav + drums.crepe.npy + drums.pitched.npy
      vocals.wav + vocals.crepe.npy + vocals.pitched.npy
      other.wav + other.crepe.npy + other.pitched.npy
      mixture.wav (optional)
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════
# StemDataset
# ═══════════════════════════════════════════════════════════════════

class StemDataset(Dataset):
    """
    Multi-stem dataset for MUSDB18-HQ per-track processing.

    Returns dict with:
      target:       (L,)   — target stem waveform
      other_stems:  (K, L) — OTHER stems in same track (stacked)
      target_crepe: (T, 360) — CREPE f0 probabilities for target
      other_crepe:  (K, T, 360) — CREPE for other stems
      target_pitched: float — 0.0 or 1.0 (is this stem pitched?)
      other_pitched: (K,) — per-stem pitched flags
      stem_name:    str
      valid:        bool  — False if missing stems (filtered by collate)
    """

    STEM_NAMES = ['bass', 'drums', 'vocals', 'other']

    def __init__(
        self,
        root: str,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        clip_seconds: float = 4.0,
        augment: bool = True,
        is_val: bool = False,
        cache_limit: int = 16,
        min_complete_tracks: int = 1,
    ):
        self.root = Path(root)
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.clip_len = int(clip_seconds * sample_rate)
        self.T_frames = 1 + self.clip_len // self.hop_length  # ~345 @ 4s
        self.augment = augment and not is_val
        self.is_val = is_val
        self.min_complete_tracks = min_complete_tracks

        # Discover tracks that have ALL WAV stems
        self.tracks: list[Path] = []
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            # Check all 4 stems exist
            if all((d / f"{s}.wav").exists() for s in self.STEM_NAMES):
                self.tracks.append(d)

        n_tracks = len(self.tracks)
        assert n_tracks >= min_complete_tracks, (
            f"Found {n_tracks} complete tracks in {root}, "
            f"need at least {min_complete_tracks}"
        )
        print(f"  StemDataset: {n_tracks} tracks, {n_tracks * 4} stems")

        # LRU cache
        self.cache_limit = cache_limit
        self._cache_queue: list[str] = []
        self._cache: dict[str, dict] = {}  # {track_key: {'stems': {}, 'crepe': {}}}

    def __len__(self) -> int:
        """4 crops per stem per track."""
        return len(self.tracks) * len(self.STEM_NAMES) * 4

    # ── Loading ─────────────────────────────────────────────────────

    def _load_stem(self, track_dir: Path, stem_name: str) -> Optional[Tensor]:
        """Load stem WAV, resample to target SR, return (L,) tensor."""
        key = f"{track_dir}/{stem_name}"

        # Check cache
        if track_dir.name in self._cache:
            if stem_name in self._cache[track_dir.name]['stems']:
                return self._cache[track_dir.name]['stems'][stem_name]

        # Load from disk
        path = track_dir / f"{stem_name}.wav"
        if not path.exists():
            return None

        try:
            data, sr = sf.read(str(path), dtype='float32', always_2d=True)
            # Mix to mono
            if data.shape[1] > 1:
                data = data.mean(axis=1)
            else:
                data = data[:, 0]
            wav = torch.from_numpy(data).float()  # (L,)

            # Resample if needed
            if sr != self.sr:
                wav = torchaudio.functional.resample(
                    wav.unsqueeze(0), sr, self.sr
                ).squeeze(0)
        except Exception as e:
            print(f"  WARNING: failed to load {path}: {e}")
            return None

        # Cache
        self._cache_stem(track_dir.name, stem_name, wav)
        return wav

    def _load_crepe(self, track_dir: Path, stem_name: str,
                    target_len: int) -> tuple[Tensor, float]:
        """
        Load pre-extracted CREPE f0.

        Returns:
          probs:   (T_model, 360) float32 or zeros if missing
          pitched: 0.0 or 1.0 (from .pitched.npy)
        """
        key = f"{track_dir}/{stem_name}"

        # Check cache
        if track_dir.name in self._cache:
            if 'crepe' in self._cache[track_dir.name] and \
               stem_name in self._cache[track_dir.name]['crepe']:
                return self._cache[track_dir.name]['crepe'][stem_name]

        crepe_path = track_dir / f"{stem_name}.crepe.npy"
        pitched_path = track_dir / f"{stem_name}.pitched.npy"

        T_expected = self.T_frames

        if not crepe_path.exists():
            result = (torch.zeros(T_expected, 360, dtype=torch.float32), 0.0)
        else:
            try:
                probs = np.load(str(crepe_path)).astype(np.float32)
                pitched = float(np.load(str(pitched_path)))
                result = (torch.from_numpy(probs).float(), pitched)
            except Exception:
                result = (torch.zeros(T_expected, 360, dtype=torch.float32), 0.0)

        self._cache_crepe(track_dir.name, stem_name, result)
        return result

    # ── Cache management ───────────────────────────────────────────

    def _cache_stem(self, track_key: str, stem_name: str, wav: Tensor):
        if track_key not in self._cache:
            self._evict_if_full()
            self._cache[track_key] = {'stems': {}, 'crepe': {}}
            self._cache_queue.append(track_key)
        self._cache[track_key]['stems'][stem_name] = wav

    def _cache_crepe(self, track_key: str, stem_name: str,
                     result: tuple[Tensor, float]):
        if track_key not in self._cache:
            self._evict_if_full()
            self._cache[track_key] = {'stems': {}, 'crepe': {}}
            self._cache_queue.append(track_key)
        self._cache[track_key]['crepe'][stem_name] = result

    def _evict_if_full(self):
        if len(self._cache_queue) >= self.cache_limit:
            oldest = self._cache_queue.pop(0)
            if oldest in self._cache:
                del self._cache[oldest]

    # ── Crop / align ───────────────────────────────────────────────

    def _crop_with_crepe(self, wav: Tensor, crepe: Tensor
                         ) -> tuple[Tensor, Tensor]:
        """
        Sync-crop waveform and CREPE tensor so frames stay aligned.

        wav:   (L,) at sample_rate
        crepe: (T_full, 360) — full-track CREPE probs

        Returns (wav_crop, crepe_crop) aligned to clip_len.
        """
        L = wav.shape[-1]
        T_clip = self.T_frames

        if L <= self.clip_len:
            # Pad waveform
            wav_padded = F.pad(wav, (0, self.clip_len - L))
            # Pad/truncate CREPE
            T_have = crepe.shape[0]
            if T_have >= T_clip:
                crepe_out = crepe[:T_clip]
            else:
                crepe_out = F.pad(crepe, (0, 0, 0, T_clip - T_have))
            return wav_padded, crepe_out

        # Random sample-aligned start
        start = random.randint(0, L - self.clip_len)
        wav_crop = wav[start:start + self.clip_len]

        # Convert sample start → frame start (center=True STFT)
        frame_start = start // self.hop_length
        frame_end = frame_start + T_clip
        T_have = crepe.shape[0]

        if frame_end <= T_have:
            crepe_crop = crepe[frame_start:frame_end]
        else:
            crepe_crop = crepe[frame_start:]
            pad_amt = T_clip - crepe_crop.shape[0]
            crepe_crop = F.pad(crepe_crop, (0, 0, 0, pad_amt))

        return wav_crop, crepe_crop

    # ── Pitch shift augmentation ───────────────────────────────────

    @staticmethod
    def _pitch_shift_waveform(wav: Tensor, semitones: float, sr: int
                              ) -> Tensor:
        """Pitch shift via resampling. wav: (L,). Returns (L,)."""
        if abs(semitones) < 0.01:
            return wav
        factor = 2 ** (semitones / 12.0)
        orig_len = wav.shape[-1]
        new_sr = int(sr * factor)
        shifted = torchaudio.functional.resample(
            wav.unsqueeze(0), sr, new_sr
        ).squeeze(0)
        if shifted.shape[-1] > orig_len:
            shifted = shifted[..., :orig_len]
        else:
            shifted = F.pad(shifted, (0, orig_len - shifted.shape[-1]))
        return shifted

    @staticmethod
    def _pitch_shift_crepe(crepe: Tensor, semitones: float) -> Tensor:
        """
        Roll CREPE 360-D probability vector to match pitch shift.
        20 cents per bin → 5 bins per semitone.
        crepe: (T, 360)
        """
        if abs(semitones) < 0.01:
            return crepe
        bins_shift = int(round(5 * semitones))
        if bins_shift == 0:
            return crepe

        rolled = torch.zeros_like(crepe)
        if bins_shift > 0 and bins_shift < 360:
            rolled[..., bins_shift:] = crepe[..., :-bins_shift]
        elif bins_shift < 0 and -bins_shift < 360:
            rolled[..., :bins_shift] = crepe[..., -bins_shift:]
        else:
            rolled = crepe
        return rolled

    # ── Standard augmentations (gain, polarity) ────────────────────

    @staticmethod
    def _augment_wav(wav: Tensor) -> Tensor:
        """Random gain ±6 dB + random polarity flip."""
        gain_db = random.uniform(-6, 6)
        wav = wav * (10 ** (gain_db / 20))
        if random.random() < 0.5:
            wav = -wav
        return wav.clamp(-1.0, 1.0)

    # ── Get item ────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        track_idx = (idx // (len(self.STEM_NAMES) * 4)) % len(self.tracks)
        stem_idx = (idx // 4) % len(self.STEM_NAMES)
        track_dir = self.tracks[track_idx]
        target_name = self.STEM_NAMES[stem_idx]

        # Load target stem
        target = self._load_stem(track_dir, target_name)
        if target is None:
            return self._fallback(target_name)

        # Load target CREPE (full length, then sync-crop)
        target_crepe, target_pitched = self._load_crepe(
            track_dir, target_name, target.shape[-1]
        )
        target, target_crepe = self._crop_with_crepe(target, target_crepe)

        # Load other stems + their CREPE
        other_wavs: list[Tensor] = []
        other_crepes: list[Tensor] = []
        other_pitched: list[float] = []
        for name in self.STEM_NAMES:
            if name == target_name:
                continue
            wav = self._load_stem(track_dir, name)
            if wav is not None:
                cr, pit = self._load_crepe(track_dir, name, wav.shape[-1])
                w_crop, c_crop = self._crop_with_crepe(wav, cr)
                other_wavs.append(w_crop)
                other_crepes.append(c_crop)
                other_pitched.append(pit)
            else:
                other_wavs.append(torch.zeros(self.clip_len))
                other_crepes.append(torch.zeros(self.T_frames, 360))
                other_pitched.append(0.0)

        # ── Augmentations ──
        if self.augment:
            # Pitch shift ALL stems together (keep them in key)
            semitones = random.uniform(-2.0, 2.0)

            target = self._pitch_shift_waveform(target, semitones, self.sr)
            target_crepe = self._pitch_shift_crepe(target_crepe, semitones)

            for i in range(len(other_wavs)):
                other_wavs[i] = self._pitch_shift_waveform(
                    other_wavs[i], semitones, self.sr
                )
                other_crepes[i] = self._pitch_shift_crepe(
                    other_crepes[i], semitones
                )

            # Random gain + polarity (per stem)
            target = self._augment_wav(target)
            for i in range(len(other_wavs)):
                other_wavs[i] = self._augment_wav(other_wavs[i])

        return {
            'target':          target,                            # (L,)
            'other_stems':     torch.stack(other_wavs, dim=0),    # (K, L)
            'target_crepe':    target_crepe,                      # (T, 360)
            'other_crepe':     torch.stack(other_crepes, dim=0),  # (K, T, 360)
            'target_pitched':  float(target_pitched),
            'other_pitched':   torch.tensor(other_pitched, dtype=torch.float32),  # (K,)
            'stem_name':       target_name,
            'valid':           True,
        }

    def _fallback(self, stem_name: str) -> dict:
        return {
            'target':          torch.zeros(self.clip_len),
            'other_stems':     torch.zeros(len(self.STEM_NAMES) - 1, self.clip_len),
            'target_crepe':    torch.zeros(self.T_frames, 360),
            'other_crepe':     torch.zeros(len(self.STEM_NAMES) - 1, self.T_frames, 360),
            'target_pitched':  0.0,
            'other_pitched':   torch.zeros(len(self.STEM_NAMES) - 1),
            'stem_name':       stem_name,
            'valid':           False,
        }


# ═══════════════════════════════════════════════════════════════════
# collate_valid — filter out invalid batches
# ═══════════════════════════════════════════════════════════════════

def collate_valid(batch: list[dict]) -> Optional[dict]:
    """Filter invalid samples and collate."""
    batch = [b for b in batch if b.get('valid', True)]
    if not batch:
        return None
    return {
        'target':          torch.stack([b['target'] for b in batch]),
        'other_stems':     torch.stack([b['other_stems'] for b in batch]),
        'target_crepe':    torch.stack([b['target_crepe'] for b in batch]),
        'other_crepe':     torch.stack([b['other_crepe'] for b in batch]),
        'target_pitched':  torch.tensor([b['target_pitched'] for b in batch]),
        'other_pitched':   torch.stack([b['other_pitched'] for b in batch]),
        'stem_names':      [b['stem_name'] for b in batch],
    }


# ═══════════════════════════════════════════════════════════════════
# Helper: make train/val DataLoaders
# ═══════════════════════════════════════════════════════════════════

def make_dataloaders(
    data_root: str,
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    clip_seconds: float = 4.0,
    batch_size: int = 4,
    val_split: float = 0.1,
    num_workers: int = 0,
    cache_limit: int = 16,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Create train and val DataLoaders with deterministic track split.

    Returns: (train_loader, val_loader, n_train_tracks)
    """
    # Create datasets
    full_dataset = StemDataset(
        root=data_root, sample_rate=sample_rate,
        n_fft=n_fft, hop_length=hop_length, clip_seconds=clip_seconds,
        augment=True, is_val=False, cache_limit=cache_limit,
    )
    val_dataset = StemDataset(
        root=data_root, sample_rate=sample_rate,
        n_fft=n_fft, hop_length=hop_length, clip_seconds=clip_seconds,
        augment=False, is_val=True, cache_limit=cache_limit,
    )

    # Split by TRACK (not by sample) — prevent leakage
    n_tracks = len(full_dataset.tracks)
    n_val = max(1, int(n_tracks * val_split))
    rng = random.Random(seed)
    rng.shuffle(full_dataset.tracks)
    val_tracks = full_dataset.tracks[-n_val:]
    train_tracks = full_dataset.tracks[:-n_val]
    full_dataset.tracks = train_tracks
    val_dataset.tracks = val_tracks

    print(f"  Tracks: {len(train_tracks)} train, {len(val_tracks)} val")

    train_loader = DataLoader(
        full_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_valid,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0,  # No fork for val (lower overhead)
        pin_memory=True,
        collate_fn=collate_valid,
        drop_last=False,
    )

    return train_loader, val_loader, len(train_tracks)
