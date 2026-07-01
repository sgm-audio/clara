#!/usr/bin/env python3
"""CLARA Mamba-3 Loss Functions — loss.py

Loss stack (mamba3 reference):
  Phase 1: NT-Xent contrastive (semantic encoder pretraining)
  Phase 2/3: MR-STFT (4 scales) + Mel-L1 (3 scales) + SI-SDR (ramped)

Architecture reference: Mamba-3 MIMO R=8, complex poles, spectral domain.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor, nn

# ═══════════════════════════════════════════════════════════════════
# NT-Xent Contrastive Loss (Phase 1)
# ═══════════════════════════════════════════════════════════════════


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross-entropy for contrastive
    semantic encoder pretraining.

    Two augmented views of the same stem = positive pair.
    All other stems in batch = negatives.

    Reference: Chen et al., "A Simple Framework for Contrastive
    Learning of Visual Representations" (SimCLR), ICML 2020.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.T = temperature

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """z1, z2: (B, d_model) — L2-normalized embeddings from semantic_enc
        Returns: scalar loss
        """
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # (2B, d)
        sim = torch.mm(z, z.T) / self.T  # (2B, 2B)

        # Label: z1[i] positive = z2[i] = z[B+i]
        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels + B, labels])  # (2B,)

        # Mask out self-similarity diagonal
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        return F.cross_entropy(sim, labels)


# ═══════════════════════════════════════════════════════════════════
# Multi-Resolution STFT Loss (4 scales)
# ═══════════════════════════════════════════════════════════════════


class SingleResolutionSTFTLoss(nn.Module):
    """Single-resolution STFT loss with spectral convergence + log-magnitude L1.

    Spectral convergence: normalized Frobenius norm (robust to gain changes).
    Log-magnitude L1: perceptually motivated spectral shape loss.
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """pred, target: (B, L) mono audio
        Returns: (sc_loss, lm_loss) each scalar
        """
        # Support (B, 1, L) shape
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        stft_pred = (
            torch.stft(
                pred,
                self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=True,
            )
            .abs()
            .clamp(min=1e-8)
        )
        stft_tgt = (
            torch.stft(
                target,
                self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=True,
            )
            .abs()
            .clamp(min=1e-8)
        )

        # Spectral convergence: ‖S_pred - S_target‖_F / ‖S_target‖_F
        sc_loss = torch.norm(stft_tgt - stft_pred, p="fro") / torch.norm(stft_tgt, p="fro").clamp(
            min=1e-8
        )

        # Log-magnitude L1
        lm_loss = F.l1_loss(torch.log(stft_tgt), torch.log(stft_pred))

        return sc_loss, lm_loss


class MultiResolutionSTFTLoss(nn.Module):
    """4-scale MR-STFT: 512, 1024, 2048, 4096 FFT sizes.

    Each scale captures different time-frequency tradeoffs:
    - 512: transient detail (high time resolution)
    - 1024: mid-range
    - 2048: primary spectral balance
    - 4096: low-frequency / sub-bass detail (high freq resolution)
    """

    def __init__(
        self,
        fft_sizes: list[int] = None,
        hop_sizes: list[int] = None,
        win_lengths: list[int] = None,
    ):
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [512, 1024, 2048, 4096]
            hop_sizes = [128, 256, 512, 1024]
            win_lengths = [512, 1024, 2048, 4096]
        self.losses = nn.ModuleList(
            [
                SingleResolutionSTFTLoss(n, h, w)
                for n, h, w in zip(fft_sizes, hop_sizes, win_lengths, strict=False)
            ]
        )

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        sc_total = torch.tensor(0.0, device=pred.device)
        lm_total = torch.tensor(0.0, device=pred.device)
        for loss in self.losses:
            sc, lm = loss(pred, target)
            sc_total = sc_total + sc
            lm_total = lm_total + lm
        return sc_total + lm_total


# ═══════════════════════════════════════════════════════════════════
# Mel-L1 Loss (3 filterbank scales)
# ═══════════════════════════════════════════════════════════════════


class MelL1Loss(nn.Module):
    """Multi-scale Mel-L1 loss on log-mel spectrograms.

    3 mel filterbank scales:
    - 512 FFT → 64 mel bands (wide-band envelope)
    - 1024 FFT → 80 mel bands (mid detail)
    - 2048 FFT → 128 mel bands (fine spectral detail)

    Using L1 on log-mel (not MSE) for robustness to outliers.
    """

    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.mel_transforms = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n,
                    hop_length=n // 4,
                    n_mels=m,
                    f_min=0.0,
                    f_max=sample_rate / 2.0,
                    norm="slaney",
                    mel_scale="htk",
                )
                for n, m in [(512, 64), (1024, 80), (2048, 128)]
            ]
        )

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """pred, target: (B, L) mono audio
        Returns: scalar L1 loss on log-mel
        """
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        loss = torch.tensor(0.0, device=pred.device)
        for mel_tf in self.mel_transforms:
            p = torch.log1p(mel_tf(pred))
            t = torch.log1p(mel_tf(target))
            loss = loss + F.l1_loss(p, t)
        return loss / len(self.mel_transforms)


# ═══════════════════════════════════════════════════════════════════
# SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
# ═══════════════════════════════════════════════════════════════════


def si_sdr_loss(pred: Tensor, target: Tensor) -> Tensor:
    """SI-SDR in dB. Returns negative (minimize = maximize SI-SDR).

    pred, target: (B, L) mono audio
    """
    eps = 1e-8
    target_energy = (target**2).sum(dim=-1, keepdim=True).clamp(min=eps)
    dot = (pred * target).sum(dim=-1, keepdim=True)
    proj = dot / target_energy * target
    noise = pred - proj
    si_sdr = 10 * torch.log10((proj**2).sum(-1) / ((noise**2).sum(-1) + eps) + eps)
    return -si_sdr.mean()


# ═══════════════════════════════════════════════════════════════════
# Combined TrackProcessorLoss (mamba3 reference)
# ═══════════════════════════════════════════════════════════════════


class TrackProcessorLoss(nn.Module):
    """Multi-objective loss for PerTrackProcessor.
    Mirrors mamba3 reference's TrackProcessorLoss.

    Loss formula:
      total = 1.0 × MR-STFT + 0.5 × Mel-L1 + (ramped) SI-SDR

    SI-SDR weight schedule:
      Steps 0-5k:   0.0 (disabled — destabilizes early training)
      Steps 5k-15k: ramped 0 → 0.1
      Steps 15k+:   0.1
    """

    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.mr_stft = MultiResolutionSTFTLoss()
        self.mel_l1 = MelL1Loss(sample_rate)
        self.register_buffer("_step_counter", torch.zeros(1, dtype=torch.long))

    @property
    def global_step(self) -> int:
        return int(self._step_counter.item())

    @global_step.setter
    def global_step(self, val: int):
        self._step_counter.fill_(val)

    def _si_sdr_weight(self) -> float:
        """Ramp SI-SDR weight from 0 → 0.1 between steps 5k and 15k."""
        s = self.global_step
        if s < 5000:
            return 0.0
        if s > 15000:
            return 0.1
        return 0.1 * (s - 5000) / 10000.0

    def forward(
        self, pred_stft: Tensor, target_wave: Tensor, model: nn.Module
    ) -> dict[str, Tensor]:
        """pred_stft:   (B, T, F) complex — model output STFT
        target_wave: (B, L) — target waveform
        model:       PerTrackProcessor (for ISTFT via waveform_from_stft)

        Returns dict with keys: total, mr_stft, mel_l1, si_sdr
        """
        # Reconstruct waveform from predicted STFT
        pred_wave = model.waveform_from_stft(pred_stft)

        # Align lengths (ISTFT may differ by a few samples)
        L = min(pred_wave.shape[-1], target_wave.shape[-1])
        pred_wave = pred_wave[..., :L]
        target_wave = target_wave[..., :L]

        losses = {}
        losses["mr_stft"] = self.mr_stft(pred_wave, target_wave)
        losses["mel_l1"] = self.mel_l1(pred_wave, target_wave)

        si_w = self._si_sdr_weight()
        if si_w > 0:
            losses["si_sdr"] = si_sdr_loss(pred_wave, target_wave) * si_w
        else:
            losses["si_sdr"] = torch.tensor(0.0, device=pred_wave.device)

        losses["total"] = 1.0 * losses["mr_stft"] + 0.5 * losses["mel_l1"] + losses["si_sdr"]

        self._step_counter += 1
        return losses
