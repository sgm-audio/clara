"""
CLARA Mamba SSM training — FIXED: operates on STFT frames like mamba3.
======================================================================
Before: raw audio [B, 88200] → Linear → Mamba layers (88200 steps) → Linear → output
After:  raw audio [B, 88200] → STFT [B, T, F] → Mamba layers (~86 steps) → ISTFT → output

This matches the proven mamba3 architecture that trains at 9 steps/sec.

Architecture (inside model):
  audio_in → [STFT] → frames [B,T,F] → proj [B,T,D] → Mamba×4 → out_proj [B,T,2F]
          → complex STFT → [ISTFT] → audio_out

Usage:
    python train_clara_now.py --data "C:/dev/mamba3/mamba3/Mamba-3-training/datasets/musdb18hq/train"
"""

from __future__ import annotations

import random
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchaudio

warnings.filterwarnings("ignore", message="At least one mel filterbank has all zero values")

if sys.platform != "win32":
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


# ══════════════════════════════════════════════════════════════════════════
# STFT-based Mamba Model — processes frames, not raw audio
# ══════════════════════════════════════════════════════════════════════════


class STFTMambaLayer(nn.Module):
    """
    Single Mamba layer operating on STFT frame representations.
    Matches the Mamba paper: pre-SSM normalization prevents numerical explosion.
    """

    def __init__(
        self, d_model: int = 64, d_state: int = 16, conv_width: int = 3, dt_rank: int = 16
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.conv_width = conv_width

        # Linear projections at frame level
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, conv_width, padding=conv_width - 1, groups=d_model
        )

        # Pre-SSM normalization (critical for stability)
        self.norm_before_ssm = nn.LayerNorm(d_model)

        # SSM core
        self.log_A_real = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1))
            * (math.log(0.1 / 0.001) / max(d_state - 1, 1))
            - torch.log(torch.tensor(0.1, dtype=torch.float32))
        )
        self.D = nn.Parameter(torch.ones(d_model))
        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        dt_init = torch.exp(
            torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Initialize output projection to near-zero to prevent explosion at start
        with torch.no_grad():
            self.out_proj.weight.mul_(0.1)

    def _ssm_scan(self, conv_out: Tensor, chunk_size: int = 64) -> Tensor:
        """
        Numerically stable chunked parallel scan.
        Processes in chunks of chunk_size, serial within chunk, state carried between chunks.
        Matches export path (ONNX processes 64-sample blocks with state carry).
        """
        B, T, D = conv_out.shape
        N = self.d_state

        xz = self.x_proj(conv_out)
        dt_latent = xz[:, :, : self.dt_rank]
        B_proj = xz[:, :, self.dt_rank : self.dt_rank + N]
        C_proj = xz[:, :, self.dt_rank + N :]

        dt = F.softplus(self.dt_proj(dt_latent)).clamp(max=0.1)
        A = -torch.exp(self.log_A_real)

        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dBdx = dt.unsqueeze(-1) * B_proj.unsqueeze(2) * conv_out.unsqueeze(-1)

        h = torch.zeros(B, 1, D, N, device=conv_out.device)
        outputs = []

        for t_start in range(0, T, chunk_size):
            t_end = min(t_start + chunk_size, T)
            chunk_dA = dA[:, t_start:t_end]
            chunk_dBdx = dBdx[:, t_start:t_end]
            chunk_C = C_proj[:, t_start:t_end]
            chunk_T = t_end - t_start

            # Serial within chunk (stable for small chunk_size ≤ 64)
            hs = []
            for t in range(chunk_T):
                h = chunk_dA[:, t : t + 1] * h + chunk_dBdx[:, t : t + 1]
                h = h.clamp(-100.0, 100.0)
                hs.append(h)
            h_chunk = torch.cat(hs, dim=1)

            y_chunk = torch.sum(chunk_C.unsqueeze(2) * h_chunk, dim=-1)
            outputs.append(y_chunk)

        y = torch.cat(outputs, dim=1)
        return y + self.D.unsqueeze(0).unsqueeze(0) * conv_out

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, T, D]  — frame-level features (T = frame count)
        Returns: [B, T, D]
        """
        residual = x
        seq_len = x.shape[1]

        # Gated linear unit
        projected = self.in_proj(x)
        x_ssm = projected[:, :, : self.d_model]
        x_gate = projected[:, :, self.d_model :]

        # Causal Conv1d on frame sequence
        x_conv_seq = x_ssm.transpose(1, 2)  # [B, D, T]
        x_conv_seq = F.pad(x_conv_seq, (self.conv_width - 1, 0))  # causal padding
        conv_out = self.conv1d(x_conv_seq)
        conv_out = conv_out[:, :, :seq_len].transpose(1, 2)  # [B, T, D]
        conv_out = conv_out * F.silu(x_gate)

        # Pre-SSM normalization prevents the scan from exploding
        conv_out = self.norm_before_ssm(conv_out)

        # SSM scan
        y = self._ssm_scan(conv_out)

        return self.out_proj(y) + residual


class STFTMambaModel(nn.Module):
    """
    Full STFT-based Mamba model.

    audio [B, L] → STFT [B, T, F] → proj → [B, T, D] → Mamba×4 → [B, T, D]
        → decode → [B, T, 2F] → complex STFT → ISTFT → audio [B, L]

    For ONNX export, STFT/ISTFT are included in the graph (opset 18+ supports STFT).
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 4,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_fft_half = n_fft // 2 + 1  # F_bins

        # STFT frame encoder: F_bins → d_model
        self.frame_proj = nn.Linear(self.n_fft_half, d_model)
        # Learnable skip connection
        self.skip_weight = nn.Parameter(torch.tensor(1.0))

        # Mamba layers at frame rate
        self.layers = nn.ModuleList(
            [STFTMambaLayer(d_model, d_state, conv_width=3, dt_rank=16) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        # Decoder: d_model → 2 * F_bins (real + imag)
        self.decode = nn.Linear(d_model, 2 * self.n_fft_half)

        # Register Hann window as buffer
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

    def forward(self, audio: Tensor) -> Tensor:
        L_in = audio.shape[-1]

        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        stft = stft.transpose(1, 2)

        stft_real = stft.real
        stft_imag = stft.imag
        mag = torch.sqrt(stft_real.pow(2) + stft_imag.pow(2))

        x = self.frame_proj(mag)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        delta = self.decode(x)

        delta_real = delta[:, :, : self.n_fft_half]
        delta_imag = delta[:, :, self.n_fft_half :]

        stft_out = torch.complex(
            stft_real + delta_real * self.skip_weight,
            stft_imag + delta_imag * self.skip_weight,
        ).transpose(1, 2)

        return torch.istft(
            stft_out,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            length=L_in,
        )


# ══════════════════════════════════════════════════════════════════════════
# Dataset — flat WAV files
# ══════════════════════════════════════════════════════════════════════════


class FlatWavDataset(Dataset):
    def __init__(
        self,
        root: str,
        sample_rate: int = 44100,
        clip_seconds: float = 2.0,
        augment: bool = True,
        cache_in_ram: bool = False,
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.clip_len = int(clip_seconds * sample_rate)
        self.augment = augment
        self.cache_in_ram = cache_in_ram
        self._ram_cache: dict[str, Tensor] = {}

        self.audio_files = sorted([f for f in self.root.rglob("*.wav") if f.is_file()])
        assert len(self.audio_files) > 0, f"No *.wav files found in {root}"

        if self.cache_in_ram:
            self._preload()

    def _preload(self):
        print(f"Pre-loading {len(self.audio_files)} files into RAM...", flush=True)
        for path in self.audio_files:
            wav = self._load_file_uncached(path)
            if wav is not None:
                self._ram_cache[str(path)] = wav
        total_gb = sum(v.numel() * 4 for v in self._ram_cache.values()) / 1e9
        print(f"  RAM cache: {len(self._ram_cache)} files, {total_gb:.2f} GB", flush=True)

    def _load_file_uncached(self, path: Path) -> Optional[Tensor]:
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception:
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

    def set_shared_cache(self, cache: dict):
        self._ram_cache = cache
        self.cache_in_ram = True

    def __len__(self) -> int:
        return len(self.audio_files) * 4

    def _crop(self, wav: Tensor) -> Tensor:
        L = wav.shape[-1]
        if L <= self.clip_len:
            return F.pad(wav, (0, self.clip_len - L))
        start = random.randint(0, L - self.clip_len)
        return wav[start : start + self.clip_len]

    def _augment(self, wav: Tensor) -> Tensor:
        gain_db = random.uniform(-6, 6)
        wav = wav * (10 ** (gain_db / 20))
        if random.random() < 0.5:
            wav = -wav
        return wav.clamp(-1.0, 1.0)

    def __getitem__(self, idx: int) -> dict:
        file_idx = (idx // 4) % len(self.audio_files)
        path = self.audio_files[file_idx]
        wav = self._load_file(path)
        if wav is None:
            return {"audio": torch.zeros(self.clip_len), "valid": False}
        wav = self._crop(wav)
        if self.augment:
            wav = self._augment(wav)
        return {"audio": wav, "valid": True}


def collate_valid(batch: list[dict]) -> Optional[dict]:
    batch = [b for b in batch if b["valid"]]
    if not batch:
        return None
    return {"audio": torch.stack([b["audio"] for b in batch])}


# ══════════════════════════════════════════════════════════════════════════
# Loss — matches train_laptop LeanTrackLoss exactly
# ══════════════════════════════════════════════════════════════════════════


class LeanClaraLoss(nn.Module):
    """Identical pattern to LeanTrackLoss in train_laptop.py."""

    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.sr = sample_rate
        self.n_fft = 1024
        self.hop = 256
        # Single mel (like laptop edition)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=64
        )
        self.global_step = 0

    def _si_sdr(self, pred: Tensor, target: Tensor) -> Tensor:
        eps = 1e-8
        target_energy = (target**2).sum(dim=-1, keepdim=True)
        dot = (pred * target).sum(dim=-1, keepdim=True)
        proj = dot / (target_energy + eps) * target
        noise = pred - proj
        si_sdr = 10 * torch.log10((proj**2).sum(-1) / ((noise**2).sum(-1) + eps) + eps)
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
        L = min(pred_wave.shape[-1], target_wave.shape[-1])
        pred_wave = pred_wave[..., :L]
        target_wave = target_wave[..., :L]

        losses = {}

        # Single STFT magnitude L1
        window = torch.hann_window(self.n_fft, device=pred_wave.device)
        p_stft = torch.stft(pred_wave, self.n_fft, self.hop, window=window, return_complex=True)
        t_stft = torch.stft(target_wave, self.n_fft, self.hop, window=window, return_complex=True)
        losses["stft"] = F.l1_loss(p_stft.abs(), t_stft.abs())

        # Mel-L1
        losses["mel_l1"] = self._mel_l1(pred_wave, target_wave)

        # SI-SDR (ramped in)
        si_w = self._si_sdr_weight()
        if si_w > 0:
            losses["si_sdr"] = self._si_sdr(pred_wave, target_wave) * si_w
        else:
            losses["si_sdr"] = torch.tensor(0.0, device=pred_wave.device)

        # Total — same weights as LeanTrackLoss
        losses["total"] = 1.0 * losses["stft"] + 0.5 * losses["mel_l1"] + losses["si_sdr"]

        self.global_step += 1
        return losses


# ══════════════════════════════════════════════════════════════════════════
# Training — matches train_laptop structure exactly
# ══════════════════════════════════════════════════════════════════════════


def train(
    data_root: str,
    save_dir: str,
    n_steps: int = 50_000,
    batch_size: int = 2,
    lr: float = 3e-4,
    device: str = "auto",
    resume_from: Optional[str] = None,
    val_split: float = 0.1,
    ckpt_every: int = 200,
    log_every: int = 25,
    val_every: int = 1000,
    num_workers: int = 2,
    grad_clip: float = 1.0,
    grad_accum: int = 8,
    early_stop_patience: int = 20,
    amp: bool = False,
    compile_model: bool = False,
    profile_steps: int = 0,
    fresh_best_val: bool = False,
    cache_ram: bool = False,
    clip_seconds: float = 2.0,
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Auto-detected GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("No GPU detected, using CPU")
    elif device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    dev = torch.device(device)
    print(f"Device: {dev}")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
        print(f"CPU threads: {torch.get_num_threads()}")

    if amp and device == "cpu":
        amp = False
        print("AMP disabled on CPU")

    autocast_dtype = torch.bfloat16 if amp else torch.float32
    print(f"AMP: {amp} ({autocast_dtype})")

    if device == "cuda" and num_workers == 2 and sys.platform != "win32":
        num_workers = 4
        print(f"num_workers auto-bumped to {num_workers} for GPU")
    if sys.platform == "win32" and num_workers > 0:
        num_workers = 0
        print("num_workers set to 0 (Windows CUDA multiprocessing is unstable)")

    # ── Model: STFT-based (mamba3 pattern) ──
    model = STFTMambaModel(d_model=64, d_state=16, n_layers=4, n_fft=1024, hop_length=256).to(dev)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print("  Architecture: STFT(n_fft=1024, hop=256) + Mamba(d_model=64, n_layers=4, d_state=16)")
    print(
        f"  Frame rate: {44100 // 256} fps, Sequence: ~{int(clip_seconds * 44100 // 256)} frames per clip"
    )

    if compile_model and hasattr(torch, "compile"):
        print("Compiling model with torch.compile (mode='reduce-overhead')...")
        model = torch.compile(model, mode="reduce-overhead")
    elif compile_model:
        print("torch.compile not available — skipping")

    # ── Loss (matches LeanTrackLoss from train_laptop.py) ──
    loss_fn = LeanClaraLoss(sample_rate=44100).to(dev)

    # ── Optimizer ──
    decay_params = [
        p
        for n, p in model.named_parameters()
        if "bias" not in n
        and "norm" not in n
        and "LayerNorm" not in type(p).__name__
        and p.requires_grad
    ]
    nodecay_params = [
        p for n, p in model.named_parameters() if ("bias" in n or "norm" in n) and p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 1e-4},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    def lr_lambda(step: int) -> float:
        warmup = 2000
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, n_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ──
    start_step = 0
    best_val_loss = float("inf")
    if resume_from is not None:
        ckpt_path = Path(resume_from)
        if not ckpt_path.exists():
            print(f"ERROR: --resume path does not exist: {ckpt_path}")
            sys.exit(1)
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        if "best_val_loss" in ckpt and not fresh_best_val:
            best_val_loss = ckpt["best_val_loss"]
            print(f"  resumed at step {start_step}, best_val_loss={best_val_loss:.4f}")
        else:
            print(f"  resumed at step {start_step}, best_val_loss reset")

    # ── Dataset ──
    full_dataset = FlatWavDataset(
        data_root,
        sample_rate=44100,
        clip_seconds=clip_seconds,
        augment=True,
        cache_in_ram=cache_ram,
    )
    if cache_ram:
        full_dataset._preload()
    val_dataset = FlatWavDataset(
        data_root, sample_rate=44100, clip_seconds=clip_seconds, augment=False, cache_in_ram=False
    )
    if cache_ram:
        val_dataset.set_shared_cache(full_dataset._ram_cache)

    n_files = len(full_dataset.audio_files)
    n_val = max(1, int(n_files * val_split))
    random.Random(42).shuffle(full_dataset.audio_files)
    val_files = full_dataset.audio_files[-n_val:]
    train_files = full_dataset.audio_files[:-n_val]

    full_dataset.audio_files = train_files
    val_dataset.audio_files = val_files

    print(f"Files: {len(train_files)} train, {len(val_files)} val")

    # ── DataLoaders ──
    dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_valid,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_valid,
        drop_last=False,
    )

    # ── Signal handling ──
    import signal

    _state = {"step": start_step, "interrupted": False}

    def signal_handler(signum, frame):
        if _state["interrupted"]:
            print("\n! Second interrupt — exiting hard.")
            sys.exit(130)
        print(f"\n! Caught signal {signum} — will save after current step.")
        _state["interrupted"] = True

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM") and sys.platform != "win32":
        signal.signal(signal.SIGTERM, signal_handler)
    if sys.platform == "win32" and hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal_handler)

    # ── Checkpointing ──
    def save_checkpoint(
        step: int, loss_val: float, is_best: bool = False, name: Optional[str] = None
    ):
        ckpt = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss": loss_val,
            "best_val_loss": best_val_loss,
            "architecture": "STFTMambaModel",
        }
        fname = name if name else f"checkpoint_{step:06d}.pt"
        tmp_path = save_path / f".{fname}.tmp"
        final_path = save_path / fname
        torch.save(ckpt, tmp_path)
        tmp_path.replace(final_path)
        if is_best:
            best_tmp = save_path / ".best.pt.tmp"
            torch.save(ckpt, best_tmp)
            best_tmp.replace(save_path / "best.pt")

    # ── Validation ──
    @torch.no_grad()
    def validate() -> float:
        model.eval()
        total_loss = 0.0
        n_batches = 0
        max_val_batches = 20
        for v_idx, v_batch in enumerate(val_loader):
            if v_idx >= max_val_batches:
                break
            if v_batch is None:
                continue
            v_audio = v_batch["audio"].to(dev)
            pred = model(v_audio)
            vl_dict = loss_fn(pred, v_audio)
            vl = vl_dict["total"]
            if not (torch.isnan(vl) or torch.isinf(vl)):
                total_loss += vl.item()
                n_batches += 1
        model.train()
        return total_loss / max(1, n_batches)

    # ── Training loop ──
    step = start_step
    plateau_count = 0
    last_log_time = time.time()
    last_log_step = step
    accum_step = 0

    print(f"\n{'='*60}")
    print("  CLARA TRAINING (STFT-based, mamba3 pattern)")
    print(f"{'='*60}")
    print(f"  steps={n_steps}, batch={batch_size}, grad_accum={grad_accum}")
    print(f"  effective batch = {batch_size * grad_accum}")
    print(f"  lr={lr}, grad_clip={grad_clip}")
    print(f"  ckpt every {ckpt_every} steps, val every {val_every} steps")
    print(f"  clip_seconds={clip_seconds} ({int(clip_seconds * 44100)} samples)")
    print("  STFT: n_fft=1024, hop=256")
    print(f"  Frame count per clip: ~{int(clip_seconds * 44100 // 256)}")
    if start_step > 0:
        print(f"  Resuming at step {start_step}")
    print()

    if device == "cuda":
        print("Warming up GPU kernels...", flush=True)
        warm_batch = next(iter(dataloader))
        if warm_batch is not None:
            wa = warm_batch["audio"][:1].to(dev)
            model.train()
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=amp):
                wp = model(wa)
                wl = wp.square().mean()
            wl.backward()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(dev)
        print("GPU warmup complete.", flush=True)

    prof = None
    if profile_steps > 0:
        print(f"Profiler enabled — will capture {profile_steps} steps after warmup")
        prof = torch.profiler.profile(
            activities=(
                [
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
                if device == "cuda"
                else [torch.profiler.ProfilerActivity.CPU]
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=profile_steps, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(save_path / "profile_trace")
            ),
        )
        prof.start()

    empty_batch_streak = 0

    while step < n_steps:
        for batch in dataloader:
            if batch is None:
                empty_batch_streak += 1
                if empty_batch_streak > len(dataloader) * 2:
                    print(f"\nERROR: all batches empty. Dataset root: {data_root}")
                    sys.exit(1)
                continue
            empty_batch_streak = 0

            audio = batch["audio"].to(dev, non_blocking=True)

            loss_scale = 1.0 / grad_accum

            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=amp):
                pred = model(audio)
                loss_dict = loss_fn(pred, audio)

            loss = loss_dict["total"]
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[step {step}] NaN/Inf loss — skipping")
                optimizer.zero_grad(set_to_none=True)
                accum_step = 0
                continue

            (loss * loss_scale).backward()
            accum_step += 1

            if accum_step >= grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum_step = 0

            if prof is not None:
                prof.step()

            if step % log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                now = time.time()
                elapsed = max(1e-6, now - last_log_time)
                steps_per_sec = (step - last_log_step) / elapsed
                last_log_time = now
                last_log_step = step

                if steps_per_sec > 0.001:
                    eta_h = (n_steps - step) / steps_per_sec / 3600
                    eta_str = f"ETA {eta_h:.1f}h"
                else:
                    eta_str = "ETA -"

                log_losses = (
                    f"  stft={loss_dict['stft'].item():.4f} mel={loss_dict['mel_l1'].item():.4f}"
                )
                if "si_sdr" in loss_dict and loss_dict["si_sdr"].item() != 0:
                    log_losses += f" si_sdr={loss_dict['si_sdr'].item():.4f}"

                log_str = (
                    f"[step {step:6d}] loss={loss.item():.4f} lr={lr_now:.2e} "
                    f"{steps_per_sec:.3f} step/s {eta_str}{log_losses}"
                )
                if device == "cuda":
                    mem = torch.cuda.max_memory_allocated(dev) / 1024**3
                    log_str += f"  gpu_mem={mem:.2f}GB"
                print(log_str, flush=True)

            if step > 0 and step % val_every == 0:
                print(f"  Running validation at step {step}...")
                val_loss = validate()
                print(f"  val_loss = {val_loss:.4f}  (best={best_val_loss:.4f})")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(step, loss.item(), is_best=True)
                    print("  new best val loss — saved best.pt")
                    plateau_count = 0
                else:
                    plateau_count += 1
                    if plateau_count >= early_stop_patience:
                        print("  Val plateaued — early stopping")
                        save_checkpoint(step, loss.item(), name="final_earlystop.pt")
                        if prof is not None:
                            prof.stop()
                        return model

            if step > 0 and step % ckpt_every == 0:
                save_checkpoint(step, loss.item())

            if _state["interrupted"]:
                print(f"\n! Saving interrupt checkpoint at step {step}...")
                save_checkpoint(step, loss.item(), name=f"interrupt_{step:06d}.pt")
                print(f"  saved. Resume with: --resume {save_path}/interrupt_{step:06d}.pt")
                if prof is not None:
                    prof.stop()
                return model

            step += 1
            _state["step"] = step
            if step >= n_steps:
                break

    if prof is not None:
        prof.stop()
        print(f"Profile trace saved to {save_path / 'profile_trace'}.json")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    save_checkpoint(step, loss.item(), name="final.pt")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLARA STFT-based Mamba training")
    parser.add_argument("--data", required=True, help="Path to WAV dataset directory")
    parser.add_argument("--save", default="./checkpoints_clara")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--amp", action="store_true", help="Enable bfloat16 AMP (may cause NaN on ROCm 860M)"
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fresh-best-val", action="store_true")
    parser.add_argument("--cache", action="store_true", help="Enable RAM pre-caching")
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument(
        "--profile",
        type=int,
        default=0,
        metavar="STEPS",
        help="Run torch profiler for STEPS steps, save Chrome trace to save dir",
    )

    args = parser.parse_args()

    train(
        data_root=args.data,
        save_dir=args.save,
        n_steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
        resume_from=args.resume,
        val_split=args.val_split,
        ckpt_every=args.ckpt_every,
        log_every=args.log_every,
        val_every=args.val_every,
        num_workers=args.num_workers,
        grad_accum=args.grad_accum,
        early_stop_patience=args.early_stop_patience,
        amp=args.amp,
        compile_model=args.compile,
        profile_steps=args.profile,
        fresh_best_val=args.fresh_best_val,
        cache_ram=args.cache,
        clip_seconds=args.clip_seconds,
    )
