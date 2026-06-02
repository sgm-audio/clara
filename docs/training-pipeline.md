# Training Pipeline — CLARA Mamba SSM

## 1. Model Architecture Overview

### Hyperparameters

| Parameter   | Value | Description |
|-------------|-------|-------------|
| `d_model`   | 64    | Embedding dimension — 64-wide vector per sample per layer |
| `n_layers`  | 4     | Stacked Mamba SSM layers |
| `d_state`   | 16    | SSM state dimension per channel |
| `conv_width`| 4     | Causal depthwise convolution kernel width |
| `dt_rank`   | 16    | Discretization projection rank |

### Parameter Count: ~72K

Calculated from the architecture in `export_mamba_onnx.py`:

| Component | Params | Notes |
|-----------|--------|-------|
| InputProjection | 128 | `Linear(1, 64)` with bias |
| Per MambaLayer (×4) | 17,856 | See breakdown below |
| LayerNorm (top-level) | 128 | `LayerNorm(64)` weight + bias |
| OutputProjection | 128 | `Linear(64, 1)` with bias |
| **Total** | **71,808** | Exact count printed by `export_mamba_onnx.py` |

Per-layer breakdown:

| Submodule | Params | Details |
|-----------|--------|---------|
| `in_proj` | 8,192 | `Linear(64, 128, bias=False)` — splits into SSM + gate paths |
| `conv1d` | 320 | Depthwise `Conv1d(64, 64, kernel=4, groups=64)` + bias |
| `ssm_proj.log_A_real` | 1,024 | Learned log-discretization matrix `[64, 16]` |
| `ssm_proj.D` | 64 | Skip connection vector `[64]` |
| `ssm_proj.x_proj` | 3,072 | `Linear(64, 48, bias=False)` — projects to dt/B/C latents |
| `ssm_proj.dt_proj` | 1,088 | `Linear(16, 64, bias=True)` — projects dt latent to d_model |
| `out_proj` | 4,096 | `Linear(64, 64, bias=False)` |
| **Layer total** | **17,856** | |

This is small enough for single-core CPU inference at 48 kHz with headroom. ONNX Runtime further optimizes via graph fusion and constant folding.

### ONNX I/O Contract

**9 inputs → 3 outputs** (must never change without updating `export_mamba_onnx.py` and `MambaInferenceEngine` in lockstep):

| Index | Input Name | Shape | Description |
|-------|-----------|-------|-------------|
| 0 | `audio_chunk` | `{1, seq_len, 1}` | Mono audio samples, float32, dynamic seq_len |
| 1-4 | `conv{0..3}` | `{1, 64, 4}` | Per-layer causal conv state |
| 5-8 | `ssm{0..3}` | `{1, 64, 16}` | Per-layer SSM recurrent state |

| Index | Output Name | Shape | Description |
|-------|------------|-------|-------------|
| 0 | `processed_audio` | `{1, seq_len, 1}` | Processed mono audio |
| 1 | `new_conv_state` | `{1, 4, 64, 4}` | Stacked conv states (layer dim=1) |
| 2 | `new_ssm_state` | `{1, 4, 64, 16}` | Stacked SSM states (layer dim=1) |

Output states are stacked to simplify the ONNX graph (8 separate outputs would complicate the trace). The C++ engine (`MambaInferenceEngine`) unstacks them into per-layer buffers after each `processBlock()` call.

`seq_len` is dynamic (`dynamic_axes={'audio_chunk': {1: 'seq_len'}}`). This prevents SSM state corruption from zero-padded inputs — using a fixed large shape and padding unused portions would feed zeros through the SSM's recurrent state.

### Architecture Diagram (Data Flow)

```
audio_chunk [1, seq_len, 1]
    │
    ▼
InputProjection (Linear 1→64)
    │
    ▼
┌─────────────────────────────────────────────┐
│  MambaLayer 0                               │
│  in_proj → Conv1d → SiLU gate → SSM → out   │
│  conv_state[0] ────► conv1d (prepended)     │
│  ssm_state[0] ────► scan_step (recurrent)   │
│  outputs: x [1, L, 64], new_conv[0], new_ssm[0] │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  MambaLayer 1 (same structure)              │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  MambaLayer 2 (same structure)              │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  MambaLayer 3 (same structure)              │
└─────────────────────────────────────────────┘
    │
    ▼
LayerNorm(64)
    │
    ▼
OutputProjection (Linear 64→1)
    │
    ▼
processed_audio [1, seq_len, 1]
```

### Per-MambaLayer Detail

```
x_in [1, L, 64]
    │
    ▼
in_proj: Linear(64 → 128, bias=False)
    │
    ├──► x_ssm [1, L, 64] ──► transpose(1,2) ──► [1, 64, L]
    │                              │
    │                    prepend conv_state [1, 64, 4]
    │                              │
    │                    Conv1d(64, 64, k=4, groups=64)
    │                              │
    │                    extract causal output positions
    │                              │
    │                    transpose(1,2) ──► [1, L, 64]
    │
    └──► x_gate [1, L, 64] ──► SiLU
              │
              ▼
        element-wise multiply (conv_out × SiLU(x_gate))
              │
              ▼
        MambaSSMCore:
          x_proj: Linear(64 → 48) splits into dt_latent [16], B [16], C [16]
          dt = softplus(dt_proj(dt_latent))  // [1, L, 64]
          A = -exp(log_A_real)               // [64, 16]
              │
              ▼
        SelectiveScanStep (unrolled for t in range(L)):
          dA = exp(dt_t × A)                    // [B, 64, 16]
          dB = dt_t × B_t                       // [B, 64, 16]
          h = dA × h + dB × x_t                 // [B, 64, 16]
          y = sum(C_t × h, dim=-1) + D × x_t    // [B, 64]
              │
              ▼
        out_proj: Linear(64 → 64, bias=False)
              │
              ▼
        x_out [1, L, 64]
```

### Selective Scan (S6) Core

The S6 selective scan is the heart of the Mamba architecture. Unlike a standard RNN with fixed transition matrices, S6 makes the state transition **input-dependent**:

```
Given at timestep t:
  x_t:  [d_model]          — current input (post-conv, post-gate)
  dt_t: [d_model]          — learned discretization step (input-dependent)
  A:    [d_model, d_state] — negative-reparametrized, input-independent
  B_t:  [d_state]          — input-dependent projection
  C_t:  [d_state]          — input-dependent projection
  D:    [d_model]          — learned skip connection
  h_in: [d_model, d_state] — previous state

Compute:
  dA = exp(dt_t × A)                    — discretized A (always stable, A < 0)
  dB = dt_t × B_t                       — discretized B
  h_out = dA ⊙ h_in + dB × x_t         — state update (hadamard + outer product)
  y_t = C_t^T h_out + D ⊙ x_t          — output with skip connection
```

The `log_A_real` parameter enforces negative A (via `A = -exp(log_A_real)`), guaranteeing a stable discretized system regardless of dt. The `dt_proj.bias` is initialized to match a uniform random dt distribution via `inv_softplus` to stabilize early training.

### Key Design Properties

1. **Causal processing**: Conv1d is causal — state is prepended to the input, and only outputs corresponding to each input position are extracted. No future context leaks.

2. **Input-dependent state transitions**: Unlike fixed-RNN architectures, both the state update (A→dA via dt) and the IO mappings (B, C) depend on the current input, giving the model more expressivity per parameter.

3. **Linear-time in sequence length**: The loop `for t in range(seq_len)` is O(L) with per-step O(d_model × d_state) compute. No quadratic attention.

4. **State carry across blocks**: Conv state (rolling buffer of last 4 samples) and SSM state (accumulated recurrent state) persist across `processBlock()` calls. Output states from one block become input states for the next.

5. **Latency = conv_width**: The causal conv1d contributes `conv_width - 1 = 3` samples of latency. The SSM scan is synchronous (no lookahead). Total model latency: 4 samples (<0.1 ms at 48 kHz).

## 2. Ideal Training Data

### Task Definition

The MambaRT model is an **audio-to-audio effect processor**. It learns a transformation from input audio to output audio. The training task is supervised: given an input audio chunk and its corresponding output (target) chunk, the model learns to approximate the mapping.

This is distinct from generative modeling (audio generation from noise/latent) or classification tasks. The model preserves the temporal structure of the input and applies a learned transformation.

### Use Cases

| Application | Input | Target | Example Data |
|-------------|-------|--------|--------------|
| **Audio restoration / denoising** | Noisy recording | Clean recording | DNS Challenge (noisy/clean speech) |
| **Timbre transfer** | Dry instrument | Processed/amped signal | Guitar amp sim (DI → mic'd cab) |
| **Compression/limiting** | Uncompressed | Compressed with known settings | Custom generated pairs |
| **EQ/filtering** | Full-range | Filtered | Sweep-based synthetic pairs |
| **Creative effects** | Dry audio | Any desired transformation | Custom studio recordings |
| **Source separation** | Mixture | Individual stem | MUSDB18 (mix → vocal/drums/bass/other) |

### Recommended Datasets

| Dataset | Size | Use Case | Format | License |
|---------|------|----------|--------|---------|
| **MUSDB18** | ~10 hours | Multi-track stems, instrument separation/transformation | WAV, 44.1 kHz | CC BY-NC-SA 4.0 |
| **DNS Challenge** | ~750 hours | Noisy/clean speech pairs for denoising | WAV, 16 kHz | CC BY-SA 4.0 |
| **MAESTRO** | ~200 hours | Piano performances with aligned MIDI | WAV, 44.1 kHz | CC BY-NC-SA 4.0 |
| **FSD50K** | ~100 hours | General sound events (200+ classes) | WAV, 44.1 kHz | CC BY 4.0 |
| **PedalNet / Guitar datasets** | Varies | Dry/amped guitar pairs | WAV, 44.1 kHz | Various |
| **Custom studio recordings** | Yours | Specific studio processing targets | WAV | Proprietary |

For initial training (proof of concept), MAESTRO or DNS Challenge provide the easiest starting point with readily available paired data.

### Data Preparation Pipeline

```
raw audio files (*.wav, *.flac)
    │
    ▼
[1] Resample to target sample rate (44100 or 48000 Hz)
    │
    ▼
[2] Convert to mono (average channels)
    │
    ▼
[3] Normalize peak amplitude to -1 dBFS (prevent clipping after processing)
    │
    ▼
[4] Segment into fixed-length chunks (e.g., 88200 samples = 2 s @ 44100 Hz)
       • Overlap: 0% for training (stateless chunks), or
       • 50% overlap for data augmentation (more chunks per file)
    │
    ▼
[5] Train/val/test split (80/10/10)
       • Split at file level, not chunk level (prevent data leakage)
       • Stratified if using multi-class datasets
    │
    ▼
[6] Serialize to .npy or .wav files
       • .npy: fastest loading via np.load + torch.from_numpy
       • .wav: more portable, slightly slower to decode
```

### Pre-processing Script Structure

```python
# preprocess.py — converts raw audio files to serialized chunks

def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    sample_rate: int = 44100,
    chunk_samples: int = 88200,  # 2 seconds
    peak_normalize_db: float = -1.0,
    val_split: float = 0.1,
    test_split: float = 0.1,
):
    # 1. Recursively find all audio files in input_dir
    # 2. For each file:
    #    a. Load with librosa (or torchaudio), resample to target SR
    #    b. Convert to mono
    #    c. Normalize peak to peak_normalize_db
    #    d. Segment into chunk_samples-length windows (no overlap)
    #    e. Save each chunk as {output_dir}/{split}/{filename}_chunk{n}.npy
    # 3. Split files (not chunks) into train/val/test
    # 4. Write split index files (train_files.txt, val_files.txt, test_files.txt)
    pass
```

### Data Customization for Specific Tasks

**Denoising**: Use DNS Challenge directly (paired noisy/clean). For custom denoising, generate synthetic pairs by adding noise sources (white, pink, brown, babble, fans, traffic) to clean audio at random SNRs (-5 to +20 dB).

**Timbre transfer**: For amp sim, record DI (dry) and processed (amped) signals simultaneously via a re-amp box. For vocal processing, use dry vocals and processed vocals through a known chain.

**Creative effects**: If you want the model to learn a specific processing style, prepare pairs by running audio through a fixed analog/DSP chain (or an existing plugin chain in a DAW).

## 3. Training Infrastructure — RunPod A100 80GB

### Instance Configuration

| Spec | Value | Notes |
|------|-------|-------|
| **GPU** | 1× NVIDIA A100 80GB (SXM) | Single GPU — no multi-GPU complexity |
| **vCPU** | 16 cores | Ample for data loading |
| **RAM** | 120 GB | Enables full dataset RAM caching |
| **Storage** | Network volume (persistent) | For datasets and checkpoints |
| **Bandwidth** | ~25 Gbps | Fast model/dataset download |

### Network Volume Structure

```
/workspace/
├── clara/                    # Git repository
│   ├── models/               # ONNX export destination
│   ├── export_mamba_onnx.py
│   └── train.py              # Training script (see §4)
├── data/
│   ├── raw/                  # Downloaded datasets
│   ├── processed/            # Pre-processed chunks on network volume
│   └── preprocess.py
├── logs/                     # TensorBoard logs, checkpoints
│   ├── checkpoint_best.pt
│   ├── checkpoint_last.pt
│   └── events.out.tfevents.*
└── venv/                     # Python virtual environment
```

### Session Management

- **tmux**: All training runs inside a tmux session for persistence across SSH disconnects.
- **Detach**: `Ctrl+B, D`
- **Re-attach**: `tmux attach -t train`
- **Multiple windows**: Use `Ctrl+B, C` for monitoring (nvidia-smi, htop, tensorboard).

### Memory Strategy

| Component | Size | Location |
|-----------|------|----------|
| Dataset (50 hours, 44.1 kHz mono, ~300M samples) | ~1.2 GB as float32 | `/dev/shm` (tmpfs in RAM) |
| Model (72K params × 4 bytes) | ~0.3 MB | GPU VRAM |
| ONNX Runtime session | ~40 MB | GPU memory for execution |
| Training state (optimizer, gradients) | ~3× model = ~1 MB | GPU VRAM |
| **Total GPU usage** | **< 2 GB** | Well within 80 GB capacity |

The dataset fits entirely in RAM. For datasets larger than RAM (e.g., DNS Challenge at 750 hours), use NumPy `memmap` or lazy loading with `torchdata`.

### Software Setup

| Tool | Purpose |
|------|---------|
| PyTorch 2.x | Training framework, torch.compile |
| torchaudio | Audio I/O, resampling |
| librosa | Audio analysis utilities |
| onnx / onnxruntime | ONNX export and verification |
| tensorboard | Training metrics logging |
| matplotlib | Plotting spectrograms, loss curves |
| numpy | Data manipulation |
| aria2c | Fast multi-connection downloads |

## 4. Training Script Architecture

### Recommended Approach: PyTorch Lightning

PyTorch Lightning simplifies checkpointing, logging, mixed precision, and early stopping without boilerplate.

#### Model Module

```python
# train.py — LightningModule for MambaRT

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from export_mamba_onnx import MambaRTModel


class MambaRTLightning(pl.LightningModule):
    def __init__(
        self,
        d_model=64,
        d_state=16,
        n_layers=4,
        conv_width=4,
        dt_rank=16,
        lr=3e-4,
        weight_decay=1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MambaRTModel(d_model, d_state, n_layers, conv_width, dt_rank)
        self.loss_fn = CombinedLoss()

    def forward(self, audio_chunk, conv_states, ssm_states):
        return self.model(audio_chunk, *conv_states, *ssm_states)

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: input audio, y: target audio
        B, L = x.shape[0], x.shape[1]

        # Zero states for independent chunks (training mode)
        conv_states = tuple(torch.zeros(B, 64, 4, device=x.device) for _ in range(4))
        ssm_states = tuple(torch.zeros(B, 64, 16, device=x.device) for _ in range(4))

        pred, _, _ = self.model(x, *conv_states, *ssm_states)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        conv_states = tuple(torch.zeros(x.shape[0], 64, 4, device=x.device) for _ in range(4))
        ssm_states = tuple(torch.zeros(x.shape[0], 64, 16, device=x.device) for _ in range(4))
        pred, _, _ = self.model(x, *conv_states, *ssm_states)
        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
```

#### Data Module

```python
class MambaAudioDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=128, num_workers=4, prefetch_factor=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        # Load pre-processed .npy files
        # Each sample: (input_chunk, target_chunk) as float32 arrays
        self.train_data = AudioDataset(os.path.join(self.data_dir, "train"))
        self.val_data = AudioDataset(os.path.join(self.data_dir, "val"))
        self.test_data = AudioDataset(os.path.join(self.data_dir, "test"))

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
```

#### Training Entry Point

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mixed-precision", choices=["bf16", "fp16", "no"], default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--log-dir", default="/workspace/logs/")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    args = parser.parse_args()

    model = MambaRTLightning(lr=args.lr)

    # torch.compile for speed (PyTorch 2.x)
    if args.compile:
        model.model = torch.compile(model.model, mode="reduce-overhead")

    precision = 32 if args.mixed_precision == "no" else args.mixed_precision

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision=precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=10,
        logger=pl.loggers.TensorBoardLogger(args.log_dir),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=args.log_dir,
                filename="checkpoint_best",
                monitor="val_loss",
                save_top_k=1,
                mode="min",
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=args.log_dir,
                filename="checkpoint_last",
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, mode="min"
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    dm = MambaAudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    trainer.fit(model, dm)
```

### Alternative: Raw PyTorch with torch.compile

If you prefer minimal dependencies, use raw PyTorch directly. The structure is similar but requires manual loop, checkpointing, and logging:

```python
# train_raw.py — minimal training loop without Lightning

def train_epoch(model, dataloader, optimizer, scaler, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            conv_states = tuple(torch.zeros(B, 64, 4, device=x.device) for _ in range(4))
            ssm_states = tuple(torch.zeros(B, 64, 16, device=x.device) for _ in range(4))
            pred, _, _ = model(x, *conv_states, *ssm_states)
            loss = combined_loss(pred, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    return total_loss / len(dataloader)
```

### Training Configuration Summary

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Optimizer** | AdamW | Standard for transformer/SSM architectures |
| **Learning rate** | 3e-4 | Baseline for AdamW on audio tasks |
| **Weight decay** | 1e-5 | Light regularization |
| **Batch size** | 128 (maximize for A100 80GB) | Limited by chunk length, not model size |
| **Gradient clipping** | 1.0 (global norm) | Prevents SSM state explosion |
| **Mixed precision** | bfloat16 | A100 native, 2× memory efficiency |
| **Epochs** | 200 | With early stopping patience 20 |
| **Scheduler** | CosineAnnealingWarmRestarts (T₀=50, T_mult=2) | Periodic high LR for escaping local minima |
| **Gradient accumulation** | 1 (if needed, increase for effective batch size) | Not needed on A100 80GB |

## 5. Speed Optimization

### Data Loading Pipeline

```
Disk (.npy files)          RAM (/dev/shm)            GPU
    │                           │                      ▲
    ▼                           ▼                      │
[Pre-process once] → [Cache entire dataset] → [DataLoader] → [Training loop]
                         in tmpfs at pod start    num_workers=4
                                                   pin_memory=True
                                                   prefetch_factor=4
```

#### Pre-process All Data to .npy

Convert all audio chunks to `float32 .npy` files during setup. This avoids on-the-fly resampling, normalization, or segmentation during training. The tradeoff is disk space (~1.2 GB per 50 hours of mono audio at 44.1 kHz), which is negligible.

```python
# On-the-fly transforms are disabled during training:
class AudioDataset(Dataset):
    def __getitem__(self, idx):
        input_path = self.input_files[idx]
        target_path = self.target_files[idx]

        # Fast path: memory-mapped read
        # .npy files can be loaded with np.load, which is fast
        x = np.load(input_path).astype(np.float32)
        y = np.load(target_path).astype(np.float32)

        # No augmentations, no resampling, no normalization
        return torch.from_numpy(x), torch.from_numpy(y)
```

#### Cache Entire Dataset in RAM

At pod start, copy processed data to `/dev/shm` (tmpfs backed by RAM):

```bash
cp -r /workspace/data/processed/ /dev/shm/clara_data/
```

This eliminates all disk I/O during training. If the dataset exceeds available RAM, use NumPy memmap:

```python
# Alternative for datasets larger than RAM:
x = np.load(input_path, mmap_mode='r')
```

#### DataLoader Configuration

```python
DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,          # 4 worker processes for I/O
    pin_memory=True,        # Faster GPU transfer (pinned memory)
    prefetch_factor=4,      # Overlap I/O and compute (4 batches ahead)
    persistent_workers=True, # Keep workers alive between epochs
)
```

### Compute Optimization

#### torch.compile

PyTorch 2.x's `torch.compile` fuses the forward pass graph, reducing Python overhead and kernel launch latency:

```python
model = torch.compile(model, mode="reduce-overhead")
```

The `"reduce-overhead"` mode is best for small models (like 72K params) where kernel launch overhead is a significant fraction of total time. The first batch will be slow (compilation), but subsequent batches will be 20-40% faster.

**Important for Mamba**: The `for t in range(seq_len)` loop inside `MambaLayer.forward()` will be unrolled by the compiler. For `seq_len=88200` (2 s at 44.1 kHz), the unrolled graph will be very large. Compile may struggle or produce excessive memory usage for long sequences. Test with smaller seq_len first (1024 or 2048) and verify compilation succeeds before scaling up.

#### Mixed Precision

```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    pred, _, _ = model(x, *conv_states, *ssm_states)
    loss = loss_fn(pred, y)
```

bfloat16 is preferred over float16 for the Mamba SSM because:
- The SSM recurrence involves multiplication and addition of small-magnitude values (the `dA = exp(dt * A)` term). bfloat16 retains the same exponent range as float32, preventing underflow in the exponential.
- A100 has native bfloat16 tensor core support (2× throughput vs float32).
- No loss scaling needed (bfloat16 matches float32 exponent range).

#### cuDNN Benchmarking

```python
torch.backends.cudnn.benchmark = True
```

Allows cuDNN to find the fastest convolution algorithm for the fixed input sizes. Set once at the start of training.

### Expected Training Speed

| Configuration | Samples/second | Time per epoch (200K samples) |
|---------------|---------------|------------------------------|
| float32, no compile | ~125K | ~160 seconds |
| bf16, no compile | ~200K | ~100 seconds |
| bf16 + compile | ~250K+ | ~80 seconds |
| **Total (200 epochs, bf16 + compile)** | | **~4.4 hours** |

The model is small enough (72K params) that training is dominated by data loading and kernel launch overhead, not compute. Even with very modest hardware (RTX 3090), training should complete in under 12 hours.

## 6. Loss Functions

### Primary: L1 Loss (Mean Absolute Error)

```python
l1_loss = F.l1_loss(pred, y)
```

- Simple, stable, robust to outliers
- Converges reliably for audio tasks
- Preserves phase information (unlike spectral losses alone)

### Spectral: Multi-Resolution STFT Loss

STFT loss measures the difference between the short-time Fourier transform magnitude of the prediction and target. Multi-resolution means computing STFT at multiple window sizes to capture both fine detail and broad structure.

```python
class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512],
                 win_lengths=[512, 1024, 2048], window="hann_window"):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, x, y):
        loss = 0.0
        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            x_spec = torch.stft(
                x.squeeze(-1), fft_size, hop_size, win_length, window=self.window,
                return_complex=True
            )
            y_spec = torch.stft(
                y.squeeze(-1), fft_size, hop_size, win_length, window=self.window,
                return_complex=True
            )
            x_mag = x_spec.abs()
            y_mag = y_spec.abs()
            loss += F.l1_loss(x_mag, y_mag)
        return loss / len(self.fft_sizes)
```

### Combined Loss

```python
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.85, beta=0.15):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.stft_loss = MultiResolutionSTFTLoss()

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        spec = self.stft_loss(pred, target)
        return self.alpha * l1 + self.beta * spec
```

| Weight | Loss Component | Effect |
|--------|---------------|--------|
| α = 0.85 | L1 time-domain | Preserves waveform shape, phase, transients |
| β = 0.15 | Multi-resolution STFT | Improves spectral envelope, reduces artifacts |

### Optional: Feature Loss (Perceptual)

For tasks where perceptual quality matters (e.g., timbre transfer, vocal processing), add a feature loss from a pre-trained audio encoder:

```python
class FeatureLoss(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        pred_feat = self.encoder(pred)
        target_feat = self.encoder(target)
        return F.mse_loss(pred_feat, target_feat)
```

Suitable pre-trained encoders:
- **VGGish**: AudioSet-trained, 128-dim embeddings, good for general audio similarity
- **CLAP**: Contrastive Language-Audio Pretraining, aligns audio and text embeddings
- **OpenL3**: Audio embedding from sound + vision

### Loss Landscape Summary

| Loss | Domain | Weight | Pros | Cons |
|------|--------|--------|------|------|
| L1 | Time | 0.85 | Simple, phase-preserving, stable | Ignores frequency structure |
| Multi-resolution STFT | Frequency | 0.15 | Captures spectral detail | Phase-invariant (magnitude only) |
| Feature (optional) | Latent | 0.1 | Perceptual quality | Requires pre-trained encoder, compute overhead |

Start with L1 + STFT loss. Add feature loss only if results are perceptually poor.

## 7. RunPod Setup Commands (Copy-Paste Ready)

### Quick Start

```bash
# Launch a RunPod pod with A100 80GB, attach network volume
# Use the "PyTorch 2.x - CUDA 12.x" template

# Start tmux session
tmux new-session -s train

# Clone repository
cd /workspace
git clone <your-repo-url> clara
cd clara

# Setup Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning tensorboard onnx onnxruntime numpy matplotlib librosa

# Create project directories
mkdir -p /workspace/data/raw /workspace/data/processed /workspace/logs
```

### Dataset Download

```bash
# MUSDB18 (multi-track stems)
aria2c -x 16 -s 16 https://zenodo.org/records/1117372/files/musdb18.zip \
    -d /workspace/data/raw/

# DNS Challenge (noisy/clean speech pairs)
aria2c -x 16 -s 16 https://dns-challenge.blob.core.windows.net/datasets/...
    -d /workspace/data/raw/

# MAESTRO (piano performances)
aria2c -x 16 -s 16 https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip \
    -d /workspace/data/raw/
```

### Pre-processing

```bash
# Pre-process dataset to /dev/shm for RAM-speed loading
python3 preprocess.py \
    --input /workspace/data/raw/ \
    --output /dev/shm/clara_data/ \
    --sample-rate 44100 \
    --chunk-samples 88200 \
    --peak-normalize-db -1.0

# Copy pre-processing script to workspace
cp preprocess.py /workspace/clara/
```

### Launch Training

```bash
cd /workspace/clara
source venv/bin/activate

# Training (lightning version)
python3 train.py \
    --data-dir /dev/shm/clara_data/ \
    --epochs 200 \
    --batch-size 128 \
    --lr 3e-4 \
    --mixed-precision bf16 \
    --compile \
    --log-dir /workspace/logs/

# Detach from tmux: Ctrl+B, D
# Re-attach: tmux attach -t train
```

### Monitoring

```bash
# In a separate tmux window (Ctrl+B, C):
watch -n 1 nvidia-smi

# Or start TensorBoard:
tensorboard --logdir /workspace/logs/ --bind_all

# View training logs:
tail -f /workspace/logs/train.log
```

### Full Session Script

```bash
#!/bin/bash
# runpod_train.sh — full setup and training script for RunPod

set -euo pipefail

REPO_URL="<your-repo-url>"
DATASET_URL="<dataset-url>"

echo "=== CLARA Mamba SSM Training Setup ==="

# Clone
cd /workspace
if [ ! -d "clara" ]; then
    git clone "$REPO_URL" clara
fi
cd clara

# Venv
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Dependencies
pip install --quiet --upgrade pip
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --quiet pytorch-lightning tensorboard onnx onnxruntime numpy matplotlib librosa

# Dataset
mkdir -p /workspace/data/raw /workspace/data/processed /workspace/logs
if [ ! -f "/workspace/data/raw/dataset.zip" ]; then
    aria2c -x 16 -s 16 "$DATASET_URL" -d /workspace/data/raw/
fi

# Pre-process
python3 preprocess.py \
    --input /workspace/data/raw/ \
    --output /dev/shm/clara_data/ \
    --sample-rate 44100 \
    --chunk-samples 88200

# Train
python3 train.py \
    --data-dir /dev/shm/clara_data/ \
    --epochs 200 \
    --batch-size 128 \
    --lr 3e-4 \
    --mixed-precision bf16 \
    --compile \
    --log-dir /workspace/logs/

echo "=== Training complete ==="
```

## 8. Expected Training Time

### Estimates for A100 80GB with 50 Hours of Audio

| Scenario | Batch Size | Steps/Epoch | Sec/Step | Time/Epoch | Total (200 epochs) |
|----------|-----------|-------------|----------|------------|-------------------|
| float32, no compile | 128 | 1,562 | ~3.0 | ~78 min | ~260 hours (11 days) |
| bf16, no compile | 128 | 1,562 | ~1.5 | ~39 min | ~130 hours (5.4 days) |
| bf16 + compile | 128 | 1,562 | ~1.0 | ~26 min | ~87 hours (3.6 days) |
| **Recommended: bf16 + compile** | **128** | **1,562** | **~1.0** | **~26 min** | **~87 hours (3.6 days)** |

### Scaling to Different Dataset Sizes

| Dataset Size | Chunks (2 s) | Steps/epoch (batch 128) | Time/epoch (bf16+compile) | 200 epochs |
|-------------|-------------|------------------------|--------------------------|------------|
| 10 hours | 180,000 | 1,406 | ~23 min | ~77 hours |
| 25 hours | 450,000 | 3,516 | ~59 min | ~196 hours |
| **50 hours** | **900,000** | **7,031** | **~117 min** | **~390 hours** |
| 100 hours | 1,800,000 | 14,062 | ~234 min | ~780 hours |

**Correction**: The above assumes 200K samples. With 50 hours @ 44.1 kHz at 2 s chunks = 90,000 chunks (not 200K). Let me recalculate:

50 hours × 3600 seconds/hour = 180,000 seconds
180,000 seconds / 2 seconds per chunk = 90,000 chunks
With no overlap: 90,000 chunks

| Dataset Size | Chunks (2 s) | Steps/Epoch (batch 128) | Time/Epoch (1.0 s/step) |
|-------------|-------------|------------------------|------------------------|
| 10 hours | 18,000 | 141 | ~2.4 min |
| 25 hours | 45,000 | 352 | ~5.9 min |
| 50 hours | 90,000 | 704 | ~11.7 min |
| 100 hours | 180,000 | 1,407 | ~23.5 min |

Total for 50 hours, 200 epochs: 11.7 min × 200 = ~39 hours. With compile: potentially ~28-32 hours.

However, the per-step time of 1.0 s assumes the model is the bottleneck. At 72K params, the bottleneck is data handling and Python overhead, not compute. Expect 0.3-0.5 s/step in practice, making total training for 50 hours × 200 epochs closer to 12-16 hours.

These are approximations. Actual time depends on dataset size, chunk length, and batch size.

### Bottleneck Analysis

For the 72K-parameter MambaRT on A100:

| Component | Time | Notes |
|-----------|------|-------|
| Forward pass | ~50 μs | Two matrix multiplies per layer × 4 layers + SSM scan |
| Backward pass | ~150 μs | ~3× forward for gradient computation |
| Data loading | ~100 μs | .npy load + GPU transfer (with pinned memory) |
| Python overhead | ~200 μs | Loop, scheduler, logging |
| **Total per sample** | **~500 μs** | Batch of 128: ~64 ms |

The model is **compute-bound** only for very long sequences (seq_len > 8192). For typical chunk sizes (88200 = 2 s), the SSM loop dominates: 88200 iterations of the scan step, each doing small matrix ops. This is GPU-efficient only if the torch compiler fuses the loop body.

To maximize GPU utilization at very long seq_len, consider:
- Increasing batch size (GPU has 80 GB — batch size 512-1024 is feasible)
- Reducing chunk length (try 1 s = 44100 samples)
- Using gradient accumulation to spread memory

## 9. ONNX Export Post-Training

After training completes, export the trained weights back to ONNX format for use with the C++ inference engine.

### Export Command

```bash
# Export trained model to ONNX
# The export_mamba_onnx.py script loads the same architecture,
# then we can optionally load weights from checkpoint

python3 export_mamba_onnx.py \
    --checkpoint /workspace/logs/checkpoint_best.pt \
    --output /workspace/clara/models/mamba_rt.onnx \
    --seq-len 256
```

### Modified Export Script (with checkpoint loading)

```python
# Add to export_mamba_onnx.py or create export.py:

def export_trained(checkpoint_path, output_path, seq_len=256):
    # Rebuild the model architecture
    model = MambaRTModel(d_model=64, d_state=16, n_layers=4,
                         conv_width=4, dt_rank=16)

    # Load trained weights
    # If using Lightning: checkpoint saves full module
    # If using raw PyTorch: checkpoint saves state_dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        # Lightning format: strip "model." prefix
        sd = {}
        for k, v in state_dict["state_dict"].items():
            if k.startswith("model."):
                sd[k[6:]] = v
            else:
                sd[k] = v
        model.load_state_dict(sd)
    else:
        model.load_state_dict(state_dict)

    model.eval()

    # Export to ONNX with the same I/O contract
    batch_size = 1
    dummy_audio = torch.randn(batch_size, seq_len, 1)
    dummy_convs = tuple(torch.zeros(batch_size, 64, 4) for _ in range(4))
    dummy_ssms = tuple(torch.zeros(batch_size, 64, 16) for _ in range(4))

    torch.onnx.export(
        model,
        (dummy_audio, *dummy_convs, *dummy_ssms),
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['audio_chunk', 'conv0', 'conv1', 'conv2', 'conv3',
                     'ssm0', 'ssm1', 'ssm2', 'ssm3'],
        output_names=['processed_audio', 'new_conv_state', 'new_ssm_state'],
        dynamic_axes={
            'audio_chunk': {1: 'seq_len'},
            'processed_audio': {1: 'seq_len'},
        },
    )
    print(f"Exported trained model to {output_path}")
```

### Verification

The export script already includes verification. After export, run:

```bash
# Run the verification embedded in export_mamba_onnx.py
python3 -c "
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('models/mamba_rt.onnx')

# Check shapes
audio = np.random.randn(1, 64, 1).astype(np.float32)
feeds = {'audio_chunk': audio}
for i in range(4):
    feeds[f'conv{i}'] = np.zeros((1, 64, 4), dtype=np.float32)
    feeds[f'ssm{i}'] = np.zeros((1, 64, 16), dtype=np.float32)

out_audio, out_conv, out_ssm = sess.run(None, feeds)

assert out_audio.shape == (1, 64, 1), f'Audio shape: {out_audio.shape}'
assert out_conv.shape == (1, 4, 64, 4), f'Conv shape: {out_conv.shape}'
assert out_ssm.shape == (1, 4, 64, 16), f'SSM shape: {out_ssm.shape}'
assert np.all(np.isfinite(out_audio)), 'Output has NaN!'
print('ONNX verification PASSED')

# Verify state continuity (critical for real-time use)
# Process two chunks back-to-back
chunk1 = np.random.randn(1, 32, 1).astype(np.float32)
chunk2 = np.random.randn(1, 32, 1).astype(np.float32)

feeds1 = {'audio_chunk': chunk1}
for i in range(4):
    feeds1[f'conv{i}'] = np.zeros((1, 64, 4), dtype=np.float32)
    feeds1[f'ssm{i}'] = np.zeros((1, 64, 16), dtype=np.float32)

out1, conv_out, ssm_out = sess.run(None, feeds1)

feeds2 = {'audio_chunk': chunk2}
for i in range(4):
    feeds2[f'conv{i}'] = conv_out[:, i, :, :]
    feeds2[f'ssm{i}'] = ssm_out[:, i, :, :]

out2, _, _ = sess.run(None, feeds2)

assert not np.allclose(out2, 0.0), 'State continuity FAILED (all zeros)'
assert np.all(np.isfinite(out2)), 'State continuity FAILED (NaN)'
print('State continuity verification PASSED')
print(f'Output range: [{out2.min():.6f}, {out2.max():.6f}]')
"
```

### Deploy

After successful verification, the ONNX model is ready for deployment:

```bash
# Copy to the project's models/ directory (adjacent to plugin)
cp /workspace/clara/models/mamba_rt.onnx /workspace/clara/models/
# The .onnx.data sidecar is generated automatically during export
# Both files must be present in the models/ directory

# Rebuild the C++ plugins with the new model
cd /workspace/clara/build
cmake .. && cmake --build . --parallel

# Verify the plugin loads the new model
./clap-validator build/plugins/MambaSuiteClap.clap || true
```

### Export Checklist

- [ ] Model exported with opset 18 (matches `MambaInferenceEngine` expectations)
- [ ] `dynamic_axes` set for `audio_chunk`/{1: 'seq_len'} and `processed_audio`/{1: 'seq_len'}
- [ ] Input/output names match C++ side: `audio_chunk`, `conv0`-`conv3`, `ssm0`-`ssm3`, `processed_audio`, `new_conv_state`, `new_ssm_state`
- [ ] State shapes match: conv `{1, 64, 4}`, SSM `{1, 64, 16}`, stacked outputs `{1, 4, 64, 4}` and `{1, 4, 64, 16}`
- [ ] External data format preserves ONNX graph (uses `.onnx.data` sidecar)
- [ ] `export_mamba_onnx.py` runs without errors in `--verify` mode
- [ ] State continuity verified across 2+ chunks (output states feed next input states)
- [ ] Output values are finite and non-zero for random input
- [ ] Model and `.onnx.data` deployed to `build/plugins/models/`
- [ ] C++ plugins rebuild and load the new model

### Versioning

Consider versioning trained ONNX models:

```
models/
├── mamba_rt.onnx              # Symlink to latest trained model
├── mamba_rt.onnx.data         # Symlink to latest trained weights
├── v1_mamba_rt.onnx           # Trained v1 weights
├── v1_mamba_rt.onnx.data
├── v2_mamba_rt.onnx           # Trained v2 weights
├── v2_mamba_rt.onnx.data
└── ...
```

This allows rolling back to a previous model if a new training run produces worse results.

---

# License: Apache 2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
