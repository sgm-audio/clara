# CLARA Training Guide — Mamba-3

**Architecture:** Mamba-3 SSM with multi-head MIMO R=4  
**Parameters:** 1,662,384  
**Training script:** `python/training/train_clara.py`  
**Weight format:** Flat binary `model.bin` (version 2)  
**Inference engine:** Custom C++ (`libMamba3Core.a`), no ONNX Runtime

This document is the single source of truth for training CLARA. For a one-command quick start, see [`TRAINING_QUICKSTART.md`](TRAINING_QUICKSTART.md).

---

## Table of Contents

1. [Model Architecture](#1-model-architecture)
2. [Training Pipeline Overview](#2-training-pipeline-overview)
3. [Stage 0: Self-Supervised Pretraining](#3-stage-0-self-supervised-pretraining)
4. [Stage 1: Supervised Effect Training](#4-stage-1-supervised-effect-training)
5. [Stage 2: Adversarial Fine-Tuning](#5-stage-2-adversarial-fine-tuning)
6. [Stage 3: RLHF / Preference Optimization](#6-stage-3-rlhf--preference-optimization)
7. [Datasets](#7-datasets)
8. [Training Commands](#8-training-commands)
9. [Loss Functions](#9-loss-functions)
10. [Precomputed Feature Pipeline](#10-precomputed-feature-pipeline)
11. [Export & Validation](#11-export--validation)
12. [Configuration Reference](#12-configuration-reference)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Model Architecture

CLARA uses a Mamba-3 Selective State Space Model with multi-head MIMO (Multiple-Input Multiple-Output) rank 4. Each of the 4 heads has 4 slots, giving 16 independent SSM streams per layer.

### Dimensions

| Constant | Value | Description |
|----------|-------|-------------|
| `d_model` | 128 | Embedding dimension |
| `d_inner` | 256 | 2× d_model expansion |
| `d_state` | 32 | SSM state dimension |
| `nheads` | 4 | Parallel heads |
| `headdim` | 64 | Per-head dimension (= d_inner / nheads) |
| `mimo_rank` | 4 | Slots per head (sub / bass / mids / highs) |
| `n_layers` | 6 | Stacked SSM layers |
| `cond_dim` | 128 | Conditioning vector dimension |
| `chunk_size` | 64 | Sub-block size matching C++ engine |
| `num_rope_angles` | 16 | = d_state / 2 |

### Parameter Count

| Component | Parameters |
|-----------|------------|
| Per-layer weights (×6) | 277,064 each |
| 6 layers | 1,662,384 |
| **Total** | **1,662,384** |

*Note: The migration plan's ~757K claim was stale — the architecture produces 1.66M params due to full B/C projections (nheads×mimo_rank×d_state = 512 each) and film/dt_film layers.*

Each layer exports 13 float32 arrays in the flat binary weight file:

| # | Array | Shape |
|---|-------|-------|
| 0 | `in_proj_w` | `[in_proj_dim, d_model]` |
| 1 | `dt_bias` | `[nheads]` |
| 2 | `B_bias` | `[nheads, mimo_rank, d_state]` |
| 3 | `C_bias` | `[nheads, mimo_rank, d_state]` |
| 4 | `B_norm_weight` | `[d_state]` |
| 5 | `C_norm_weight` | `[d_state]` |
| 6 | `mimo_x` | `[nheads, mimo_rank, headdim]` |
| 7 | `mimo_z` | `[nheads, mimo_rank, headdim]` |
| 8 | `mimo_o` | `[nheads, mimo_rank, headdim]` |
| 9 | `D_proj` | `[nheads]` |
| 10 | `out_proj_w` | `[d_inner, d_model]` |
| 11 | `film_proj_w` | `[2*d_model, cond_dim]` |
| 12 | `dt_film_proj_w` | `[2*nheads, cond_dim]` |

The input/output projection and LayerNorm are intentionally **not** learned: the C++ engine embeds scalar audio into the first channel of `d_model` and reads the first channel as output, with a fixed `gamma=1, beta=0` LayerNorm. This keeps Python and C++ bit-exact compatible.

### SSM Block Pipeline

Each `SSMBlock` runs an 8-step per-sample pipeline:

1. **FiLM conditioning** — `gamma, beta = film_proj @ cond`; `u_film = u * (1 + gamma) + beta`
2. **Fused `in_proj`** — single matvec producing `z`, `x`, `B`, `C`, `dd_dt`, `dd_A`, `trap`, `angles`
3. **A and Δ preprocessing** — `dt = softplus(dd_dt + dt_bias)`; `A = -softplus(dd_A)`
4. **RMSNorm on B and C**
5. **RoPE rotation** on `B` and `C` using cumulative angles
6. **Multi-head MIMO SSM recurrence** — trapezoidal discretization across `mimo_rank` slots
7. **SiLU z-gating** — `z_gated = z * sigmoid(z)`
8. **Output projection + D skip + residual**

State is normalized every 4096 samples if any element exceeds ±1.0.

---

## 2. Training Pipeline Overview

Training follows a multi-stage curriculum. Each stage builds on the previous one.

```
Stage 0: SSL Pretraining
    │
    ▼
Stage 1: Supervised Per-Effect Training
    │
    ▼
Stage 2: Adversarial Fine-Tuning
    │
    ▼
Stage 3: RLHF / Preference Optimization (post-MVP)
```

**Only `python/training/train_clara.py` produces exportable weights.** Older scripts (`train_clara_parallel.py`, `train_clara_export.py`, `export_mamba_onnx.py`, `train_clara_now.py`) implement incompatible Mamba-1 architectures and should not be used.

---

## 3. Stage 0: Self-Supervised Pretraining

Stage 0 teaches the model general audio structure before any effect-specific training. For the MVP this is typically an **identity autoencoder** on clean musical audio: the model learns to pass audio through cleanly, establishing phase coherence, transient preservation, and a stable baseline.

| Aspect | Detail |
|--------|--------|
| **Task** | input == target (identity mapping) |
| **Data** | MUSDB18-HQ mixture tracks |
| **Conditioning** | Random 128-dim vectors (the model learns to ignore or condition on them) |
| **Loss** | MSE + Multi-Resolution STFT (default 0.5/0.5) |
| **Steps** | 100,000 |
| **Output** | Base checkpoint for Stage 1 fine-tuning |

### Quick start

```bash
python python/training/train_clara.py \
  --data ~/datasets/musdb18hq/train \
  --device cuda \
  --steps 100000 \
  --batch 4 \
  --grad-accum 4
```

For fastest training with the precomputed feature pipeline:

```bash
# One-time preprocessing
python python/tools/precompute_features.py \
  --input ~/datasets/musdb18hq/train \
  --output ~/datasets/musdb18hq/precomputed

# Train using precomputed target features
python python/training/train_clara.py \
  --precomputed ~/datasets/musdb18hq/precomputed \
  --cache \
  --device cuda \
  --steps 100000
```

---

## 4. Stage 1: Supervised Effect Training

Each effect is trained as a dedicated model. There is no multi-task compromise.

| Plugin | Data (paired) | Loss | Steps | Notes |
|--------|---------------|------|-------|-------|
| **Mamba Transparent** | clean → clean | MSE + STFT | 50K | Proves zero-latency transparency |
| **Mamba Saturator** | clean → tape/tube/digital sat | MSE + STFT + SI-SDR + harmonic distortion | 100K | MIMO slots learn sub/bass/mids/highs |
| **Mamba Compressor** | clean → compressed | MSE + STFT + RMS envelope + crest factor | 100K | Dynamics prediction without lookahead |
| **Mamba Denoiser** | noisy → clean | MSE + STFT + perceptual (CDPAM) | 150K | Preserves musical character |
| **Mamba Remixer** | original → stylized | Style + content + adversarial | 200K | Semantic style transfer |

Each plugin loads the Stage 0 checkpoint as initialization and fine-tunes on task-specific paired data. Conditioning slots 8-39 can carry CLAP text/audio embeddings; slots 40-71 carry user knob values.

---

## 5. Stage 2: Adversarial Fine-Tuning

After Stage 1, add a spectrogram discriminator and continue training for ~20K steps per plugin.

| Aspect | Detail |
|--------|--------|
| **Input** | Stage 1 checkpoint |
| **Loss** | Stage 1 losses + adversarial loss |
| **Effect** | Reduces digital artifacts, improves perceptual realism |
| **Steps** | ~20,000 per plugin |

---

## 6. Stage 3: RLHF / Preference Optimization

Post-MVP stage once user feedback data is available.

| Aspect | Detail |
|--------|--------|
| **When** | After public beta / user data collection |
| **Method** | DPO (Direct Preference Optimization) — no reward model needed |
| **Data** | A/B comparisons of model outputs rated by users |
| **Effect** | Learns subjective quality: warmth, punch, clarity |

---

## 7. Datasets

### Required for MVP

| Dataset | Size | Use | Location |
|---------|------|-----|----------|
| **MUSDB18-HQ** | ~41 GB | Stage 0 identity training, base musical audio | `~/datasets/musdb18hq/train/` |

### Optional / Future Stages

| Dataset | Size | Use |
|---------|------|-----|
| FMA Full | ~100 GB | SSL pretraining diversity |
| Slakh2100 Redux | ~30 GB | Stem separation, multi-instrument |
| FSD50K | ~10 GB | General sound events |
| AudioSet subset | varies | Contrastive CLAP pretraining |
| DNS Challenge | ~100+ GB | Denoising pairs |

### Data format

Training expects mono WAV files at the target sample rate (default 44.1 kHz). The dataset loader reads files recursively, converts multi-channel audio to mono, and extracts random `chunk_samples`-length crops.

---

## 8. Training Commands

### Local Training

**Fresh Stage 0 (on-the-fly STFT):**

```bash
python python/training/train_clara.py \
  --data ~/datasets/musdb18hq/train \
  --device cuda \
  --steps 100000 \
  --batch 4 \
  --grad-accum 4 \
  --lr 1e-3 \
  --save checkpoints_clara
```

**With precomputed features (faster):**

```bash
python python/tools/precompute_features.py \
  --input ~/datasets/musdb18hq/train \
  --output ~/datasets/musdb18hq/precomputed

python python/training/train_clara.py \
  --precomputed ~/datasets/musdb18hq/precomputed \
  --cache \
  --device cuda \
  --steps 100000 \
  --batch 4 \
  --grad-accum 4
```

**Resume from checkpoint:**

```bash
python python/training/train_clara.py \
  --resume checkpoints_clara/step_050000.pt \
  --precomputed ~/datasets/musdb18hq/precomputed \
  --cache \
  --device cuda \
  --steps 100000
```

**CPU smoke test:**

```bash
python python/training/train_clara.py \
  --data ~/datasets/musdb18hq/train \
  --device cpu \
  --steps 10 \
  --batch 2
```

### Cloud Training (MI300X)

See `python/training/train_clara.py` for training options. For the launch workflow:

```bash
# On the cloud VM — the launcher is archived, not in scripts/ (verified 2026-10-04:
# the URL this block used to print pointed at a different account and a missing file)
git clone https://github.com/sgm-audio/clara.git && cd clara
bash archive/old-training-files/setup_cloud.sh
```

This installs ROCm + PyTorch, downloads MUSDB18-HQ, precomputes features, starts cost monitoring (warns at $75, kills at $90), syncs checkpoints to S3 every 5 minutes, and begins Stage 0 training.

### Monitoring

```bash
# GPU status (inside distrobox or on cloud VM)
watch -n 2 rocm-smi --showtemp --showpower --showuse --showmemuse

# Training log
tail -f checkpoints_clara/train.log

# Most recent checkpoint
ls -lt checkpoints_clara/ | head -5
```

---

## 9. Loss Functions

### `CombinedLoss` (on-the-fly)

Default when training without `--precomputed`:

```python
loss = 0.5 * MSE(pred, target) + 0.5 * MultiResolutionSTFT(pred, target)
```

The multi-resolution STFT uses FFT sizes `[512, 1024, 2048]` with hops `[128, 256, 512]` and Hann windows.

### `PrecomputedMRSTFTLoss` (precomputed pipeline)

Default when training with `--precomputed`:

```python
loss = 0.5 * MSE(pred, target)
       + 0.5 * MSE(|STFT(pred)|, target_stft)
       + 0.5 * MSE(log_mel(pred), target_mel)
```

Target STFT magnitude and log-mel spectrogram are precomputed once by `python/tools/precompute_features.py`, so target-side DSP consumes zero GPU time during training.

### Optional: SI-SDR

For Stage 1 effect tasks, add SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) to the loss mix to improve perceptual reconstruction quality.

---

## 10. Precomputed Feature Pipeline

The precomputed pipeline removes the CPU STFT/mel bottleneck from the training loop.

### One-time preprocessing

```bash
python python/tools/precompute_features.py \
  --input ~/datasets/musdb18hq/train \
  --output ~/datasets/musdb18hq/precomputed \
  --sr 44100 \
  --n-fft 1024 \
  --hop-length 256 \
  --n-mels 64 \
  --chunk-samples 22050 \
  --num-workers 8
```

This writes `.npz` files containing:

- `audio`: mono waveform chunk
- `stft`: precomputed target STFT magnitude
- `mel`: precomputed log-mel spectrogram

### Training with precomputed data

```bash
python python/training/train_clara.py \
  --precomputed ~/datasets/musdb18hq/precomputed \
  --cache \
  --device cuda \
  --steps 100000
```

`--cache` loads all precomputed tensors into RAM as pinned torch tensors for fastest GPU transfer.

---

## 11. Export & Validation

### Export to flat binary

```bash
python python/training/train_clara.py \
  --export checkpoints_clara/final.pt \
  --output models/model_trained.bin
```

Or use the dedicated export script:

```bash
python python/tools/export_mamba_weights.py \
  --output models/model.bin
```

Or export directly from the training script:

```bash
python python/training/train_clara.py \
  --export checkpoints/final.pt \
  --output models/model.bin
```

### Validate in C++ engine

```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The C++ test suite includes:

- Construct/destroy lifecycle
- `init()` with model path
- `processBlock()` with silence input
- `resetState()` thread safety
- Python↔C++ numerical equivalence (tolerance ~0.05 for float32 cross-library difference)
- Long-run stability (60s @ 44.1 kHz, no NaN/Inf)

### Flat binary format

The exported `model.bin` has a 64-byte header:

| Field | Bytes | Value |
|-------|-------|-------|
| magic | 4 | `0x4D424D41` ("MBMA") |
| version | 4 | 2 |
| d_state | 4 | 32 |
| d_model | 4 | 128 |
| d_inner | 4 | 256 |
| nheads | 4 | 4 |
| headdim | 4 | 64 |
| mimo_rank | 4 | 4 |
| n_layers | 4 | 6 |
| cond_dim | 4 | 128 |
| in_proj_dim | 4 | 1612 |
| weight_data_size | 4 | total bytes |
| reserved | 12 | 0 |

Followed by the concatenated float32 weight arrays (13 per layer).

---

## 12. Configuration Reference

### Model hyperparameters (`Mamba3Config`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 128 | Embedding dimension |
| `d_inner` | 256 | Expanded dimension |
| `d_state` | 32 | SSM state dimension |
| `nheads` | 4 | Number of heads |
| `headdim` | 64 | Dimension per head |
| `mimo_rank` | 4 | MIMO slots per head |
| `n_layers` | 6 | Number of layers |
| `cond_dim` | 128 | Conditioning vector size |
| `in_proj_dim` | 1612 | Fused projection output size |

### Training flags (`python/training/train_clara.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | None | Path to WAV dataset dir (on-the-fly loading) |
| `--precomputed` | None | Path to precomputed `.npz` directory (faster) |
| `--cache` | off | RAM-cache precomputed tensors |
| `--cache-limit` | 16 | Max tracks preloaded into RAM (0=all) |
| `--save` | `checkpoints` | Checkpoint directory |
| `--resume` | None | Resume from checkpoint path (preserves optimizer/scheduler) |
| `--steps` | 100000 | Total training steps |
| `--batch` | 4 | Batch size |
| `--grad-accum` | 4 | Gradient accumulation steps |
| `--lr` | 1e-3 | Learning rate |
| `--weight-decay` | 1e-5 | Weight decay |
| `--clip-grad-norm` | 1.0 | Gradient clipping norm |
| `--chunk-size` | 64 | SSM sub-block size matching C++ `kSubBlockSize` |
| `--scan` | `parallel` | SSM scan: `parallel` (fast) or `sequential` (bit-exact C++) |
| `--clip-seconds` | 0.5 | Audio clip duration in seconds |
| `--num-workers` | 0 | DataLoader workers (0=safest for iGPU) |
| `--log-every` | 50 | Log interval (steps) |
| `--ckpt-every` | 1000 | Checkpoint interval (steps) |
| `--val-every` | 2000 | Validation interval (steps) |
| `--device` | cuda | `cuda` or `cpu` |
| `--warmup` | 2000 | Linear warmup steps |
| `--val-split` | 0.1 | Fraction held out for validation |
| `--early-stop-patience` | 10 | Stop after N val checks without improvement |
| `--export` | None | Export checkpoint to `model.bin` |
| `--output` | `models/model_trained.bin` | Output `.bin` path |

### Optimizer & scheduler

- **Optimizer:** AdamW
- **Learning rate:** cosine decay from `--lr` after `--warmup` linear ramp
- **Gradient clipping:** global norm `--clip-grad-norm`
- **Effective batch size:** `--batch` × `--grad-accum`

---

## 13. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CUDA not available` on AMD APU (860M) | gfx1152 not in ROCm allowlist | `export HSA_OVERRIDE_GFX_VERSION=11.0.0` before Python import |
| `CUDA not available` on MI300X | Wrong env override | Do **NOT** set `HSA_OVERRIDE_GFX_VERSION`; ROCm detects gfx942 natively |
| NaN loss | bfloat16 AMP on 860M ROCm | Default is float32 (no AMP flag); switch to CPU if persistent |
| OOM | Batch/clip too large | Reduce `--batch` or `--chunk-samples` |
| Slow training | CPU STFT bottleneck | Use `--precomputed` + `--cache` |
| Hang at warmup | MIOpen compilation deadlock | Already reduced to 1 chunk; kill and retry if persistent |
| C++ load fails | Wrong weight format | Ensure export used version 2 format from `python/training/train_clara.py` |
| Python↔C++ mismatch >0.05 | Architecture divergence | Check that input/output projections and LayerNorm are stripped in Python |

---

## See Also

- [`MAMBA3_MIGRATION_PLAN.md`](MAMBA3_MIGRATION_PLAN.md) — Full architecture, C++ engine spec, product roadmap, risk register
- [`TRAINING_QUICKSTART.md`](TRAINING_QUICKSTART.md) — One-command quick start
- [`reference_forward.py`](../python/reference_forward.py) — PyTorch bit-exact gold standard
- [`MAMBA3_DRIVE_AUDIT.md`](MAMBA3_DRIVE_AUDIT.md) — mamba3 external drive audit
