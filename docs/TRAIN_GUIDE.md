# CLARA Training Guide

How to train the PerTrackProcessor (Mamba-3 SSM, R=8 spectral-domain, ~4.26M params).

## Prerequisites

- **Dataset:** MUSDB18-HQ at `/run/media/sgm/Data/datasets/musdb18hq`
  - 43 complete tracks (vocals, drums, bass, other per track)
  - CREPE f0 features pre-extracted (`.crepe.npy` + `.pitched.npy` next to each `.wav`)
- **Python env:** `/home/sgm/micromamba/envs/rocm_torch` (PyTorch 2.11 ROCm build)
- **GPU:** AMD Radeon 860M (15.2 GB UMA VRAM, shared with system RAM)

## Quick Start

### 1. Smoke Test (5 checks, ~10 seconds)

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 HIP_VISIBLE_DEVICES=0 ROC_VISIBLE_DEVICES=0 \
  HSA_ENABLE_SDMA=0 \
  PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
  MIOPEN_FIND_MODE=5 \
  PYTHONPATH="$(pwd)/python/training" \
  /home/sgm/micromamba/envs/rocm_torch/bin/python3 \
  python/tools/test_model.py --device cuda
```

Expect: `5/5 passed`.

### 2. CPU Training (reliable, no GPU required)

```bash
PYTHONPATH="$(pwd)/python/training" \
  /home/sgm/micromamba/envs/rocm_torch/bin/python3 \
  python/training/train_clara.py \
  --data /run/media/sgm/Data/datasets/musdb18hq/train \
  --save checkpoints_clara \
  --device cpu \
  --batch 2 --grad-accum 4 \
  --clip-seconds 0.5 \
  --steps 100000 \
  --warmup 2000 \
  --val-every 2000 \
  --ckpt-every 1000 \
  --log-every 50 \
  --skip-contrastive \
  --num-workers 0 \
  --cache-limit 4
```

Effective batch = `batch × grad-accum` = 8. Speed: ~0.1–0.5 step/s.

### 3. GPU Training (faster, needs clean system state)

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 HIP_VISIBLE_DEVICES=0 ROC_VISIBLE_DEVICES=0 \
  HSA_ENABLE_SDMA=0 \
  PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:64 \
  MIOPEN_FIND_MODE=5 \
  PYTHONPATH="$(pwd)/python/training" \
  /home/sgm/micromamba/envs/rocm_torch/bin/python3 \
  python/training/train_clara.py \
  --data /run/media/sgm/Data/datasets/musdb18hq/train \
  --save checkpoints_clara \
  --device cuda \
  --batch 2 --grad-accum 4 \
  --clip-seconds 0.5 \
  --steps 100000 \
  --warmup 2000 \
  --val-every 2000 \
  --ckpt-every 1000 \
  --log-every 50 \
  --skip-contrastive \
  --num-workers 0 \
  --cache-limit 4 \
  --amp
```

Speed: ~0.5–2.0 step/s with AMP (bf16).

## GPU Training with AMP (bf16)

Add `--amp` to enable bfloat16 mixed precision. Gradients are in fp32, forward pass in bf16.

**Known issue:** Gradient checkpointing (`torch.utils.checkpoint`) is **disabled** under AMP because ROCm's `torch.autocast` + `torch.checkpoint` produce `CheckpointError` (dtype mismatch during recompute). The model is small enough (81 MB peak VRAM) that checkpointing is unnecessary.

No extra flags needed — the training script auto-disables checkpointing when `--amp` is on.

## System Limits

| Resource | Limit | Notes |
|----------|-------|-------|
| GPU VRAM | 15.2 GB | Shared with system RAM (UMA) |
| System RAM | 30 GB total | ~15 GB available for GPU + process |
| Swap | 15 GB (zram) | OOM-killer triggers at ~25 GB RSS |
| ROCm overhead | ~1.3 GB RSS | Just importing torch with ROCm |
| Model peak VRAM | ~81 MB | Without gradient checkpointing |
| Clip length ceiling | 0.5s @ batch=1 | Longer clips → OOM on this APU |

### Why 0.5s clips?

The 860M APU uses Unified Memory Architecture (UMA) — GPU and CPU share physical RAM. ROCm init maps a large virtual address region (~10 GB). When swap fills up from failed runs, even modest allocations trigger the OOM killer (exit code 137 = SIGKILL).

**Workarounds if 0.5s clips feel too short:**
- Use `--grad-accum 4` (effective batch = 8 with `--batch 2`)
- Reboot to clear swap, then GPU training may tolerate 1–2s clips
- Train on CPU (slower but no memory pressure)

## Phase Schedule

Training has 3 phases controlled by `--steps`:

| Phase | Step Range | Description |
|-------|-----------|-------------|
| 1 | 0 → 20% | Contrastive pretraining (disabled by default with `--skip-contrastive`) |
| 2 | 20% → 50% | Single-track reconstruction (no cross-track conditioning) |
| 3 | 50% → 100% | Cross-track conditioned reconstruction |

Phase 1 requires `--skip-contrastive` to be **removed** (it defaults to `True`).

## Key CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | *required* | Path to MUSDB18-HQ train/ root |
| `--save` | `./checkpoints` | Checkpoint directory |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--batch` | 4 | Batch size per step |
| `--grad-accum` | 1 | Gradient accumulation steps (effective = batch × accum) |
| `--steps` | 100000 | Total training steps |
| `--lr` | 3e-4 | Learning rate (AdamW) |
| `--warmup` | 2000 | Linear warmup steps |
| `--clip-seconds` | 4.0 | Audio clip duration (reduce to 0.5 for this APU) |
| `--amp` | off | Enable bf16 mixed precision |
| `--skip-contrastive` | on | Skip Phase 1 (contrastive pretraining) |
| `--val-every` | 2000 | Validation interval (steps) |
| `--ckpt-every` | 1000 | Checkpoint interval (steps) |
| `--log-every` | 50 | Logging interval (steps) |
| `--cache-limit` | 16 | LRU cache size for loaded stems (tracks) |
| `--num-workers` | 0 | DataLoader workers (0 = main thread) |
| `--resume` | *none* | Resume from checkpoint path |
| `--export` | *none* | Export checkpoint → model.bin (no training) |
| `--output` | `models/model_trained.bin` | Output path for `--export` |
| `--early-stop-patience` | 10 | Early stopping patience (validation checks) |

## Checkpointing & Resume

Checkpoints saved as `checkpoint_XXXXXX.pt` in `--save` directory. Best model saved as `best.pt`.

```bash
# Resume from last checkpoint
python python/training/train_clara.py \
  --data ... \
  --resume checkpoints_clara/checkpoint_005000.pt \
  --steps 100000
```

Interrupt with Ctrl+C — saves `interrupt_XXXXXX.pt` automatically.

## Export to C++ Engine Format

```bash
python python/training/train_clara.py \
  --export checkpoints_clara/best.pt \
  --output models/model.bin
```

Exports flat binary weights (Version 3 format, complex pole parameters) for the C++ inference engine.

## Known Issues & Fixes Applied (July 2026)

1. **`CheckpointError` under AMP** — `torch.checkpoint` + `torch.autocast` incompatible on ROCm. **Fix:** Disabled gradient checkpointing when `--amp` is on.

2. **Phase 3 OOM** — Other-stem embedding extraction leaked autograd graph (4× model passes tracked). **Fix:** Wrapped in `torch.no_grad()` + `.detach()`.

3. **Gradient checkpointing memory leak** — `use_reentrant=False` checkpoint leaks memory over steps (25 GB RSS in 10 steps on real data). **Fix:** Disabled checkpointing universally (model is only 81 MB peak).

4. **Import errors** — `train_mamba3.py` renamed to `train_clara.py`. `test_model.py` and `dump_reference_output.py` imports updated.

## File Layout

```
python/training/
  train_clara.py      # Canonical training pipeline (PerTrackProcessor, R=8)
  dataset.py          # StemDataset (MUSDB18-HQ, CREPE f0)
  loss.py             # TrackProcessorLoss (MR-STFT + Mel-L1 + SI-SDR)

python/tools/
  test_model.py       # 5-section smoke test (GPU/CPU)
  dump_reference_output.py  # Dump reference outputs for export validation
  extract_crepe.py    # CREPE f0 extraction (run once per dataset)
  validate_export.py  # Validate model.bin round-trip
  inspect_ckpt.py     # Inspect checkpoint contents
  export_mamba_weights.py  # Export checkpoint → model.bin

scripts/
  bootstrap_local.sh  # Local ROCm training bootstrap (setup + smoke + train)

checkpoints_clara/    # Default checkpoint directory
models/               # Exported model.bin files
```

## Architecture

```
PerTrackProcessor (~4.26M params)
├── STFT (n_fft=2048, hop=512)
├── R8FeatureExtractor → (B, T, 128)
├── SemanticEncoder → (B, T, 128)
├── F0Conditioner (CREPE) → (B, T, 32)
├── CrossTrackAggregator → (B, T, 64)
├── CondMLP → (B, T, 128)
├── Mamba3MIMOLayer × 6 (d_model=128, d_state=32, nheads=4, R=8)
│   └── Complex-pole trapezoidal SSM (no RoPE)
└── Decoder → complex STFT → ISTFT
```
