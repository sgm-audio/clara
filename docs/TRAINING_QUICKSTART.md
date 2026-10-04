# CLARA Training Quick Start — Morning Guide

## One Command Does Everything (local)

<!-- STATUS: in progress — this section used to document `scripts/setup_cloud.sh`,
     which is not in the repository. The flags below were re-checked against
     `python/training/train_clara.py`; the rest of this file was not. -->

```bash
bash scripts/setup_local.sh     # create venv_rocm + install ROCm PyTorch/deps (once)
bash scripts/run_test.sh        # smoke-test the training stack before a long run
bash scripts/run_train.sh [DATA_DIR]   # train; auto-resumes from checkpoints_clara/
```

`scripts/bootstrap_local.sh` chains dataset download, feature extraction, smoke
test and training on a Radeon APU (`--train`, `--smoke`, `--crepe-only`).

## What It Does

1. **system** — Install deps + ROCm PyTorch (`scripts/setup_local.sh`)
2. **datasets** — Download/verify MUSDB18-HQ
3. **precompute** — Extract CREPE/mel features (removes the CPU bottleneck)
4. **train** — Identity training, 100K steps by default
5. **export** — `python/training/train_clara.py --export <ckpt> --output models/model.bin`

## Local Setup (Ubuntu / ROCm)

For local training on an AMD APU (860M) or GPU:

```bash
# 1. Set up environment (once)
bash scripts/setup_local.sh

# 2. Verify the training stack BEFORE long runs
bash scripts/run_test.sh

# 3. Start training (auto-resumes from checkpoint if exists)
bash scripts/run_train.sh /path/to/musdb18hq/train
```

## Run Individual Phases

```bash
bash scripts/setup_cloud.sh system      # Install deps only
bash scripts/setup_cloud.sh datasets    # Download dataset only
bash scripts/setup_cloud.sh precompute  # Precompute features only
bash scripts/setup_cloud.sh train       # Train only (requires precomputed)
bash scripts/setup_cloud.sh export      # Export only
```

## Resume Training

If interrupted, resume from checkpoint:

```bash
python python/training/train_clara.py \
  --resume checkpoints_clara/checkpoint_050000.pt \
  --steps 100000 \
  --data ~/datasets/musdb18hq/train \
  --device cuda
```

## Monitor Training

Open a second terminal:

```bash
# GPU status
watch -n 2 rocm-smi --showtemp --showpower --showuse --showmemuse

# Training log
tail -f checkpoints_clara/train.log
```

## After Training

### Export Weights

```bash
python python/training/train_clara.py \
  --export checkpoints_clara/final.pt \
  --output models/model_trained.bin
```

### Verify C++ Engine

```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA not available` on AMD APU (860M) | `export HSA_OVERRIDE_GFX_VERSION=11.0.0` before Python import |
| `CUDA not available` on MI300X | Do **NOT** set `HSA_OVERRIDE_GFX_VERSION`; ROCm detects gfx942 natively |
| OOM (out of memory) | Reduce `--batch 2` or `--clip-seconds 0.25` |
| Slow training | Add `--cache` (uses ~20GB RAM) |
| NaN loss | Training is float32 only (no `--amp` on 860M) |
| Dataset not found | Copy from external drive (see One-Time Setup C) |

## Key Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | required (except `--export`) | Path to MUSDB18-HQ `train/` |
| `--save` | `./checkpoints` | Checkpoint directory |
| `--resume` | none | Resume from a checkpoint path |
| `--steps` | 100000 | Total training steps |
| `--batch` | 4 | Batch size (reduce if OOM) |
| `--grad-accum` | 1 | Gradient accumulation (`effective_batch = batch × accum`) |
| `--lr` | 3e-4 | Learning rate |
| `--clip-seconds` | 4.0 | Audio clip duration in seconds |
| `--amp` | off | bf16 AMP (fp32 SSM core stays fp32) |
| `--precomputed` | none | Deprecated — `StemDataset` loads `.crepe.npy` automatically |
| `--cache-limit` | 16 | Dataset cache limit (there is no `--cache` flag) |
| `--val-every` / `--ckpt-every` / `--log-every` | 2000 / 1000 / 50 | Reporting + checkpoint cadence |
| `--early-stop-patience` | 10 | Validation checks without improvement before stopping |

## Success Criteria

- [ ] Loss decreases: ~0.6 → ~0.3 in first 1000 steps
- [ ] No NaN occurrences in log
- [ ] Validation loss monotonically decreasing
- [ ] Export produces valid model.bin
- [ ] C++ tests pass with trained weights
