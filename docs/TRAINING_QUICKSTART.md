# CLARA Training Quick Start — Morning Guide

## One Command Does Everything (Cloud)

```bash
cd ~/Desktop/Dev/clara
bash scripts/setup_cloud.sh all
```

That's it. Downloads MUSDB18-HQ (21GB), extracts it, precomputes features, runs training. 

## What It Does

1. **system** — Install deps + ROCm PyTorch
2. **repo** — Clone/pull CLARA repo
3. **datasets** — Download MUSDB18-HQ via aria2c
4. **precompute** — Precompute mel + STFT features (eliminates CPU bottleneck)
5. **train** — Stage 0 identity training (100K steps, --precomputed --cache)
6. **export** — Export + validate model.bin

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
python train_mamba3.py \
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
python train_mamba3.py \
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
| `--data` | required | Path to MUSDB18-HQ `train/` directory |
| `--steps` | 100000 | Total training steps |
| `--batch` | 4 | Batch size (reduce if OOM) |
| `--grad-accum` | 4 | Gradient accumulation steps |
| `--clip-seconds` | 0.5 | Audio clip duration (0.5=faster, 2.0=better) |
| `--lr` | 1e-3 | Learning rate |
| `--cache` | off | RAM pre-cache dataset (~20GB) |
| `--resume` | None | Resume from checkpoint path |

## Success Criteria

- [ ] Loss decreases: ~0.6 → ~0.3 in first 1000 steps
- [ ] No NaN occurrences in log
- [ ] Validation loss monotonically decreasing
- [ ] Export produces valid model.bin
- [ ] C++ tests pass with trained weights
