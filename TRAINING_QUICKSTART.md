# CLARA Training Quick Start — Morning Guide

## Prerequisites Check

Run these to verify everything is ready:

```bash
# 1. Check distrobox exists
distrobox list | grep clara-training

# 2. Check dataset (should be ~41GB)
ls -lh ~/datasets/musdb18hq/train/ 2>/dev/null || echo "DATASET MISSING"

# 3. Check GPU
rocm-smi --showuse 2>/dev/null || echo "No GPU (will use CPU)"
```

## One-Time Setup (Skip if Already Done)

### A. Start Training Environment

```bash
distrobox enter clara-training
```

### B. Install Dependencies (Inside Distrobox)

```bash
sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
python3 -m venv ~/clara-training/venv
source ~/clara-training/venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
pip install numpy scipy soundfile tqdm
```

### C. Copy Dataset to NVMe

```bash
# From host (outside distrobox)
cp -a /run/media/sgm/Linux\ Backup/datasets/musdb18hq/. ~/datasets/musdb18hq/
```

## Training Commands

### Option 1: Quick Smoke Test (5 min)

Tests the entire pipeline with synthetic data:

```bash
distrobox enter clara-training
source ~/clara-training/venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
cd ~/Github\ Repo\'s/clara
bash scripts/smoke_test.sh
```

### Option 2: Full Training (4-8 hours)

```bash
distrobox enter clara-training
source ~/clara-training/venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HSA_ENABLE_SDMA=0
export PYTORCH_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:256"
export MIOPEN_FIND_MODE=5
cd ~/Github\ Repo\'s/clara

# Run training
python train_clara_mamba3.py \
  --data ~/datasets/musdb18hq/train \
  --device cuda \
  --batch 4 \
  --grad-accum 4 \
  --steps 100000 \
  --clip-seconds 0.5 \
  --ckpt-every 5000 \
  --val-every 1000 \
  --log-every 100 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --clip-grad-norm 1.0 \
  --warmup 2000 \
  --cache
```

### Option 3: Use the Script (Easier)

```bash
distrobox enter clara-training
source ~/clara-training/venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
cd ~/Github\ Repo\'s/clara
bash scripts/run_train.sh
```

## Resume Training

If interrupted, resume from checkpoint:

```bash
python train_clara_mamba3.py \
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
python train_clara_mamba3.py \
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
| `CUDA not available` | `export HSA_OVERRIDE_GFX_VERSION=11.0.0` |
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
