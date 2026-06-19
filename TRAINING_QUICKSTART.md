# CLARA Training Quick Start — Morning Guide

## One Command Does Everything

```bash
cd ~/Desktop/Dev/clara
bash scripts/train_local.sh
```

That's it. Downloads MUSDB18-HQ (21GB), extracts it, runs training. Add flags to customize:

```bash
bash scripts/train_local.sh --steps 50000 --batch 2
```

## Manual Steps (If You Prefer)

### Prerequisites

```bash
# Check distrobox exists
distrobox list | grep clara-training

# Check GPU
rocm-smi --showuse 2>/dev/null || echo "No GPU (will use CPU)"
```

### One-Time Setup (Inside Distrobox)

```bash
distrobox enter clara-training
sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip aria2
python3 -m venv ~/clara-training/venv
source ~/clara-training/venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
pip install numpy scipy soundfile tqdm
```

### Download Dataset

```bash
mkdir -p ~/datasets
cd ~/datasets
aria2c -x 4 -s 4 --continue=true \
    "https://zenodo.org/record/5061438/files/musdb18hq.tar.gz" \
    -o musdb18hq.tar.gz
tar -xzf musdb18hq.tar.gz
```

### Run Training

```bash
cd ~/Desktop/Dev/clara
export HSA_OVERRIDE_GFX_VERSION=11.0.0
python train_clara_mamba3.py \
  --data ~/datasets/musdb18hq/train \
  --device cuda \
  --batch 4 \
  --steps 100000 \
  --cache
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
