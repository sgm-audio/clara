# CLARA Training V2 — MambaRTModelParallel (76K Parameter Exportable Model)

> **⚠️ WHY V2?** The old pipeline (`train_clara_now.py`) trains `STFTMambaModel` which is **incompatible** with the ONNX export the C++ engine needs. Only 2 of 50 model keys matched `MambaRTModelSequential`. V2 trains `MambaRTModelParallel` — the model that can actually be exported to `mamba_rt.onnx` via `train_clara_export.py`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       TRAINING PIPELINE V2                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MUSDB18-HQ WAVs  ──→  MambaRTModelParallel  ──→  Checkpoint .pt   │
│  (86 tracks × 5      (76K params, time-       (P1: identity step)  │
│   stems, 44.1kHz)     domain, parallel scan)                        │
│                                                                     │
│                              │                                      │
│                              ▼                                      │
│                                                                     │
│  Checkpoint .pt  ──→  MambaRTModelSequential  ──→  mamba_rt.onnx   │
│  (train_clara_       (step-by-step export      (drop into models/   │
│   export.py)          model, match ONNX graph)   for C++ engine)    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Environment Setup (One Time)

**Prerequisites:** Bazzite with ROCm, `distrobox`, `podman`.

```bash
# 1. Create the training distrobox (already done — skip if exists)
distrobox create --name clara-training --image ubuntu:24.04 \
  --additional-flags "--security-opt seccomp=unconfined"

# 2. Enter and install
distrobox enter clara-training
sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
python3 -m venv ~/clara-training/venv
source ~/clara-training/venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
pip install numpy scipy soundfile onnx onnxruntime tqdm

# 3. Verify GPU
export HSA_OVERRIDE_GFX_VERSION=11.0.0
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: AMD Radeon 860M Graphics
```

## Training Commands

### 0. One-time: Copy Dataset to NVMe

> The MUSDB18-HQ dataset lives on an external drive at `/run/media/sgm/Linux Backup/datasets/musdb18hq/`. Copy to NVMe for training speed (~5 min).

```bash
cp -a /run/media/sgm/Linux\ Backup/datasets/musdb18hq/. ~/datasets/musdb18hq/
```

Expect: 41 GB, 44 tracks in `musdb18hq/train/` (each with bass/drums/vocals/other/mixture.wav).

### 1. Quick Smoke Test (5 minutes)

Verifies the entire pipeline end-to-end with synthetic data:

```bash
distrobox enter clara-training
source ~/clara-training/venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
cd ~/Github\ Repo\'s/clara
bash scripts/smoke_test.sh
```

**What it tests:**
- Creates 3 synthetic WAV files (white noise, 5 sec each)
- Trains `MambaRTModelParallel` for 50 steps
- Exports to ONNX
- Loads ONNX in onnxruntime, verifies output shapes
- Confirms output is non-zero (weights learned something)

**Expected output:** All steps PASS, final ONNX model at `/tmp/clara_smoke_test/mamba_rt.onnx`.

### 1b. Quick Launch

```bash
distrobox enter clara-training
export HSA_OVERRIDE_GFX_VERSION=11.0.0
~/Github\ Repo\'s/clara/scripts/run_train.sh
```

For GPU monitoring, open a second terminal:
```bash
watch -n 2 rocm-smi --showtemp --showpower --showuse --showmemuse
```

> See **`Train_commands.mmd`** for the comprehensive command reference, including MI300X configs, all training flags, and troubleshooting.

### 2. Phase 1 Identity Training (Estimated ~4-8 hours for 50K steps)

Trains the 76K-param model to reconstruct its input (identity autoencoder).

**Basic command:**
```bash
distrobox enter clara-training
source ~/clara-training/venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HSA_ENABLE_SDMA=0
export PYTORCH_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:256"
export MIOPEN_FIND_MODE=5
cd ~/Github\ Repo\'s/clara
python train_clara_parallel.py \
  --data ~/datasets/musdb18hq/train \
  --device cuda \
  --batch 4 \
  --grad-accum 4 \
  --steps 50000 \
  --clip-seconds 0.5 \
  --ckpt-every 500 \
  --val-every 500 \
  --log-every 25
```

**With RAM cache (faster but uses ~20 GB):**
```bash
python train_clara_parallel.py \
  --data ~/datasets/musdb18hq/train \
  --device cuda \
  --batch 4 \
  --grad-accum 4 \
  --steps 50000 \
  --clip-seconds 0.5 \
  --cache
```

### 3. Resume from Checkpoint

The training loop auto-saves every `--ckpt-every` steps. To resume:

```bash
python train_clara_parallel.py \
  --data ~/datasets/musdb18hq/train \
  --resume checkpoints_clara/checkpoint_025000.pt \
  --resume-optimizer \
  --device cuda
```

### 4. Export to ONNX

After training completes (`checkpoints_clara/final.pt`), export:

```bash
source ~/clara-training/venv/bin/activate
cd ~/Github\ Repo\'s/clara
python train_clara_export.py \
  --checkpoint checkpoints_clara/final.pt \
  --output models/mamba_rt.onnx \
  --seq-len 64
```

### 5. Verify in C++ Engine

```bash
cd ~/Github\ Repo\'s/clara
cmake -B build -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

**Or manually test the ONNX model against the C++ contract:**
```bash
python scripts/verify_onnx_contract.py
```

## Parameter Reference

### Training Flags (`train_clara_parallel.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | **required** | Path to MUSDB18-HQ `train/` directory |
| `--steps` | 50000 | Total training steps |
| `--batch` | 4 | Batch size (reduce to 2 if OOM) |
| `--grad-accum` | 4 | Gradient accumulation steps |
| `--clip-seconds` | 0.5 | Audio clip duration (0.5=faster, 2.0=better quality) |
| `--lr` | 3e-4 | Max learning rate (cosine warmup) |
| `--cache` | off | RAM pre-cache dataset (~20 GB) |
| `--resume` | None | Resume from checkpoint path |
| `--resume-optimizer` | off | Also restore optimizer state |
| `--num-workers` | 0 | DataLoader threads (0=safe for ROCm/APU, 2-4=faster) |
| `--ckpt-every` | 500 | Save checkpoint every N steps |
| `--val-every` | 500 | Validate every N steps |
| `--log-every` | 25 | Log every N steps |

### Model Architecture (fixed for ONNX compatibility)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `d_model` | 64 | Embedding dimension |
| `d_state` | 16 | SSM state per channel |
| `n_layers` | 4 | Stacked SSM layers |
| `conv_width` | 4 | Causal conv kernel (latency = 4 samples) |
| `dt_rank` | 16 | Discretization rank |
| `cond_dim` | 8 | Conditioning dimension (zeros for Phase 1) |
| Total params | **76,865** | Matches ONNX export model exactly |

## Known Constraints

1. **No `--amp` on AMD 860M** — bfloat16 produces NaN at step ~8 on ROCm. All training is float32.
2. **3.5 step/s typical** — Chunked SSM scan on 860M iGPU. 50K steps ≈ 4 hours @ 0.5s clips.
3. **Phase 1 is identity only** — model learns to pass audio through. No conditioning, no denoising.
4. **CPU training available** — Set `--device cpu`. Much slower but deterministic.
5. **State reset per batch** — Conv/SSM states are zeroed at the start of each training clip. This matches inference where states persist across blocks.

## Monitor Training

```bash
# GPU temperature, power, utilization
watch -n 2 'rocm-smi --showtemp --showpower --showuse --showmemuse'

# Tail training log
tail -f checkpoints_clara/train.log
```

## Success Criteria

| Check | How | Target |
|-------|-----|--------|
| Loss decreases | First 1000 steps | ~0.6 → ~0.3 |
| No NaN | Scan log for "NaN" | Zero occurrences |
| Validation improves | `val_loss` trend | Monotonically down |
| ONNX export passes | `train_clara_export.py` | No errors |
| ONNX runs in C++ | `ctest --output-on-failure` | All tests pass |
| Non-zero output | Model produces variance > 1e-6 | Trained ≠ random |
