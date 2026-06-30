# CLARA Training Setup — Agent Context File

**For any agent tasked with: running training, fixing ROCm/PyTorch issues, debugging
training pipeline, or continuing from an existing setup on this machine.**

---

## Machine & OS

- **OS:** Bazzite 44 (atomic Fedora-based, rpm-ostree)
- **GPU:** AMD Ryzen AI 7 350 / Radeon 840M/860M (integrated APU)
  - PCI ID: `1002:1114` (gfx1152, RDNA3.5)
- **Distrobox:** `clara-training` (Ubuntu 24.04 container)
- **VRAM:** ~16GB UMA (shared with system RAM)
- **Host:** Bazzite (immutable, apt/dnf unreliable for system packages)

**ROCm:** Installed on host via rpm-ostree layered packages:
`rocm-hip`, `rocm-opencl`, `rocm-runtime`, `rocm-clinfo`, `rocm-smi` (ROCm 7.1.1)

---

## ROCm Environment Variables (ALL 6 REQUIRED for gfx1152)

```
HSA_OVERRIDE_GFX_VERSION=11.0.0   # MUST spoof gfx1152 → gfx1100 (FL worst-case)
PYTORCH_ROCM_ARCH=gfx1100            # compile for gfx1100 target
HIP_VISIBLE_DEVICES=0               # enumerate GPU 0 (CRITICAL — without this torch.cuda shows 0 devices)
ROC_VISIBLE_DEVICES=0              # AMD ROCm device visibility (CRITICAL)
HSA_ENABLE_SDMA=0                   # DISABLE SDMA — APU shared RAM, SDMA causes OOM
PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb=256
MIOPEN_FIND_MODE=5                  # exhaustive MIOpen kernel search, 1-2s init cost
```

**ALL 6 vars are required.** `HSA_OVERRIDE_GFX_VERSION` alone is NOT sufficient —
`HIP_VISIBLE_DEVICES` and `ROC_VISIBLE_DEVICES` must also be set for PyTorch to enumerate GPU 0.

**Where these go:**
- `train_mamba3.py` — hardcoded at top via `os.environ.setdefault()`
- `scripts/bootstrap_local.sh` — `export` before any Python call
- Any manual `python` / `torchrun` invocation

**Do NOT use `gfx1103` or `11.0.3`** — user confirmed gfx1100 works, gfx1103 caused issues.

---

## Python Environment

### The Problem (Bazzite specific)
PyTorch ROCm wheels (`download.pytorch.org/whl/rocm6.2`) only ship for Python 3.8–3.12.
System has Python 3.14 (Bazzite host). Solution: use distrobox with its own Python.

### Correct Setup
```bash
# Inside distrobox clara-training
eval "$(micromamba shell hook --shell bash)"  # init micromamba shell hook
micromamba activate rocm_torch                # activate env (Python 3.12)

# Install ROCm PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Verify — MUST show True
HSA_OVERRIDE_GFX_VERSION=11.0.0 HIP_VISIBLE_DEVICES=0 ROC_VISIBLE_DEVICES=0 \
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# → 2.5.1+rocm6.2  True
```

**GPU torch is already installed** in `~/micromamba/envs/rocm_torch/`. Just activate and verify.

**If `pip install` for ROCm torch fails:** Build from source:
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git submodule update --init --recursive
USE_ROCM=1 ROCM_HOME=/opt/rocm HIP_CLANG_PATH=$(which clang) \
  python3.12 setup.py install
```
(~1-2hr build, most reliable)

---

## Project Structure

```
clara/                          # Main repo (this machine: /var/home/sgm/Desktop/Dev/clara)
├── train_mamba3.py       # PRIMARY training script (PerTrackProcessor R=8)
├── loss.py         # Loss stack: MR-STFT + Mel-L1 + SI-SDR
├── dataset.py      # StemDataset + CachedMelDataset
├── scripts/
│   ├── extract_crepe.py        # CREPE pitch extraction (port from mamba3 drive)
│   ├── bootstrap_local.sh      # Full setup + training launcher
│   ├── test_model.py           # Smoke test (5 sections)
│   ├── precompute_features.py  # Precompute STFT/mel chunks
│   ├── validate_export.py       # Weight format validator
│   └── dump_reference_output.py # PerTrackProcessor reference output
├── src/                        # C++ audio plugin (separate from training)
├── models/                     # Binary weights (*.bin)
└── venv_rocm/                  # OLD venv (broken pip, ignore)

/home/sgm/micromamba/envs/rocm_torch/  # CORRECT Python env for training
```

---

## Model Architecture: PerTrackProcessor (R=8)

**Mamba-3 MIMO R=8 spectral-domain — NOT the old R=4 time-domain model.**

```
PerTrackProcessor (R=8 spectral)
├── R8FeatureExtractor          # 8 soft triangular band energies + transient + centroid
├── SemanticEncoder             # conv stem → 2× Mamba3MIMOLayer → projection
├── F0Conditioner                # CREPE 360-D → d_f0=32
├── CrossTrackAggregator         # Causal MHA over K other stems
├── CondMLP                     # [d_sem+d_f0+d_cross+4] → d_cond=128
├── mamba_layers[6]             # 6× Mamba3MIMOLayer (gradient checkpointed)
│   └── PerBandSSM: R=8, d_model=128, d_state=32, nheads=4
└── Decoder                     # d_model → 2×F_bins (complex STFT) → ISTFT → waveform

Key dims: d_model=128, d_state=32, nheads=4, n_layers=6, R=8, d_f0=32, d_cond=128
Total params: ~4.26M
in_proj_dim: 1552
```

**Incompatible with C++ engine (R=4 time-domain)** — separate tracks, do not mix.

---

## Training Script: train_mamba3.py

### CLI Flags (all verified present)
```
--data PATH                     # MUSDB18-HQ train/ root
--save DIR                      # checkpoint directory
--resume PATH                   # resume from checkpoint
--steps N                       # total steps (default 100000)
--batch N                       # per-device batch (default 4)
--grad-accum N                  # gradient accumulation steps (default 1)
--lr FLOAT                      # learning rate (default 3e-4)
--weight-decay FLOAT            # (default 1e-4)
--clip-grad-norm FLOAT          # grad clip (default 1.0)
--warmup N                      # LR warmup steps (default 2000)
--clip-seconds FLOAT            # audio clip duration (default 4.0)
--amp                           # enable bf16 AMP (fp32 SSM core)
--num-workers N                 # DataLoader workers (default 0)
--cache-limit N                 # stem RAM cache (default 16 tracks)
--precomputed PATH              # DEPRECATED passthrough (StemDataset loads .crepe.npy auto)
--val-split FRAC                # train/val split fraction (default 0.1)
--val-every N                   # validate every N steps (default 2000)
--ckpt-every N                  # checkpoint every N steps (default 1000)
--log-every N                   # log every N steps (default 50)
--early-stop-patience N         # early stop (default 10)
--device cuda|cpu               # (default cuda)
--export PATH                   # export checkpoint to model.bin
--output PATH                   # output path for export (default models/model_trained.bin)
```

### 3-Phase Training Schedule
```
Phase 1 (0-20% steps):  Contrastive pretraining — NT-Xent on augmented pairs
Phase 2 (20-50% steps): Single-track reconstruction (no cross-track)
Phase 3 (50-100% steps): Cross-track conditioned reconstruction
```

### Loss Stack (per phase)
```
Phase 1:        NT-Xent contrastive loss only
Steps 0-5k:    MR-STFT + Mel-L1
Steps 5k-15k:   MR-STFT + Mel-L1 + SI-SDR (SI-SDR ramps in 0→0.1)
Step 15k+:     1.0×MR-STFT + 0.5×Mel-L1 + SI-SDR (full weight)
```

### Gradient Accumulation Implementation
```python
accum = args.grad_accum
loss_scaled = loss / accum        # scale BEFORE backward
loss_scaled.backward()            # accumulates in .grad
if (step + 1) % accum == 0:       # optimizer.step() every N batches
    torch.nn.utils.clip_grad_norm_(...)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
```
**Effective batch = `--batch` × `--grad-accum`**
Conservative for 16GB UMA: `batch=2, accum=4` → effective=8

---

## Dataset

### MUSDB18-HQ Location
`~/datasets/musdb18hq/train` (symlinked or actual)

### StemDataset (used by train_mamba3.py)
- Discovers tracks: `root/TrackName/{bass,drums,vocals,other,...}.wav`
- Loads `.crepe.npy` + `.pitched.npy` if present, else **zeros (graceful degradation)**
- On-the-fly STFT/mel computation (CPU, no GPU bottleneck for small batch)
- Augmentations: gain ±6dB, polarity flip, pitch shift ±2 semitones, time shift
- Bounded FIFO track cache (`cache_limit=16`)

**CREPE pre-extraction is OPTIONAL** — StemDataset degrades to zeros for f0 if missing.
`python/tools/extract_crepe.py` extracts it offline (~30-60min for full dataset).

### CREPE Extraction (optional but recommended)
```bash
python python/tools/extract_crepe.py --data ~/datasets/musdb18hq/train --device cuda --model full
# --model tiny for faster (lower accuracy)
```
Saves alongside each WAV:
- `<stem>.crepe.npy` — (T_model, 360) float16, 100Hz→86Hz resampled
- `<stem>.pitched.npy` — float32, 1.0=pitched, 0.0=unpitched (drums)

### CachedMelDataset (alternative, not used by default)
- Uses precomputed `.npz` chunks from `python/tools/precompute_features.py`
- Faster DataLoader, different from StemDataset
- **NOT used by `train_mamba3.py`** — only `StemDataset`

---

## Bootstrap Script: scripts/bootstrap_local.sh

Full workflow:
```bash
bash scripts/bootstrap_local.sh           # full: setup → crepe → precompute → smoke → train
bash scripts/bootstrap_local.sh --train   # skip crepe/precompute, go straight to train
bash scripts/bootstrap_local.sh --smoke   # smoke test only
bash scripts/bootstrap_local.sh --crepe-only  # extract CREPE only
```

Sets ALL ROCm env vars, creates venv if needed, installs deps, runs smoke test,
then launches training with conservative defaults:
```bash
python3 train_mamba3.py \
  --data ~/datasets/musdb18hq/train \
  --save ./checkpoints_clara \
  --device cuda \
  --batch 2 \
  --grad-accum 4 \
  --steps 100000 \
  --val-every 2000 \
  --ckpt-every 1000 \
  --log-every 50 \
  --warmup 2000 \
  --early-stop-patience 10 \
  --num-workers 0 \
  --cache-limit 16 \
  --precomputed ~/datasets/musdb18hq/precomputed   # if available
```

---

## Smoke Test

```bash
cd /var/home/sgm/Desktop/Dev/clara
python3 python/tools/test_model.py --device cpu    # CPU smoke
python3 python/tools/test_model.py --device cuda    # GPU smoke (after ROCm torch)
```

Expected output: **5/5 sections pass**
- [0] ROCm/HSA env — ✓ (GPU mode only checks `torch.cuda.is_available()`)
- [1] Model build — ✓ (4.26M params, <30s build)
- [2] Forward pass — ✓ (finite output, correct shape)
- [3] bf16 AMP — ✓ (GPU mode)
- [4] Forward+backward — ✓ (finite loss, nonzero grads)

---

## VRAM / Memory Optimization (16GB UMA)

1. **Gradient checkpointing** — ON for all `mamba_layers` (trades compute for memory)
2. **`batch=2`** — conservative single-track batch for 16GB
3. **Effective batch via accumulation** — `batch=2 × accum=4 = 8`
4. **`HSA_ENABLE_SDMA=0`** — prevents SDMA pinning conflicts on APU shared RAM
5. **`PYTORCH_ALLOC_CONF=max_split_size_mb=256`** — reduces fragmentation in UMA buffer
6. **`MIOPEN_FIND_MODE=5`** — exhaustive kernel search, 1-2s init, 10-30% faster convolutions
7. **`num_workers=0`** — DataLoader workers disabled (APU, avoid IPC overhead)
8. **`cache_limit=16`** — stem RAM cache (prevents repeated disk I/O)

---

## Weight Export Format (Version 3)

`train_mamba3.py --export checkpoints/best.pt --output models/model.bin`

Version 3 (complex pole format, R=8):
```
Header: 64 bytes — magic, version, dims, weight_data_size
R8 extractor weights: shared frontend
Per-layer Mamba weights (18 arrays/layer):
  [0] in_proj.weight    [in_proj_dim, d_model]
  [1] dt_bias           [nheads]
  [2-3] B_proj          [N*2, R], [N*2]  (R=8)
  [4-5] C_proj          [head_dim, N*2], [head_dim]
  [6-8] mimo_x/z/o     [nheads, R, headdim]
  [9] D_proj            [nheads]
  [10-11] out_proj     [d_model, d_inner], bias
  [12-13] film_proj    [2*d_model, cond_dim], bias
  [14-15] dt_film_proj [2*nheads, cond_dim], bias
  [16] log_a_real      [nheads]   ← complex pole, v3 new
  [17] a_imag          [nheads]   ← complex pole, v3 new
  [18-19] norm         [d_model], [d_model]
```

**C++ engine is R=4 — does NOT support v3 weights.** H3 (C++ R=8 update) is separate work.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `python/training/train_mamba3.py` | Main training script (PerTrackProcessor R=8) |
| `python/training/loss.py` | MR-STFT + Mel-L1 + SI-SDR + NT-Xent losses |
| `python/training/dataset.py` | StemDataset, CachedMelDataset, make_dataloaders |
| `scripts/bootstrap_local.sh` | Full setup + training launcher |
| `python/tools/extract_crepe.py` | CREPE pitch extraction |
| `python/tools/test_model.py` | Smoke test |
| `python/tools/precompute_features.py` | STFT/mel chunk precomputation |
| `python/tools/validate_export.py` | Binary weight format validator |
| `src/core/SsmInferenceCore.cpp` | C++ Mamba-3 engine (R=4, time-domain) — INCOMPATIBLE with PerTrackProcessor |
| `TODO.md` | Current task list |
| `MAMBA3_MIGRATION_PLAN.md` | Full architecture and migration plan |
| `SESSION_LOG.md` | Session-by-session history (append-only) |

---

## Reference Implementations (mamba3 drive)

`/run/media/sgm/01DCA22D5C1C2A50/mamba3/` (external drive):
- `model.py` — PerTrackProcessor reference (R=8)
- `train.py` — canonical training loop (no grad accum, batch=4)
- `dataset.py` — StemDataset inline in train.py
- `extract_crepe.py` — CREPE extraction (ported to `python/tools/extract_crepe.py`)
- `run_train.sh` — native training runner with gfx1103 spoof

---

## Known Issues & Solutions

| Issue | Cause | Fix |
|-------|-------|-----|
| `torch.cuda.is_available() = False` with ROCm torch | Missing `HIP_VISIBLE_DEVICES=0` | Set `HIP_VISIBLE_DEVICES=0` AND `ROC_VISIBLE_DEVICES=0` + `HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| `torch.cuda.is_available() = False` | Wrong PyTorch build (NVIDIA CUDA) | Install ROCm build in distrobox: `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` |
| ROCm wheel not found for Python 3.14 | PyTorch ROCm only ships py3.8-3.12 | Use distrobox with Python 3.12, or build from source |
| `gfx1152 not supported` | ROCm doesn't know gfx1152 | `HSA_OVERRIDE_GFX_VERSION=11.0.0` (gfx1100) |
| `hipRuntimeError: out of memory` | APU shared RAM + SDMA conflict | `HSA_ENABLE_SDMA=0` |
| Training NaN/Inf loss | Numeric instability | `--amp` (bf16 autocast, fp32 SSM) |
| Slow DataLoader | CPU STFT bottleneck | Use `--precomputed` + `python/tools/precompute_features.py` |
| C++ engine crash on load | Weight format mismatch | C++ is R=4, weights are R=8 v3 — H3 update needed |

---

## Complete Minimal Training Command

```bash
cd /var/home/sgm/Desktop/Dev/clara

# Activate correct environment (distrobox clara-training)
eval "$(micromamba shell hook --shell bash)"
micromamba activate rocm_torch

# Verify ROCm torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Must be: 2.7.1+rocm6.2.4  True

# Quick smoke test
HSA_OVERRIDE_GFX_VERSION=11.0.0 HIP_VISIBLE_DEVICES=0 ROC_VISIBLE_DEVICES=0 \
  HSA_ENABLE_SDMA=0 \
  PYTHONPATH=. python3 python/tools/test_model.py --device cuda
# Expected: 5/5 passed, VRAM ~35 MB

# Full training
HSA_OVERRIDE_GFX_VERSION=11.0.0 HIP_VISIBLE_DEVICES=0 ROC_VISIBLE_DEVICES=0 \
  HSA_ENABLE_SDMA=0 PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb=256 \
  MIOPEN_FIND_MODE=5 \
  python3 train_mamba3.py \
  --data ~/datasets/musdb18hq/train \
  --save ./checkpoints_clara \
  --device cuda \
  --batch 2 \
  --grad-accum 4 \
  --steps 100000 \
  --amp \
  --val-every 2000 \
  --ckpt-every 1000 \
  --log-every 50 \
  --warmup 2000 \
  --early-stop-patience 10 \
  --num-workers 0 \
  --cache-limit 16
```
