# CLARA Training Commands

**Verified 2026-06-05** — 50-step run on AMD Radeon 860M ROCm, zero NaN, ~3.5 step/s.

## Quick Start

```powershell
$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
.\venv\Scripts\python.exe train_clara_now.py --data "C:\dev\mamba3\mamba3\datasets\musdb18hq\train"
```

AMP is **off by default** (`bfloat16` triggers NaN at step 8 on 860M ROCm). Enable with `--amp` if running on A100/4090.

**Verified output:**
```
AMP: False (torch.float32)
GPU warmup complete.
[step      0] loss=0.6031 stft=0.2869 mel=0.6322  gpu_mem=0.09GB
[step     10] loss=0.6617 2.892 step/s
[step     50] Training complete. Best val loss: inf
```

## All Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | **required** | Path to WAV dataset dir |
| `--save` | `./checkpoints_clara` | Checkpoint dir |
| `--resume` | None | Resume from `.pt` checkpoint |
| `--steps` | 50000 | Total training steps |
| `--batch` | 2 | Batch size |
| `--lr` | 3e-4 | Learning rate (cosine warmup 2000 steps) |
| `--device` | auto | `auto`, `cuda`, or `cpu` |
| `--grad-accum` | 8 | Gradient accumulation (eff batch = batch × grad_accum) |
| `--val-split` | 0.1 | Fraction of files held out for val |
| `--val-every` | 1000 | Validate every N steps |
| `--ckpt-every` | 200 | Save checkpoint every N steps |
| `--log-every` | 25 | Log every N steps |
| `--early-stop-patience` | 20 | Early stop after N val checks without improvement |
| `--num-workers` | 2 | DataLoader workers (forced 0 on Windows — multiprocessing unstable) |
| `--amp` | off | Enable bfloat16 AMP (NaN on 860M ROCm, safe on A100/4090) |
| `--compile` | off | Enable `torch.compile` |
| `--profile STEPS` | off | Capture torch profiler Chrome trace for N steps |
| `--fresh-best-val` | off | Reset best val loss on resume |
| `--cache` | off | RAM pre-cache dataset (~20 GB for MUSDB18-HQ 500 files) |
| `--clip-seconds` | 2.0 | Audio clip duration in seconds |

## Examples

**Full identity training with RAM cache (recommended):**
```powershell
$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
.\venv\Scripts\python.exe train_clara_now.py --data "C:\dev\mamba3\mamba3\datasets\musdb18hq\train" --cache
```

**Smoke test (50 steps, short clips):**
```powershell
.\venv\Scripts\python.exe train_clara_now.py --data "C:\dev\mamba3\mamba3\datasets\musdb18hq\train" --steps 50 --clip-seconds 0.5 --batch 2 --log-every 10
```

**Resume from checkpoint:**
```powershell
.\venv\Scripts\python.exe train_clara_now.py --data "C:\dev\mamba3\mamba3\datasets\musdb18hq\train" --resume checkpoints_clara/checkpoint_50000.pt
```

**Profile training (Chrome trace for perf debugging):**
```powershell
.\venv\Scripts\python.exe train_clara_now.py --data "C:\dev\mamba3\mamba3\datasets\musdb18hq\train" --steps 100 --profile 10
# Trace → checkpoints_clara/profile_trace/
```

**CPU training:**
```powershell
.\venv\Scripts\python.exe train_clara_now.py --data "C:\dev\mamba3\mamba3\datasets\musdb18hq\train" --device cpu
```

## Architecture Changes (vs mamba3)

### SSM Scan

The **log-cumsum parallel scan** (Heinsen) is numerically unstable for real-valued SSM — `exp(log_cumsum) * cumsum(b * exp(-log_cumsum))` creates `0 × inf = NaN` on long sequences because cumulative product of `dA < 1` underflows.

**Fix:** Chunked parallel scan — processes sequence in chunks of 64, does serial scan within each chunk, carries state between chunks. Matches the ONNX export path behavior. Numerically stable, ~3-3.5 step/s.

### Full diff from mamba3 `train_laptop.py`

| Feature | mamba3 | CLARA `train_clara_now.py` |
|---------|--------|---------------------------|
| Params | 785K | 171K |
| SSM poles | Complex-valued | Real-valued |
| Discretization | Bilinear (trapezoidal) | Euler |
| SSM scan | Vectorized parallel | Chunked parallel (C=64) |
| STFT | GPU | GPU (fixed — was CPU) |
| cuDNN benchmark | Yes | Yes |
| GPU warmup | Forward+backward | Forward+backward |
| Parameter groups | decay + nodecay | decay + nodecay |
| Loss | MR-STFT + Mel-L1 + SI-SDR | STFT + Mel-L1 + SI-SDR (lean) |
| F0 conditioning | CREPE | None (identity task) |
| Cross-track attention | Yes | None (identity task) |
| `--profile` | Yes | Yes |

## Environment

**Canonical Python:** `.\venv\Scripts\python.exe` (PyTorch 2.11.0+rocm7.13.0)

**Mandatory:**
```powershell
$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
```

**Full ROCm env:**
```powershell
$env:HSA_ENABLE_SDMA = "0"
$env:PYTORCH_ALLOC_CONF = "garbage_collection_threshold:0.8,max_split_size_mb:256"
$env:MIOPEN_LOG_LEVEL = "3"
$env:MIOPEN_FIND_MODE = "2"
$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
```

**If venv breaks** (`did not find executable at 'C:\ProgramData\miniforge3\python.exe'`):
```powershell
# Fix pyvenv.cfg → home = C:\Program Files\Python313
robocopy "C:\dev\mamba3\mamba3\venv\Lib\site-packages" "venv\Lib\site-packages" /E /COPY:DAT /SL
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| NaN at step 0 | Unstable parallel scan (log-cumsum underflow) | Fixed — now uses chunked scan |
| NaN at step 8+ | bfloat16 AMP on 860M ROCm | Don't use `--amp` on this hardware |
| HIP error | Missing `HSA_OVERRIDE_GFX_VERSION` | Set env var before Python |
| No CUDA | Wrong Python | Use `.\venv\Scripts\python.exe` |
| Hung at warmup | DataLoader deadlock | `num_workers=0` (auto-forced) |
| Slow (~2.5-3.5 step/s) | Chunked serial scan + 171K params | Expected on 860M iGPU |

## Paths

| Resource | Path |
|----------|------|
| Project venv | `C:\Dev\Audio Development\0 CLARA\venv\` |
| Ref venv (ROCm) | `C:\dev\mamba3\mamba3\venv\` |
| MUSDB18-HQ | `C:\dev\mamba3\mamba3\datasets\musdb18hq\train\` |
| Checkpoints | `C:\Dev\Audio Development\0 CLARA\checkpoints_clara\` |
| ONNX models | `C:\Dev\Audio Development\0 CLARA\models\` |
