# CLARA Training Plan — v1.0

**Date:** 2026-06-03
**Hardware:** Ryzen 7 AI 350 + Radeon 860M iGPU (ROCm)
**Model:** Mamba RT SSM, ~77K params (d_model=64, n_layers=4, d_state=16, COND_DIM=8)

---

## 1. Executive Summary

Train the CLARA Mamba SSM audio effect processor through staged curriculum: identity autoencoder → saturation → compression → denoising → (TBD additional effects). Each phase fine-tunes from the previous checkpoint. Conditioning (COND_DIM=8) is folded into the ONNX model as a 10th input with per-layer FiLM modulation.

**Estimated throughput:** 30-50 steps/sec on 860M ROCm → ~33-55 min per 100K-step phase.

---

## 2. Architecture Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| d_model | 64 | Embedding dimension | 
| d_state | 16 | SSM state per channel |
| n_layers | 4 | Stacked Mamba layers |
| conv_width | 4 | Causal conv kernel (4-sample latency) |
| dt_rank | 16 | Discretization projection rank |
| COND_DIM | 8 | Conditioning input (tempo, key, time sig, effect params) |
| Total params | ~77K | ~72K base + ~5K cond FiLM |

### Data flow (with conditioning)

```
audio_chunk [B, L, 1]          cond [B, 8]
    │                              │
    ▼                              │
InputProjection(1+8 → 64)  ◄──────┤  cond concatenated with audio
    │                              │
    ▼                              │
┌─ MambaLayer 0 ◄─────────────────┤  per-layer FiLM (gamma, beta)
│   in_proj → Conv1d → SiLU → SSM → out_proj
│   conv_state[0] ──► causal conv
│   ssm_state[0] ──► parallel scan
└─────────────────────────────────┘
    │                              │
    ▼                              │
┌─ MambaLayer 1 ◄─────────────────┤
└─────────────────────────────────┘
    │                              │
    ▼                              │
┌─ MambaLayer 2 ◄─────────────────┤
└─────────────────────────────────┘
    │                              │
    ▼                              │
┌─ MambaLayer 3 ◄─────────────────┘
└─────────────────────────────────┘
    │
    ▼
LayerNorm(64) → OutputProjection(64→1) → processed_audio [B, L, 1]
```

### ONNX I/O contract (10 inputs, 3 outputs)

| Index | Name | Shape | Description |
|-------|------|-------|-------------|
| 0 | `audio_chunk` | `{1, seq_len, 1}` | Mono audio, dynamic seq_len |
| 1-4 | `conv{0..3}` | `{1, 64, 4}` | Per-layer conv state |
| 5-8 | `ssm{0..3}` | `{1, 64, 16}` | Per-layer SSM state |
| 9 | `cond` | `{1, 8}` | Conditioning vector |

| Index | Name | Shape | Description |
|-------|------|-------|-------------|
| 0 | `processed_audio` | `{1, seq_len, 1}` | Processed mono audio |
| 1 | `new_conv_state` | `{1, 4, 64, 4}` | Stacked conv states |
| 2 | `new_ssm_state` | `{1, 4, 64, 16}` | Stacked SSM states |

---

## 3. Training Phases

### Phase 1: Identity Autoencoder

**Goal:** Verify training pipeline. Model learns passthrough (input = target). Baseline weights for all subsequent fine-tuning.

| Parameter | Value |
|-----------|-------|
| Task | input == target (same audio chunk) |
| Dataset | MUSDB18-HQ mixture tracks |
| Conditioning | Zeros (cond = [0,...,0]) — model learns identity regardless of cond |
| Loss | 0.85×L1 + 0.15×MR-STFT |
| Steps | 50,000 |
| Success metric | L1 loss < 0.01, audible passthrough |
| Init from | Random (fresh) |

**State handling:** Zero-initialized conv/ssm states per independent 2-second chunk.

### Phase 2: Saturation / Harmonic Distortion

**Goal:** Model learns musical saturation (tanh, soft clip, hard clip, asymmetrical).

| Parameter | Value |
|-----------|-------|
| Task | clean → saturated pairs |
| Dataset | MUSDB18-HQ, MAESTRO |
| Data generation | Synthetic: apply random saturation DSP with varied drive/curve/asymmetry |
| Conditioning | librosa-estimated tempo+key; drive amount encoded in cond[4] |
| Loss | 0.80×L1 + 0.15×MR-STFT + 0.05×SI-SDR |
| Steps | 100,000 |
| Init from | Phase 1 checkpoint (transfer learning) |

### Phase 3: Compression

**Goal:** Model learns dynamics compression (threshold, ratio, attack, release). Causal only (no lookahead).

| Parameter | Value |
|-----------|-------|
| Task | uncompressed → compressed pairs |
| Dataset | MUSDB18-HQ, MAESTRO, DNS Challenge |
| Data generation | Synthetic: RMS feed-forward compressor with randomized params |
| Conditioning | Compressor params encoded in cond[4-6] (threshold, ratio, makeup) |
| Loss | 0.80×L1 + 0.15×MR-STFT + 0.05×SI-SDR |
| Steps | 100,000 |
| Init from | Phase 1 or Phase 2 checkpoint |

### Phase 4: Denoising

**Goal:** Model learns to remove noise from audio.

| Parameter | Value |
|-----------|-------|
| Task | noisy → clean pairs |
| Dataset | DNS Challenge + synthetic noise on MUSDB18/MAESTRO |
| Data generation | Mix clean audio with noise (white/pink/brown/environmental) at random SNR [-5,+20] dB |
| Conditioning | SNR encoded in cond[4] |
| Loss | 0.70×L1 + 0.20×MR-STFT + 0.10×SI-SDR |
| Steps | 150,000-200,000 |
| Init from | Phase 1 checkpoint |

### Future Tasks

| Task | Data Source | Generation Method |
|------|------------|-------------------|
| EQ matching | Any musical audio | Apply random parametric EQ curves; encode params in cond |
| Transient shaping | MUSDB18 drums | Apply attack/sustain envelope modification |
| De-essing | DNS Challenge vocals | Detect sibilance (4-10 kHz), apply targeted gain reduction |
| Stereo widening | MUSDB18 stereo | Requires model upgrade to stereo input. Deferred. |
| Creative multi-effect | Any audio | Chain saturation→EQ→compression; encode all params in cond |

---

## 4. Critical Implementation: Parallel Scan

The ONNX export uses a sequential scan (`for t in range(seq_len)`) which is fine for ORT inference at 256-sample blocks. Training uses a **parallel scan** (Heinsen log-cumsum) that is 25-50x faster on GPU.

Both variants share identical weight names/layout — checkpoints load into either.

### Parallel scan math (CLARA real-valued SSM)

**Sequential** (current ONNX export):
```python
for t in range(L):
    dA = exp(dt[:,t,:].unsqueeze(-1) * A.unsqueeze(0))    # [B, D, N]
    dB = dt[:,t,:].unsqueeze(-1) * B[:,t,:].unsqueeze(1)   # [B, D, N]
    h = dA * h + dB * x[:,t,:].unsqueeze(-1)               # [B, D, N]
    ys.append((C[:,t,:].unsqueeze(1) * h).sum(-1) + D*x)   # [B, D]
y = stack(ys, dim=1)  # [B, L, D]
```

**Parallel** (training only):
```python
dA = exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B, L, D, N]
dB = dt.unsqueeze(-1) * B.unsqueeze(2)                     # [B, L, D, N]
b_input = dB * x.unsqueeze(-1)                             # [B, L, D, N]

log_dA = log(dA + 1e-12)
log_cumsum = cumsum(log_dA, dim=1)
h = exp(log_cumsum) * cumsum(b_input * exp(-log_cumsum), dim=1)
y = (C.unsqueeze(2) * h).sum(dim=-1) + D * x              # [B, L, D]
```

- Zero additional parameters
- Mathematically identical to sequential
- Complex math always runs in fp32 (autocast disabled locally)
- h_last = h[:, -1, :, :] → next block's ssm_state

---

## 5. Dataset Strategy

| Dataset | Size | SR | Use | Status |
|---------|------|-----|-----|--------|
| MUSDB18-HQ | ~30 GB, 86 train + 50 test | 44.1 kHz stereo stems | Primary for all phases | Already at `C:\Dev\mamba3\mamba3\datasets\musdb18hq\train\` |
| MAESTRO v3 | ~100 GB, ~200 hrs piano | 44.1 kHz stereo | Musical diversity, MIDI metadata | Download before Phase 2 |
| DNS Challenge | ~100+ GB, ~750 hrs speech | 16 kHz | Denoising task | Download before Phase 4 |

### Data Loading Strategy (from mamba3 reference)

- RAM pre-caching all stems (~4 GB for MUSDB18-HQ) → eliminates disk I/O → 9x speed boost
- 2-second random crops (88,200 samples @ 44.1 kHz)
- Augmentation: ±6 dB gain, 50% polarity flip, clamp to [-1, 1]
- Train/val split: 90/10 at track level, seed 42
- DataLoader: batch=4, num_workers=2, pin_memory=True, persistent_workers=True

### Conditioning Label Extraction

For phases where cond ≠ zero, librosa estimates musical features:
```python
tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
key = chroma.mean(axis=1).argmax()  # 0-11

# COND_DIM=8 encoding:
# cond[0] = tempo / 200.0         # normalized tempo (30-300 BPM → 0.15-1.5)
# cond[1] = key / 12.0            # key root (0-11 → 0-0.917)
# cond[2] = major/minor flag      # 0=major, 1=minor
# cond[3] = time sig numerator/8  # 3/4→0.375, 4/4→0.5, 6/8→0.75
# cond[4] = effect param 1        # phase-dependent (drive, threshold, SNR)
# cond[5] = effect param 2        # phase-dependent
# cond[6] = effect param 3        # phase-dependent  
# cond[7] = reserved / section pos
```

---

## 6. Training Infrastructure

### ROCm Environment (860M iGPU)

```powershell
$env:HSA_ENABLE_SDMA = "0"
$env:PYTORCH_ALLOC_CONF = "garbage_collection_threshold:0.8,max_split_size_mb:256"
$env:MIOPEN_LOG_LEVEL = "3"
$env:MIOPEN_FIND_MODE = "2"
$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
```

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch size | 4 | 2× mamba3 batch (model is 10× smaller) |
| Gradient accum | 4 | Effective batch = 16 |
| Chunk duration | 2.0 sec (88,200 samples) | Enough context for SSM, fits in shared VRAM |
| AMP | bf16 | ROCm supports bf16, model ~154 KB in bf16 |
| Workers | 2 | Matches reference (4 Zen 5c cores available) |
| Ckpt every | 500 steps | Power-loss protection, GPU hang resilience |
| Val every | 1000 steps | Capped at 20 batches |
| Log every | 25 steps | |

### Optimizer

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.999, ε=1e-8) |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 (no decay on bias/norm params) |
| Schedule | Cosine annealing, 2000-step linear warmup |
| Grad clipping | 1.0 (global norm) |
| Early stop | Patience 20 val checks |

### Performance Estimate

| Component | mamba3 laptop | CLARA | Factor |
|-----------|--------------|-------|--------|
| Params | 785K | 77K | 10.2× smaller |
| Steps/sec | 9 | 30-50 | 3-5× faster |
| 100K steps | ~3 hrs | ~33-55 min | |
| VRAM usage | <3 GB | <1 GB | |

---

## 7. Loss Functions

### Identity Autoencoder (Phase 1)
```
total = 0.85 × L1(time-domain) + 0.15 × MR-STFT(magnitude)
```
FFT sizes: [512, 1024, 2048], hops: [128, 256, 512]

### Effect Tasks (Phases 2-4)
```
total = α×L1 + β×MR-STFT + γ×SI-SDR
```
| Phase | α | β | γ (ramps 0→value over steps 5K-15K) |
|-------|---|---|------------------------------------------|
| Saturation | 0.80 | 0.15 | 0.05 |
| Compression | 0.80 | 0.15 | 0.05 |
| Denoising | 0.70 | 0.20 | 0.10 |

---

## 8. ONNX Export Post-Training

After each phase, export trained weights to ONNX via `train_clara_export.py`:

```bash
python train_clara_export.py \
    --checkpoint checkpoints_clara/phase1_identity_best.pt \
    --output models/mamba_rt.onnx
```

The export:
1. Loads checkpoint into `MambaRTModel` (sequential scan variant from `export_mamba_onnx.py`)
2. Verifies ONNX-vs-PyTorch output match (cosine >0.999, max abs diff <1e-4)
3. Verifies state continuity across 2+ chunks
4. Generates `models/manifest.json` with SHA256 + architecture contract
5. Outputs `models/mamba_rt.onnx` + `models/mamba_rt.onnx.data`

---

## 9. C++ Engine Updates (Synchronized)

When ONNX contract changes from 9→10 inputs:

1. `MambaInferenceEngine.hpp`:
   - `kNumInputs = 10` (was 9)
   - Add `"cond"` to `inputNames_`
   - Add `condInput_` buffer
   - Remove `condWeights_`, `condWeightsLoaded_`, `loadCondWeights()`
   - Keep `setConditioning()` (still double-buffered for RT-safety)

2. `MambaInferenceEngine.cpp`:
   - Feed cond tensor as ONNX input in `processBlock()`
   - Remove external gain modulation post-processing loop
   - Remove `loadCondWeights()` implementation

3. `MambaClapPlugin.cpp`:
   - Remove `loadCondWeights()` call in `mamba_activate()`
   - Keep `setConditioning()` call (now feeds ONNX, not post-processing)

4. `export_mamba_onnx.py`:
   - Add `cond` parameter to `MambaRTModel.forward()`
   - Add `cond` input to `MambaLayer` + per-layer FiLM
   - Update input names, preflight checks, verification code

---

## 10. Files to Deploy

```
C:\Dev\Audio Development\0 CLARA\
├── train_clara.py                 # Main training script
├── train_clara_model.py           # Parallel-scan training model + sequential export model
├── train_clara_dataset.py         # Dataset loader (MUSDB18/MAESTRO/DNS)
├── train_clara_dsp.py             # Synthetic DSP (saturation, compression)
├── train_clara_loss.py            # L1 + MR-STFT + SI-SDR
├── train_clara_export.py          # ONNX export from checkpoint
├── train_clara_launcher.py        # Auto-detect dataset, resume, launch
├── train_clara_config.py          # Configuration dataclass
├── train_clara.ps1                # PowerShell wrapper
├── train_clara_requirements.txt   # Python dependencies
├── docs/training-plan.md          # This document
├── models/                        # ONNX model outputs
├── checkpoints_clara/             # Training checkpoints
├── src/core/MambaInferenceEngine.hpp  # (modified) kNumInputs=10
├── src/core/MambaInferenceEngine.cpp  # (modified) cond tensor, remove ext gain
├── src/clap/MambaClapPlugin.cpp       # (modified) remove loadCondWeights
└── export_mamba_onnx.py               # (modified) cond input + FiLM
```

---

## 11. Risks and Mitigations

| Risk | Sev | Mitigation |
|------|-----|------------|
| ONNX batch-size mismatch (train batch=4, infer batch=1) | Med | ONNX dynamic axes handle any batch. Verify at export. |
| Sequential vs parallel scan divergence | Med | Validate ONNX-vs-PyTorch output at export (cosine >0.999). |
| 860M GPU hangs (gfx1152 ROCm bug) | Med | Checkpoint every 500 steps. Interrupt handler. |
| dt underflow: large A × large dt → exp→0 | Low | log_A init produces reasonable A values. Monitor for zero output. |
| FiLM dominates identity (gamma/beta init) | Low | Initialize FiLM weights to zero → identity at training start. |
| DNS resampling artifacts (16→44.1 kHz) | Low | High-quality torchaudio resample. Verify spectrally. |
| C++ backward compat (old models won't load) | Med | Version model files. Old 9-input models rejected by manifest check. |
| Zero-state training vs stateful inference mismatch | Low | Conv state is 4 samples (<0.1ms context). SSM mismatch acceptable for effect processing. |

---

## 12. Implementation Order

1. `train_clara_config.py` — Configuration dataclass
2. `train_clara_model.py` — Parallel scan model + cond FiLM
3. `train_clara_dataset.py` — MUSDB18 loader with RAM caching
4. `train_clara_loss.py` — L1 + MR-STFT + SI-SDR
5. `train_clara.py` — Full training loop
6. Phase 1 training — Identity autoencoder on MUSDB18
7. `train_clara_export.py` — ONNX export from checkpoint
8. C++ updates — kNumInputs=10, cond tensor, remove ext gain
9. `export_mamba_onnx.py` update — cond input + per-layer FiLM
10. `train_clara_dsp.py` — Synthetic DSP generators
11. Phase 2+ training — Saturation, compression, denoising
