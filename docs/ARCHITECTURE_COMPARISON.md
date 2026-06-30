# Architecture Comparison — Clara Mamba-3 vs mamba3 Drive PerTrackProcessor

**Date:** 2026-06-20

---

## TL;DR

These are **different models**. The mamba3 drive's `PerTrackProcessor` (3.9M params, spectral domain, chunk_size=128) is incompatible with Clara's time-domain Mamba-3 (1.6M params, no STFT, chunk_size=64). No architecture port is planned.

---

## 1. Model Dimensions

| Parameter | Clara | mamba3 Drive | Delta |
|-----------|-------|-------------|-------|
| `d_model` | 128 | 256 | 2× |
| `d_inner` | 256 | 512 | 2× |
| `d_state` | 32 | 64 | 2× |
| `nheads` | 4 | 8 | 2× |
| `headdim` | 64 | 128 | 2× |
| `mimo_rank` | 4 | 4 | Same |
| `n_layers` | 6 | 6 | Same |
| `cond_dim` | 128 | 128 | Same |
| `chunk_size` | **64** | **128** | **2× — BLOCKER** |
| `in_proj_dim` | 1,612 | ~2,400+ | Larger |

---

## 2. Parameter Count

| Component | Clara | mamba3 Drive |
|-----------|-------|-------------|
| Per-layer | ~123,356 | ~650,000 |
| 6 layers | ~740,136 | ~3,900,000 |
| LayerNorm + output | ~384 | ~512 |
| **Total** | **~757K** (1,662,384 in export) | **~3.9M** |

Clara's model is ~2.4× smaller.

---

## 3. Signal Domain

### Clara: Time-Domain

```
Input audio (float32 samples)
    ↓
in_proj (scalar → d_model embedding)
    ↓
6× SSMBlock layers (Mamba-3 MIMO R=4)
    ↓
LayerNorm (fixed γ=1, β=0)
    ↓
Read d_model[0] as output sample
```

- No STFT, no mel spectrogram, no frequency domain
- Input: raw audio samples
- Output: raw audio samples
- C++ engine processes samples directly

### mamba3 Drive: Spectral-Domain

```
Input audio (float32 samples)
    ↓
STFT → complex spectrogram
    ↓
PerTrackProcessor (Mamba-3, chunk_size=128)
    ↓
iSTFT → output audio samples
```

- Has STFT frontend + iSTFT output wrapper
- Processes in frequency domain internally
- Different weight layout, different state representation

---

## 4. C++ Engine Constraints

### Clara's Engine (`SsmInferenceCore`)

```cpp
// src/core/MambaConfig.h
constexpr size_t MAX_CHUNK_SIZE    = 64;    // Sub-block size
constexpr size_t MAX_BLOCK_SIZE    = 512;   // Max processBlock() call
constexpr size_t MAX_D_STATE       = 32;
constexpr size_t MAX_D_MODEL       = 128;
constexpr size_t MAX_D_INNER       = 256;
constexpr size_t MAX_NHEADS        = 4;
constexpr size_t MAX_HEADDIM       = 64;
constexpr size_t MAX_IN_PROJ_DIM   = 1612;
```

The engine reads audio in blocks of ≤512 samples, processing sub-blocks of ≤64 samples each. The `SsmBlock` runs an 8-step per-sample pipeline:

1. FiLM conditioning
2. Fused `in_proj` (1,612 → 128 per sample)
3. A and Δ preprocessing
4. RMSNorm on B and C
5. RoPE rotation
6. Multi-head MIMO SSM recurrence
7. SiLU z-gating
8. Output projection + D skip + residual

### mamba3 Drive's Engine (`PerTrackProcessor`)

```python
# model.py
class PerTrackProcessor(nn.Module):
    def __init__(self, ...):
        self.in_proj = nn.Linear(d_model, in_proj_dim)  # larger
        self.blocks = nn.ModuleList([SSMBlock(...) for _ in range(n_layers)])
        # chunk_size=128 hardcoded
```

- Custom C++ with `MAX_CHUNK_SIZE=128`
- Larger weight arrays (different shapes)
- Different export format

---

## 5. Weight Format Comparison

### Clara (Version 2, 13 arrays per layer)

| # | Array | Shape |
|---|-------|-------|
| 0 | `in_proj_w` | `[1612, 128]` |
| 1 | `dt_bias` | `[4]` |
| 2 | `B_bias` | `[4, 4, 32]` |
| 3 | `C_bias` | `[4, 4, 32]` |
| 4 | `B_norm_weight` | `[32]` |
| 5 | `C_norm_weight` | `[32]` |
| 6 | `mimo_x` | `[4, 4, 64]` |
| 7 | `mimo_z` | `[4, 4, 64]` |
| 8 | `mimo_o` | `[4, 4, 64]` |
| 9 | `D_proj` | `[4]` |
| 10 | `out_proj_w` | `[256, 128]` |
| 11 | `film_proj_w` | `[256, 128]` |
| 12 | `dt_film_proj_w` | `[8, 128]` |

**Total per layer:** 277,064 floats (1,108,256 bytes)
**Total file:** 64-byte header + 6 × 277,064 = 1,662,384 floats (6,649,600 bytes)

### mamba3 Drive (Different layout)

- `in_proj` shape: `[in_proj_dim, 256]` (larger d_model)
- `mimo_x/z/o` shapes: `[8, 4, 128]` (more heads, larger headdim)
- `out_proj_w`: `[512, 256]` (larger dimensions)
- Different header format, different array count

**These weight files are NOT interchangeable.**

---

## 6. Training Pipeline Comparison

| Aspect | Clara | mamba3 Drive |
|--------|-------|-------------|
| Script | `train_mamba3.py` (1,258 lines) | `train.py` (1,164 lines) |
| Dataset | MUSDB18-HQ (WAV) | MUSDB18-HQ (WAV) |
| Loss | MSE + Multi-Resolution STFT | MSE + Multi-Resolution STFT + SI-SDR |
| Optimizer | AdamW | AdamW |
| LR Schedule | Cosine with warmup | Cosine with warmup |
| Resume | ✅ | ✅ |
| Validation | ✅ | ✅ |
| Early Stopping | ✅ | ✅ |
| Signal Handler | ✅ | ✅ |
| Atomic Writes | ✅ | ✅ |
| Precomputed Pipeline | ✅ | ❌ |
| Chunked Scan | ✅ (C=64) | ✅ (C=128) |

**Training pipelines are functionally equivalent.** Clara's has additional features (precomputed pipeline, better validation logging).

---

## 7. Why No Port

1. **chunk_size mismatch** — Clara's C++ engine requires `chunk_size=64`. mamba3 drive uses 128. Changing the C++ engine would require updating `MambaConfig.h`, `SsmBlock.cpp`, `SsmInferenceCore.cpp`, weight format, and export pipeline.

2. **Parameter count mismatch** — mamba3 drive's 3.9M params would require larger weight arrays, different `in_proj_dim`, and more memory in the C++ engine (currently stack-allocated at fixed sizes).

3. **Signal domain mismatch** — mamba3 drive wraps Mamba-3 in STFT/iSTFT. Clara is purely time-domain. Porting would require adding spectral processing to the C++ engine.

4. **No benefit** — Clara's 1.6M time-domain model is designed for zero-latency real-time processing. The mamba3 drive's 3.9M spectral model has higher latency (STFT window) and higher compute cost.

5. **Already has all improvements** — The 6 training-loop improvements from the mamba3 drive are already in Clara's `train_mamba3.py`.

---

## 8. What IS Shared

Despite the architecture differences, the following are genuinely transferable:

| Item | Shared | Notes |
|------|--------|-------|
| Training loop features | ✅ | Resume, validation, early stopping, signal handler, atomic writes |
| Loss functions | ✅ | MSE + Multi-Resolution STFT |
| Optimizer config | ✅ | AdamW + cosine LR + warmup |
| CREPE extraction | ✅ | Adapted with hop=256 for Clara's frame rate |
| Smoke test structure | ✅ | Adapted for Mamba3Model (not PerTrackProcessor) |
| MUSDB18-HQ dataset | ✅ | Same dataset, same format |
| SI-SDR loss concept | ✅ | Adoption planned for Phase 2 |

---

## 9. Future: Phase 2 Conditioning

When Clara adds f0/CREPE conditioning (Phase 2), the model will gain:

- **Input:** +1 channel (f0 embedding concatenated to audio)
- **Conditioning vector:** f0 features go into `cond[40-71]` (reserved slots)
- **Loss:** +SI-SDR ramping (from mamba3 drive)
- **Dataset:** `StemDataset` class (from mamba3 drive concept)

This does NOT require porting the mamba3 drive's architecture — only the training techniques.

---

## 10. Summary Table

| Dimension | Clara | mamba3 Drive | Compatible? |
|-----------|-------|-------------|-------------|
| Params | 757K | 3.9M | ❌ |
| Domain | Time | Spectral | ❌ |
| chunk_size | 64 | 128 | ❌ |
| C++ engine | SsmInferenceCore | PerTrackProcessor | ❌ |
| Weight format | v2, 13 arrays | Different layout | ❌ |
| Training loop | Full | Full | ✅ |
| Loss functions | MSE+STFT | MSE+STFT+SI-SDR | ✅ (adopt SI-SDR) |
| Dataset | MUSDB18-HQ | MUSDB18-HQ | ✅ |
| CREPE extraction | hop=256 | hop=512 | ✅ (adapted) |
