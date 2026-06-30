# CLARA TODO — Reality-Based Task List

**Last updated:** 2026-06-22
**Context:** Model is R=8 PerTrackProcessor (spectral-domain, complex poles). Training is LOCAL on AMD AI 7 350 / Radeon 860M. C++ engine (SsmInferenceCore) is R=4 time-domain — fundamentally incompatible with PerTrackProcessor.

---

## ✅ VERIFICATION CHECKLIST (Current — 2026-06-22)

All checks pass (verified this session):

| # | Check | Status |
|---|-------|--------|
| 1 | test_model.py — PerTrackProcessor smoke | ✅ fixed imports |
| 2 | validate_export.py — v3 format | ✅ updated for R=8, 18 arrays |
| 3 | dump_reference_output.py — PerTrackProcessor | ✅ rewritten for spectral model |
| 4 | prep_dataset.py — dead train_clara_parallel.py ref | ✅ fixed |
| 5 | bootstrap_local.sh | ✅ created for Radeon 860M |

---

## 🔴 BLOCKING — Must Do Before Any Real Training

### H3. C++ Engine Update for R=8 (complex poles, spectral domain)

**Status:** Not started
**Severity:** Blocking — PerTrackProcessor cannot run in C++ until engine is updated

PerTrackProcessor (R=8, spectral) is fundamentally different from the current C++ engine (R=4, time-domain). The C++ engine must first be updated to support R=8 SSM with complex poles before a trained PerTrackProcessor can be loaded as a plugin.

**Required changes to C++ engine:**
1. Add R=8 support (MAX_MIMO_RANK=8, update SSMState, SSMBlock, WeightLoader)
2. Add complex pole trapezoidal discretization (log_a_real, a_imag per head)
3. Remove per-sample RoPE (PerTrackProcessor doesn't use RoPE)
4. Add STFT frontend + ISTFT output wrapper (spectral domain model)
5. Update WeightLoader for v3 format (18 arrays/layer, R8 extractor weights)
6. Update validate_export.py for v3

**Files needing changes:**
- `src/core/MambaConfig.h` — MAX_MIMO_RANK=8, add complex pole params
- `src/core/SsmBlock.cpp` — complex pole SSM recurrence
- `src/core/WeightLoader.cpp` — v3 weight layout
- `src/core/SsmInferenceCore.cpp` — add STFT/ISTFT wrapper

### H5 ~~Cloud Bootstrap~~ → Local Bootstrap

**Status:** ✅ Done — `scripts/bootstrap_local.sh` created
**Was:** Cloud bootstrap for MI300X
**Now:** Local bootstrap for AMD AI 7 350 / Radeon 860M

---

## 🟡 MEDIUM — Needed for Quality Training

### M1. Validation loop

**File:** `train_mamba3.py`
**Status:** Exists (run_validation fn) — needs real audio dataset to be fully useful

### M2. Gradient checkpointing

**File:** `train_mamba3.py`
**Status:** Already ON for all Mamba layers (gradient_checkpointing=True)

### M3. AMP (bf16, opt-in)

**File:** `train_mamba3.py`
**Status:** `--amp` flag exists, disable if bf16 NaNs occur on 860M

### M4. Early stopping

**File:** `train_mamba3.py`
**Status:** Already implemented (--early-stop-patience)

---

## 🟢 LOW — Post-MVP

### L1. Real WAV dataset

**Status:** AudioDataset exists, CachedMelDataset exists
**Needs:** MUSDB18-HQ downloaded to ~/datasets/musdb18hq/train

### L2. CREPE f0 extraction + wiring

**File:** `scripts/extract_crepe.py`
**Status:** Ported from mamba3 drive, hop=256 for Clara's frame rate
**Needs:** Wiring into PerTrackProcessor conditioning slots

### L3. CLAP text encoder integration

**Status:** Phase 2/3 — needs trained model first

### L4. ARA DAW context → conditioning

**Status:** Phase 2 — needs trained model first

### L5. GUI (CLAP_EXT_GUI)

**Status:** Phase 2 — needs JUCE or pugl

---

## Architecture Summary

| Aspect | Python (train) | C++ (inference) |
|--------|--------------|-----------------|
| Model | PerTrackProcessor | SsmInferenceCore |
| MIMO Rank | R=8 | R=4 |
| Domain | Spectral (STFT→SSM→ISTFT) | Time-domain (per-sample) |
| Poles | Complex (log_a_real, a_imag) | Real (dd_A) |
| RoPE | None | Yes |
| Export | v3 (18 arrays/layer) | v2 (13 arrays/layer) |
| Compatible? | — | ❌ |

---

## ⏱️ NEXT ACTION

1. `bash scripts/bootstrap_local.sh` — setup + smoke test
2. Download MUSDB18-HQ if not present
3. **H3: Update C++ engine to R=8** (blocking for plugin deployment)
4. Launch training: `bash scripts/run_train.sh`
5. Export: `python train_mamba3.py --export checkpoints/best.pt --output models/model.bin`

(End of file - total 83 lines)
