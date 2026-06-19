# CLARA — Architecture & Shipping Plan

**Version:** 1.0 | **Date:** 2026-06-12 | **Author:** Hermes Agent (clara-architect skill)

---

## Phased Shipping Plan

```
Phase 1 (POC) ──→ Phase 2 (Alpha) ──→ Phase 3 (Context-Aware) ──→ Phase 4 (Production)
   You are here          Next ship               Feature ship            Polish ship
```

---

## Phase 1 — Proof of Concept (Current State)

### Complete ✅
| Component | Status |
|-----------|--------|
| `MambaCore` static library (ONNX inference) | ✅ Implemented, tested |
| `MambaSuiteClap` CLAP plugin (3 params, bypass, latency reporting) | ✅ Implemented |
| `MambaSuiteARA` ARA2 library (DocumentController, PlaybackRenderer, archive) | ✅ Implemented |
| `MambaRTModelParallel` / `MambaRTModelSequential` (76K-param models) | ✅ Implemented |
| `train_clara_parallel.py` (training loop) | ✅ Implemented + chunked |
| `train_clara_export.py` (ONNX export) | ✅ Verified via smoke test |
| Distrobox `clara-training` with ROCm PyTorch | ✅ GPU verified |
| MUSDB18-HQ dataset on NVMe | ✅ 41 GB copied |
| `clara-architect` + `clara-training-launch` skills | ✅ Created |
| CI/CD (Linux + Windows build/test) | ✅ Updated with quality gate |
| Tmux integration in training launch | ✅ Implemented |
| Chunked sub-block processing (kSubBlockSize=64) | ✅ Implemented |

### Needed to close Phase 1
| # | Task | Effort | Priority |
|---|------|--------|----------|
| P1-01 | **Run Phase 1 identity training** (50K steps) | ~4-8 hr on 860M, ~20 min on RunPod H100 | 🔴 Blocking |
| P1-02 | **Export trained model to ONNX** `train_clara_export.py` | 5 min | 🔴 Blocking |
| P1-03 | **Fix stale CMake cache** (`rm -rf build && cmake -B build ...`) | 2 min | 🔴 Blocking |
| P1-04 | **Run C++ tests with trained model** (`ctest --output-on-failure`) | 5 min | 🔴 Blocking |
| P1-05 | **DAW load test** (Carla/Bitwig/REAPER — load plugin, verify audio) | 30 min | 🟡 High |
| P1-06 | **Update ONNX manifest.json** with new export timestamp + SHA256 | 5 min | 🟢 Med |

---

## Phase 2 — Alpha (Next Ship)

### G6: GUI (CLAP_EXT_GUI)
**What:** 3 sliders (bypass, wet/dry, intensity) + status indicator. Resizable window.
**Approach:** JUCE or pugl. JUCE has CLAP support built-in but adds ~20MB binary. pugl is minimal but needs manual widget drawing.
**Files to create:** `src/gui/MambaGuiPlugin.cpp`, `src/gui/MambaGuiView.hpp`
**ADR needed:** ADR-011 — GUI Framework Decision
**Effort:** ~1 week

### G7: ARA Conditioning Bridge (REAL — NOT STUB)
**What:** `doUpdateMusicalContextContent()` currently pushes neutral zeros. Must extract real musical context from ARA and encode it into the cond vector.

**Implementation plan:**
1. **Host content access controller setup** — In `MambaDocumentController`, request host content access via `ARADocumentControllerHostInstance`:
   - Request `kARAContentTypeTempoEntries` for tempo
   - Request `kARAContentTypeKeySignatures` for key/mode
   - Request `kARAContentTypeBarSignatures` for time signature

2. **Content reader implementation** — Create `ARA_CONTENT_READER_CLASS` for each content type:
   ```cpp
   // In MambaAraController.hpp
   class MambaContentReader : public ARA::PlugIn::ContentReader {
       void doReadMusicalContextContent(
           ARA::PlugIn::MusicalContext* context,
           ARAContentType contentType,
           const ARAContentTimeRange* range,
           ARA::ContentReadingType readingType) noexcept override;
   };
   ```

3. **Conditioning update flow:**
   ```
   ARA host signals content change
       → doUpdateMusicalContextContent() called
       → Create MambaContentReader for the changed context
       → Read tempo entries, key signatures, bar signatures
       → Encode into cond[8] via ConditioningGenerator::encode()
       → Call engine->setConditioning(cond, 8)
       → Audio thread picks up new cond via atomic pointer swap
   ```

4. **ConditioningGenerator improvements** — Current `encode()` maps only 5 of 8 dims. For full musical context:
   | Dim | Source | Encoding |
   |-----|--------|----------|
   | 0 | Tempo (BPM) | Log-normalized, centered at 120bpm |
   | 1 | Key pitch class | 0=C, 0.083=C#, ... 0.916=B |
   | 2 | Key mode | 0=major, 1=minor, 0.5=unknown |
   | 3 | Time signature | 0=3/4, 0.5=4/4, 1=other |
   | 4 | Section type | 0=intro, 0.2=verse, 0.5=chorus, 0.8=bridge, 1=outro |
   | 5 | Chord density | Future: count of unique chords per bar |
   | 6-7 | Reserved | Future use |

5. **Tests:**
   - Unit: mock content reader, verify cond encoding
   - Integration: set cond from known tempo/key → run model → verify output differs
   - ThreadSanitizer: concurrent ARA thread write + audio thread read

**Files to create/modify:**
- `src/ara/MambaAraController.hpp` — add ContentReader class
- `src/ara/MambaAraController.cpp` — implement reading + encoding
- `src/core/ConditioningGenerator.hpp` — verify dim layout
- `tests/test_ara_controller.cpp` — add conditioning tests
- `ADR-011-ara-conditioning-bridge.md` — new ADR

**Effort:** ~1 week

### G8: ARA Analysis Pipeline (REAL — NOT STUB)
**What:** `contentDirty_` flag is set but never consumed. Must actually analyze audio source content.

**Implementation plan:**
1. **Analysis trigger** — When `didEnableAudioSourceSamplesAccess(true)` fires:
   - Push `AnalysisQueueDrain` task to SPSC queue
   - In `mamba_on_main_thread`, pop the task and:
     - Read audio samples from ARA content access controller
     - Chunk into 64-sample blocks (matching kSubBlockSize)
     - Run inference through `MambaInferenceEngine`
     - Extract features (spectral centroid, RMS energy, onset density)
     - Write analysis content back to host via `ARAContentWriter`

2. **SPSC queue analysis drain** — Fill in the empty case body at `MambaClapPlugin.cpp:523-530`:
   ```cpp
   case MambaPluginState::DeferredTask::AnalysisQueueDrain: {
       // Analysis results available from ARA.
       // Read pending analysis data, write to ARA content.
       break;
   }
   ```

3. **Content writing** — Use `ARAContentWriter` to publish:
   - `kARAContentTypeTempoEntries` (if tempo detected)
   - Analysis metadata as custom content type

**Files to modify:**
- `src/clap/MambaClapPlugin.cpp` — AnalysisQueueDrain case body
- `src/core/MambaInferenceEngine.hpp/cpp` — analysis API (if needed)
- `src/ara/MambaAraController.cpp` — analysis triggering

**Effort:** ~3-4 days

---

## Phase 3 — Context-Aware Features

### G9: Phases 2-5 Training Design
**The gap:** Phase 1 (identity) trains the model to reconstruct its input. It has NO effect (output = input). The actual product capabilities come from Phases 2-5.

**Architecture decision:** The model learns via supervised contrastive training:
- Each phase teaches a different transformation
- The same 76K-param `MambaRTModel` learns all phases sequentially
- Earlier phases provide pre-trained features for later phases

#### Phase 2: Saturation / Harmonic Enhancement
- **Goal:** Model learns to add harmonic saturation (soft clipping, tape warmth)
- **Data:** Clean audio → apply `apply_saturation()` (from `train_clara_dsp.py`)
- **Loss:** STFT-L1 + Mel-L1 + perceptual (SI-SDR not applicable for nonlinear FX)
- **Conditioning:** Zeros (no ARA context needed for saturation)
- **Target quality:** +3dB RMS loudness at 0.1% THD max

#### Phase 3: Compression
- **Goal:** Model learns dynamic range compression
- **Data:** Clean audio → apply `RMSCompressor()` (from `train_clara_dsp.py`)
- **Loss:** STFT-L1 + Mel-L1 + envelope L1 (match attack/release curve)
- **Conditioning:** Zeros
- **Target quality:** ±1dB gain reduction tracking accuracy

#### Phase 4: Denoising
- **Goal:** Model learns to remove additive noise
- **Data:** Clean audio + noise dataset → noisy = clean * 0.8 + noise * 0.2
- **Loss:** STFT-L1 + Mel-L1 + SI-SDR
- **Conditioning:** Zeros (noise profile is implicitly learned)
- **Target quality:** SI-SDR > 10dB on held-out noise types

#### Phase 5: Full Context-Aware Processing
- **Goal:** Model uses cond vector to modulate behavior based on musical context
- **Data:** Same as Phase 2-4 but with non-zero conditioning vectors
- **Conditioning:** ARA musical context encoded into cond[8]
- **Loss:** STFT-L1 + Mel-L1 + SI-SDR + conditioning contrastive loss
- **Target quality:** Different cond → different output for same input

**Files to create:**
- `ADR-012-training-phases.md` — full architecture doc for multi-phase training
- `train_clara_phase2.py` — saturation training
- `train_clara_phase3.py` — compression training
- `train_clara_phase4.py` — denoising training
- `train_clara_phase5.py` — full context-aware training

**Effort:** ~2-4 weeks total

---

## Phase 4 — Production Polish

| # | Task | Effort | Priority |
|---|------|--------|----------|
| G10 | ONNX manifest enforcement (reject mismatched models, not just warn) | 1 day | 🟡 High |
| G11 | clap-validator blocking in CI (remove `continue-on-error`) | 1 day | 🟢 Med |
| G12 | macOS + Windows build verification | 2 days | 🟡 High |
| G13 | Install/packaging (DEB, distrobox-export, install script) | 1 day | 🟢 Med |
| G14 | Train step rate optimization (H100 RunPod) | 1 day | 🟢 Med |

---

## Training Configuration (for next-morning launch)

```bash
# Recommended: 860M overnight run
# With chunked processing (chunk_size=64), step rate should hit ~2-5 step/s
distrobox enter clara-training
source ~/clara-training/venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HSA_ENABLE_SDMA=0
export PYTORCH_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:256"
export MIOPEN_FIND_MODE=5
cd ~/Github\ Repo\'s/clara

# Full Phase 1 identity training (50K steps)
python train_clara_parallel.py \
  --data ~/datasets/musdb18hq/train \
  --device cuda \
  --batch 4 \
  --grad-accum 4 \
  --steps 50000 \
  --clip-seconds 0.5 \
  --cache \
  --num-workers 4

# Alternative: RunPod H100 ($15-20, ~20 min for 50K steps)
# Requires: copying code + data to RunPod instance
```

---

## Reference: All Gaps (from audit)

| Gap | Phase | Description | Status |
|-----|-------|-------------|--------|
| G1 | P1 | Model is untrained (random weights) | ❌ |
| G2 | P1 | Stale CMake cache (ARADebug.c.o path) | ⚠️ |
| G3 | P1 | C++ tests never run with real model | ⚠️ |
| G4 | P1 | No model quality acceptance criteria | ✅ CI gate added |
| G5 | P1 | No DAW smoke test | ❌ |
| G6 | P2 | No GUI (CLAP_EXT_GUI) | ❌ |
| G7 | P2 | ARA conditioning is a stub | ❌ |
| G8 | P2 | ARA analysis pipeline is a stub | ❌ |
| G9 | P3 | Phases 2-5 training don't exist | ❌ |
| G10 | P4 | No ONNX manifest enforcement | ⚠️ |
| G11 | P4 | clap-validator not blocking in CI | ⚠️ |
| G12 | P4 | Cross-platform builds untested | ❌ |
| G13 | P4 | No install/packaging | ⚠️ |
| G14 | P4 | Train step rate is very slow | ⚠️ |
