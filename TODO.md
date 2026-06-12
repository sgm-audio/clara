# CLARA — Master TODO (2026-06-12)

**Repo:** `SGM-Studios/clara` | **Branch:** `main` | **Status key:** `⬜ pending` `🔄 in progress` `✅ done` `❌ cancelled`

---

## 1. Project Overview

### What CLARA Is

CLARA is a real-time Mamba SSM (Selective State Space Model) audio processor distributed as a **CLAP plugin** (`MambaSuiteClap.clap`) and an **ARA2 shared library** (`MambaSuiteARA.so/.dll/.dylib`). Both share a single inference core (`MambaCore` static library → `MambaInferenceEngine`).

**What it does:** Real-time audio denoising / transformation via a trained neural network running inside the DAW.

**How it works:**
- Audio enters as `[B, seq_len, 1]` float samples
- ONNX model (d_model=64, n_layers=4, d_state=16, conv_width=4, cond_dim=8) applies Mamba-SSM selective scan with FiLM conditioning
- Output returns processed audio + updated per-layer conv state + SSM state for next block

### Key Numbers

| Dimension | Value |
|-----------|-------|
| `d_model` | 64 |
| `n_layers` | 4 |
| `d_state` | 16 |
| `conv_width` | 4 |
| `dt_rank` | 16 |
| `cond_dim` | 8 |
| ONNX inputs | 10 (audio_chunk + 4×conv + 4×ssm + cond) |
| ONNX outputs | 3 (processed_audio + new_conv_state + new_ssm_state) |
| Sub-block size | 64 samples |
| Reported latency | 4 samples (conv_width) |
| Total params | ~171K (STFT-based `STFTMambaModel`) |
| Checkpoints | 248 files, best.pt = step 49,200 |

### Build System

- **CMake 3.14+, C++17**
- Three dependencies via `FetchContent` (no manual install):
  - CLAP SDK (`29ffcc27`)
  - ARA SDK (`1ca0b0aa`)
  - ONNX Runtime v1.17.1
- Outputs: `build/plugins/MambaSuiteClap.clap` + `MambaSuiteARA.so`
- Cross-platform: Linux (GCC/Clang), macOS (Clang, universal), Windows (MSVC)
- WSL Ubuntu build verified

---

## 2. PRD Obligations

The **Product Requirements Document** defines what CLARA must do as a product. Each obligation must be verified.

### PRD-1: Real-Time Audio Processing
- [ ] Audio input-to-output latency ≤ 10ms at 48kHz (currently: 4 samples = 0.083ms — far below threshold)
- [ ] No audio dropouts, clicks, or artifacts during normal operation
- [ ] `processBlock()` completes within one audio buffer period (≤ 256 samples @ 48kHz)
- [ ] Zero heap allocations on the audio thread (enforced via code review)
- [ ] **Verification:** Load plugin in Bitwig/REAPER, run 30min stress test at 48kHz/256 samples, monitor for dropouts

### PRD-2: CLAP Plugin Compliance
- [ ] All required CLAP vtable callbacks implemented (`init`, `destroy`, `get_extension`, `activate`, `deactivate`, `process`, `params_*`)
- [ ] Extension `CLAP_EXT_PARAMS` registered with 3 params: bypass (stepped), wet/dry, intensity
- [ ] Extension `CLAP_EXT_AUDIO_PORTS` registered with stereo I/O
- [ ] Extension `CLAP_EXT_STATE` for preset save/restore
- [ ] Extension `CLAP_EXT_LATENCY` reporting 4 samples
- [ ] Extension `CLAP_EXT_RENDER` (realtime-only)
- [ ] Extension `CLAP_EXT_TAIL` (0 samples)
- [ ] Extension `CLAP_EXT_STATE_CONTEXT` (project-preserved SSM state)
- [ ] **Verification:** `clap-validator` passes all required categories (audio-ports, params, state, latency)

### PRD-3: ARA2 Library Compliance
- [ ] `ARAGetFactory()` exported with `visibility("default")`
- [ ] `MambaDocumentController` with archive save/restore (magic `0x4D424152` + version)
- [ ] `MambaPlaybackRenderer` constructs `MambaInferenceEngine`; catches init exceptions
- [ ] `doGetContentTypes()` / `doGetContentGrades()` implemented
- [ ] `willEnable` / `didEnable` semantics correct
- [ ] `doUpdateMusicalContextContent()` — real musical context extraction (tempo/key/chord)
- [ ] **Verification:** Load in Studio One / REAPER with ARA, load/archive/save cycle works

### PRD-4: Click-Free Bypass
- [ ] Bypass engages/disengages with 50ms linear crossfade (no audible pop)
- [ ] Crossfade ramp step accumulated once per sample, outside channel loop (no stereo drift)
- [ ] **Verification:** Inject 1kHz sine, toggle bypass via host, check waveform scope for transient — should be clean

### PRD-5: Trained Model Quality
- [ ] ONNX model produces non-noise output (variance > 1e-6 with random weights; much higher with trained)
- [ ] `best.pt` at step 49,200 shows measurable loss improvement over step 0
- [ ] Phase 2 (contrastive noise training) implemented and executed
- [ ] Exported ONNX produces audible difference from untrained model in DAW
- [ ] **Verification:** Load trained `.onnx` vs random `.onnx` in plugin, compare output on known audio signal

### PRD-6: Cross-Platform Deployment
- [ ] Linux: builds and installs cleanly, RPATH `$ORIGIN` set
- [ ] macOS: universal binary (x86_64 + arm64), `@loader_path` RPATH, code signed (or notarized)
- [ ] Windows: MSVC build, `.dll` deployed adjacent to plugin, UTF-8 paths handled
- [ ] **Verification:** CI matrix: Linux + macOS + Windows builds pass; plugin loads in Bitwig (Linux), REAPER (Windows), Bitwig/Logic (macOS)

### PRD-7: GUI (Deferred)
- [ ] `CLAP_EXT_GUI` implemented with parameter visualization
- [ ] ARA `EditorRenderer` / `EditorView` populated
- [ ] **Blocked on:** Training complete + JUCE/pugl integration plan

### PRD-8: Conditioning System
- [ ] 8-dim conditioning vector passed as 10th ONNX input (FiLM per layer)
- [ ] Double-buffered atomic swap for RT-safe updates from non-RT thread
- [ ] `deltaWeights` + `bWeights` loaded from config, pre-computed as `combinedWeights_`
- [ ] ARA host bridge: real musical context (tempo/key) → `setConditioning()` → plugin
- [ ] **Verification:** `test_conditioning_with_model()` proves non-zero cond changes output

---

## 3. ARD Obligations

**Architecture Decision Records** define the implementation contract. Every ADRs compliance item must hold.

### ADR-001: Shared ONNX Runtime Inference Engine
- [ ] 10-input / 3-output ONNX I/O contract never changes without updating `export_mamba_onnx.py` + engine in lockstep
- [ ] `processBlock()` uses 64-sample sub-blocks with state carry-over
- [ ] All buffers sized in `init()`, no heap allocation in RT path
- [ ] `Ort::Value` tensors created via `thread_local std::vector` + `reserve()`, not `std::array`
- [ ] ORT session: `IntraOpNumThreads=1`, `ORT_ENABLE_ALL` graph optimization
- [ ] `Ort::Exception` caught in both `init()` and `processBlock()`, never propagates across C boundary
- [ ] `libonnxruntime.so.1.17.1` deployed adjacent to plugin via `$ORIGIN` RPATH

### ADR-002: Deferred Atomic State Reset
- [ ] `resetRequested_` is `std::atomic<bool>`, set with `release` ordering, checked/cleared with `acq_rel` exchange
- [ ] Zeroing happens on audio thread, safe from any caller
- [ ] No mutex, no SPSC queue, no double-buffered state — just one atomic flag

### ADR-003: Click-Free Bypass via 50ms Linear Crossfade
- [ ] Step = `1.0 / (0.05 * sample_rate)`, computed once at `activate()`
- [ ] Step accumulated outside channel loop (same `bypassVal` for all channels per sample)
- [ ] 50ms duration regardless of sample rate (step adapts to sample rate)

### ADR-004: CLAP + ARA Dual-Target Build
- [ ] `MambaCore` static library, linked into both plugin targets
- [ ] ARA PlaybackRenderer does NOT share engine instance with CLAP (separate instances allowed)
- [ ] Cross-platform install paths: Linux `/usr/local/lib/clap`, macOS `/Library/Audio/Plug-Ins/CLAP/`, Windows `C:/Program Files/Common Files/CLAP/`

### ADR-005: thread_local + reserve() for Ort::Value
- [ ] `inputTensors_` and `outputTensors_` are `thread_local std::vector<Ort::Value>`
- [ ] Both `reserve(kNumInputs)` / `reserve(kNumOutputs)` called before use
- [ ] `Ort::Value` never stored as class member or `std::array`

### ADR-006: dladdr Model Path Resolution
- [ ] `mambaResolveModulePath()` uses `dladdr` on Unix, `GetModuleFileNameW` on Windows
- [ ] Model path resolved relative to plugin binary location → `models/mamba_rt.onnx`
- [ ] Cached in `static` local to avoid repeated syscalls

### ADR-007: Cross-Platform Portability
- [ ] `Platform.hpp` with `MAMBA_EXPORT`, platform guards for all platform-specific code
- [ ] `CMakePresets.json` with Linux/macOS/Windows presets
- [ ] CI matrix: Linux + macOS + Windows build + test
- [ ] `HSA_OVERRIDE_GFX_VERSION=11.0.0` for AMD ROCm (Strix Point / 860M)

### ADR-008: Conditioning Vector (8-dim FiLM + External Gain)
- [ ] `COND_DIM = 8`, `setConditioning()` uses double-buffered atomic pointer swap
- [ ] FiLM: cond passed as 10th ONNX input, `CondFiLM` per layer modulates SSM features
- [ ] External gain: `condScale = 1.0 + sum(cond[d] * combinedWeights_[d])` applied post-ONNX
- [ ] `loadCondWeights()` called from `init()`, computes `combinedWeights_` once

### ADR-009: 64-Sample Sub-Block Processing
- [ ] `kSubBlockSize = 64`, each ONNX inference call gets exactly 64 samples (zero-padded if needed)
- [ ] Partial final block handled: `memset` padding for trailing samples
- [ ] Conv state prepended from rolling buffer, SSM state carried from previous block

### ADR-010: Dual-Path Conditioning
- [ ] Path A: ONNX input 9 (cond) → FiLM per layer → modulates SSM features
- [ ] Path B: `combinedWeights_` scalar gain applied post-ONNX inference
- [ ] Both read same `condFrame` snapshot (no TOCTOU within a block)

---

## 4. Training Pipeline

### 4.1 Checkpoint Status

| Checkpoint | Step | Status |
|-----------|------|--------|
| `best.pt` | 49,200 | **Use this** — best validation loss |
| `checkpoint_049800.pt` | 49,800 | Final regular checkpoint |
| `interrupt_045825.pt` | 45,825 | Interrupted run |
| `final.pt` | 49,200 | Final, from completed run |

Training ran at ~3.5 step/s on AMD 860M ROCm. Steps 0 → 49,200 completed.

### 4.2 Training Script Architecture

**`train_clara_now.py`** (canonical script — USE THIS):
- STFT-based Mamba (matches mamba3 proven architecture)
- `STFTMambaModel`: audio → STFT → proj → Mamba×4 → decode → ISTFT → audio
- `chunk_size=64` for SSM scan (numerically stable, matches ONNX export)
- `d_model=64, d_state=16, n_layers=4, conv_width=3, dt_rank=16`
- Loss: STFT magnitude L1 + Mel-L1 + SI-SDR (ramped in after step 5,000)
- AMP: **OFF BY DEFAULT** (bfloat16 NaN on 860M ROCm)

**`export_mamba_onnx.py`** (ONNX export):
- Uses `MambaRTModel` (the sequential export model, not `STFTMambaModel`)
- Different architecture from training model — **separate export model class**
- 10 inputs (audio_chunk + 4×conv + 4×ssm + cond) → 3 outputs (processed_audio + stacked conv + stacked ssm)
- seq_len=256 default for export

**`train_clara_model.py`**:
- `MambaRTModelParallel` — parallel scan for training
- `MambaRTModelSequential` — step-by-step for ONNX export
- `make_zero_states()` — creates conv/ssm initial states

### 4.3 Resume + Continue Training

```bash
# Resume from best.pt, continue to step 50,000 (identity phase complete)
python train_clara_now.py \
  --data /path/to/musdb18hq/train \
  --resume checkpoints_clara/best.pt \
  --steps 50000 \
  --batch 2 \
  --grad-accum 8 \
  --no-amp \
  --clip-seconds 0.5 \
  --log-every 25 \
  --ckpt-every 200
```

**Flags to add to `train_clara_now.py`:**

| Flag | Default | Description |
|------|---------|-------------|
| `--cache` | off | RAM pre-cache entire dataset (~20GB for 500 files) |
| `--cache-chunked` | off | NEW: Chunked RAM caching — preload in segments to avoid OOM |
| `--num-workers` | 2 | DataLoader workers (forced 0 on Windows) — set to 4 for Linux ROCm |
| `--steps` | 50,000 | Total steps |

**Linux ROCm recommended command:**
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HSA_ENABLE_SDMA=0
python train_clara_now.py \
  --data /path/to/musdb18hq/train \
  --resume checkpoints_clara/best.pt \
  --steps 50000 \
  --batch 2 \
  --grad-accum 8 \
  --num-workers 4 \
  --cache \
  --cache-chunked \
  --no-amp \
  --clip-seconds 0.5
```

### 4.4 Chunked RAM Caching Implementation

The `--cache` flag pre-loads all files into RAM before training. For large datasets this can OOM. The new `--cache-chunked` flag should:

1. Load files in chunks of N files (e.g., N=50)
2. Keep only the current chunk in RAM, release previous chunk
3. Pre-fetch next chunk in background thread during training epoch
4. Alternatively: use `memory-mapped files` (`np.memmap` or `torch.utils.data.TensorMmapDataset`)

**Implementation approach:**
```python
class ChunkedRamCache:
    """Loads dataset in chunks, keeps chunk_count chunks in RAM."""

    def __init__(self, root, chunk_size=50, num_chunks=2):
        # chunk_size: files per chunk
        # num_chunks: chunks to keep in RAM (1=current, 1=next)
        self.files = sorted(Path(root).rglob("*.wav"))
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.cache = {}  # chunk_idx -> {filepath: tensor}
        self._load_chunk(0)

    def _load_chunk(self, idx):
        if idx in self.cache:
            return
        start = idx * self.chunk_size
        end = min(start + self.chunk_size, len(self.files))
        self.cache[idx] = {}
        for f in self.files[start:end]:
            self.cache[idx][str(f)] = self._load_file(f)

    def get(self, filepath):
        for cache in self.cache.values():
            if filepath in cache:
                return cache[filepath]
        return self._load_file(Path(filepath))  # fallback
```

### 4.5 Phase 2: Contrastive Noise Training

**UPDATE 2026-06-12:** The noise dataset is **LOCAL ONLY, PROPRIETARY**. It is NOT on GitHub and must NEVER be committed. Verify `.gitignore` blocks all audio files before any push.

**Noise dataset contents (LOCAL, UNPUBLISHED):**
- Suno-generated audio (various quality levels, artifacts)
- Personal recordings (proprietary, low quality)
- Live band recordings: Coore's Event Centre, Saskatoon (line array PA recordings)

**Dataset path:** User must configure. Add to `train_clara_now.py`:
```bash
# Example local paths — REPLACE with actual paths:
--clean-data /path/to/clean/reference/audio
--noise-data /path/to/noise/dataset  # Suno + recordings + live bands
```

**Goal:** Train CLARA to output clean audio when given noisy input — denoising.

**Architecture for Phase 2:**
- `train_clara_model.py` → `MambaRTModelParallel` / `MambaRTModelSequential` already support this
- Dataset: (clean, noisy) pairs from real-world noise dataset
- Loss: SI-SDR(clean_pred, clean_target) — scale-invariant to match Phase 1

**IMPORTANT:** Do NOT add noise dataset paths to any committed files. Use command-line args or environment variables for local paths.

### 4.6 ONNX Export After Training

```bash
# Export best checkpoint to ONNX
python export_mamba_onnx.py \
  --checkpoint checkpoints_clara/best.pt \
  --output models/mamba_rt_trained.onnx \
  --seq-len 256

# Verify output
python -c "
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession('models/mamba_rt_trained.onnx')
# Run inference, check output variance
"
```

---

## 5. Verification Plan

### 5.1 PRD Verification Tests

| Test | Method | Pass Criteria |
|------|--------|---------------|
| V1: RT safety | `rg "malloc\|free\|new\|delete\|mutex\.lock" src/` | Zero real-time-unsafe calls found |
| V2: Latency | Impulse response test in REAPER | First non-zero output within 4 samples of input |
| V3: Bypass click-free | Inject 1kHz sine, toggle bypass | No visible transient on scope |
| V4: CLAP validator | `clap-validator` CTest target | Pass required categories (audio-ports, params, state, latency) |
| V5: ONNX output variance | Random input, compare trained vs random weights | Trained model output variance >> random model variance |
| V6: Loss improvement | Load best.pt, run verify script | Loss curve shows decreasing trend |
| V7: State continuity | ONNX inference on 2 consecutive chunks | Second chunk output differs from first (state is being used) |
| V8: ARA archive round-trip | Save/archive/load in REAPER | All 3 params (bypass, wetDry, intensity) restored correctly |
| V9: Conditioning effect | Run with zero cond vs non-zero cond | Output differs measurably |
| V10: Cross-platform build | CI Linux + macOS + Windows | All 3 build + pass tests |

### 5.2 Existing Test Infrastructure

| Test File | Coverage | Status |
|-----------|----------|--------|
| `tests/test_engine.cpp` | 23 tests: construct/destroy, init, processBlock (256/64/0/512), reset, stereo, conditioning, partial block, state carry-over | ✅ All pass |
| `tests/test_clap_plugin.cpp` | 10 tests: SPSC, param flush, bypass, state save/load | ✅ Pass |
| `tests/test_ara_controller.cpp` | 4 tests: model path resolution, init | ✅ Pass |
| `verify_training_pipeline.py` | 56 checks: config, model forward/backward, optimizer, ONNX export, losses, DSP, dataset | ✅ All PASS |

### 5.3 Missing Verification Tests

| Test | File | Priority | Description |
|------|------|----------|-------------|
| T1: Latency impulse | `tests/test_latency.cpp` | P0 | Inject impulse, measure first output sample delay at 48kHz |
| T2: ONNX trained vs random | `tests/test_onnx_quality.cpp` | P0 | Prove trained checkpoint output ≠ random output |
| T3: State continuity ONNX | `tests/test_state_continuity.cpp` | P0 | Process 2 ONNX chunks consecutively, verify state carried |
| T4: CLAP validator CI | `.github/workflows/ci.yml` | P1 | Full validator run in CI (not continue-on-error) |
| T5: Conditioning with non-zero | `tests/test_conditioning_nonzero.cpp` | P1 | Verify non-zero cond produces different output than zero cond |
| T6: Bypass crossfade timing | `tests/test_bypass.cpp` | P1 | Verify 50ms crossfade math at 48kHz, 96kHz |
| T7: ARA archive versioning | `tests/test_ara_archive.cpp` | P1 | Save old archive format, load in new version — graceful handle |
| T8: Chunked cache stability | `tests/test_chunked_cache.cpp` | P2 | RAM cache with 2+ chunks, no OOM, no file descriptor leaks |
| T9: ROCm GPU warmup | `tests/test_gpu_warmup.cpp` | P2 | First 3 forward passes on GPU don't produce NaN |
| T10: Stereo correlation | `tests/test_stereo.cpp` | P2 | Left/right channels don't drift during bypass crossfade |

---

## 6. C++ Source File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `src/core/MambaInferenceEngine.hpp` | 109 | Class interface, Config struct, Ort forward-declares |
| `src/core/MambaInferenceEngine.cpp` | 318 | init, processBlock (64-sample sub-block), conditioning, manifest |
| `src/core/ConditioningFrame.h` | ~100 | Double-buffered conditioning frame with atomic seq counter |
| `src/core/ConditioningGenerator.hpp` | ~80 | ARA → cond encoding (tempo/key/chord → 8-dim vector) |
| `src/core/Platform.hpp` | ~100 | Cross-platform: MAMBA_EXPORT, dladdr, GetModuleFileName, UTF-8 |
| `src/core/SPSCQueue.hpp` | ~150 | Lock-free single-producer single-consumer queue |
| `src/clap/MambaClapPlugin.hpp` | ~200 | CLAP plugin class, DeferredTask enum, voice state |
| `src/clap/MambaClapPlugin.cpp` | ~500 | Full CLAP implementation: process, params, bypass crossfade |
| `src/ara/MambaAraController.hpp` | ~200 | ARA DocumentController + PlaybackRenderer declarations |
| `src/ara/MambaAraController.cpp` | ~400 | Full ARA implementation: archive, enable/disable, content |

**Total: ~2157 lines C++**

---

## 7. Python File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `train_clara_now.py` | 803 | **Canonical training script** — STFT Mamba, chunked scan |
| `train_clara_model.py` | ~400 | `MambaRTModelParallel` + `MambaRTModelSequential` + `make_zero_states` |
| `train_clara_export.py` | ~200 | ONNX export from checkpoint |
| `export_mamba_onnx.py` | 754 | Standalone ONNX export (separate export model class) |
| `train_clara_config.py` | ~150 | `ClaraTrainingConfig` dataclass with phase presets |
| `train_clara_dataset.py` | ~250 | `ClaraAudioDataset` with MUSDB/MAESTRO/DNS loading, RAM cache |
| `train_clara_dsp.py` | ~200 | `apply_saturation`, `RMSCompressor`, `add_noise`, TRANSFORM_REGISTRY |
| `train_clara_loss.py` | ~200 | `MultiResolutionSTFTLoss`, `SI_SDRLoss`, `ClaraLoss` |
| `verify_training_pipeline.py` | 413 | 56-check end-to-end verification |
| `__train_laptop_ref.py` | ~800 | Reference mamba3 training script (do not modify) |

**Total: ~4170 lines Python**

---

## 8. Full Prioritized Task List

### 🔴 P0 — Critical (Must do before first user-facing release)

| # | Task | Details | Status |
|---|------|---------|--------|
| T-1 | **Add `--cache-chunked` flag** | Chunked RAM caching to avoid OOM. Load N files, keep M chunks, prefetch next. See Section 4.4. | ⬜ Pending |
| T-2 | **Add `--num-workers` flag to `train_clara_now.py`** | Set to 4 for Linux ROCm. Auto-force 0 on Windows. | ⬜ Pending |
| T-3 | **Verify `best.pt` is valid checkpoint** | Run: `python -c "import torch; ckpt=torch.load('checkpoints_clara/best.pt', weights_only=False); print(ckpt.keys(), ckpt.get('step'))"` | ⬜ Pending |
| T-4 | **Resume training from `best.pt` to step 50,000** | Run identity phase to completion (step 49,200→50,000 is only 800 steps) | ⬜ Pending |
| T-5 | **Export `best.pt` → ONNX** | `python export_mamba_onnx.py --checkpoint checkpoints_clara/best.pt --output models/mamba_rt_trained.onnx` | ⬜ Pending |
| T-6 | **Verify trained ONNX output quality** | Compare variance/cosine similarity: trained vs random. Pass criteria: cosine sim < 0.95 for different audio inputs | ⬜ Pending |
| T-7 | **Copy trained ONNX to `models/mamba_rt.onnx`** | Backup existing, replace with trained version, update manifest | ⬜ Pending |
| T-8 | **Build plugin with trained ONNX** | `cmake --build build --config Release`, verify no link errors | ⬜ Pending |
| T-9 | **Run existing test suite** | `ctest --test-dir build --output-on-failure` — all must pass | ⬜ Pending |
| T-10 | **Run `verify_training_pipeline.py`** | All 56 checks must PASS | ⬜ Pending |

### 🟠 P1 — High Priority (Before shipping)

| # | Task | Details | Status |
|---|------|---------|--------|
| T-11 | **Implement Phase 2 contrastive noise training** | Continue from best.pt, add noise augmentation, SI-SDR loss (see Section 4.5) | ⬜ Pending |
| T-12 | **Write `test_onnx_quality.cpp`** | Load trained ONNX vs random ONNX, prove trained produces meaningfully different output | ⬜ Pending |
| T-13 | **Write `test_state_continuity.cpp`** | Process 2 ONNX chunks, prove state carries (2nd chunk output differs from 1st with same input) | ⬜ Pending |
| T-14 | **Write `test_latency.cpp`** | Impulse response at 48kHz, prove 4-sample latency | ⬜ Pending |
| T-15 | **Run `clap-validator` in CI properly** | Currently `continue-on-error: true` — fix so failures block merge | ⬜ Pending |
| T-16 | **Fix clap-validator failures** | Root cause: model path resolution when validator runs from wrong CWD | ⬜ Pending |
| T-17 | **Verify conditioning effect** | Run engine with zero cond vs non-zero cond, measure output difference | ⬜ Pending |
| T-18 | **Test ARA archive versioning** | Save archive with v1 format, verify graceful handling on load | ⬜ Pending |
| T-19 | **GPU warmup test** | First 3 `processBlock()` calls on GPU produce finite (non-NaN) output | ⬜ Pending |
| T-20 | **Verify ROCm environment** | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` — must show 860M | ⬜ Pending |

### 🟡 P2 — Important (Before beta)

| # | Task | Details | Status |
|---|------|---------|--------|
| T-21 | **Implement real ARA musical context extraction** | `doUpdateMusicalContextContent()` → extract tempo/key → `setConditioning()` | ⬜ Pending |
| T-22 | **Populate EditorRenderer / EditorView stubs** | Currently empty shells — implement ARA visual editor interface | ⬜ Pending |
| T-23 | **Fix Windows CI** | MSVC build, symbol export verification, .dll deployment | ⬜ Pending |
| T-24 | **Fix macOS CI** | Universal binary, code signing, `clap-validator` on macOS | ⬜ Pending |
| T-25 | **Upgrade ONNX Runtime 1.17.1 → 1.21.0** | 4 versions behind, security + perf fixes available | ⬜ Pending |
| T-26 | **Write `test_stereo.cpp`** | Verify left/right bypass crossfade doesn't drift | ⬜ Pending |
| T-27 | **Write `test_chunked_cache.cpp`** | RAM cache stability test with multiple chunks, file descriptor check | ⬜ Pending |
| T-28 | **Implement `--cache-chunked` prefetch thread** | Background thread loads next chunk while current is being consumed | ⬜ Pending |

### 🟢 P3 — Polish (Before production release)

| # | Task | Details | Status |
|---|------|---------|---------|
| T-29 | **Implement GUI (`CLAP_EXT_GUI`)** | JUCE or pugl-based parameter visualization | ⬜ Pending |
| T-30 | **Implement ARA EditorRenderer view** | Show analysis waveform / conditioning visualization | ⬜ Pending |
| T-31 | **CPack packaging** | `.deb` for Linux, `.pkg` for macOS, installer for Windows | ⬜ Pending |
| T-32 | **Code coverage** | `gcov`/`lcov` step in CI | ⬜ Pending |
| T-33 | **Sanitizer builds** | ASan/UBSan/TSan in CI (separate build config) | ⬜ Pending |
| T-34 | **ONNX Runtime CPU provider tuning** | `OrtThreadingHints` for performance | ⬜ Pending |
| T-35 | **INT8 quantization** | `onnxruntime.perdoder` for ~2× speedup on CPU | ⬜ Pending |

### 🔵 P4 — Future (Post-v1)

| # | Task | Status |
|---|------|--------|
| T-36 | Phase 3 compression training | ⬜ Pending |
| T-37 | Phase 4 denoising (DNS Challenge dataset) | ⬜ Pending |
| T-38 | Multi-effect training (EQ, transient, de-essing) | ⬜ Pending |
| T-39 | CUDA/TensorRT execution provider | ⬜ Pending |
| T-40 | ARA ContentAnalyzer class for audio analysis | ⬜ Pending |

---

## 9. Immediate Next Actions (Tonight)

```
1. Verify best.pt:          python -c "import torch; print(torch.load('checkpoints_clara/best.pt', weights_only=False).get('step'))"
2. Add --cache-chunked flag to train_clara_now.py
3. Add --num-workers flag to train_clara_now.py
4. Resume training:         python train_clara_now.py --data /path/to/musdb18hq/train --resume checkpoints_clara/best.pt --steps 50000 --batch 2 --grad-accum 8 --num-workers 4 --cache --cache-chunked --no-amp --clip-seconds 0.5
5. Export ONNX:              python export_mamba_onnx.py --checkpoint checkpoints_clara/best.pt --output models/mamba_rt_trained.onnx
6. Run verify script:       python verify_training_pipeline.py
7. Build plugin:            cmake --build build --config Release
8. Run C++ tests:           ctest --test-dir build --output-on-failure
```

---

## 10. File Locations

| Resource | Path |
|----------|------|
| Project | `/home/sgm/Github Repo's/clara/` |
| Training script | `train_clara_now.py` |
| ONNX export | `export_mamba_onnx.py` |
| Training model | `train_clara_model.py` (MambaRTModelParallel/Sequential) |
| Checkpoints | `checkpoints_clara/` (248 files, best.pt at step 49,200) |
| ONNX models | `models/` (multiple variants: mamba_rt.onnx, mamba_rt_1x1_complete.onnx, etc.) |
| C++ source | `src/core/MambaInferenceEngine.{hpp,cpp}`, `src/clap/`, `src/ara/` |
| Tests | `tests/` (test_engine.cpp, test_clap_plugin.cpp, test_ara_controller.cpp) |
| ADRs | `ADR-00*.md` (10 ADRs) |
| Build | `CMakeLists.txt`, `CMakePresets.json` |
| GitHub | `https://github.com/SGM-Studios/clara` |
| MUSDB18-HQ | `C:/dev/mamba3/mamba3/datasets/musdb18hq/train/` (Windows) |
| Canonical Python (ROCm) | `C:/dev/mamba3/mamba3/venv/Scripts/python.exe` (Windows) |

---

*Last updated: 2026-06-12*