# CLARA — Mamba Audio Suite (CLAP + ARA2)

**Migration status (July 2026): Phases 0-6 COMPLETE.** Both CLAP and ARA plugins run on the custom C++ engine. C++ engine is being updated to R=8 PerTrackProcessor (H3). Training pipeline uses PerTrackProcessor R=8 spectral-domain. No ONNX Runtime dependency.

**Current build:** `cmake -B build -DBUILD_TESTING=ON && cmake --build build --parallel && ctest --test-dir build` — 3/3 tests pass.

**Migration plan:** `docs/MAMBA3_MIGRATION_PLAN.md` — complete architecture, model config, C++ engine spec, training pipeline, product roadmap, risk register, and phase-by-phase implementation plan.

**Architecture reference:** See `docs/TRAINING_SETUP_CONTEXT.md` for PerTrackProcessor architecture diagram and `docs/TRAINING.md` for training pipeline.
**Training launch:** See `scripts/bootstrap_local.sh` for ROCm local training workflow.

C++17 audio plugin running a PerTrackProcessor (Mamba-3 SSM with multi-head MIMO R=8, spectral-domain, complex poles) via custom C++ inference engine (no ONNX Runtime). Supports Linux (stable), macOS and Windows (in progress). Produces `MambaSuiteClap.clap` (per-effect model variants planned for Phase 2+).

**Key architecture decisions:**
- PerTrackProcessor R=8 (8 spectral bands: sub_bass, bass, low_mid, mid, high_mid, air, transient, centroid)
- d_model=128, d_state=32, 6 layers, ~4.26M total params (~3.9M mamba core)
- Complex-pole trapezoidal SSM (no RoPE in PerTrackProcessor)
- STFT frontend + ISTFT decoder (n_fft=2048, hop_length=512, latency=n_fft/2)
- CREPE f0 conditioning, cross-track attention, semantic encoder, CondMLP
- Custom C++ engine with flat binary weights V3 format (18 arrays/layer) — no ONNX Runtime
- CLAP text + audio encoders shipped with plugin (Option C, full quality, no distillation) — Phase 2/3 feature
- Precomputed CLAP preset embeddings for instant preset loading — Phase 2/3 feature
- CLAP audio analysis on transport stop — plugin "listens" between takes — Phase 2/3 feature
- Per-effect model variants (Saturator, Compressor, Denoiser, etc.) — each its own CLAP plugin (planned)
- Training on MI300X via AMD AI Developer Cloud ($100 credits, ~50 hours)
- Python reference forward pass written BEFORE any C++ (prevents Python↔C++ divergence)
- Multi-stage training: SSL pretraining → supervised per-effect → adversarial → RLHF (later)

See `ADR-001` through `ADR-010` in `docs/adr/` for architectural decisions and `docs/adr/ADR-WORKFLOW.md` for the ADR drafting backlog and system prompts.

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

With tests:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build . --parallel && ctest --output-on-failure
```

Cross-platform builds via CMake presets:
```bash
cmake --preset default             # Linux (Release)
cmake --preset macos-release       # macOS
cmake --preset windows-msvc-release  # Windows
cmake --build --preset <preset>
```

Install (default prefix `/usr/local`):
```bash
cmake --install .          # both CLAP + ARA components
cmake --install . --component clap   # CLAP bundle + bare .clap only
cmake --install . --component ara    # ARA .so only
```

Outputs go to `build/plugins/`:
- `MambaSuiteClap.clap` — CLAP audio effect
- `MambaSuiteARA.so` — ARA2 library
- `libMamba3Core.a` — static C++ engine (no ONNX Runtime dependency)
- `models/*.bin` — flat binary weights copied from source

Two dependencies via CMake FetchContent (no manual install):
- CLAP SDK (`29ffcc27`)
- ARA SDK (`1ca0b0aa`)

ONNX Runtime is REMOVED. All inference is custom C++ with flat binary weights.

## Model Export (V3 — PerTrackProcessor)

The **canonical** training pipeline is `python/training/train_clara.py` which trains `PerTrackProcessor` (~4.26M params, spectral-domain, Mamba-3 multi-head MIMO R=8 with complex poles).

```bash
# Train
python python/training/train_clara.py --data ~/datasets/musdb18hq/train --device cuda --steps 100000

# Export to flat binary weights (V3 format, 18 arrays/layer)
python python/training/train_clara.py --export checkpoints/final.pt --output models/model.bin
```

Exports `models/model.bin` using the 18-array version 3 weight format with complex pole parameters (`log_a_real`, `a_imag`) and R8 extractor weights. The C++ engine is being updated to match (see H3 in `docs/TODO.md`). Requires `torch`, `numpy`.

**Older Mamba-1 scripts (`train_clara_parallel.py`, `train_clara_export.py`, `export_mamba_onnx.py`, `train_clara_now.py`) are ALL DEAD — incompatible architectures.**
**Legacy R=4 time-domain script:** `python/training/train_local_radeon.py` — not the canonical pipeline.

## Architecture

`src/core/SsmInferenceCore` — RT-safe custom C++ PerTrackProcessor engine (being updated to R=8). Currently 6 `SSMBlock` layers (stack-allocated) at R=4; being migrated to PerBandSSM with complex-pole trapezoidal SSM and R=8 spectral bands. No ONNX Runtime dependency. Uses stack-allocated `alignas(64)` state arrays with `#pragma unroll` for zero heap allocations. Deferred state reset via `resetRequested_` atomic flag. Both CLAP and ARA link the `Mamba3Core` static library.

`src/clap/MambaClapPlugin` — Full CLAP plugin. Registers extensions: params, state, latency (n_fft/2 samples — STFT window), audio-ports (stereo). 3 atomics: bypass (stepped), wet/dry, intensity. Click-free bypass via 50ms linear crossfade (per-sample ramp, shared across channels to prevent stereo aliasing). Uses `mambaResolveModulePath()` (Platform.hpp) for model path resolution relative to `.clap` binary. Uses `std::make_unique` + `.release()` for C-ABI lifetime.

`src/ara/MambaAraController` — DocumentController + PlaybackRenderer + `extern "C" ARAGetFactory()` with `visibility("default")`. ARA 2.0 has NO `renderSamples()` — audio processing is exclusively through the companion CLAP plugin. Content stubs documented with proper `enable` handling in `willEnableAudioSourceSamplesAccess`. Archive state persists 3 floats (bypass, wetDry, intensity) via `HostArchiveReader/Writer`.

`src/core/Platform.hpp` — cross-platform abstractions for module path resolution (`mambaResolveModulePath()`), symbol visibility (`MAMBA_EXPORT`/`MAMBA_IMPORT`), and thread naming. Each target platform (Linux, macOS, Windows) provides a conforming implementation via `#ifdef` guards.

Flat binary model: 1 input file (`model.bin`, version 3) loaded via mmap at init. The engine reads 18 weight arrays per layer (including complex pole parameters and R8 extractor weights) and constructs PerBandSSM objects with placement-new. State is stack-allocated — no heap allocations in audio thread.

Conditioning vector `cond[COND_DIM]` (`COND_DIM=128`): f0 projection (slots 0-31 from CREPE), semantic embedding (32-159 from SemanticEncoder), cross-track attention (160-223), and loudness features (224-227). CondMLP combines these into per-layer FiLM conditioning. Double-buffered with atomic pointer swap for RT-safe updates from non-RT threads.

`src/core/SPSCQueue.hpp` — fixed-capacity lock-free SPSC ring buffer for deferred task communication between audio thread and main thread. Capacity 64 entries, supports push from any thread, pop from main thread. Used for `ModelReload` and `AnalysisQueueDrain` deferred tasks.

## C++ Conventions

- C++17, no RTTI in release (`-ffast-math -O3 -DNDEBUG -fno-rtti`)
- RT-safe: no allocations on audio thread — state lives in stack-allocated `alignas(64)` arrays; `resetState()` uses deferred atomic clear
- `-Wall -Wextra -Werror=return-type` always on
- Symbols hidden (`CXX_VISIBILITY_PRESET hidden`), `$ORIGIN` RPATH only (no build-tree leak via `BUILD_WITH_INSTALL_RPATH ON`)
- `init()` returns bool (never throws); callers use `if (!engine->init())` pattern
- Release builds stripped via `LINK_FLAGS "-s"`

## Code Quality

Pre-commit hooks enforce consistent formatting and catch common issues:

```bash
pre-commit install              # one-time setup
pre-commit run --all-files      # run on all tracked files
```

Hooks: `clang-format` (C++, `--style=file` via `.clang-format`), `black` (Python, 100-char), `ruff` (Python lint, auto-fix), `trailing-whitespace`, `end-of-file-fixer`, `mixed-line-ending`, `check-added-large-files`, `check-merge-conflict`, `check-case-conflict`.

C++ formatting (`.clang-format`): LLVM style with 4-space indent, 120-column limit, right-aligned pointers/references, C++17.

## CI/CD

GitHub Actions workflow at `.github/workflows/ci.yml`:
- Builds Release on `ubuntu-22.04`, `macos-14`, and `windows-2022` with both FetchContent dependencies
- Verifies symbol exports (`clap_entry`, `ARAGetFactory`), RPATH / install name / DLL search path, model deployment
- Runs `clap-validator` against built `.clap` (continues on error — validator may fail without GUI)
- Validates model round-trip via `python3 python/tools/validate_export.py`
- Uploads plugin artifacts (`.clap`, `.so`/`.dylib`/`.dll`, `models/`)
- Test job builds Debug on all three platforms with `BUILD_TESTING=ON` and runs CTest

## Verification

```bash
# Build and test (requires model files in models/)
cmake -B build -DBUILD_TESTING=ON && cmake --build build --parallel && ctest --test-dir build --output-on-failure
```

Tests cover: construct/destroy lifecycle, `init()` with model path, `processBlock()` with silence input (checking finite output), `resetState()` thread safety (deferred clear), Python↔C++ numerical equivalence (bit-exact), and long-run stability (60s @ 44.1kHz with no NaN/Inf). Python model export includes inline validation. For runtime testing, load `build/plugins/MambaSuiteClap.clap` in a CLAP host (Bitwig, Carla, clap-host); models must be in `models/` adjacent to the plugin binary.

## Known Gaps

See `docs/MAMBA3_MIGRATION_PLAN.md` for the complete phased migration plan. Quick reference (updated 2026-06-16):

| Gap | Phase | Severity | Status |
|-----|-------|----------|--------|
| G1 | P0 | ✅ Done | `python/training/train_clara.py` implemented and vectorized; needs cloud training run |
| G2 | P0 | ✅ Done | CMake build fixed — no ONNX Runtime, `Mamba3Core` static library |
| G5 | P1 | 🟡 High | No DAW load test — engine works in tests, needs DAW validation |
| G6 | P2 | 🟡 High | GUI (CLAP_EXT_GUI) — JUCE or pugl needed |
| G7 | P2 | 🟡 Low | ARA conditioning — `doUpdateMusicalContextContent()` encodes default context to 128-dim cond. Real host reading deferred. |
| G8 | P2 | 🟡 Low | ARA analysis — `contentDirty_` flag set, SPSC AnalysisQueueDrain stub ready |
| G9 | P3 | 🔴 Blocking | **Training pipeline** — needs `python/training/train_clara.py` + Phase 2-5 per-effect training |
| G10 | P4 | 🟢 Med | ONNX manifest enforcement — will be removed (ONNX is dead) |
| G12 | P4 | 🟡 High | macOS/Windows builds — defined in CMake but untested |
| G13 | P4 | 🟢 Med | Install/packaging — no DEB, distrobox-export, or install script |

**Training pipeline status:** `python/training/train_clara.py` exists and matches the Mamba-3 architecture. Next step is a cloud training run on MI300X. Needs MUSDB18-HQ dataset (~100K steps, ~4-8 hr on MI300X via AMD AI Developer Cloud).

## Audit & Documentation Files

| File | Purpose |
|------|---------|
| `docs/MAMBA3_DRIVE_AUDIT.md` | Full audit of findings from mamba3 external drive — what was transferred, what wasn't, and why |
| `docs/ARCHITECTURE_COMPARISON.md` | Clara Mamba-3 vs mamba3 drive PerTrackProcessor — architecture, dimensions, weight format, compatibility |
| `docs/PHASE2_ROADMAP.md` | Phase 2 conditioning work items — f0, SI-SDR, StemDataset, CLAP encoders, dependency graph |

## New Scripts (Phase 1)

| Script | Purpose |
|--------|---------|
| `python/tools/test_model.py` | 7-section smoke test: ROCm env, model build, forward, bf16 AMP, backward, memory, chunked equivalence |
| `scripts/run_test.sh` | Smoke test launcher with ROCm env config (activates `venv_rocm`, sets `HSA_OVERRIDE_GFX_VERSION`) |
| `scripts/setup_local.sh` | Local Ubuntu environment setup — creates `venv_rocm`, installs ROCm PyTorch + deps |

## Session Logging — Mandatory at Session Close

A `SESSION_LOG.md` file is maintained in this project's root directory. It is an
**append-only log** of every OpenCode session. At the END OF EVERY SESSION (when
the user says goodbye, wrap up, or when no more tasks remain), the agent MUST:

1. Load the `session-log` skill (`skill` tool)
2. Use it to write a new entry to `SESSION_LOG.md`
3. Update `AGENTS.md` if the logging section is missing
4. Save key findings to agentmemory

The user should be able to start a session by reading ONLY `SESSION_LOG.md` —
no need to ask "what did we do last time?".

## Key Design Constraints

- All SSM state is stack-allocated with `alignas(64)` — zero heap allocations on audio thread
- `ARA::PlugIn::PlugInEntry::getPlugInEntry<F, D>()` creates static singletons — cannot inject custom constructor args
- ARA 2.0 `PlaybackRendererInterface` has no `renderSamples()` method; audio processing is companion CLAP plugin only
- Plugins resolve model path via `mambaResolveModulePath()` (Platform.hpp abstraction over `dladdr`/`GetModuleFileName`/`_NSGetExecutablePath`) relative to their own binary; models must be in `models/` subdirectory adjacent to plugin
- `MAMBA_EXPORT` macro (Platform.hpp) handles symbol visibility across platforms: ELF hidden/visibility(default), DLL export, macOS weak import
- Flat binary weights (`model.bin`, version 3) loaded via mmap at init — no ONNX Runtime, no model parsing at runtime
- CLAP text + audio encoders are a future Phase 2/3 feature; current conditioning uses a 128-dim `cond` vector with slots reserved for DAW context, semantic embedding, and user knobs
