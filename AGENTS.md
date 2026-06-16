# CLARA — Mamba Audio Suite (CLAP + ARA2)

**Migration status (June 2026): Phases 0-6 COMPLETE.** Both CLAP and ARA plugins run on the custom C++ Mamba-3 engine. No ONNX Runtime dependency in plugins. Training pipeline (Phase 8) is the next blocking deliverable.

**Current build:** `cmake -B build -DBUILD_TESTING=ON && cmake --build build --parallel && ctest --test-dir build` — 3/5 tests pass (engine/latency tests are ONNX-era, expected failures).

**Migration plan:** `MAMBA3_MIGRATION_PLAN.md` — complete architecture, model config, C++ engine spec, training pipeline, product roadmap, risk register, and phase-by-phase implementation plan.

**Canonical architecture:** Load the `clara-architect` skill (`skill_view('clara-architect')`) — replaces all stale prompts.
**Training launch:** Load the `clara-training-launch` skill (`skill_view('clara-training-launch')`) for ROCm + tmux training workflow.

C++17 audio plugin running a Mamba-3 SSM via custom C++ inference engine (no ONNX Runtime). Supports Linux (stable), macOS and Windows (in progress). Produces dedicated CLAP plugins (one per effect model) sharing a common ARA2 library and inference core.

**Key architecture decisions (from MAMBA3_MIGRATION_PLAN.md):**
- Mamba-3 with multi-head MIMO R=4 (4 heads × 4 slots = 16 SSM streams per layer)
- d_model=128, d_state=32, 6 layers, ~757K params
- Custom C++ engine with flat binary weights (no ONNX, no onnxruntime)
- CLAP text + audio encoders shipped with plugin (Option C, full quality, no distillation)
- Precomputed CLAP preset embeddings for instant preset loading
- CLAP audio analysis on transport stop — plugin "listens" between takes
- Dedicated model per task (Saturator, Compressor, Denoiser, etc.) — each its own CLAP plugin
- Zero-latency (no conv1d), trapezoidal SSM discretization, SiLU z-gating
- Training on MI300X via AMD AI Developer Cloud ($100 credits, ~50 hours)
- Python reference forward pass written BEFORE any C++ (prevents Python↔C++ divergence)
- Multi-stage training: SSL pretraining → supervised per-effect → adversarial → RLHF (later)

See `ADR-001` through `ADR-010` in `ADR-*.md` for architectural decisions and `ADR-WORKFLOW.md` for the ADR drafting backlog and system prompts.

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
cmake --preset linux-release      # Linux
cmake --preset macos-release      # macOS
cmake --preset windows-release    # Windows
cmake --build --preset <preset>
```

Install (default prefix `/usr/local`):
```bash
cmake --install .          # both CLAP + ARA components
cmake --install . --component clap   # CLAP bundle + bare .clap only
cmake --install . --component ara    # ARA .so only
```

Outputs go to `build/plugins/`:
- `Mamba<Effect>.clap` — CLAP audio effect (one per model)
- `MambaSuiteARA.so` — ARA2 library
- `libmamba_core.so` — shared C++ engine (no ONNX Runtime dependency)
- `libmamba_clap.so` — shared CLAP text + audio encoders (Option C)
- `models/*.bin` — flat binary weights copied from source

Two dependencies via CMake FetchContent (no manual install):
- CLAP SDK (`29ffcc27`)
- ARA SDK (`1ca0b0aa`)
- ONNX Runtime is REMOVED. All inference is custom C++ with flat binary weights.

## Model Export (V2 — Correct Pipeline)

The **only** training pipeline that produces exportable weights is `train_clara_mamba3.py` which trains `Mamba3Model` (757K params, time-domain, Mamba-3 multi-head MIMO R=4). **Do NOT use older Mamba-1 scripts** — they produce incompatible architectures.

```bash
# Train (with chunked 64-sample sub-blocks matching C++ engine)
python train_clara_mamba3.py --data ~/datasets/musdb18hq/train --device cuda --steps 100000

# Export to flat binary weights
python scripts/export_mamba_weights.py --checkpoint checkpoints/final.pt --output models/model.bin
```

Exports `models/model.bin` using the 13-array version 2 weight format. The model has chunked sub-block processing (chunk_size=64) that exactly matches the C++ engine's `kSubBlockSize`. Requires `torch`, `numpy`.

**Older Mamba-1 scripts (`train_clara_parallel.py`, `train_clara_export.py`, `export_mamba_onnx.py`, `train_clara_now.py`) are ALL DEAD — incompatible architectures.**

## Architecture

`src/core/SsmInferenceCore` — RT-safe custom C++ Mamba-3 engine. Contains 6 `SSMBlock` layers (stack-allocated), each implementing multi-head MIMO R=4 with trapezoidal SSM, RoPE, FiLM conditioning, and SiLU z-gating. No ONNX Runtime dependency. Uses stack-allocated `alignas(64)` state arrays with `#pragma unroll` for zero heap allocations. Deferred state reset via `resetRequested_` atomic flag. Both CLAP and ARA link it via `libmamba_core` shared library.

`src/clap/MambaClapPlugin` — Full CLAP plugin. Registers extensions: params, state, latency (0 samples — no conv1d), audio-ports (stereo). 3 atomics: bypass (stepped), wet/dry, intensity. Click-free bypass via 50ms linear crossfade (per-sample ramp, shared across channels to prevent stereo aliasing). Uses `mambaResolveModulePath()` (Platform.hpp) for model path resolution relative to `.clap` binary. Uses `std::make_unique` + `.release()` for C-ABI lifetime.

`src/ara/MambaAraController` — DocumentController + PlaybackRenderer + `extern "C" ARAGetFactory()` with `visibility("default")`. ARA 2.0 has NO `renderSamples()` — audio processing is exclusively through the companion CLAP plugin. Content stubs documented with proper `enable` handling in `willEnableAudioSourceSamplesAccess`. Archive state persists 3 floats (bypass, wetDry, intensity) via `HostArchiveReader/Writer`.

`src/core/Platform.hpp` — cross-platform abstractions for module path resolution (`mambaResolveModulePath()`), symbol visibility (`MAMBA_EXPORT`/`MAMBA_IMPORT`), and thread naming. Each target platform (Linux, macOS, Windows) provides a conforming implementation via `#ifdef` guards.

Flat binary model: 1 input file (`model.bin`, version 2) loaded via mmap at init. The engine reads 13 weight arrays per layer and constructs SSMBlock objects with placement-new. State is 16,960 floats per layer (101,760 total across 6 layers) — all stack-allocated, no heap allocations in audio thread.

Conditioning vector `cond[COND_DIM]` (`COND_DIM=128`): slots 0-7 (DAW context from ARA), slots 8-39 (CLAP semantic embedding), slots 40-71 (user knobs), slots 72-127 (reserved). Double-buffered with atomic pointer swap for RT-safe updates from non-RT threads. FiLM conditioning applied per-layer via `film_proj_w` (γ,β per d_model channel) and `dt_film_proj_w` (per-head Δ modulation).

`src/core/SPSCQueue.hpp` — fixed-capacity lock-free SPSC ring buffer for deferred task communication between audio thread and main thread. Capacity 64 entries, supports push from any thread, pop from main thread. Used for `ModelReload` and `AnalysisQueueDrain` deferred tasks.

## C++ Conventions

- C++17, no RTTI in release (`-ffast-math -O3 -DNDEBUG -fno-rtti`); exceptions allowed (ORT C++ API `Ort::Session::Run()` throws `Ort::Exception`)
- RT-safe: no allocations on audio thread — `thread_local std::vector` with `reserve()` in `processBlock()`; `resetState()` uses deferred atomic clear
- `-Wall -Wextra -Werror=return-type` always on
- Symbols hidden (`CXX_VISIBILITY_PRESET hidden`), `$ORIGIN` RPATH only (no build-tree leak via `BUILD_WITH_INSTALL_RPATH ON`)
- `init()` returns bool (never throws); callers use `if (!engine->init())` pattern
- Release builds stripped via `LINK_FLAGS "-s"`

## CI/CD

GitHub Actions workflow at `.github/workflows/ci.yml`:
- Builds Release on `ubuntu-22.04`, `macos-14`, and `windows-2022` with all two FetchContent dependencies
- Verifies symbol exports (`clap_entry`, `ARAGetFactory`), RPATH / install name / DLL search path, model deployment
- Runs `clap-validator` against built `.clap` (continues on error — validator may fail without GUI)
- Validates model round-trip via `python3 scripts/validate_export.py`
- Uploads plugin artifact (`.clap`, `.so`/`.dylib`/`.dll`, `libmamba_core.*`, `models/`)
- Test job builds Debug on all three platforms with `BUILD_TESTING=ON` and runs CTest

## Verification

```bash
# Build and test (requires model files in models/)
cmake -B build -DBUILD_TESTING=ON && cmake --build build --parallel && ctest --test-dir build --output-on-failure
```

Tests cover: construct/destroy lifecycle, `init()` with model path, `processBlock()` with silence input (checking finite output), `resetState()` thread safety (deferred clear), Python↔C++ numerical equivalence (bit-exact), and long-run stability (60s @ 44.1kHz with no NaN/Inf). Python model export includes inline validation. For runtime testing, load `build/plugins/Mamba<Effect>.clap` in a CLAP host (Bitwig, Carla, clap-host); models must be in `models/` adjacent to the plugin binary.

## Known Gaps

See `MAMBA3_MIGRATION_PLAN.md` for the complete phased migration plan. Quick reference (updated 2026-06-16):

| Gap | Phase | Severity | Status |
|-----|-------|----------|--------|
| G1 | P0 | ✅ Done | Training pipeline stub — `train_clara_mamba3.py` needs implementation (Phase 8) |
| G2 | P0 | ✅ Done | CMake build fixed — CMake 4.3.3 + GCC 16 workarounds in place |
| G5 | P1 | 🟡 High | No DAW load test — engine works in tests, needs DAW validation |
| G6 | P2 | 🟡 High | GUI (CLAP_EXT_GUI) — JUCE or pugl needed |
| G7 | P2 | 🟡 Low | ARA conditioning — `doUpdateMusicalContextContent()` encodes default context to 128-dim cond. Real host reading deferred. |
| G8 | P2 | 🟡 Low | ARA analysis — `contentDirty_` flag set, SPSC AnalysisQueueDrain stub ready |
| G9 | P3 | 🔴 Blocking | **Training pipeline** — needs `train_clara_mamba3.py` + Phase 2-5 per-effect training |
| G10 | P4 | 🟢 Med | ONNX manifest enforcement — will be removed (ONNX is dead) |
| G12 | P4 | 🟡 High | macOS/Windows builds — defined in CMake but untested |
| G13 | P4 | 🟢 Med | Install/packaging — no DEB, distrobox-export, or install script |

**Training pipeline status:** Pipeline does NOT yet exist. Phase 8 will create `train_clara_mamba3.py` — a PyTorch training script matching the 757K-param Mamba-3 architecture from `reference_forward.py`. Needs MUSDB18-HQ dataset (~100K steps, ~4-8 hr on MI300X via AMD AI Developer Cloud).

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
- Flat binary weights (`model.bin`, version 2) loaded via mmap at init — no ONNX Runtime, no model parsing at runtime
- CLAP text + audio encoders shipped as ONNX models (loaded via Ort::Session on main thread only — never on audio thread)
