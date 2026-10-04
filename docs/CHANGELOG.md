# Changelog

<!-- STATUS: historical -->
> **Historical — the pre-0.1.0 internal changelog.** These 1.x version numbers
> predate the first release and overstated maturity (see the note in
> [`../CHANGELOG.md`](../CHANGELOG.md) under v0.1.0). The live changelog is
> [`/CHANGELOG.md`](../CHANGELOG.md); the ONNX-era entries further down describe
> code that now lives in [`../archive/mamba1-onnx-era/`](../archive/mamba1-onnx-era/).

## 1.2.1 — 2026-07-01

Pre-commit hooks, C++ formatting config, and GCC 16 compatibility fix.
3/3 CTest pass, clang-tidy: 0 CRITICAL/HIGH issues.

### Infrastructure
- New: `.pre-commit-config.yaml` — hooks for clang-format, black, ruff (Python),
  trailing-whitespace, end-of-file-fixer, mixed-line-ending, check-added-large-files,
  check-merge-conflict, check-case-conflict
- New: `.clang-format` — LLVM style, 4-space indent, 120-column limit,
  right-aligned pointers/references, C++17 standard
- New: `ruff.toml` — practical lint config (pycodestyle + pyflakes + pyupgrade),
  excludes `archive/` directory
- Updated: `README.md` + `AGENTS.md` — added Code Quality section with
  pre-commit install/run instructions; fixed stale CI artifact references
- Applied auto-formatting across 60 source, test, and archive files

### Fixes
- `src/core/SPSCQueue.hpp`: added missing `#include <cstddef>` for `size_t`
  (GCC 16 removed transitive include from `<atomic>`; template failed to compile)

## In Progress — 2026-06-02

### CLARA-06: Latency Reporting Verification
- Investigating: Does the 4-sample latency claim hold at all sample rates?
- Adding: `tests/test_latency.cpp` — impulse-response test at 44.1k/48k/96k/192k
- Adding: `scripts/verify_latency.py` — Python ONNX-level latency verification
- Fixing: P0/A1 silent engine failure path (processBlock now has optional `bool* ok` out-param)
- Fixing: P0/A2 duplicate param flush (removed redundant param handling from mamba_process)

### CLARA-09: Conditioning Vector from Musical Context
- New: `src/core/ConditioningGenerator.hpp` — encodes musical context to cond[8]
- Changed: `mamba_activate()` now loads non-zero test weights and sets default conditioning
- Changed: ARA stubs updated with ConditioningGenerator API references for future host-bridge integration
- Tests: 4 new ConditioningGenerator tests in test_engine.cpp
- Known gap: Real ARA integration requires host bridge — conditioning currently uses hardcoded default context

### CLARA-08: ONNX Export Pipeline Hardening
- New: Pre-flight checks in export_mamba_onnx.py (Python, torch, onnx, onnxruntime, numpy versions; opset availability; dry-run forward pass)
- New: ONNX vs PyTorch output validation (cosine similarity, max absolute difference)
- New: models/manifest.json generation (schema_version, architecture contract, SHA256 hashes, export timestamp)
- New: MambaInferenceEngine::validateManifest() — validates model architecture at init() against manifest
- Changed: Export script flow reorganized to [1/4] Export → [2/4] Verify → [3/4] Validate → [4/4] Manifest

### CLARA-07: Cross-Platform Build Matrix
- Changed: Windows CLAP install path now uses CLAP convention (lib/clap → CLAP/)
- Changed: CLAP_INSTALL_PREFIX and ARA_INSTALL_PREFIX CMake cache variables for customizable install paths
- Changed: Windows MSVC CRT constraint documented (/MD required because onnxruntime.dll is built with /MD)
- Known gap: macOS universal binary (arm64+x86_64) not yet wired; code signing not implemented
- Known gap: ONNX Runtime built with /MD — static CRT (/MT) would require building ORT from source

## 1.2.0 — 2026-06-02

Conditioning vector, SPSC deferred task queue, and ARA stub resolution.

### Features
- New: Conditioning vector `cond[COND_DIM=8]` — double-buffered with atomic pointer swap for RT-safe updates from any thread. Projection weights loaded at `activate()`, immutable in RT. External gain modulation — no ONNX I/O contract change.
- New: `src/core/SPSCQueue.hpp` — lock-free single-producer single-consumer ring buffer for deferred tasks (ModelReload, AnalysisQueueDrain).
- New: `mamba_on_main_thread()` now drains SPSC queue for `ModelReload` and `AnalysisQueueDrain` deferred tasks.
- New: `ADR-008-conditioning-vector.md` — documents conditioning vector architecture.

### Changed
- `MambaInferenceEngine` — added `setConditioning()` (double-buffered atomic swap), `loadCondWeights()` (non-RT), and conditioning gain modulation in `processBlock()`.
- `MambaClapPlugin` — replaced `std::atomic<DeferredTask>` with full `SPSCQueue<DeferredTask, 64>`. Engine null guard added to `mamba_process()` for dry-pass on invalid engine. Version bumped to 1.1.0.
- ARA controller — updated 6 stub methods with proper documentation and intent. `getInformationURL()` now returns GitHub URL.
- CLAP plugin descriptor — added GitHub URLs for plugin, manual, and support.
- Tests added for SPSC queue, conditioning vector, plugin descriptor URLs, ARA information URL, and engine edge cases.

### Fixes
- Multi-config generator model deployment: models and ONNX Runtime now deploy to `<plugin-binary-dir>/` via per-target `POST_BUILD` (previously deployed to `build/plugins/` only, breaking Windows multi-config builds)
- ARA `getInformationURL()` and CLAP descriptor URLs now point to the actual project URL (previously fictional `elite-dsp/clara`)
- ARA `getVersion()` synced to "1.1.0" to match CLAP descriptor
- ARA `AudioSourceReader` → `HostAudioReader` comment references in `MambaAraController.cpp:86,98` (the class `AudioSourceReader` does not exist in the plugin SDK; the correct class is `HostAudioReader`)

### Infrastructure
- CI: added `Verify new source files` and `Verify conditioning infrastructure` steps.
- `CMakeLists.txt`: SPSCQueue documented as header-only dependency.

## 1.1.0 — 2026-05-31

Cross-platform portability — CLARA now builds on macOS and Windows in addition to Linux.

### Features
- New: `src/core/Platform.hpp` with `MAMBA_EXPORT` (dllexport/dllimport), `mambaResolveModulePath` (cross-platform `dladdr`/`GetModuleFileName`), `mambaResolveModelDir`, `mambaFileExists`
- New: `CMakePresets.json` for multi-platform builds (Linux GCC, macOS Clang, Windows MSVC)
- Phase 0: Cross-platform portability scaffolding complete

### Changed
- Replaced `dladdr` / `access` / POSIX-only calls with cross-platform equivalents via `Platform.hpp`
- Updated `CMakeLists.txt` with platform-conditional guards (`WIN32`, `__APPLE__`)
- ARA library now installs as `LIBRARY RUNTIME` — enables Windows `.dll` support
- Updated tests for cross-platform compatibility (`std::filesystem` path separators, platform-conditional includes)

### Infrastructure
- Added macOS CI job (Clang + `@loader_path` RPATH)
- Added Windows CI job (MSVC + `.dll` deployment)
- `ADR-007-cross-platform-portability.md` — documents cross-platform strategy

### Status
- Phase 0 (scaffolding): ✅
- Phase 1 (macOS): 🔄
- Phase 2 (Windows): 🔄
- Phase 3 (CI matrix + docs): ⬜

## 1.0.0 — 2026-05-30

Initial release of CLARA — Mamba Audio Suite (CLAP + ARA2).

### Features
- CLAP audio effect plugin (`MambaSuiteClap.clap`) with Mamba SSM neural network inference via ONNX Runtime
- ARA2 shared library (`MambaSuiteARA.so`) for non-real-time playback rendering
- Shared `MambaInferenceEngine` C++ class powering both CLAP and ARA targets
- 3 plugin parameters: bypass (stepped), wet/dry, intensity — all atomic, sample-ramped
- Click-free bypass via linear crossfade (50ms, sample-rate-independent)
- Full CLAP state extension with versioned preset save/load
- CLAP audio ports extension (stereo I/O)
- CLAP latency reporting (4 samples)
- ARA2 DocumentController with binary archive state persistence
- ARA2 factory entry point (`ARAGetFactory`) with correct symbol visibility
- Model path resolution via `dladdr` for both CLAP and ARA (no hardcoded paths)
- Python ONNX export script (`export_mamba_onnx.py`) — opset 18, `torch.jit.trace`
- Pre-exported model: `mamba_rt.onnx` (untrained, random weights)
- CMake FetchContent for all 3 dependencies (CLAP SDK, ARA SDK, ONNX Runtime)

### Fixes (pre-release audit)
- **CLAP activate**: Wrapped `engine->init()` in try/catch — returns `false` instead of throwing across C boundary (UB fix)
- **CLAP create_plugin**: Changed raw `new` to `std::make_unique` + `.release()` — prevents leak on allocation failure
- **CLAP bypass crossfade**: Moved step accumulation outside per-channel loop — fixed stereo image aliasing during bypass transitions
- **InferenceEngine RT safety**: Replaced per-call `std::vector` allocation with `thread_local std::vector` + `reserve` — zero heap allocations on audio thread
- **ARA model path**: Replaced hardcoded `"models/mamba_rt.onnx"` with `dladdr`-based resolution
- **ARA factory export**: Added `extern "C" ARAGetFactory()` with `visibility("default")`
- **ARA archive state**: Implemented binary read/write for bypass/wet-dry/intensity persistence
- **CMake Release flags**: Added `-fno-exceptions -fno-rtti` for Release builds
- **Python opset**: Changed default from 17 to 18
- **Python dt_bias**: Removed double-bias add (`nn.Linear` already includes bias)

### Gap Audit (2026-05-30)
38 gaps found across P0–P4. 30 fixed in this session.

**P0 Runtime (2/2 fixed)**
- ARA architecture documented (no `renderSamples` in ARA 2.0)
- `libonnxruntime.so` deployed alongside plugins; `$ORIGIN`-only RPATH

**P1 Functional (6/7 fixed)**
- Crossfade stereo aliasing repaired
- ARA content update stubs parameterized and documented
- `mamba_analysis.onnx` removed (14 MB, never used)
- `readerwriterqueue` dead dependency removed
- Remaining: GUI (`CLAP_EXT_GUI`) — deferred

**P2 Correctness (8/9 fixed)**
- SSM state corruption from zero-padding fixed (dynamic `seq_len`)
- `resetState()` thread-safe via deferred atomic clear
- `init()` returns `bool` instead of throwing
- Placeholder URLs removed (CLAP + ARA)
- Remaining: untrained models (awaiting training pipeline)

**P3 Build (5/8 fixed)**
- Core compiled once as `MambaCore` STATIC library
- Release builds strip symbols
- `.gitignore` created, repo initialized, pushed to GitHub
- Remaining: install rules, license, CLAP bundle dir

**P4 Testing (1/5 fixed)**
- Python export simplified (single model verification)
- Remaining: unit tests, CLAP validator, CI/CD

### Known Gaps
See the comprehensive master TODO list at `TODO.md` for all known issues organized by priority (P0–P5, 38 items as of 2026-05-31).
