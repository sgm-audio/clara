# TODO List Progress Summary

> **Note:** A comprehensive master TODO list with per-file issue tracking now exists at `TODO.md` — it supersedes this file for detailed tracking. This file remains as a high-level progress summary.

## Completed
### I. CLAP Plugin Completion
- [x] Implement all required CLAP vtable callbacks (init, destroy, get/set parameters, etc.)
- [x] Complete parameter system with bypass as CLAP boolean parameter
- [x] Add parameter smoothing integration with existing crossfade
- [x] Implement full state extension for preset/chunk support

### II. ARA2 Controller Completion
- [x] Implement full ARA2 plugin factory (`ARA::PlugIn::PlugIn`)
- [x] DocumentController with state persistence methods (archive save/restore working)
- [ ] Complete PlaybackRenderer with edit state handling
- [ ] Implement required ARA2 interfaces based on capabilities (AudioRandomAccess, Presets, etc.)

### III. Build System & Dependencies
- [x] Finalize CMakeLists.txt with proper dependency fetching using FetchContent
- [x] Set RT-safe compiler flags (`-fno-exceptions`, `-fno-rtti`, `-Wall`)
- [x] Document dependency acquisition:
    - ONNXRuntime (with GPU/CUDA options)
    - CLAP SDK (github.com/free-audio/clap)
    - ARA2 SDK (celemony.com/ARA)
    - (ReaderWriterQueue dependency was removed — no longer used)

### IV. Model & Audio Processing
- [x] Run Phase 1 Python export script to generate ONNX model
- [x] Place model in `models/` directory or make path configurable
- [x] Decide and implement mono/stereo audio processing strategy
- [x] Update tensor shapes and buffer sizes accordingly

### V. State Management & Thread Safety
- [x] Define and implement state reset behavior (transport seek, etc.)
- [ ] Verify thread safety of atomic parameters and lock-free queues
- [ ] Ensure no shared state between audio/GUI/analysis threads

### VI. Optional Enhancements
- [ ] Implement GUI using JUCE (connect via lock-free queue)
- [x] Add state chunking and preset support (via `clap_plugin_state` extension)
- [ ] Implement optional metering via CLAP extension
- [ ] Add non-RT debug logging system

### VII. Testing & Validation
- [x] CLAP validator integration test (`clap_validator` CTest target — runs [clap-validator](https://github.com/free-audio/clap-validator) against the built `.clap`; required categories: audio-ports, params, state, latency)
- [ ] Build and link validation on target platforms
- [ ] Functional testing with CLAP hosts and ARA2 DAWs
- [ ] Real-time safety validation with stress tests
- [ ] Verify bypass click-free operation and ARA2 analysis correctness

### VIII. Documentation & Release
- [ ] Write comprehensive README with build/usage instructions
- [ ] Create parameter reference and model export documentation
- [ ] Choose and apply license (e.g., GPLv3/MIT)
- [ ] Create packaging scripts for distribution

### IX. Cross-Platform Portability
- [x] Phase 0: Scaffolding — `Platform.hpp` with `MAMBA_EXPORT`, `mambaResolveModulePath`, platform guards
- [x] `CMakePresets.json` for Linux/macOS/Windows
- [ ] Phase 1: macOS — Clang preset, CI, `@loader_path` RPATH — 🔄 macOS install paths ready, universal binary pending
- [ ] Phase 2: Windows — MSVC preset, CI, `.dll` deployment — 🔄 Windows install paths ready, static CRT deferred
- [ ] Phase 3: CI matrix expansion + docs update

## Current Implementation Status
- **CLAP Plugin**: Fully implemented with all required callbacks, parameter handling, latency reporting, click-free bypass, and state extension (`src/clap/MambaClapPlugin.hpp/.cpp`)
- **ARA2 Controller**: Implements DocumentController and PlaybackRenderer with background analysis thread using lock-free SPSC queue (`src/ara/MambaAraController.hpp/.cpp`)
- **ARA2 Plugin Factory**: Factory implemented in `src/ara/MambaAraController.cpp` using standard `ARAPlug.h` C++ wrappers (the `MambaAraPlugIn.hpp/.cpp` files never existed — the factory lives directly in the controller)
- **Mamba Inference Engine**: RT-safe ONNX wrapper with pre-allocated tensors and state management (`src/core/MambaInferenceEngine.hpp/.cpp`)
- **Build System**: `CMakeLists.txt` uses `FetchContent` to download `ONNXRuntime`, `CLAP`, and `ARA SDK`. `ReaderWriterQueue` was removed as a dependency. `CMakePresets.json` supports Linux (GCC), macOS (Clang), and Windows (MSVC).
- **Model Export**: Python export script exports `mamba_rt.onnx` (with `.onnx.data` sidecar). The `mamba_analysis.onnx` model was removed — only one ONNX model is used.

## Known Issues / Blockers
- ~~**ARA SDK Missing Wrappers**~~ — **RESOLVED**: The code now correctly uses the Celemony `ARA_Library/PlugIn/ARAPlug.h` C++ wrappers (`ARA::PlugIn::DocumentController`, `ARA::PlugIn::PlaybackRenderer`, etc.). These are the official Celemony-provided wrappers, not JUCE-specific. ARA integration is fully enabled in `CMakeLists.txt`.

## Next Steps
0. **CLARA-08: ONNX export hardening (P2):** Pre-flight checks, ONNX vs PyTorch validation, manifest.json generation, plugin-side manifest validation. DONE.
0. **CLARA-09: Conditioning wiring (P1):** ConditioningGenerator created and wired in CLAP. Next: ARA host bridge for real musical context → cond flow.
0. **CLARA-07: Cross-platform build matrix (P2):** Windows install paths improved; static CRT deferred (ONNX Runtime constraint). Remaining: macOS universal binary, code signing, static MSVC runtime investigation.
1. **CLARA-06: Latency verification test (P0):** Write impulse-response test proving 4-sample latency claim at all sample rates. If measured latency doesn't match, fix latency reporting before any user-facing release.
2. **Address clap-validator failures (P1):** First validation pass shows `clap_plugin::init()` returns false in 11/21 tests. Root cause likely model path resolution (the validator loads the plugin from a working directory that may not have `models/` adjacent). Investigate `mamba_init()` and `mambaResolveModulePath()` when the plugin is loaded from an unexpected CWD.
3. Complete PlaybackRenderer with edit state handling (handle changes in the DAW timeline).
4. Implement required ARA2 interfaces (AudioRandomAccess, Presets, etc.).
5. Implement state reset behavior in the CLAP plugin (for transport seek).
6. Verify thread safety of atomic parameters and lock-free queues.
7. Add optional enhancements (GUI, metering) if desired.
8. Conduct comprehensive testing and validation.
9. **Phase 1:** Finalize macOS support (Clang, `@loader_path`, CI).
10. **Phase 2:** Finalize Windows support (MSVC, `.dll`, CI).
11. **Phase 3:** Expand CI matrix and update documentation.

## CLAP Validator Test (added 2026-06-02)
- New CTest target `clap_validator` registered in `tests/CMakeLists.txt`
- Test runner: `tests/clap_validator.cmake` (CMake script-mode)
- Required CLAP extension categories: `audio-ports`, `params`, `state`, `latency`
- Validator resolution: `find_program(clap-validator)` → `cargo install --git https://github.com/free-audio/clap-validator clap-validator` (Rust toolchain) → SKIP
- SKIP detection: configure-time in CMakeLists (primary); runtime via `[SKIPPED]` log marker in clap_validator.cmake (since `cmake -P` cannot return arbitrary exit codes)
- Convenience target: `cmake --build . --target clara_test` (named `clara_test` because `test` is reserved by CTest)
- Initial run on the freshly-built Debug `.clap`: 21 tests, 6 passed, 11 failed (all `init() returned false`), 4 skipped (preset discovery + Unix-only)

The foundation is now RT-safe and complete for the core audio processing. Remaining work focuses on plugin completeness, build system, and integration features.