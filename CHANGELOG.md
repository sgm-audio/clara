# Changelog

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
- Pre-exported models: `mamba_rt.onnx` + `mamba_analysis.onnx` (untrained, random weights)
- CMake FetchContent for all 4 dependencies (CLAP SDK, ARA SDK, ONNX Runtime, ReaderWriterQueue)

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

### Known Gaps (tracked in TODO.md)
- ARA PlaybackRenderer has no audio output — ARA 2.0 has no `renderSamples()`; audio comes from companion plugin
- `libonnxruntime.so` not deployed alongside plugins — will fail to load outside build tree
- Models are untrained (random weights) — audio output is noise
- No test suite, no CI/CD pipeline
- No GUI (`CLAP_EXT_GUI`)
- Various correctness, build, and documentation gaps (32 total — see `TODO.md`)
