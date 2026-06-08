I'm going to describe a project so you understand what we're building and where things stand. Read carefully — I need you fully up to speed.

# Project: CLARA — Mamba Audio Suite (CLAP + ARA2)

A real-time audio plugin that runs a Mamba SSM (Selective State Space Model) neural network inside a DAW via ONNX Runtime. It produces two artifacts:
1. **MambaSuiteClap.clap** — CLAP audio effect plugin (stereo in/out, real-time processing)
2. **MambaSuiteARA.so** — ARA2 shared library for non-real-time playback rendering inside DAWs

The same `MambaInferenceEngine` C++ class powers both — no code duplication.

## Build System
- CMake 3.14+, C++17, no exceptions, no RTTI in release
- Three dependencies fetched via FetchContent (no manual install):
  - CLAP SDK (commit `29ffcc27`) — plugin format
  - ARA SDK (commit `1ca0b0aa`) — Audio Random Access protocol
  - ONNX Runtime v1.17.1 — prebuilt packages for Linux/macOS/Windows
- Cross-platform via Platform.hpp abstractions (uses `mambaResolveModulePath()`, `MAMBA_EXPORT`)
- Outputs to `build/plugins/`
- WSL Ubuntu build env (g++ 15.2.0, cmake 4.2.3)

## Architecture
```
src/core/MambaInferenceEngine  — RT-safe ONNX wrapper
src/clap/MambaClapPlugin       — Full CLAP plugin
src/ara/MambaAraController     — ARA2 DocumentController + PlaybackRenderer + FactoryConfig
src/core/Platform.hpp        — Cross-platform abstractions (module path, visibility, threading)
```

**MambaInferenceEngine** (`src/core/`):
- Pre-allocates all tensors in `init()` — zero allocations on audio thread
- ONNX model has 9 inputs (`audio_chunk` + 4×conv state + 4×ssm state) → 3 outputs (`processed_audio`, stacked conv state, stacked ssm state)
- Uses `thread_local std::vector<Ort::Value>` with `reserve` to avoid per-call heap allocs
- `processBlock()` is RT-safe: memcpy, memset, no allocations, no throws across C boundary

**CLAP plugin** (`src/clap/`):
- Parameters: bypass (stepped), wet/dry, intensity — all atomic, sample-ramped
- Click-free bypass via linear crossfade over 50ms
- Reports 4 samples latency (conv_width)
- State save/load with version header for forward compat
- `clap_create_plugin` uses `dladdr` to resolve model path relative to `.clap` binary
- Audio ports extension: stereo I/O
- `mamba_activate` catches exceptions, returns false instead of throwing across C

**ARA2 library** (`src/ara/`):
- `MambaDocumentController` — holds atomic params (bypass, wetDry, intensity), archive save/restore via `HostArchiveReader/Writer` using `SizedStruct` binary format (3 floats)
- `MambaPlaybackRenderer` — constructs `MambaInferenceEngine`; catches init exceptions; flag-guarded process
- `getMambaARAFactory()` / `extern "C" ARAGetFactory()` — `visibility("default")` exported
- Model path resolved via `dladdr` relative to `.so`, cached in static local
- **ARA 2.0 has NO `renderSamples`** in `ARAPlaybackRendererInterface` — audio is processed by companion plugin API, not through ARA

**Python export script** (`export_mamba_onnx.py`):
- Exports PyTorch Mamba model to ONNX via `torch.jit.trace` (not dynamo), opset 18
- Produces `models/mamba_rt.onnx` + `.onnx.data` sidecar
- `MambaSSMCore.forward()` had a `dt_proj.bias` double-add bug (fixed: `nn.Linear` already includes bias)

## Recent TODO Fixes (all done, builds clean)

| Area | What was wrong | Fix |
|------|---------------|-----|
| CLAP activate | threw exceptions across C boundary | try/catch, return false |
| CLAP create_plugin | raw `new` could leak | `std::make_unique` + `.release()` |
| CLAP crossfade | step accumulation aliased stereo channels | moved outside per-sample loop |
| CLAP audio ports | no extension registered | added `clap/ext/audio-ports.h` + stereo I/O |
| CLAP state versioning | no forward compat | `int32_t version` header |
| CLAP state load | engine not reset | calls `resetState()` after load |
| CMake Release | had exceptions/RTTI | `-fno-exceptions -fno-rtti` |
| InferenceEngine RT | `std::vector<Ort::Value>` allocated per `processBlock` call | `thread_local` + `reserve` |
| ARA model path | hardcoded `"models/mamba_rt.onnx"` | `dladdr`-based resolution |
| ARA archive state | no persistence | binary read/write of 3 floats |
| ARA factory export | no `extern "C"` entry point, hidden visibility | `ARAGetFactory()` + `visibility("default")` |
| Python opset | opset 17 | bumped to 18 |
| Python dt_bias | `nn.Linear(bias=True)` + explicit `+ self.dt_proj.bias` = double bias | removed explicit add |

## Build Status
- Both targets build clean from scratch: `cmake --build . --parallel`
- `clap_entry` and `ARAGetFactory` exported as GLOBAL DEFAULT symbols
- `libonnxruntime.so.1.17.1` lives in `_deps/onnxruntime-src/lib/` — needs `$ORIGIN` deploy
- No test suite exists; no CLAP validator or ARA validation runs wired

## Known Blockers / Risks
- **ARA2 integration may need rework against raw C API** if `ARAPlug.h` C++ wrappers are incomplete (see `TODO.md` item pending for verification)
- No CI/CD pipeline
- Cross-platform support in progress — Platform.hpp abstracts module path, visibility, threading for macOS and Windows
- ONNX model files must be deployed adjacent to plugin binary at runtime

## Important Caveats
- `ARA::PlugIn::PlugInEntry::getPlugInEntry<F, D>()` creates static singletons — you cannot inject custom constructor arguments to DocumentController or PlaybackRenderer
- `Ort::Value` has NO default constructor — only created via `Ort::Value::CreateTensor<T>(...)` static factories
- Model file: `models/mamba_rt.onnx` with sidecar `models/mamba_rt.onnx.data` (external data format)

## Master TODO List
The comprehensive master TODO list is at `TODO.md` — 34 items organized by priority (P0–P5). It supersedes the now-deprecated `TODO_PROGRESS.md`.
