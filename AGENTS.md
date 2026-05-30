# CLARA — Mamba Audio Suite (CLAP + ARA2)

C++17 audio plugin using Mamba SSM (Selective State Space Model) inference via ONNX Runtime. Produces a CLAP plugin and an ARA2 shared library, both sharing the same inference engine.

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

Outputs land in `build/plugins/`:
- `MambaSuiteClap.clap` — CLAP audio effect plugin
- `MambaSuiteARA.so` — ARA2 playback renderer library
- `models/` — ONNX models (copied from source `models/`)

All four dependencies are fetched via CMake FetchContent (no manual install needed):
- CLAP SDK (pinned commit `29ffcc27`)
- ReaderWriterQueue v1.0.6 (header-only)
- ARA SDK (pinned commit `1ca0b0aa`)
- ONNX Runtime v1.17.1 (prebuilt Linux x64 tarball)

**No tests exist yet.** Build validation is: `make` succeeds and both `.clap`/`.so` appear in `build/plugins/`.

## Model Export

```bash
python3 export_mamba_onnx.py
```

Requires: `torch`, `onnxruntime`, `onnx`, `numpy`. Exports `models/mamba_rt.onnx` and `models/mamba_analysis.onnx` (both with `.onnx.data` sidecar). Uses `torch.jit.trace` with opset 18, not dynamo. Models already exist in `models/` — re-export only if architecture changes.

## Architecture

```
src/core/MambaInferenceEngine  — RT-safe ONNX wrapper (pre-allocated tensors, zero alloc on process)
src/clap/MambaClapPlugin       — Full CLAP plugin with params, state, latency, click-free bypass crossfade
src/ara/MambaAraController     — ARA2 DocumentController + PlaybackRenderer + FactoryConfig
```

- `MambaInferenceEngine` is the single shared inference core. Both CLAP and ARA targets link it.
- CLAP plugin resolves model path relative to the `.clap` binary via `dladdr` at `clap_create_plugin` time.
- ARA PlaybackRenderer currently hardcodes `models/mamba_rt.onnx` path.
- ONNX model I/O: 9 inputs (`audio_chunk` + 4×conv state + 4×ssm state) → 3 outputs (`processed_audio`, stacked conv state, stacked ssm state). The engine unstacks output states back into per-layer buffers each block.
- CLAP reports latency of 4 samples (conv_width).
- Parameters: bypass (stepped), wet/dry, intensity — all atomic, smoothed per-sample in the process loop.

## Known Blockers

- **ARA2 does not build with the raw Celemony SDK.** The `MambaAraController` uses `ARA::PlugIn::*` C++ wrapper classes from `ARAPlug.h`, but the fetched ARA SDK may not expose all expected wrappers consistently. The `.so` currently builds, but ARA integration may need rework against the raw C API if `ARAPlug.h` wrappers are insufficient. See `TODO_PROGRESS.md` section II.
- No test suite. No CLAP validator or ARA validation runs are wired.

## C++ Conventions

- C++17, no exceptions, no RTTI in release (`-ffast-math -O3 -DNDEBUG`)
- RT-safe: no allocations on audio thread; all buffers pre-allocated in `init()`
- `-Wall -Wextra -Werror=return-type` always on
- Plugin symbol visibility: hidden (`CXX_VISIBILITY_PRESET hidden`)
- RPATH set to `$ORIGIN` so plugins find `libonnxruntime.so*` next to themselves

## Plugin Testing

- CLAP: load `build/plugins/MambaSuiteClap.clap` in a CLAP host (Bitwig, Carla, clap-host)
- ARA2: load in an ARA-compatible DAW (Studio One, Reaper with ARA extension)
- Must have `models/mamba_rt.onnx` + `.onnx.data` adjacent to the plugin binary (inside the `.clap` bundle or in `models/` dir)
