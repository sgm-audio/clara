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

Requires: `torch`, `onnxruntime`, `onnx`, `numpy`. Exports `models/mamba_rt.onnx` (with `.onnx.data` sidecar). Uses `torch.jit.trace` with opset 18, not dynamo. Models already exist in `models/` — re-export only if architecture changes.

## Architecture

```
src/core/MambaInferenceEngine  — RT-safe ONNX wrapper (pre-allocated tensors, zero alloc on process)
src/clap/MambaClapPlugin       — Full CLAP plugin with params, state, latency, click-free bypass crossfade
src/ara/MambaAraController     — ARA2 DocumentController + PlaybackRenderer + FactoryConfig
```

- `MambaInferenceEngine` is the single shared inference core. Both CLAP and ARA targets link it.
- Both CLAP and ARA resolve model path relative to their binary via `dladdr`.
- ARA 2.0 has no `renderSamples()` — audio processing is done by the companion CLAP plugin instance. The ARA library provides timeline metadata and audio source access for offline analysis (currently unused).
- ONNX model I/O: 9 inputs (`audio_chunk` + 4×conv state + 4×ssm state) → 3 outputs (`processed_audio`, stacked conv state, stacked ssm state). The engine unstacks output states back into per-layer buffers each block.
- CLAP reports latency of 4 samples (conv_width).
- Parameters: bypass (stepped), wet/dry, intensity — all atomic, smoothed per-sample in the process loop.

## Known Blockers

- **ARA2 has no `renderSamples()`.** The ARA 2.0 `ARAPlaybackRendererInterface` only provides `addPlaybackRegion`/`removePlaybackRegion`. Audio processing is exclusively handled by the companion CLAP plugin instance. The ARA library provides timeline metadata and audio source access for offline analysis (currently unused).
- **Models are untrained** — `mamba_rt.onnx` weights are randomly initialized. Audio output is noise until training is implemented.
- No test suite. No CLAP validator or ARA validation runs are wired.

## C++ Conventions

- C++17, no RTTI in release (`-ffast-math -O3 -DNDEBUG -fno-rtti`); exceptions allowed (ORT C++ API requires them)
- RT-safe: no allocations on audio thread; all buffers pre-allocated in `init()`
- `-Wall -Wextra -Werror=return-type` always on
- Plugin symbol visibility: hidden (`CXX_VISIBILITY_PRESET hidden`)
- RPATH set to `$ORIGIN` so plugins find `libonnxruntime.so*` next to themselves

## Plugin Testing

- CLAP: load `build/plugins/MambaSuiteClap.clap` in a CLAP host (Bitwig, Carla, clap-host)
- ARA2: load in an ARA-compatible DAW (Studio One, Reaper with ARA extension)
- Must have `models/mamba_rt.onnx` + `.onnx.data` adjacent to the plugin binary (inside `models/` dir)
