# CLARA — Mamba Audio Suite

[![CI](https://github.com/SGM-Studios/clara/actions/workflows/ci.yml/badge.svg)](https://github.com/SGM-Studios/clara/actions/workflows/ci.yml)

C++17 audio plugin that runs a Mamba-3 SSM (Selective State Space Model) through a
custom C++ inference engine. Ships as a CLAP plugin (`MambaSuiteClap.clap`), a VST3
wrapper (`MambaSuite.vst3`, via clap-wrapper) and an ARA 2.0 extension library
(`MambaSuiteARA.so`), all sharing one inference core. Linux is the tested platform;
macOS and Windows presets exist but are not verified.

No ONNX Runtime. Weights are a flat binary file (`model.bin`) loaded via `mmap` at init.

**Read this first:** [`docs/BLUEPRINT_COMPARISON.md`](docs/BLUEPRINT_COMPARISON.md) —
a verification pass over this repository (build, tests, benchmarks, model files)
that documents what works, what does not, and what the remaining work is. The
short version: **the engine is real and tested, but it is ~12x too slow for
real time and the shipped weights are an untrained fixture.**

## Status — 2026-10-04

| Capability | Status |
|---|---|
| Builds on Linux (CLAP + VST3 + ARA) | ✅ requires `libx11-dev` (native GUI) and CMake ≥ 3.14 |
| Weights present in a fresh clone | ✅ committed to Git (6.6 MB) — LFS removed, see [`models/README.md`](models/README.md) |
| Engine tests (`test_mamba3`) | ✅ 21 checks pass, incl. Python↔C++ equivalence and a drift guard |
| Python↔C++ equivalence | ✅ within float32 noise floor: 1.0–1.8e-3 abs / 5–9e-4 relative on a 2.14 peak |
| Compute budget guard (`test_perf_budget`) | ✅ exists and passes; it reports ~1,250% of one core per mono stream |
| **Real-time capable** | ❌ **~0.26 ms/sample → ~1,250% of one core per 48 kHz mono stream** |
| Numerical stability | ⚠️ float32 output drifts ~2% vs a float64 oracle within 12 ms; see *Stability* below |
| Trained model | ❌ `model.bin` is an untrained random fixture; the trained blobs were lost with the LFS budget |
| GUI | ✅ native `CLAP_EXT_GUI` (X11/Win32 framebuffer) + `CLAP_EXT_WEBVIEW` |
| ARA real-time conditioning | 🟡 scaffolding only — no host content reader |
| macOS / Windows builds | 🟡 presets defined; not verified in this environment |
| CI | 🟡 clap-validator download fixed, `libx11-dev` added; needs a run to confirm green |

## Performance (measured, not estimated)

Xeon @ 2.60 GHz, 2 vCPU container, 6-layer R=4 model, 512-sample blocks:

| Build | Per sample | 1 stream @48 kHz | Real-time factor |
|---|---|---|---|
| Original code, repo Release flags | 0.379 ms | 1,820% of one core | 0.055x |
| **Current, repo Release flags** | **0.262 ms** | **1,256%** | **0.080x** |
| Current, `-O3 -march=native` | 0.188 ms | 903% | 0.111x |

The 2026-10-04 optimization pass (batched projections + blocked GEMM kernels +
hoisted RoPE trigonometry) bought **1.45x mono / 1.6x stereo**, not the 3–5x
originally hoped: per-component profiling shows the `in_proj` GEMM is already at
this CPU's floating-point issue limit, so the remaining lever is **model size**
(`d_model`, `in_proj` width, MIMO rank), not more kernel tuning. See
`docs/BLUEPRINT_COMPARISON.md` §5.

Where the time goes now, per sample, per layer: `in_proj` 0.0246 ms (57%),
SSM recurrence 0.0148 ms (34%), `out_proj` 0.0037 ms (9%), FiLM ≈ 0.

### Stability (known issue)

Comparing the float32 engine against a float64 oracle of the identical pipeline:
2.0% of peak divergence within 12 ms, 3.6% at 23 ms, 7.3% at 93 ms. A 1-ULP
change to one input sample grows to 0.1% of peak over 85 ms. Peak `|ssm_real|`
sits at exactly 1.000, i.e. the periodic state clamp is active every 4096 samples.
The output is finite and stable, but float32 precision is not sufficient for this
recurrence, and different CPUs/compilers will produce slightly different audio.
`test_numeric_sensitivity` tracks this; the fix (double accumulation or a
less saturating state) is tracked in [`docs/TODO.md`](docs/TODO.md).

## Build

Dependencies are fetched by CMake `FetchContent`: CLAP SDK (`29ffcc27`), ARA SDK
(`2.3.0`), clap-wrapper (`v0.9.1`). On Ubuntu the build also needs `libx11-dev`
(the native GUI talks to Xlib directly) and `unzip`/`wget` are only needed for
the optional clap-validator.

```bash
sudo apt-get install -y cmake g++ libx11-dev      # Debian/Ubuntu

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

> With CMake 4.x, pass `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`: clap-wrapper's
> bundled dependencies still declare `cmake_minimum_required` values below 3.5.

The default build produces the CLAP plugin and ARA library. The VST3 wrapper is a
separate target:

```bash
cmake --build build --target MambaSuite_as_vst3 --parallel
```

Debug build with tests:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build --parallel && ctest --test-dir build --output-on-failure
```

Install (default prefix `/usr/local`), cross-platform presets, and the `clara_test`
aggregate target work as before: `cmake --preset macos-release`,
`cmake --install build --component clap`, `cmake --build build --target clara_test`.

Outputs:

- `build/plugins/MambaSuiteClap.clap` — CLAP audio effect
- `build/Release/MambaSuite.vst3` — VST3 bundle (`MambaSuite_as_vst3` target)
- `build/plugins/MambaSuiteARA.so` — ARA2 library
- `build/libMamba3Core.a` — static engine
- `build/plugins/models/*.bin` — weights, deployed next to the binary

## Running

**Ableton Live (VST3):** copy `build/Release/MambaSuite.vst3` to your VST3 folder
(`~/.vst3/`, `~/Library/Audio/Plug-Ins/VST3/`, `C:\Program Files\Common Files\VST3\`).

**CLAP hosts (Bitwig, Carla, clap-host):** load `build/plugins/MambaSuiteClap.clap`.

The weight file must sit in `models/` next to the plugin binary; the build
deploys it for the CLAP target. The VST3 bundle does not embed it yet — copy
`build/plugins/models/` next to the `.so` inside the bundle.

Each plugin exposes three parameters: bypass (stepped, click-free 50 ms
crossfade), wet/dry, intensity.

> **No DAW load test has been performed on this codebase.** The engine is
> verified by unit tests and by direct benchmark, but nobody has confirmed audio
> output inside a running host.

## Model

**Shipping format:** Mamba-3 R=4 time-domain (multi-head MIMO, RoPE, trapezoidal
SSM). `models/model.bin` is a 6.6 MB V2 flat-binary file.

| Parameter | Value |
|---|---|
| d_model / d_inner | 128 / 256 |
| n_layers / d_state | 6 / 32 |
| nheads / headdim | 4 / 64 |
| mimo_rank | 4 |
| cond_dim | 128 |
| Total params | 1,662,384 (~1.66M) |

**Not a trained effect.** The blobs that were labelled trained were lost when the
account's LFS budget was exceeded; the current file is an untrained fixture with
the correct shape. To train and install a real one, see
[`models/README.md`](models/README.md) and [`docs/TRAINING.md`](docs/TRAINING.md).

**Target architecture:** PerTrackProcessor R=8 spectral-domain (~4.26M params) —
see [`docs/MAMBA3_MIGRATION_PLAN.md`](docs/MAMBA3_MIGRATION_PLAN.md). The C++
engine rewrite to match it (H3) has not been started.

### Execution model

The host block is processed in chunks of `PROCESS_CHUNK_SIZE` (128) samples,
layer-major. Within a chunk the FiLM/Δ conditioning projections are evaluated once,
and `in_proj`/`out_proj` run as register-blocked GEMMs; the SSM recurrence stays
per-sample because it is causal. This is a scheduling decision only — it does not
change the model. Weights are `mmap`'d, state and scratch are engine-owned, and the
audio path performs **zero heap allocations** (verified with an allocation counter).

## Testing

```bash
cmake -B build -DBUILD_TESTING=ON && cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

| CTest name | Scope |
|---|---|
| `test_mamba3` | init, silence/sine response, conditioning changes output, determinism, state continuity, long-run finiteness, **Python↔C++ equivalence**, **1-ULP sensitivity guard** |
| `test_perf_budget` | per-sample cost vs a same-machine calibration loop; fails on regression (budget ratio 3.0) |
| `test_clap_plugin` | CLAP descriptor, lifecycle, parameters, buffer handling |
| `test_ara_controller` | ARA document controller lifecycle, archive round-trip |
| `clap_validator` | clap-validator (pinned 0.4.1 ubuntu-22.04 asset) against the built `.clap`; SKIPs (77) if the binary is absent |

The Python↔C++ fixture pair in `models/` is generated by
`python/tools/dump_engine_reference.py`, which runs the same model through
`python/reference_forward.py`. Those two files are committed, so the test suite
does **not** need torch. Regenerating them does (plus numpy).

### What the equivalence test does and does not prove

It proves the C++ engine and the Python reference implement the same arithmetic:
they agree to ~1e-3 absolute (~5e-4 relative) over 512 samples, and both sit
~2% away from a float64 oracle of the same pipeline. That residual is float32
rounding amplified by the recurrence, not a modelling difference — which is why
the tolerance is 5e-3 rather than 1e-6, and why `test_numeric_sensitivity` exists
to make the drift visible.

## Numerical layout and provenance

`python/reference_forward.py` was a `SyntaxError` (a multi-line generator inside
an f-string) and therefore unimportable until 2026-10-04; it also disagreed with
the engine in five places (RoPE pairing convention, missing `B_bias`/`C_bias`,
missing Δ conditioning, a broadcast bug in the D-skip, and state-normalization
scope). It has been corrected to match `src/core/SsmBlock.cpp`, which is the
shipping implementation. See the module docstring for the list.

## Project structure

```
├── CMakeLists.txt              # FetchContent deps, dual-target build
├── AGENTS.md                   # Build/architecture reference for agents
├── models/                     # Weights + reference fixtures (see models/README.md)
├── docs/
│   ├── BLUEPRINT_COMPARISON.md # Verification pass: measurements, gaps, priorities
│   ├── MAMBA3_MIGRATION_PLAN.md# R=8 target architecture and roadmap
│   ├── TRAINING.md             # Training reference
│   ├── TODO.md                 # Open work, incl. stability and compute budget
│   └── adr/                    # ADR-001..013
├── python/
│   ├── reference_forward.py    # R=4 time-domain reference (engine-equivalent)
│   ├── training/               # train_clara.py (R=8), train_local_radeon.py (R=4)
│   └── tools/                  # export, validate, dump_engine_reference, ...
├── src/
│   ├── core/                   # Mamba3Core: SsmInferenceCore, SsmBlock, WeightLoader
│   ├── clap/                   # CLAP plugin entry + extensions
│   ├── ara/                    # ARA document controller
│   └── gui/                    # Native (X11/Win32) and webview GUIs
├── tests/                      # test_mamba3, test_perf_budget, test_clap_plugin, ...
└── archive/mamba1-onnx-era/    # Deprecated Mamba-1 / ONNX code and docs
```

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) builds Release on `ubuntu-22.04`,
`macos-14` and `windows-2022`, verifies exported symbols and model deployment
(including that `model.bin` is not an LFS pointer), runs `clap-validator`, runs
CTest on all three platforms, and uploads artifacts.

## License

Apache 2.0. See [`LICENSE`](LICENSE).

## Roadmap

| Phase | What | Status |
|---|---|---|
| Now | R=4 engine, tests, fixtures, compute-budget guard | ✅ |
| Next (P1) | Fix numerical drift; cut per-sample cost ~12–24x so a stereo instance fits a CPU budget | 🔴 blocking for real-time |
| P2 | One flagship trained effect + DAW demo + CPU numbers | 🔴 not started |
| P3 | R=8 PerTrackProcessor engine rewrite (H3) | 🔴 not started |
| P4 | Per-effect variants, CLAP text/audio encoders, presets, installers | 🔴 not started |
