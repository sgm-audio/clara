# CLARA — Mamba Audio Suite

[![CI](https://github.com/sgm-audio/clara/actions/workflows/ci.yml/badge.svg)](https://github.com/sgm-audio/clara/actions/workflows/ci.yml)

C++17 audio plugin that runs a Mamba-3 SSM (Selective State Space Model) through a
custom C++ inference engine. Ships as a CLAP plugin (`MambaSuite.clap`), a VST3
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
| Builds on Linux (CLAP + VST3 + ARA) | ✅ clean-container verified (`cmake` + `g++` only; `libx11-dev` optional, X11 auto-detected) |
| Weights present in a fresh clone | ✅ committed to Git (6.6 MB) — LFS removed, see [`models/README.md`](models/README.md) |
| Engine tests (`test_mamba3`) | ✅ 21 checks pass, incl. Python↔C++ equivalence and a drift guard |
| Python↔C++ equivalence | ✅ within float32 noise floor: 1.0–1.8e-3 abs / 5–9e-4 relative on a 2.14 peak |
| Compute budget guard (`test_perf_budget`) | ✅ exists and passes; it reports ~1,250% of one core per mono stream |
| **Real-time capable** | ❌ **~0.26 ms/sample → ~1,250% of one core per 48 kHz mono stream** |
| Numerical stability | ⚠️ float32 output drifts ~2% vs a float64 oracle within 12 ms; see *Stability* below |
| Trained model | ❌ `model.bin` is an untrained random fixture; the trained blobs were lost with the LFS budget |
| GUI | ✅ native `CLAP_EXT_GUI` (X11/Win32 framebuffer) + `CLAP_EXT_WEBVIEW`; auto-disabled when X11 headers are absent |
| ARA real-time conditioning | 🟡 scaffolding only — no host content reader |
| macOS / Windows builds | 🟡 presets defined; not verified in this environment |
| CI | 🔴 **not running**: every run since 2026-10-04 ends in `startup_failure` with 0 jobs (workflow parses and is registered active, so this is account/repo-level, not a YAML error) — see *CI/CD* below |
| VST3 in a host | 🟡 **loads now, still not DAW-tested**: build output renamed to `MambaSuite.clap` so clap-wrapper's name match succeeds — `GetPluginFactory()` verified, see *Running* |

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
(`2.3.0`), clap-wrapper (`v0.9.1`), and — for the VST3 target — the Steinberg
VST3 SDK (`v3.7.6_build_18`). Only a C++17 toolchain, CMake ≥ 3.14, `git` and a
network connection to GitHub are required (the project is built with
`CMAKE_CXX_STANDARD 17`; the VST3 SDK is the only dependency that would accept
a newer standard):

```bash
sudo apt-get install -y cmake g++                 # Debian/Ubuntu

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel                    # CLAP + ARA (+ VST3 wrapper)
```

Tests are opt-in: add `-DBUILD_TESTING=ON` to the configure line, otherwise no
test targets exist and `build/tests/` is empty.

> CI passes `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` on every configure. With the
> pinned dependency revisions it is **not required** — verified 2026-10-04: a
> fresh configure (all four dependencies fetched) succeeds on CMake 4.4.3
> without it, because clap-wrapper v0.9.1 builds the VST3 SDK from a source glob
> rather than `add_subdirectory`, so the SDK's
> `cmake_minimum_required(VERSION 3.4.3)` files are never processed. Passing it
> is harmless, and it protects against older clap-wrapper/dependency pins.

Two build knobs matter on restricted machines:

- **`libx11-dev` is optional.** The native framebuffer editor talks to Xlib
  directly. CMake detects X11 and, when it is absent, defines
  `CLARA_NO_NATIVE_GUI` and compiles the editor out — the CLAP plugin, ARA
  library and all tests still build, the plugin just advertises no GUI extension
  and hosts show their generic editor. Install `libx11-dev` to get the editor.
- **`-DCLARA_BUILD_VST3=OFF`** skips clap-wrapper and the VST3 SDK entirely
  (the CLAP plugin, ARA library and tests need neither).

The VST3 wrapper is a separate target, because the SDK is large:

```bash
cmake --build build --target MambaSuite_as_vst3 --parallel
```

clap-wrapper normally pulls the VST3 SDK through CPM, whose bootstrap downloads
`CPM.cmake` from a **GitHub release asset** — a class of URL that some CI
sandboxes and corporate networks block even when `git clone` works. CLARA avoids
that bootstrap: it fetches the same SDK revision by git and passes it to
clap-wrapper as `VST3_SDK_ROOT`, so no release-asset download happens. If you
already have an SDK checkout, skip the fetch with
`-DVST3_SDK_ROOT=/path/to/vst3sdk`.

Debug build with tests:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build --parallel && ctest --test-dir build --output-on-failure
```

Install (default prefix `/usr/local`), cross-platform presets, and the `clara_test`
aggregate target: `cmake --preset macos-release`, `cmake --build build --target clara_test`.

`cmake --install <build> [--prefix <dir>] [--component clap|vst3|ara]` installs,
prefix-relative, on Linux:

```
lib/clap/MambaSuite.clap              the CLAP plugin (bare DSO)
lib/clap/models/*.bin                 weights — must sit next to the DSO (dladdr)
lib/ara/MambaSuiteARA.so              the ARA2 library
lib/ara/models/*.bin                  weights, next to the library
lib/vst3/MambaSuite.vst3/             the clap-wrapper VST3 module
  Contents/x86_64-linux/MambaSuite.so
```

macOS/Windows install the CLAP as a bundle (`<stem>.clap/Contents/<arch>/<stem>.clap`),
which is the local convention. A *full* install also drops the fetched CLAP SDK's
own headers/CMake package files under `include/clap/` and `lib/cmake/clap/` —
those come from the CLAP SDK's install rules, not from CLARA; use a component
install if you do not want them.

The VST3 bundles carry no weights of their own: clap-wrapper loads the `.clap`
from the CLAP search paths, and *that* file reads the `models/` beside it.

Outputs:

- `build/plugins/MambaSuite.clap` — CLAP audio effect
- `build/Release/MambaSuite.vst3` — VST3 bundle (`MambaSuite_as_vst3` target)
- `build/plugins/MambaSuiteARA.so` — ARA2 library
- `build/tests/test_*` — unit tests (`-DBUILD_TESTING=ON`)
- `build/libMamba3Core.a` — static engine
- `build/plugins/models/*.bin` — weights, deployed next to the binary

## Running

**CLAP hosts (Bitwig, Carla, clap-host):** load `build/plugins/MambaSuite.clap`
directly, or copy it into a CLAP search path (`~/.clap/`, `/usr/lib/clap/`).

**Ableton Live (VST3):** copy `build/Release/MambaSuite.vst3` to your VST3 folder
(`~/.vst3/`, `~/Library/Audio/Plug-Ins/VST3/`, `C:\Program Files\Common Files\VST3\`).
The wrapper locates the CLAP by *matching file name*: `MambaSuite.vst3` → it
searches the CLAP paths for `MambaSuite.clap` (clap-wrapper's
`wrapasvst3_entry.cpp` derives the name from its own binary and looks in
`/usr/lib/clap`, `~/.clap` and `CLAP_PATH`, plus one level of subdirectories).
The build now produces exactly that name, so the pairing is:

```bash
cp build/plugins/MambaSuite.clap ~/.clap/            # + models/ beside it
cp -r build/Release/MambaSuite.vst3 ~/.vst3/
```

Verified 2026-10-04 with a `dlopen` probe: with only the old
`MambaSuiteClap.clap` present, `GetPluginFactory()` returns `nullptr`; with
`MambaSuite.clap` in `~/.clap`, the factory is created and reports
`Mamba-3 SSM Processor` (vendor `Elite DSP`, id `com.mamba.audio.suite`).

The weight file must sit in `models/` **next to the `.clap` binary that is
loaded** (the plugin resolves its own module path with `dladdr`, so it is the
`.clap` location that matters, not the host's). The build deploys it for the
CLAP target. Verified 2026-10-04 with a minimal CLAP host harness: `plugin->init()`
returns true with `models/` beside the `.clap` and false without it; with it,
`process()` produces finite audio.

Each plugin exposes three parameters: bypass (stepped, click-free 50 ms
crossfade), wet/dry, intensity.

> **No DAW load test has been performed on this codebase.** The engine is
> verified by unit tests, by direct benchmark, and (2026-10-04) by a minimal
> CLAP host harness that dlopens `MambaSuite.clap`, creates and activates
> the plugin and processes 512 frames — `init()` succeeds and the output is
> finite. Nobody has confirmed audio output inside a running DAW.

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

GitHub Actions (`.github/workflows/ci.yml`) builds Release on `ubuntu-22.04` and
`windows-2022` (there is **no macOS job**; `CMakePresets.json` defines macOS
presets, but nothing verifies them), verifies exported symbols and model
deployment (including that `model.bin` is not an LFS pointer), downloads
`clap-validator` (advisory — the step tolerates failures), runs CTest on both
platforms, and uploads the CLAP plugin, ARA library, VST3 bundle and deployed
weights as artifacts.

<!-- STATUS: in progress -->
**None of that has run since 2026-10-04.** Every workflow run created on that
date and later — the three pushes, both `pull_request` runs, and the scheduled
CodeQL runs — completed with conclusion `startup_failure`, zero jobs and
`path=BuildFailed`. The last run that actually executed was 2026-07-10
(`workflow_dispatch`, `path=.github/workflows/ci.yml`). The file itself is not
the problem: it parses as YAML, `gh api .../actions/workflows` lists `CI` as
`active` at `.github/workflows/ci.yml`, and the working tree copy is byte-identical
to `main`. A `startup_failure` with no jobs on *all* workflows at once points at
the repository/account level — Actions spending limit or billing, or Actions
disabled for the repo — which only someone with admin rights can check
(Settings → Billing → Actions, and Settings → Actions → General). Until that is
resolved the CI claims in `CHANGELOG.md` v0.1.1/v0.1.2 (clap-validator pin,
model checks, X11-free build) are written but **unverified on CI**.

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
