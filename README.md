# CLARA

A causal state-space audio effect in a small C++17 engine, built as a CLAP
plugin (`MambaSuite.clap`), a VST3 wrapper (`MambaSuite.vst3`, via clap-wrapper)
and a separate ARA 2 library (`MambaSuiteARA.so`). The engine is Mamba-3-inspired:
it uses multi-head state, RoPE on B/C and a trapezoidal (bilinear) step, but the
recurrence isn't the Mamba-3 MIMO update (see *Engine* below).

It is not usable in a session yet. It runs about 10x slower than real time on
the machine below, and `models/model.bin` holds untrained weights.

Linux is the only platform built and tested here. macOS presets and a Windows
CI job exist; neither has been checked on this branch.

## Status (checked 2026-10-31)

| | |
|---|---|
| Build, Linux | CLAP, ARA library and VST3 build from a fresh clone. Release, zig clang 22.1.8 (libc++), CMake + Ninja. |
| Tests | `ctest`: 7/7 pass (see *Tests*). |
| clap-validator 0.4.1, full run | 44 tests: 27 pass, 5 fail, 1 warning, 11 skipped. The failures are state-reproducibility-basic/binary/buffered, and param-fuzz-basic/bounds timing out at 45 s. |
| Real time | No. 0.19–0.21 ms per sample, about 1,000% of one core per 48 kHz mono stream. |
| Weights | Untrained. Generated at configure time with a fixed seed. |
| VST3 | Builds, and `GetPluginFactory()` returns a factory when `MambaSuite.clap` is in `~/.clap`. Not loaded in a DAW. |
| ARA | `MambaSuiteARA.so` exports `ARAGetFactory`, but the CLAP plugin doesn't expose the ARA factory or plug-in extension, so no host can reach it. |
| GUI | Native X11/Win32 framebuffer editor and a webview editor exist. Neither has been opened in a host. |
| CI | GitHub Actions is blocked by an account billing lock. No run has completed since 2026-07-10. |

## Performance

`tests/test_perf_budget`, 6-layer model (d_model 128, 4 heads, rank 4,
d_state 32), 512-sample blocks, mono. Release (`-O3 -fno-math-errno
-fno-trapping-math`), zig clang 22.1.8, shared 8-vCPU Intel Xeon VM
(1 thread per core, load average 1.0–1.8 during the runs), 2026-10-31.

| Build | Per sample | One 48 kHz mono stream | Real-time factor |
|---|---|---|---|
| Before (`-ffast-math`, main @ 539be30) | 0.87–0.93 ms | ~4,300% of one core | ~0.02x |
| After (no `-ffast-math`) | 0.19–0.21 ms | ~1,000% of one core | ~0.1x |

Both builds were run alternately, three times each, on the same model file.
The 4.5x difference comes from the compile flag alone (same source with
`-ffast-math` added back: 0.89–0.92 ms). I haven't found why fast-math is
slower here. Real time at 48 kHz needs about 20.8 µs per sample for one mono
stream, so the model has to shrink by roughly 10x mono / 20x stereo.

The earlier figure in this README (0.262 ms/sample, 2-vCPU container, GCC)
came from a different machine and compiler and isn't comparable.

## Engine

`src/core/SsmBlock.cpp`, 6 layers, 1,662,384 parameters. Things to know:

- The rank loop applies the state decay once per rank, so four times per
  sample, and every one of the 64 `d` rows of a head's state holds the same
  values. In practice the per-head state is `[d_state]`. That redundant work is
  a large part of the cost.
- z-gating, D-skip and the MIMO-out term collapse to one scalar per head or
  per sample.
- `A_bar = (1+aΔ/2)/(1-aΔ/2)` goes negative for aΔ < −2. Nothing clamps it.
- `normalize_if_needed` rescales the state every 4096 samples when its max
  exceeds 1. That is an audible gain step.
- float32 output drifts from a float64 oracle of the same pipeline: 2.0% of
  peak within 12 ms, 7.3% within 93 ms (measured 2026-10-04). The 1-ULP check
  in `test_mamba3` reports 0.068% of peak over 4096 samples.

`python/reference_forward.py` was corrected to match the C++ engine, so the
Python↔C++ test shows the two agree (about 1e-3 absolute). It doesn't show
either is the intended model.

## Build

Dependencies are fetched by CMake at configure time; none are vendored.

| Dependency | Pin | Licence |
|---|---|---|
| CLAP | `29ffcc27` (1.2.7) | MIT |
| ARA SDK | `releases/2.3.0` | Apache-2.0 |
| clap-wrapper (VST3 only) | `v0.14.0` | MIT |
| Steinberg VST3 SDK (VST3 only) | `v3.8.0_build_66` | MIT |

You need a C++17 compiler, CMake ≥ 3.14, git, and python3 with numpy (to
generate `models/model.bin`). `libx11-dev` is optional; without it the native
editor is compiled out.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
cmake --build build --target MambaSuite_as_vst3 --parallel   # VST3, optional
ctest --test-dir build --output-on-failure
```

`-DCLARA_BUILD_VST3=OFF` skips clap-wrapper and the VST3 SDK.
`-DVST3_SDK_ROOT=/path` uses your own SDK checkout.

Outputs: `build/plugins/MambaSuite.clap`, `build/plugins/MambaSuiteARA.so`,
`build/Release/MambaSuite.vst3`, and `build/plugins/models/` (the weights, which
must sit next to the `.clap`; the plugin finds them with `dladdr`).

`cmake --install build [--component clap|vst3|ara]` installs to
`lib/clap/`, `lib/ara/` and `lib/vst3/` under the prefix. The VST3 carries no
weights; clap-wrapper loads `MambaSuite.clap` from the CLAP search paths
(`~/.clap`, `/usr/lib/clap`, `CLAP_PATH`), and that file reads the `models/`
beside it.

## Weights

`models/model.bin` (6,649,600 bytes) isn't tracked. CMake runs
`python/tools/export_mamba_weights.py --seed 42` at configure time if the file
is missing. The output is deterministic (sha256 `bc10f763…8639e`, recorded in
`models/ref_meta.txt`). The weights are Xavier-random, so the plugin passes
audio through an untrained network. The trained blobs were lost when the
account's Git LFS budget ran out; see [`models/README.md`](models/README.md).

Training: [`docs/training/TRAINING.md`](docs/training/TRAINING.md) (R=4, the
architecture the engine runs) and
[`docs/training/PER_TRACK_PROCESSOR.md`](docs/training/PER_TRACK_PROCESSOR.md)
(R=8 spectral, which the engine doesn't implement). MUSDB18-HQ is licensed for
non-commercial research only.

## Plugin

Three parameters: bypass (50 ms crossfade), wet/dry, intensity. Parameter
events are applied at the start of each block, not sample-accurately. Latency
is reported as 0 samples (no conv1d). Tail is reported as 0, although the
state decays over time. Offline render is accepted.

Known problems:
- State load calls `reset_states()` on the main thread while the audio thread
  may be processing (data race).
- The engine runs even when bypass has settled at 1.0.
- clap-validator state-reproducibility fails (above).

## Tests

| CTest | What it checks |
|---|---|
| `test_mamba3` | init, silence/sine response, conditioning, determinism, state continuity, long-run finiteness, Python↔C++ agreement on 512 samples, 1-ULP sensitivity |
| `test_perf_budget` | per-sample cost vs a same-machine calibration loop (ratio budget 3.0) |
| `test_clap_plugin` | descriptor, params, state round-trip, bypass, process path, param events in `process()`, render modes |
| `test_ara_controller` | ARA document controller lifecycle, archive round-trip |
| `clap_validator` | clap-validator subset; skips (77) if it isn't installed. Excludes the slow tests and state-reproducibility, which fails |
| `build_vst3_for_install`, `test_install_layout` | builds the VST3, then checks the `cmake --install` layout |

Results on the machine above, 2026-10-31:

| | Before (main @ 539be30) | After (this branch) |
|---|---|---|
| ctest | 2/6 pass | 7/7 pass |
| test_perf_budget | fail (ratio 3.44) | pass (ratio 0.36–0.40) |
| test_clap_plugin | 2 failures (bypass not exact, `1e999` accepted; both from `-ffast-math`) | pass |
| clap_validator (ctest subset) | fail (`--invert-filter` rejected by 0.4.1) | pass |
| test_install_layout | fail (VST3 not built) | pass |

The perf ratio dropped partly because the calibration loop lost fast-math
vectorization (about 2x slower), so the guard is looser than before. It needs
a new budget; see [`docs/TODO.md`](docs/TODO.md).

## Layout

```
CMakeLists.txt     build; fetches CLAP, ARA SDK, clap-wrapper, VST3 SDK
src/core/          engine (SsmInferenceCore, SsmBlock, WeightLoader)
src/clap/          CLAP entry and extensions
src/ara/           ARA document controller
src/gui/           native and webview editors
tests/             C++ tests and CMake test scripts
python/            reference_forward.py, training/, tools/
models/            README, ref fixtures; model.bin is generated
scripts/           training and setup helpers
docs/              TODO, adr/, design/, training/
```

## Licence

CLARA's own code is Apache-2.0 ([`LICENSE`](LICENSE)). Third-party SDKs are
downloaded at build time under their own licences; see [`NOTICE`](NOTICE).
Before 2026-10-31 the VST3 SDK pin was v3.7.6 (GPLv3 or Steinberg proprietary);
VST3 binaries built from those revisions carry that condition. VST is a
trademark of Steinberg Media Technologies GmbH.
