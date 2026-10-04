# Incubator-Testable Checklist

**Updated 2026-10-04.** Superseded in places by `docs/BLUEPRINT_COMPARISON.md`,
which contains the measurements. Read the **Blocking** section first: this project
is not ready to be evaluated as an audio effect yet, for two reasons that have
nothing to do with polish.

## What's Done (shipping, verified)

- [x] C++ engine: R=4 time-domain Mamba-3, 1.66M params, V2 flat binary weights
- [x] CLAP plugin: `MambaSuiteClap.clap` — 3 params (bypass, wet/dry, intensity)
- [x] VST3 wrapper: `MambaSuite.vst3` via clap-wrapper — loadable in Ableton Live
- [x] ARA 2.0 extension: `MambaSuiteARA.so` — scaffolding only
- [x] GUIs: `CLAP_EXT_WEBVIEW` (inline HTML) and `CLAP_EXT_GUI` (native framebuffer, X11/Win32)
- [x] Engine tests pass, including Python↔C++ equivalence (`test_mamba3`, 21 checks)
- [x] Compute-budget test (`test_perf_budget`) and drift guard (`test_numeric_sensitivity`)
- [x] Weights committed to Git — a fresh clone is runnable without LFS
- [x] Docs honest about the model, the cost, and the drift

## Blocking (do not skip these two)

1. **It is not real time.** 0.262 ms/sample → ~1,250% of one core per 48 kHz mono
   stream (~2,430% stereo). Real time needs < 100%. Roughly 12–24x to find, and it
   has to come from a smaller model: kernel work is exhausted (the GEMM is at the
   CPU's floating-point issue limit).
2. **No trained model ships.** `models/model.bin` is an untrained random fixture
   with the correct shape. The trained blob was lost when the account's Git LFS
   budget was exceeded (see `models/README.md`). Any "does it sound good" test is
   meaningless until a trained model is installed.

Also outstanding: **no DAW load test has ever been performed** on this codebase,
and the float32 engine drifts ~2% from a float64 oracle within 12 ms (different
CPUs will produce slightly different audio).

## Historical: post-reboot tasks (2026-07-07)

### What the Agent Must Do After Reboot


### 1. Build Windows VST3 via MinGW

MinGW was installed via `rpm-ostree install mingw64-gcc-c++` and requires a reboot to activate.

**After reboot:**

```bash
cd /var/home/sgm/Desktop/Dev/clara

# Verify MinGW is available
x86_64-w64-mingw32-g++ --version

# Configure with MinGW toolchain
cmake -B build-windows \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/mingw64.cmake \
  -DBUILD_TESTING=OFF

# Build VST3 target
cmake --build build-windows --target MambaSuite_as_vst3 --parallel

# Verify output
ls -la build-windows/plugins/MambaSuite.so
```

**Expected output:** `build-windows/plugins/MambaSuite.so` — a Windows PE executable (VST3 binary).

**Known issues to fix if build fails:**
- `X11/Xlib.h` not found on MinGW: the native GUI code has `#if defined(_WIN32)` guards, but verify the include chain doesn't pull in X11 headers on Windows
- `dlfcn.h` not found: `Platform.hpp` has `#if defined(_WIN32)` guards for this
- `-s` link flag: MSVC doesn't support `-s`; the generator expression `$<$<PLATFORM_ID:Linux,Darwin>:-s>` should handle this
- clap-wrapper VST3 SDK download: may need `CLAP_WRAPPER_DOWNLOAD_DEPENDENCIES=ON` (already set)

### 2. Run 100K-Step Training

The training hangs when run in background (`nohup`/`setsid`) due to a ROCm GPU context deadlock. Must be run in an interactive terminal.

**In a terminal (not backgrounded):**

```bash
cd /var/home/sgm/Desktop/Dev/clara

HSA_OVERRIDE_GFX_VERSION=11.0.0 \
HIP_VISIBLE_DEVICES=0 \
ROC_VISIBLE_DEVICES=0 \
HSA_ENABLE_SDMA=0 \
~/micromamba/envs/rocm_torch/bin/python3 \
  python/training/train_local_radeon.py \
  --data /run/media/sgm/Data/datasets/musdb18hq/train \
  --steps 100000 --batch 4 --grad-accum 4 \
  --chunk-samples 2048 --warmup-chunk 64 \
  --log-every 500 --warmup 2000 \
  --ckpt-every 2000 \
  --save checkpoints_clara
```

**Expected:** ~8 hours at ~0.2 step/s with chunk_samples=2048. Loss should drop from ~5 to <0.1.

**After training completes:**

```bash
# Export to model.bin
~/micromamba/envs/rocm_torch/bin/python3 \
  python/training/train_local_radeon.py \
  --export checkpoints_clara/best.pt \
  --output models/model_trained.bin

# Validate export
python3 python/tools/validate_export.py models/model_trained.bin

# Test C++ engine loads it
./build/tests/test_mamba3 models/model_trained.bin

# Copy to build directories for deployment
cp models/model_trained.bin build/plugins/models/model.bin
cp models/model_trained.bin build-windows/plugins/models/model.bin 2>/dev/null || true
```

**If chunk_samples=2048 hangs despite warmup:** fall back to chunk_samples=64 (guaranteed to work, ~11 step/s, ~2.5 hours for 100K steps).

### 3. DAW Load Test (Windows)

After building the Windows VST3 and training the model:

```bash
# Copy VST3 binary to Windows partition
# (Mount Windows partition first — Bazzite auto-mounts)
cp build-windows/plugins/MambaSuite.so \
  /run/media/sgm/.../Common\ Files/VST3/MambaSuite.vst3/

# Copy model alongside VST3
mkdir -p /run/media/sgm/.../Common\ Files/VST3/MambaSuite.vst3/Contents/Resources/models
cp models/model_trained.bin \
  /run/media/sgm/.../Common\ Files/VST3/MambaSuite.vst3/Contents/Resources/models/model.bin
```

Then reboot to Windows and:
1. Open Ableton Live
2. Re-scan VST3 plugins (Options → Preferences → Plug-Ins → Rescan)
3. Find "Mamba-3 SSM Processor" under Audio Effects
4. Drag onto an audio track, hit play
5. Toggle bypass → audio should change
6. Move wet/dry and intensity sliders → audible effect
7. Open plugin window → custom GUI with 3 sliders + level meter

### 4. Verify CI

Push the current commits and verify CI passes on all three platforms:

```bash
git push origin main
# Check https://github.com/scottmill306/CLARA/actions
```

CI builds: ubuntu-22.04 (CLAP + VST3 + ARA + tests), macos-14 (CLAP + VST3 + ARA), windows-2022 (CLAP + ARA + tests).

---

## What's Still Aspirational (Not Shipping)

| Feature | Status | Notes |
|---------|--------|-------|
| R=8 PerTrackProcessor C++ engine | 🔴 Not started | H3 in TODO.md — 20-40hr C++ rewrite |
| Per-effect model variants | 🔴 Not started | Saturator, Compressor, Denoiser |
| CLAP text/audio encoders | 🔴 Not started | Phase 2/3 |
| ARA real-time conditioning | 🟡 Scaffolding only | Content reader not implemented |
| macOS native GUI | 🟡 Stub | NSView path documented but untested |
| macOS AU target | 🔴 Not started | clap-wrapper supports AUv2 |
