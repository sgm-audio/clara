# TODO — Remaining Gaps

Audited 2026-05-30. **38 items found; 35 fixed.** 3 remain.

**Legend:** P0=runtime broken, P1=incomplete, P2=correctness, P3=build, P4=testing

---

## Remaining

- [ ] **P1.7 CLAP: No GUI (`CLAP_EXT_GUI`)** — JUCE integration deferred
- [ ] **P2.8 Python: models are untrained (random weights)** — need training pipeline
- [ ] **P4.2 No CLAP validator passing in CI** — downloaded and run but may fail without GUI

---

## Fixed (this session — wave 2)

| Item | Files | Summary |
|------|-------|---------|
| P3.3 | `CMakeLists.txt` | `install()` rules: CLAP bundle, bare `.clap`, ARA `.so`, models, runtime lib |
| P3.5 | `LICENSE` | Apache 2.0 license added |
| P3.7 | `CMakeLists.txt` | CLAP bundle directory structure in install target (`MambaSuiteClap.clap/Contents/linux-x86-64/`) |
| P4.1 | `tests/test_engine.cpp` | 4 unit tests: construct, init, processBlock (silence), resetState |
| P4.4 | `.github/workflows/ci.yml` | Build + symbol check + validator + model export + artifact upload + test job |

## Fixed (wave 1)

| Item | Commit | Summary |
|------|--------|---------|
| P0.1 | f9e3a68 | ARA architecture documented (no `renderSamples` in ARA 2.0) |
| P0.2 | f9e3a68 | `libonnxruntime.so` deployed, `BUILD_WITH_INSTALL_RPATH ON` |
| P1.1 | f9e3a68 | Crossfade stereo aliasing fixed |
| P1.2 | f9e3a68 | Content update stubs documented |
| P1.3 | f9e3a68 | `willEnableAudioSourceSamplesAccess` respects `enable` param |
| P1.4 | f9e3a68 | `audioSourceAccessEnabled_` still unused (doc-only) |
| P1.5 | f9e3a68 | `mamba_analysis.onnx` removed (14 MB saved) |
| P1.6 | f9e3a68 | `readerwriterqueue` removed from CMake |
| P2.1 | 22f9c86 | SSM state corruption fixed (dynamic `seq_len` axis) |
| P2.2 | 22f9c86 | `resetState()` deferred clear via atomic flag |
| P2.3 | 22f9c86 | `init()` returns `bool` instead of throwing |
| P2.4 | 22f9c86 | `clap_params_flush` out param named |
| P2.5 | 22f9c86 | `mamba_on_main_thread` documented |
| P2.6 | 22f9c86 | Placeholder URLs removed (`nullptr`) |
| P2.7 | 22f9c86 | ARA factory URL removed (`nullptr`) |
| P2.9 | f9e3a68 | Duplicate analysis model removed |
| P3.1 | f9e3a68 | `readerwriterqueue` removed |
| P3.2 | 76b4974 | Core compiled once as STATIC library |
| P3.4 | c0a76ff | `.gitignore` created |
| P3.6 | 76b4974 | Release builds strip symbols (`-s`) |
| P3.8 | f9e3a68 | `AGENTS.md` updated (stale claims fixed) |
| P4.5 | f9e3a68 | Python inline verification simplified (single model) |
