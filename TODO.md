# TODO — Remaining Gaps

Audited 2026-05-30. **38 items found; 30 fixed.** 8 remain.

**Legend:** P0=runtime broken, P1=incomplete, P2=correctness, P3=build, P4=testing

---

## Remaining

- [ ] **P1.7 CLAP: No GUI (`CLAP_EXT_GUI`)** — JUCE integration deferred
- [ ] **P2.8 Python: models are untrained (random weights)** — add training pipeline
- [ ] **P3.3 CMake: no `install()` rules** — for packaging/deployment
- [ ] **P3.5 No `LICENSE` file** — choose and apply license
- [ ] **P3.7 No CLAP bundle directory structure**
- [ ] **P4.1 No unit tests for `MambaInferenceEngine`**
- [ ] **P4.2 No CLAP validator run**
- [ ] **P4.4 No CI/CD pipeline**

---

## Fixed (this session)

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
