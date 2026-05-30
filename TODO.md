# TODO — Gaps, Stubs & Placeholders

Audited 2026-05-30. See `GAPS.md` for full analysis context.

**Legend:** P0=runtime broken, P1=incomplete, P2=correctness, P3=build, P4=testing

---

## P0 — Runtime Broken (must fix before any DAW load)

- [ ] **P0.1 ARA: PlaybackRenderer produces no audio** `src/ara/MambaAraController.cpp:49-67`
  All region lifecycle hooks delegate to base with no custom logic. ARA 2.0 has no `renderSamples()` in `ARAPlaybackRendererInterface` — audio must come from companion plugin. Document or wire companion processing.

- [ ] **P0.2 libonnxruntime.so not deployed alongside plugins** `CMakeLists.txt:82-84`
  RUNPATH leaks absolute build path. `$ORIGIN` has no `.so` next to plugins. Fix: `file(COPY ...)` + `CMAKE_BUILD_RPATH_USE_ORIGIN ON` + deploy rule.

---

## P1 — Functional Gaps

- [ ] **P1.1 CLAP: bypass crossfade stereo aliasing** `src/clap/MambaClapPlugin.cpp:337`
  `currentBypassValue` advances inside channel loop. Fix: move increment outside.

- [ ] **P1.2 ARA: content update stubs are no-ops** `src/ara/MambaAraController.cpp:93-95,117-119,126-128`
  `doUpdateMusicalContextContent`, `doUpdateAudioSourceContent`, `didEnableAudioSourceSamplesAccess` empty.

- [ ] **P1.3 ARA: `willEnableAudioSourceSamplesAccess` ignores `enable` param**
  Always sets `audioSourceAccessEnabled_ = true`. Use actual parameter.

- [ ] **P1.4 ARA: `audioSourceAccessEnabled_` written but never read**
  Field exists but no code checks it.

- [ ] **P1.5 ARA: `mamba_analysis.onnx` never loaded** `models/mamba_analysis.onnx`
  14 MB + sidecar on disk, zero C++ references. Either wire up or delete.

- [ ] **P1.6 CMake: `readerwriterqueue` fetched but never used** `CMakeLists.txt:26-31`
  Dependency dead unless background analysis thread is implemented.

- [ ] **P1.7 CLAP: No GUI (`CLAP_EXT_GUI`)**
  Long-term: JUCE integration per TODO_PROGRESS.md.

---

## P2 — Correctness & Robustness

- [ ] **P2.1 InferenceEngine: zero-padding corrupts SSM state** `src/core/MambaInferenceEngine.cpp:60-63`
  `numSamples < maxBlockSize` feeds fake zeros into recurrent state. Use dynamic ONNX `seq_len` axis.

- [ ] **P2.2 InferenceEngine: `resetState()` not thread-safe** `src/core/MambaInferenceEngine.cpp:47-52`
  No synchronization. CLAP guarantees audio-thread reset, ARA does not.

- [ ] **P2.3 InferenceEngine: `init()` throws — breaks `-fno-exceptions`** `src/core/MambaInferenceEngine.cpp:23-27`
  Refactor to return `bool` or error status.

- [ ] **P2.4 CLAP: `clap_params_flush` no gesture begin/end** `src/clap/MambaClapPlugin.cpp:135-150`
  No `CLAP_EVENT_PARAM_GESTURE_BEGIN/END` in output. Hosts won't record automation correctly.

- [ ] **P2.5 CLAP: `mamba_on_main_thread` empty** `src/clap/MambaClapPlugin.cpp:380`
  No deferred task infrastructure.

- [ ] **P2.6 CLAP: placeholder URLs in descriptor** `src/clap/MambaClapPlugin.cpp:37-39`
  `https://example.com/mamba_audio_suite*` appears in host UIs.

- [ ] **P2.7 ARA: placeholder URLs in factory config** `src/ara/MambaAraController.cpp:242`
  Same `example.com` URLs.

- [ ] **P2.8 Python: models are untrained (random weights)** `export_mamba_onnx.py:263-266`
  Output will be random noise. Document or add training pipeline.

- [ ] **P2.9 Python: both ONNX models are identical** `export_mamba_onnx.py:353-358`
  `mamba_analysis.onnx` is a byte-for-byte duplicate of `mamba_rt.onnx`.

---

## P3 — Build & Infrastructure

- [ ] **P3.1 CMake: `readerwriterqueue` dead dependency**
- [ ] **P3.2 CMake: core sources compiled twice** — build as STATIC lib
- [ ] **P3.3 CMake: no `install()` rules**
- [ ] **P3.4 No `.gitignore`**
- [ ] **P3.5 No `LICENSE` file**
- [ ] **P3.6 Binaries not stripped in Release**
- [ ] **P3.7 No CLAP bundle directory structure**
- [ ] **P3.8 `AGENTS.md` stale** — wrong claims about ARA path, exceptions

---

## P4 — Testing

- [ ] **P4.1 No unit tests for `MambaInferenceEngine`**
- [ ] **P4.2 No CLAP validator run**
- [ ] **P4.3 No ARA validation**
- [ ] **P4.4 No CI/CD pipeline**
- [ ] **P4.5 Python inline verification not a structured test suite**
