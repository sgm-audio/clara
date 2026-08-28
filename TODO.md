# TODO — Gaps, Stubs & Placeholders

Ordered by severity. Each item identifies a real gap found in the codebase.

---

## 1. ARA: `getMambaARAFactory()` is hidden — DAWs cannot discover the plugin

`src/ara/MambaAraController.cpp:185` — `getMambaARAFactory()` has internal linkage (non-exported). ARA hosts call this function to get the factory; if it's not in the dynamic symbol table, the `.so` is dead on arrival.

- Add a visibility attribute (`__attribute__((visibility("default")))`) or a `.cpp`-level `#pragma GCC visibility push(default)` around it.
- Verify with: `readelf --dyn-syms MambaSuiteARA.so | grep getMambaARAFactory`
- The CMake target already sets `CXX_VISIBILITY_PRESET hidden`, so every symbol needs explicit opt-out.

## 2. ARA: No C entry point exported — ARA hosts cannot load the library at all

ARA requires a C-callable `ARAFactory` getter exposed from the shared library. Currently there is no `extern "C"` exported function that a DAW can `dlsym` for. `getMambaARAFactory` returns the factory struct, but the function itself is hidden (see #1) and is C++-mangled.

- Add `extern "C" __attribute__((visibility("default"))) const ARA::ARAFactory* ARAGetFactory();` that delegates to `getMambaARAFactory()`.
- Verify the symbol appears in `readelf --dyn-syms`.

## 3. CLAP: `libonnxruntime.so` not deployed alongside plugin — will fail to load at runtime

RUNPATH is `$ORIGIN:/home/SGM/.../build/_deps/onnxruntime-src/lib` — the absolute build path leaks into the binary. Outside the build tree, `libonnxruntime.so.1.17.1` won't be found next to the `.clap` (or `.so`), so `dlopen` will fail.

- CMake should `file(COPY ...)` or `install(FILES ...)` `libonnxruntime.so*` into `build/plugins/` alongside the binaries so `$ORIGIN` resolves.
- Remove the build-tree absolute path from RUNPATH (CMake `CMAKE_BUILD_RPATH` is adding it implicitly via the linked lib path; set `CMAKE_BUILD_RPATH_USE_ORIGIN ON` or use `$ORIGIN` only).
- Verify with: `ldd build/plugins/MambaSuiteClap.clap` — `libonnxruntime` should resolve via `$ORIGIN`, not an absolute path.

## 4. ARA: PlaybackRenderer has no `renderSamples()` override — no audio is ever produced

`MambaPlaybackRenderer` overrides region lifecycle hooks (add/remove) but never overrides `renderSamples()`. This is the core ARA callback where the DAW asks for processed audio. Without it, the ARA plugin loads but outputs silence.

- Implement `renderSamples()` override that reads source audio from the ARA host, runs it through `rtEngine_`, and writes output samples.
- Signature: `void renderSamples(ARA::PlugIn::AudioSource* source, ARA::ARASamplePosition position, ARA::ARASampleCount samplesCount, float* const* outputBuffers, ARA::ARAChannelCount channelCount) noexcept override`

## 5. ARA: Model path hardcoded as relative `"models/mamba_rt.onnx"` — will fail outside CWD

`src/ara/MambaAraController.cpp:8` — `config.modelPath = "models/mamba_rt.onnx"`. This resolves relative to the DAW's working directory, not the `.so` location. Unlike the CLAP plugin (which uses `dladdr`), the ARA target has no path resolution.

- Use `dladdr((void*)&getMambaARAFactory, &info)` (or similar) to resolve the `.so` path at construction time, then build the model path relative to the library location, matching the CLAP strategy.

## 6. ARA: DocumentController archive methods are no-ops — state save/load is broken

`src/ara/MambaAraController.cpp:140-148` — `doRestoreObjectsFromArchive` and `doStoreObjectsToArchive` both return `true` without reading/writing any data. DAWs that persist ARA plugin state between sessions will silently lose all settings.

- Implement actual serialization of plugin parameters (bypass, wet/dry, intensity) into the archive reader/writer.
- Mirror the `MambaPresetData` struct format used by the CLAP state extension for consistency.

## 7. ARA: DocumentController content update methods are empty stubs

Multiple overrides in `MambaDocumentController` do nothing:
- `doUpdateMusicalContextContent` (line 56) — empty body
- `doUpdateAudioSourceContent` (line 80) — empty body
- `willEnableAudioSourceSamplesAccess` (line 84) — empty body
- `didEnableAudioSourceSamplesAccess` (line 88) — empty body

These are called by the host when the DAW timeline or audio content changes. If the model's state depends on musical context (tempo, key), these need to invalidate analysis and trigger re-processing. At minimum, `willEnableAudioSourceSamplesAccess` must signal readiness for host audio access.

- Implement `will/didEnableAudioSourceSamplesAccess` to track audio source access state.
- Implement content update handlers to invalidate cached analysis when the timeline changes.

## 8. ARA: EditorRenderer and EditorView return `nullptr` — empty stubs

`src/ara/MambaAraController.cpp:122-138` — `doCreateEditorRenderer()` and `doCreateEditorView()` return `nullptr`. This is acceptable for an audio-effect-only ARA plugin, but if the plugin ever advertises editor capabilities in its factory config, the host will crash.

- Verify that `MambaFactoryConfig` / the ARA factory struct does not advertise `kARAFeatureEditorRenderer` or `kARAFeatureEditorView`.
- If editor features are not advertised, document this as intentional. If they are, these stubs must be implemented.

## 9. CLAP: Bypass crossfade is channel-aliased — stereo image shifts during bypass

`src/clap/MambaClapPlugin.cpp:310` — `state->currentBypassValue += state->crossfadeStep` runs inside the per-channel loop. The first channel advances the bypass value, so the second channel starts from a different crossfade position. This creates an audible stereo image shift during bypass transitions.

- Move the crossfade increment outside the channel loop so both channels use the same bypass ramp per sample.

## 10. CLAP: Bypass crossfade duration is frame-rate dependent

`src/clap/MambaClapPlugin.cpp:293-299` — `crossfadeStep = 0.001f` assumes a fixed step per sample regardless of sample rate. At 44.1 kHz the crossfade takes ~1 second; at 192 kHz it takes ~4 seconds.

- Compute `crossfadeStep` from the sample rate: `1.0f / (crossfadeDurationSeconds * sampleRate)`.
- Store `sampleRate` in `MambaPluginState` (available from `mamba_activate`'s `double sample_rate` parameter, currently unused/unnamed).

## 11. CLAP: `mamba_activate` ignores the `sample_rate` parameter

`src/clap/MambaClapPlugin.cpp:224` — `double` parameter is unnamed (`double,`). The sample rate is needed for frame-rate-independent crossfade (see #10) and potentially for model inference scaling.

- Name the parameter and store it in `MambaPluginState`.
- Use it to compute crossfade step and pass to the engine if needed.

## 12. CLAP: `mamba_init` is a no-op

`src/clap/MambaClapPlugin.cpp:215` — `mamba_init` just returns `true`. If model loading fails in `mamba_activate`, the host has no way to know — the plugin appears to activate successfully but outputs silence on ONNX errors.

- Move engine initialization to `mamba_init` or add error checking in `mamba_activate` that returns `false` on ONNX session creation failure (currently the exception from `init()` would propagate through the C vtable, which is UB).

## 13. CLAP: `mamba_activate` throws across C boundary — undefined behavior

`src/clap/MambaClapPlugin.cpp:239` — `state->engine->init()` can throw `std::runtime_error` (see `MambaInferenceEngine.cpp:26`). This is called from a C function pointer (`mamba_activate`). Throwing across a C frame is UB and will likely crash.

- Wrap `init()` in `try/catch` and return `false` on failure.
- Apply the same fix to `MambaPlaybackRenderer` constructor (`MambaAraController.cpp:10-11`) which also calls `init()` in a `noexcept` context.

## 14. CLAP: `clap_create_plugin` leaks the `clap_plugin_t*` on `new` failure

`src/clap/MambaClapPlugin.cpp:373-374` — If `new MambaPluginState()` succeeds but later initialization fails, the `clap_plugin_t` and `MambaPluginState` are never freed. There's no error path.

- Use `std::unique_ptr` for both allocations, release ownership only on confirmed success.

## 15. CLAP: No `CLAP_EXT_GUI` extension — plugin has no UI

The plugin exposes no GUI extension. While functional without one (hosts show generic parameter UI), this is a significant usability gap for an audio product.

- Implement `clap_gui` extension or defer to a future JUCE integration (noted in TODO_PROGRESS.md VI.1).

## 16. CLAP: No `CLAP_EXT_NOTE_PORTS` or `CLAP_EXT_AUDIO_PORTS` registration

The plugin does not declare audio ports via the `CLAP_EXT_AUDIO_PORTS` extension. Some CLAP hosts may not know how many channels the plugin supports or whether it's mono/stereo.

- Implement `clap_audio_ports` extension to declare stereo-in/stereo-out (or the desired channel config).

## 17. InferenceEngine: `processBlock` allocates `std::vector` on every call — not RT-safe

`src/core/MambaInferenceEngine.cpp:65,83` — `inputTensors` and `outputTensors` are `std::vector<Ort::Value>` constructed and destroyed each call. `std::vector` construction may heap-allocate.

- Pre-allocate these as fixed-size `std::array<Ort::Value, kNumInputs>` / `std::array<Ort::Value, kNumOutputs>` member variables.
- OR use stack-allocated C arrays `Ort::Value inputTensors[kNumInputs]`.

## 18. InferenceEngine: `processBlock` pads partial blocks with zeros but uses `maxBlockSize` tensor shape

`src/core/MambaInferenceEngine.cpp:60-63` — When `numSamples < maxBlockSize`, the engine zero-pads the input buffer and runs inference on the full `maxBlockSize` shape. This wastes compute and feeds zero-samples into the SSM state, corrupting the recurrent state for the next block.

- Use ONNX dynamic axes (`seq_len`) to run inference on the actual block size.
- OR accept the waste but at minimum: after inference, only copy `blockSamples` of output; the state corruption from zero-padding is the real bug — the SSM hidden state gets updated with fake zero-timesteps.

## 19. InferenceEngine: `resetState()` is not thread-safe

`src/core/MambaInferenceEngine.cpp:47-52` — `resetState()` writes to `convStateIn_[]` and `ssmStateIn_[]` without synchronization. If called from the main thread while `processBlock` runs on the audio thread, this is a data race.

- Use a flag (`std::atomic<bool> resetRequested_`) checked in `processBlock`, or ensure the host guarantees `reset()` is only called from the audio thread (CLAP spec does guarantee this for `mamba_reset`, but ARA does not).

## 20. InferenceEngine: `init()` throws — violates project "no exceptions" rule for Release

`src/core/MambaInferenceEngine.cpp:23-27` — The constructor and `init()` use `throw std::runtime_error`. The AGENTS.md and CMake Release flags (`-DNDEBUG`) imply no-exception mode, but `-fno-exceptions` is not set in `CMAKE_CXX_FLAGS`.

- Either add `-fno-exceptions -fno-rtti` to the Release flags (and refactor `init()` to return error codes), or document that exceptions are allowed in non-RT paths only.

## 21. CMake: `-fno-exceptions` and `-fno-rtti` are missing from compiler flags

`CMakeLists.txt:9` — Release flags include `-ffast-math -O3 -DNDEBUG` but not `-fno-exceptions -fno-rtti` despite `AGENTS.md` and `TODO_PROGRESS.md` listing them as "set".

- Add `-fno-exceptions -fno-rtti` to `CMAKE_CXX_FLAGS_RELEASE` (if exceptions are truly removed from code paths; see #20).
- If exceptions are kept for `init()`, add only `-fno-rtti`.

## 22. CMake: Core sources are compiled twice (in both CLAP and ARA targets)

`CMakeLists.txt:66-68,71-74,94-97` — `CORE_SOURCES` (which includes `MambaInferenceEngine.cpp`) is listed in both `CLAP_SOURCES` and `ARA_SOURCES`. This means the inference engine is compiled independently for each target, with no shared object. This works but means:
- No shared state between CLAP and ARA instances (intentional for RT safety, but should be documented).
- Symbol collisions if both `.clap` and `.so` are loaded into the same process.

- Consider building `MambaInferenceEngine` as a `STATIC` library linked by both targets to avoid recompilation.
- OR document that this is intentional (separate engine instances per plugin type).

## 23. CMake: `readerwriterqueue` is fetched but never used

`CMakeLists.txt:26-31` — ReaderWriterQueue is fetched and made available, but no source file includes it. The ARA controller was supposed to use a lock-free SPSC queue for background analysis (per `TODO_PROGRESS.md` line 56), but no queue exists in the current code.

- Remove the fetch if unused, or implement the background analysis queue (see #24).

## 24. ARA: Background analysis thread and lock-free queue are mentioned but not implemented

`TODO_PROGRESS.md:56` claims "background analysis thread using lock-free SPSC queue" but the actual `MambaPlaybackRenderer` has no background thread, no queue, and no analysis logic. The constructor just creates an engine and calls `init()`.

- Implement a background analysis pipeline: a non-RT thread that processes audio through `mamba_analysis.onnx` (which exists in `models/` but is never used), with results communicated to the RT thread via `readerwriterqueue`.
- OR remove the claim from documentation and the unused ReaderWriterQueue dependency.

## 25. ARA: `mamba_analysis.onnx` is never loaded or used

The `models/` directory contains both `mamba_rt.onnx` and `mamba_analysis.onnx` (+ sidecars), but only the RT model is ever loaded. The analysis model was intended for offline feature extraction in the ARA workflow.

- Implement analysis model loading in `MambaDocumentController` or a dedicated analysis runner.
- OR remove the analysis model if not needed, and simplify `export_mamba_onnx.py` to only export the RT model.

## 26. Python export: opset version mismatch between function signature and `__main__`

`export_mamba_onnx.py:262` — `export_mamba_to_onnx` defaults to `opset_version=17`, but `__main__` calls it with `opset_version=18` (line 347). The function default is misleading.

- Change the function default to `18` to match actual usage, or parameterize it from `__main__`.

## 27. Python export: Both models are identical — `mamba_analysis.onnx` is a copy of `mamba_rt.onnx`

`export_mamba_onnx.py:353-358` — The analysis model is exported with the same architecture and weights as the RT model. There's no architectural difference. This wastes ~25 MB of disk space with a duplicate model.

- Either give the analysis model a different architecture (e.g., wider hidden dim, no state streaming) or export a single shared model.

## 28. Python export: No trained weights — models use random initialization

`export_mamba_onnx.py:263-266` — `MambaRTModel` is instantiated with random weights and immediately exported. The model has never been trained. Audio output will be random noise scaled by the projection layers.

- This is expected for a scaffold project, but should be documented: "Models are untrained. Audio output is not meaningful until training is implemented."
- Add a training pipeline or placeholder script.

## 29. Python export: `dt` computation adds bias twice in `MambaSSMCore.forward`

`export_mamba_onnx.py:100` — `dt = F.softplus(self.dt_proj(dt_latent) + self.dt_proj.bias)`. `self.dt_proj` is a `nn.Linear` which already adds its bias. Then `.bias` is added again explicitly. This double-bias is likely a bug from the original Mamba reference implementation where `dt_proj` was defined without bias.

- Verify against the official Mamba implementation. If `dt_proj` has `bias=True`, remove the explicit `+ self.dt_proj.bias`.

## 30. CLAP: State save/load has no version header or migration path

`src/clap/MambaClapPlugin.cpp:176-209` — `MambaPresetData` is a packed struct with 3 floats and no version field. Any change to the parameter set or data layout will break backward compatibility with saved presets.

- Add a version header (magic + version number) before the parameter data.
- Handle unknown versions gracefully in `mamba_state_load`.

## 31. CLAP: State load does not reset engine state

`src/clap/MambaClapPlugin.cpp:193-203` — When loading state, only parameter atomics are updated. The ONNX engine's SSM/conv states are not reset, so old recurrent state from the previous session bleeds into the new session.

- Call `engine->resetState()` after loading state, or at least after any parameter change that would require a fresh state.

## 32. CLAP: `mamba_on_main_thread` is empty — no deferred task processing

`src/clap/MambaClapPlugin.cpp:341` — The main thread callback is a no-op. If the plugin ever needs to defer work (e.g., model reloading, UI updates), there's no infrastructure for it.

- Implement a lock-free task queue that `processBlock` pushes to and `on_main_thread` drains.
- OR document that no deferred work is needed in the current design.

## 33. CLAP: `clap_params_flush` does not output gesture begin/end events

`src/clap/MambaClapPlugin.cpp:134-149` — The flush callback processes incoming parameter changes but never calls `out->try_push()` for gesture begin/end events. Hosts that rely on gesture boundaries for automation recording will not record automation correctly.

- When a parameter value changes, push `CLAP_EVENT_PARAM_GESTURE_BEGIN` before the first change and `CLAP_EVENT_PARAM_GESTURE_END` after the last change for that parameter.

## 34. No test suite exists

There are zero tests — no unit tests, no integration tests, no CLAP validator, no ARA validation.

- Add CLAP validation: run `clap-validator` from the CLAP SDK against `MambaSuiteClap.clap`.
- Add unit tests for `MambaInferenceEngine` (init, process, reset, state continuity).
- Add Python-side model verification tests (already partially exist in `export_mamba_onnx.py` but are not in a test framework).
- Wire into CMake via `ctest` or a separate test runner.

## 35. Plugin binaries are not stripped — debug symbols bloat release

Both `MambaSuiteClap.clap` (147 KB) and `MambaSuiteARA.so` (795 KB) contain full symbol tables. Release builds should strip non-essential symbols.

- Add CMake install/strip rules or `set_target_properties(... LINK_FLAGS -s)` for Release builds.
- Keep only essential exported symbols (`clap_entry`, ARA factory getter).

## 36. No `.clap` bundle directory structure

CLAP plugins on Linux are typically distributed as a directory bundle: `PluginName.clap/Contents/linux-x86-64/PluginName.clap` (the binary) plus resources. The current build outputs a bare `.clap` binary file, which some hosts may not discover correctly.

- Create the proper CLAP bundle directory structure in the build output.
- Place `models/` inside the bundle so `dladdr`-based path resolution works.

## 37. Plugin descriptor has placeholder URLs

`src/clap/MambaClapPlugin.cpp:36-38` and `src/ara/MambaAraController.cpp:166-167` — All URLs are `https://example.com/mamba_audio_suite*`. These will show up in host UIs and are non-functional.

- Replace with actual project URLs or remove (CLAP allows `nullptr` for optional URL fields).

## 38. No Git repository initialized

The project has no `.git` directory, no `.gitignore`, and no version control.

- `git init` and add a `.gitignore` for `build/`, `*.o`, `.onnx.data` (if models are re-generated), and IDE files.
- Make an initial commit.

## 39. No license file

`TODO_PROGRESS.md` notes "Choose and apply license (e.g., GPLv3/MIT)" as incomplete.

- Add a `LICENSE` file with the chosen license.
- Update `clap_plugin_descriptor_t` and `MambaFactoryConfig` with the license identifier if applicable.
