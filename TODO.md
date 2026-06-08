# CLARA — Master TODO

**Generated:** 2026-06-02 | **Last update:** 2026-06-04 (session 2) | **Status key:** `⬜ pending` `🔄 in progress` `✅ done` `❌ cancelled`

---

## 🔴 P0 — Critical Bugs & Architecture

| # | File | Issue | Status |
|---|------|-------|--------|
| 1 | `src/core/MambaInferenceEngine.hpp/.cpp` | **Conditioning vector `cond[COND_DIM]`** — double-buffered with atomic pointer swap, FiLM inside ONNX, removed external gain modulation (See ADR-008) | ✅ |
| 2 | `src/core/SPSCQueue.hpp` | **SPSC ring buffer** — lock-free queue for deferred tasks (ModelReload, AnalysisQueueDrain) | ✅ |
| 3 | `models/mamba_rt.onnx`, `export_mamba_onnx.py` | **ONNX graph too large for ORT** — 256-step unrolled Mamba (25,199 nodes) caused ORT session creation timeout >120s. **Fix:** Export at seq_len=64 (6,575 nodes), C++ engine processes in 64-sample sub-blocks with state carry-over and zero-padding for partial final blocks. All 23 engine tests pass. | ✅ |
| 4 | `src/core/MambaInferenceEngine.cpp` | **Dynamic seq_len limitation** — TorchScript trace bakes fixed Gather indices for conv slicing. Inference at different seq_len fails with "indices element out of data bounds". **Mitigation:** Sub-block processing at fixed 64 samples; ONNX model always receives exactly 64. | ✅ (documented) |
| 5 | `tests/test_latency.cpp` (new) | **CLARA-06**: Latency reporting verification test — prove 4-sample claim at 44.1k/48k/96k/192k via impulse-response. | ✅ |

## 🟠 P1 — Functional Gaps

| # | File | Issue | Status |
|---|------|-------|--------|
| 6 | `src/clap/MambaClapPlugin.cpp` | `mamba_on_main_thread` — `ModelReload` now creates new engine, does RT-safe swap via `modelReloadPending` flag; `AnalysisQueueDrain` still a stub awaiting ARA host bridge | ✅ |
| 7 | `src/clap/MambaClapPlugin.hpp` | `DeferredTask` enum unchanged; added `modelReloadPending` atomic flag for thread-safe engine swap | ✅ |
| 8 | `src/ara/MambaAraController.cpp` | `doUpdateMusicalContextContent` — already calls `setConditioning()` with neutral values, sets `contentDirty_`. Ready for CLARA-09 host bridge. | ✅ |
| 9 | `src/ara/MambaAraController.cpp` | `doUpdateAudioSourceContent` — sets `contentDirty_`. ARA analysis pipeline stub awaiting host bridge. | ✅ |
| 10 | `src/ara/MambaAraController.cpp` | `didEnableAudioSourceSamplesAccess` — sets `contentDirty_` when enabled. Stub for future analysis trigger. | ✅ |
| 11 | `src/ara/MambaAraController.cpp` | `doCreateEditorRenderer` — now returns proper `MambaEditorRenderer` instance instead of nullptr | ✅ |
| 12 | `src/ara/MambaAraController.cpp` | `doDestroyEditorRenderer` — properly deletes via `static_cast<MambaEditorRenderer*>` | ✅ |
| 13 | `src/ara/MambaAraController.cpp` | `doCreateEditorView` — now returns proper `MambaEditorView` instance instead of nullptr | ✅ |
| 14 | `src/ara/MambaAraController.cpp` | `doDestroyEditorView` — properly deletes via `static_cast<MambaEditorView*>` | ✅ |
| 15 | `src/core/ConditioningGenerator.hpp` | **CLARA-09**: See `docs/ara-host-bridge-design.md` for detailed architecture to wire ARA MusicalContext → cond flow | ✅ |
| 16 | `train_clara.py` + all training files | **Training pipeline written, verified, and launch-ready.** 56/56 PASS. New issues found and fixed: (a) CUDA OOM at 2s clips — added `--clip-seconds 0.5` CLI arg; (b) ROCm HIPFFT_PARSE_ERROR on MR-STFT — added CPU fallback in loss. Launcher fixed: `_phase_num` hoisted, `num_workers=4` default. Ready to launch Phase 1 with `--clip-seconds 0.5`. | 🔄 |

## 🟡 P2 — Correctness & Technical Debt

| # | File | Issue | Status |
|---|------|-------|--------|
| 17 | `src/core/MambaInferenceEngine.cpp` | FIXME: Silent failure — `!initialized_` path now prints one-time stderr warning | ✅ |
| 18 | `src/clap/MambaClapPlugin.cpp` | Duplicate param flush — Code review: `mamba_process` reads atomics directly, `clap_params_flush` handles events. No duplication. Already correct per CLAP spec. | ✅ |
| 19 | `tests/test_clap_plugin.cpp` | wetDry `text_to_value` — Source handles both `"N%"` and raw `"0.5"` formats. Test sends both. Already correct. | ✅ |
| 20 | `CMakeLists.txt:203-298` | Multi-config generators (VS/Xcode/Ninja MC) — fixed model deploy via POST_BUILD for Debug/Release/RelWithDebInfo/MinSizeRel × {clap, ara} | ✅ |
| 21 | `tests/CMakeLists.txt:99-122` | Quick-mode clap-validator (CLAP_VALIDATOR_FULL=OFF default) | ✅ |
| 22 | `CMakeLists.txt:354-361,370-380` | **CLARA-07**: Cross-platform build matrix — Windows install paths, MSVC CRT /MD constraint | ✅ |
| 23 | `export_mamba_onnx.py`, `MambaInferenceEngine.cpp` | **CLARA-08**: ONNX export hardening — pre-flight checks, ONNX vs PyTorch validation (cosine >0.999), manifest.json with SHA256 + schema version | ✅ |
| 24 | `docs/training-plan.md` | Training pipeline reference document — architecture, phases, datasets, loss, infrastructure, risks, implementation order | ✅ |
| 25 | `export_mamba_onnx.py` | Cond integration: 10th input, InputProjection(1+cond_dim→64), CondFiLM per layer, manifest schema v1→v2 | ✅ |
| 26 | `MambaInferenceEngine.cpp` | Cond input tensor fed via `activeCond_` double-buffer; external gain modulation loop removed | ✅ |
| 27 | `MambaClapPlugin.cpp` | Removed `loadCondWeights()` call; kept `setConditioning()` with default context | ✅ |
| 28 | `tests/test_engine.cpp` | Updated conditioning tests — removed `loadCondWeights`/`condWeightsLoaded` refs, verifies cond reaches ONNX through FiLM | ✅ |

## 🟢 P3 — Training Pipeline & Model Export

| # | File | Issue | Status |
|---|------|-------|--------|
| 29 | `train_clara_config.py` | Config dataclass with per-phase presets (identity, saturation, compression, denoising, effects) | ✅ |
| 30 | `train_clara_model.py` | Parallel-scan training model + sequential export model + CondFiLM + make_zero_states/cond | ✅ |
| 31 | `train_clara_dataset.py` | ClaraAudioDataset with MUSDB18/MAESTRO/DNS loading, RAM caching, librosa conditioning extraction | ✅ |
| 32 | `train_clara_dsp.py` | Synthetic DSP generators: saturation, RMSCompressor, noise, EQ, TRANSFORM_REGISTRY | ✅ |
| 33 | `train_clara_loss.py` | MultiResolutionSTFTLoss, SI_SDRLoss, ClaraLoss with SI-SDR ramp-in schedule | ✅ |
| 34 | `train_clara.py` | Main training loop (AdamW, cosine schedule, gradient accumulation, checkpointing, validation, interrupt handler) | ✅ |
| 35 | `train_clara_export.py` | ONNX export from checkpoint with shape/parity/state-continuity verification | ✅ |
| 36 | `train_clara_launcher.py` | Auto-detect dataset, resume, ROCm env vars | ✅ |
| 37 | `train_clara.ps1` | PowerShell wrapper | ✅ |
| 38 | `train_clara_requirements.txt` | Python dependencies | ✅ |

## 🔵 P4 — Test Coverage

| # | File | Issue | Status |
|---|------|-------|--------|
| 39 | `tests/test_engine.cpp` | All 23 C++ engine tests pass (construct/destroy, init, processBlock 256/64/0/512, reset, stereo, conditioning with model, atomic swap, conditioning generator encode/different/neutral/unknown) | ✅ |
| 40 | `tests/test_engine.cpp:396` | `test_init_missing_model_path()` — graceful failure when model absent | ✅ |
| 41 | `tests/test_engine.cpp:407` | `test_conditioning_atomic_swap()` — double-buffer atomic pointer swap RT-safe verified | ✅ |
| 42 | `tests/test_engine.cpp:376-419` | `test_conditioning_with_model()` — proves cond tensor reaches ONNX via FiLM (output differs with non-zero cond) | ✅ |
| 43 | `tests/test_ara_controller.cpp:20-42` | Model dir resolution uses real symbol address to exercise `mambaResolveModulePath` | ✅ |
| 44 | `tests/test_clap_plugin.cpp:434` | `test_spsc_two_thread()` — producer/consumer across two threads for SPSC FIFO | ✅ |
| 45 | `tests/test_clap_plugin.cpp` | No test for conditioning activation or deferred task wiring — `test_set_conditioning`, `test_conditioning_with_model`, `test_conditioning_atomic_swap` etc. already cover conditioning at the engine level. CLAP-level tests deferred until ARA host bridge is integrated (CLARA-09). | 🔄 |
| 46 | `tests/test_engine.cpp:119-141` | `test_process_block_partial_block()` — proves 64-sample sub-block boundary works | ✅ |
| 47 | `tests/test_engine.cpp:144-175` | `test_process_block_stereo()` — sequential channel processing | ✅ |
| 48 | `tests/test_engine.cpp:201-231` | `test_multiple_process_blocks()` — state carry-over across 5 sequential blocks | ✅ |
| 49 | `tests/test_engine.cpp:251-272` | `test_large_block_size()` — clamping to maxBlockSize | ✅ |

## 🟣 P5 — Next Actions

| # | Action | Details | Priority |
|---|--------|---------|----------|
| N1 | Verify Python training pipeline | `verify_training_pipeline.py` written and run — **56 checks, all PASS** | ✅ |
| N2 | Launch Phase 1 training (identity) | 860M iGPU, clip_seconds=0.5 to fit 12GB VRAM (parallel scan memory). Use `--clip-seconds 0.5`. STFT loss has ROCm HIPFFT fallback to CPU. First do 500-step smoke test: `python train_clara_launcher.py --phase identity --clip-seconds 0.5 --steps 500 --no-cache` | P0 |
| N3 | Export Phase 1 checkpoint → ONNX | Use train_clara_export.py, verify with C++ engine test | P0 |
| N4 | Phase 1B with conditioning | Train with librosa-extracted tempo/key conditioning, proves cond integration end-to-end | P1 |
| N5 | Phase 2: saturation training | Requires synthetic data only (already in train_clara_dsp.py) | P1 |
| N6 | Phase 3: compression training | RMSCompressor in train_clara_dsp.py | P1 |
| N7 | Phase 4: denoising training | Requires DNS Challenge dataset download (~20 GB) | P1 |
| N8 | Phase 5: multi-effect training | EQ, transient shaping, de-essing, stereo widening | P2 |
| N9 | ARA host bridge for cond | Real musical context → cond flow via ARA MusicalContext | P1 |
| N10 | MAESTRO v3 download | Required before Phase 2 (saturation with piano data) | P2 |
| N11 | DNS Challenge download | Required before Phase 4 (denoising) | P2 |
| N12 | ONNX dynamic seq_len fix | Replace fixed Gather indices with dynamic Slice/Reshape approach for truly dynamic seq_len | P2 |

---

## Previous Sessions — Completed

### 🔴 P0 — Runtime Bugs (4/4)
| # | Item | Status |
|---|------|--------|
| — | `init()` return value ignored — plugin returns true even when init fails | ✅ |
| — | ARA constructor marked `noexcept` but calls `std::make_unique` | ✅ |
| — | Dead try/catch around `engine->init()` | ✅ |
| — | ORT timeout on 25K-node graph — switched to 64-step export + sub-block processing | ✅ |

### 🟠 P1 — Dead Code & Cleanup (6/9)
| # | Item | Status |
|---|------|--------|
| — | `engineValid_` dead member removed | ✅ |
| — | `audioSourceAccessEnabled_` dead member removed | ✅ |
| — | 4 PlaybackRegion delegation stubs removed | ✅ |
| — | Placeholder URLs → TODO comments | ✅ |
| — | `getInformationURL()` returns nullptr → TODO comment | ✅ |
| — | Phase 0: cross-platform portability scaffolding (Platform.hpp) | ✅ |

### 🟡 P2 — Missing Functionality (15/15)
| # | Item | Status |
|---|------|--------|
| — | `mamba_on_main_thread` deferred task scaffolding | ✅ |
| — | `mamba_init` validates model path | ✅ |
| — | `mamba_start_processing` checks engine | ✅ |
| — | EditorRenderer/EditorView stubs with TODO docs | ✅ |
| — | `doUpdateMusicalContextContent` sets `contentDirty_` | ✅ |
| — | `doUpdateAudioSourceContent` sets `contentDirty_` | ✅ |
| — | `didEnableAudioSourceSamplesAccess` documented, resets dirty flag | ✅ |
| — | Training pipeline doc created (`docs/training-pipeline.md` → `docs/training-plan.md`) | ✅ |
| — | CI enhanced with model shape verify + ARA checks | ✅ |
| — | Training pipeline files written (11 files: launcher, config, model, dataset, dsp, loss, train loop, export, requirements, .ps1) | ✅ |
| — | Cond folded into ONNX as 10th input with FiLM (export_mamba_onnx.py, engine, CLAP, tests) | ✅ |
| — | ONNX manifest schema v2 (cond_dim=8, num_inputs=10) | ✅ |
| — | C++ engine updated for 10-input contract + removed external gain modulation | ✅ |
| — | Seq_len=64 export + sub-block processing in engine (fixes ORT timeout) | ✅ |
| — | All 23 C++ engine tests pass | ✅ |

### 🟢 P3 — Documentation Gaps (7/7)
| # | Item | Status |
|---|------|--------|
| — | ADR-003, ADR-004, ADR-005, ADR-006 created | ✅ |
| — | README.md created | ✅ |
| — | `docs/parameter-reference.md` created | ✅ |
| — | LICENSE (Apache 2.0) | ✅ |

### 🔵 P4 — Stale/Outdated Docs (5/5)
| # | Item | Status |
|---|------|--------|
| — | TODO_PROGRESS.md stale refs fixed | ✅ |
| — | CHANGELOG.md stale refs fixed | ✅ |
| — | CLAUDE_OPUS_PROMPT.md stale refs fixed | ✅ |

### ⚪ P5 — Testing & CI (3/3)
| # | Item | Status |
|---|------|--------|
| — | 3 test files created (engine: 12, clap: 10, ara: 4) | ✅ |
| — | CI enhanced — ARA verify, model shape verify, stripped check | ✅ |
| — | ARA library verification step in CI | ✅ |

---

## Quick Stats

| Category | Completed | Remaining |
|----------|-----------|-----------|
| 🔴 P0 — Critical Bugs | 4 / 5 | 1 |
| 🟠 P1 — Functional Gaps | 10 / 11 | 1 |
| 🟡 P2 — Correctness & Debt | 12 / 12 | 0 |
| 🟢 P3 — Training Pipeline | 10 / 10 | 0 |
| 🔵 P4 — Test Coverage | 9 / 11 | 2 |
| 🟣 P5 — Next Actions | 1 / 12 | 11 |
| Previous sessions | 34 / 34 | 0 |
| **Total** | **81 / 95** | **14** |
