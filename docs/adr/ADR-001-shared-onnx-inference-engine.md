# ADR-001: Shared ONNX Runtime Inference Engine for Real-Time Mamba SSM

**Status:** Accepted (implemented) — **Historical / Superseded**
**Date:** 2026-05-30
**Note:** This ADR documents the Mamba-1 / ONNX Runtime era. CLARA has since migrated to the Mamba-3 custom C++ engine with flat binary weights. See [`MAMBA3_MIGRATION_PLAN.md`](../MAMBA3_MIGRATION_PLAN.md) for the current architecture.

## Context

CLARA needs to run a Mamba SSM (Selective State Space Model) neural network in two contexts within a DAW:

1. **CLAP plugin** — real-time audio processing (stereo in/out, variable block sizes from host)
2. **ARA2 library** — offline/non-real-time rendering, timeline-aware processing

The same model must run in both. The decision is **how** to share inference code without duplication while satisfying real-time safety constraints for the CLAP target.

### Model Architecture

The MambaRT model has a fixed architecture chosen to balance audio quality against real-time CPU feasibility:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `d_model` | 64 | Embedding dimension. Must be large enough to represent audio features but small enough for single-sample SSM recurrence. 64 is a common baseline in efficient SSM papers. Each sample through the model is a 64-wide vector matmul (~4KB state per layer). |
| `n_layers` | 4 | Depth of stacked SSM layers. 4 layers provides sufficient representational capacity for audio effects (reverb, equalization, dynamics). 1-2 layers is too shallow for timbral transformations; 8+ layers would exceed RT budget. |
| `d_state` | 16 | SSM state dimension per channel. Controls the memory span of the selective scan. 16 is typical for Mamba-SSM baselines (Gu & Dao, 2023). Total SSM state: 4 layers × 64 × 16 = 4096 floats per instance. |
| `conv_width` | 4 | Causal convolution kernel width. Determines minimum latency contribution of the model. 4 samples of latency is negligible (<0.1ms @ 48kHz). Wider kernels would increase latency without proportional quality gain. |
| `dt_rank` | 16 | Rank of the discretization projection. Controls the dimensionality of the learned timestep (delta) parameter. Set equal to `d_state` as in standard Mamba architecture. |
| `seq_len` | 256 (default) | Default export/block size. Must be moderate enough for RT processing but large enough for SSM to develop its recurrent state. 256 @ 48kHz = 5.3ms per block. |

Total parameters: ~274K. This is small enough for single-core CPU inference at 48kHz with headroom.

### Inference Backend Options

| Option | Binary Size | S6 Scan Support | RT Safety Control | Verdict |
|--------|-------------|-----------------|-------------------|---------|
| **Native C++ Mamba** | 0 MB (inline) | Manual impl needed | Full control | Rejected — S6 selective scan uses cumulative softplus, discretized 2D state recurrence, and stacked tensor operations. Correct and performant reimplementation is substantial work. |
| **LibTorch** | ~1 GB | Yes (full PyTorch) | Limited | Rejected — unacceptable binary size for audio plugin distribution. |
| **RTNeural** | ~0 MB (header) | No | Full control | Rejected — supports MLP/RNN/GRU/LSTM only. No S6 selective scan model. |
| **ONNX Runtime** | ~40 MB | Yes (via exported model) | Thread count, graph opt level | **Chosen.** |

### CPU-Only Reasoning

ONNX Runtime supports CUDA, TensorRT, DirectML, OpenVINO, and CoreML execution providers. We use only the default CPU provider because:
- Audio plugins must run on diverse hardware — cannot assume GPU presence.
- ~274K param model inference latency on modern x86 is well within a 48kHz block budget.
- ONNX Runtime's CPU graph optimizations (fusion, constant folding) provide sufficient performance.
- GPU providers add deployment complexity (CUDA redistributable, driver version requirements).

If future profiling shows CPU inference approaching the RT budget, CUDA execution provider can be enabled without changing the model or the C++ code — only the session creation path.

## Decision

### 1. Use ONNX Runtime v1.17.1

Pinned to a specific minor version for reproducibility. v1.17.1 was the latest stable at implementation time. ORT's C API has undergone ABI-breaking changes across major versions; upgrading requires retesting.

The prebuilt `onnxruntime-linux-x64` tarball is fetched via `FetchContent` at build time. No system-level installation required.

### 2. Encapsulate ORT in `MambaInferenceEngine` (static library)

The inference code lives in `src/core/`, compiled as the `MambaCore` STATIC library, then linked into both `MambaSuiteClap` and `MambaSuiteARA`.

Alternatives considered:

| Approach | Issue |
|----------|-------|
| **Header-only** (`inline` or `.hpp`-only) | Would work but bloats compile time (ORT headers are large) and increases binary size marginally. No real benefit since there are only two translation units consuming it. |
| **Compile `.cpp` into each target** (`add_library` with source files listed in each target) | Duplicates compilation AND object code. Bugs fixed in one target could be missed in the other. Increases build time. |
| **Shared library** (`.so` for MambaCore) | Adds a third `.so` to deploy alongside the plugins. No benefit — the inference engine has no reason to be hot-swappable or independently versioned. |
| **Static library** (current) | Compiled once, linked into both targets. Minimal binary size increase (linker includes only referenced symbols). No extra deployment artifact. Single fix point. **Chosen.** |

### 3. ONNX Model I/O Contract

**10 inputs → 3 outputs. This contract must never change without updating `export_mamba_onnx.py` in lockstep.**

| Index | Input Name | Shape | Description |
|-------|-----------|-------|-------------|
| 0 | `audio_chunk` | `{1, seq_len, 1}` | Mono audio samples, float32 |
| 1-4 | `conv{0..3}` | `{1, 64, 4}` | Per-layer causal conv state (rolling buffer) |
| 5-8 | `ssm{0..3}` | `{1, 64, 16}` | Per-layer SSM recurrent state (h matrix) |
| 9 | `cond` | `{1, 8}` | Conditioning vector (FiLM), 8-dim float32 |

| Index | Output Name | Shape | Description |
|-------|------------|-------|-------------|
| 0 | `processed_audio` | `{1, seq_len, 1}` | Processed mono audio |
| 1 | `new_conv_state` | `{1, 4, 64, 4}` | Stacked conv states (layer dim=1) |
| 2 | `new_ssm_state` | `{1, 4, 64, 16}` | Stacked SSM states (layer dim=1) |

The output states are stacked (layer count at dim=1) to simplify the ONNX graph — returning 8 separate outputs would complicate the trace. The C++ engine unstacks them into per-layer buffers after each `processBlock()` call.

#### Dynamic seq_len

`seq_len` is dynamic (`dynamic_axes={'audio_chunk': {1: 'seq_len'}}`). This is deliberate — using a fixed large shape and zero-padding the unused portion would feed zeros through the SSM's recurrent state, corrupting it. With dynamic shapes, ORT only operates on the actual sample count.

Tradeoff: Static shapes let ORT perform more aggressive pre-allocation and memory planning. Dynamic shapes incur a small runtime cost for shape propagation. We chose correctness over peak throughput.

#### Sub-block Processing (64 samples)

`processBlock()` decomposes the host's audio block into fixed-size sub-blocks of 64 samples (`kSubBlockSize = 64`). Each sub-block runs one ONNX inference call:
```
while (offset < totalSamples):
    chunkSamples = min(64, remaining)
    pad to 64 if needed
    session_->Run()
    copy output, unstack states
    offset += chunkSamples
```

This bounds ONNX inference to a predictable per-call size regardless of the host's block size (which can range from 64 to 1024+). The 64-sample choice balances dispatch overhead (too small → too many `Run()` calls) against latency (larger sub-blocks increase per-call inference time). The sub-block must be at least `conv_width` (4) to satisfy the causal convolution's minimum input length.

#### Cond Input

The `cond` tensor (10th input, index 9) carries an 8-dimensional conditioning vector (`COND_DIM=8`) into the ONNX graph, enabling FiLM-style feature modulation inside the model. It is double-buffered with an atomic pointer swap for RT-safe updates from non-RT threads. After ONNX inference, an additional external scalar gain modulation is applied using `combinedWeights_` (sum of delta and b projection weights) — see ADR-008 for full details.

### 4. Pre-allocate All RT Buffers in `init()`

All buffers (`inputBuffer_`, `convStateIn_[0..3]`, `ssmStateIn_[0..3]`, `processedAudioOut_`, `convStateOutStack_`, `ssmStateOutStack_`) are sized in `init()` based on `config_.maxBlockSize`. `processBlock()` performs only `memcpy`, `memset`/`std::fill`, and tensor creation — no heap allocation.

The one exception: `thread_local std::vector<Ort::Value>` for input/output tensor arrays. `Ort::Value` has no default constructor, so we cannot use `std::array<Ort::Value, N>` as class members. The `thread_local` vectors are `reserve()`d on first call and never reallocate after capacity is reached. See ADR-005.

#### maxBlockSize = 256 (default)

The default block size is 256 samples, chosen as a balance:
- **Smaller** (64-128): Lower latency but more frequent ORT `session_->Run()` calls, increasing CPU overhead from dispatch.
- **Larger** (512-1024): Fewer dispatches but higher per-block latency. DAWs commonly use 256-512 sample buffers at 48kHz.

The CLAP plugin overrides this at `activate()` time: `std::max(max_frames_count, 256)` — never smaller than 256, but grows to match the host's block size.

### 5. Ort::Env — One Per Engine Instance

Each `MambaInferenceEngine` creates its own `Ort::Env`. ORT docs recommend sharing a single `Env` across all sessions in a process for optimal resource usage. However:
- CLAP and ARA are in separate shared libraries, loaded independently by the DAW.
- ARA's PlaybackRenderer may be created/destroyed multiple times during a session.
- Sharing an `Env` would require a process-global singleton with thread-safe lazy init.
- The cost of two `Env` objects is negligible (a few KB of memory).

This is an accepted deviation from ORT best practice.

### 6. ORT Session Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| `IntraOpNumThreads` | 1 | SSM state dependencies (the unrolled `for t in range(seq_len)` scan) prevent parallelization within a single `processBlock()` call. Additional threads would contend on the same matrices. |
| `GraphOptimizationLevel` | `ORT_ENABLE_ALL` | Performs constant folding, operator fusion, and layout optimization at session creation time. Increases init time by ~10-50ms but reduces per-block inference latency. Worth it for a plugin loaded once per session. |
| `ArenaAllocator` (CPU) | Default (ORT arena) | ORT's arena allocator pre-allocates memory regions and reuses them across `Run()` calls, reducing allocation churn. The downside: allocated memory may not be returned to the OS until session destruction. For a plugin, this is acceptable — memory is retained for the session's lifetime. |

### 7. Exception Handling Strategy

ORT's C++ API throws `Ort::Exception` on errors (model load failure, shape mismatch, inference error). The project enables C++ exceptions globally because fighting ORT's API design would require wrapping every call in a C-ABI compatibility layer.

Key points:
- `init()`: Catches `Ort::Exception`, returns `false`. Callers check the return value — never throws across a C boundary.
- `processBlock()`: Catches `Ort::Exception`, zeroes the output buffer, returns early. Never throws across a C boundary.
- `-fno-rtti` is set in Release (not `-fno-exceptions`). RTTI is disabled globally; exceptions are enabled globally.
- If `-ffast-math` causes numerical issues in ORT inference (altered NaN/Inf behavior, relaxed associativity breaking SSM accumulation), this flag should be the first suspect. ORT does not officially guarantee correctness under `-ffast-math`.

### 8. Deployment Contract

At runtime, the plugin directory must contain:

```
plugin_binary(.clap or .so)/
  libonnxruntime.so.1.17.1    # resolved via $ORIGIN RPATH
  models/
    mamba_rt.onnx
    mamba_rt.onnx.data         # external data sidecar (weights)
```

The `$ORIGIN` RPATH is set at build time (`BUILD_WITH_INSTALL_RPATH ON` and `CMAKE_INSTALL_RPATH="$ORIGIN"`). The `configure_file()` command copies `libonnxruntime.so.1.17.1` to `build/plugins/` so it's adjacent during development. Install rules (lines 159-211 of CMakeLists.txt) deploy the full tree.

## Consequences

### Positive
- Single inference core, two plugin targets — no code duplication.
- ORT handles graph optimization, operator fusion, and memory planning internally.
- Dynamic seq_len prevents SSM state corruption from zero-padded input.
- Pre-allocated buffers guarantee zero heap allocations on the audio thread.
- CPU-only allows deployment on any x86 Linux host without GPU requirements.
- Pinned ORT version ensures reproducible builds and behavior.
- Static library avoids extra deployment artifact or duplicated code.

### Negative
- ORT adds ~40MB deployment payload (`libonnxruntime.so.1.17.1`).
- `Ort::Value` has no default constructor — forces `thread_local` + `reserve()` pattern (ADR-005).
- Exceptions must be globally enabled (ORT C++ API), preventing `-fno-exceptions`.
- `-ffast-math` may interact poorly with ORT's numerical assumptions — untested.
- Python toolchain required for model export (`torch`, `onnx`, `onnxruntime`).
- Model architecture is frozen at export time — changing hyperparameters requires re-export.

### Constraints
- Model I/O contract is hard: 10 inputs / 3 outputs. Any change requires lockstep update of `export_mamba_onnx.py` and `MambaInferenceEngine`.
- `processBlock()` must remain allocation-free — enforced by code review, not tooling.
- `MambaCore` must be compiled with `-fPIC` for linking into shared libraries (set via `POSITION_INDEPENDENT_CODE ON`).
- Fixed point: `d_model=64`, `n_layers=4`, `d_state=16`, `conv_width=4`. Changing any of these requires re-exporting the model and updating shape constants in `MambaInferenceEngine`.

## Compliance

- Verified: `processBlock()` contains no `new`, `malloc`, or `std::vector` resize — only `memcpy`, `std::fill`, `std::min`, and `Ort::Value::CreateTensor` (which uses pre-allocated buffer pointers).
- Verified: All buffers sized in `init()` against `config_.maxBlockSize`.
- Verified: Exceptions from `session_->Run()` caught in both `init()` and `processBlock()`. Never propagates across C boundary.
- Verified: `thread_local std::vector<Ort::Value>` with `reserve(kNumInputs)` / `reserve(kNumOutputs)` in `processBlock()`.
- Verified: Model path uses `dladdr` in both CLAP and ARA entry points.
- Verified: `libonnxruntime.so.1.17.1` copied to `build/plugins/` at build time. `$ORIGIN` RPATH set on both targets.

## Related

- AGENTS.md — build, model export, architecture summary
- `src/core/MambaInferenceEngine.hpp` — class interface with all member declarations
- `src/core/MambaInferenceEngine.cpp` — implementation
- `export_mamba_onnx.py` — model definition and ONNX export
- `CMakeLists.txt` — build config, FetchContent, install rules
- ADR-002: Deferred atomic state reset
- ADR-005: `thread_local` + `reserve()` for ORT tensor arrays
- ADR-006: `dladdr`-based model path resolution
