# CLARA — Parameter Reference & Model Export

## 1. CLAP Parameters

Three parameters are exposed via the `CLAP_EXT_PARAMS` extension. All use `nullptr` cookies (no per-parameter cookie data).

### Bypass (ID: 0)

| Field          | Value                         |
|----------------|-------------------------------|
| Identifier     | `PARAM_BYPASS` (0)            |
| Flags          | `CLAP_PARAM_IS_STEPPED`, `CLAP_PARAM_IS_BYPASS` |
| Range          | `[0.0, 1.0]` (stepped)        |
| Default        | `0.0` (off)                   |
| Module         | `"Main"`                      |
| Display        | `"Off"` / `"On"` (threshold > 0.5) |

The bypass flag triggers a click-free 50 ms linear crossfade ramp. The ramp is computed per-sample in `mamba_process()` using `crossfadeStep = 1.0f / (0.05f * sampleRate)`, shared across all channels to prevent stereo aliasing.

### Wet/Dry (ID: 1)

| Field          | Value                         |
|----------------|-------------------------------|
| Identifier     | `PARAM_WET_DRY` (1)           |
| Flags          | `CLAP_PARAM_IS_AUTOMATABLE`   |
| Range          | `[0.0, 1.0]`                  |
| Default        | `1.0` (full wet)              |
| Module         | `"Mix"`                       |
| Display        | `"XX.X%"` (percent)           |

Controls the balance between processed (wet) and dry signal in the output mix. Applied after intensity scaling.

### Intensity (ID: 2)

| Field          | Value                         |
|----------------|-------------------------------|
| Identifier     | `PARAM_INTENSITY` (2)         |
| Flags          | `CLAP_PARAM_IS_AUTOMATABLE`   |
| Range          | `[0.0, 1.0]`                  |
| Default        | `0.5`                         |
| Module         | `"Processing"`                |
| Display        | `"XX.X%"` (percent)           |

Scales the wet signal before the wet/dry mix: `wet = dry + (wet - dry) * intensity`. Applied per-sample, per-channel before the wet/dry blend.

### Parameter Storage (Atomics)

All three parameters are stored as `std::atomic<float>` in `MambaPluginState` (`MambaClapPlugin.hpp:22-24`):

```cpp
std::atomic<float> bypass{0.0f};
std::atomic<float> wetDry{1.0f};
std::atomic<float> intensity{0.5f};
```

Loads use `memory_order_acquire`, stores use `memory_order_release`. Updates arrive from both the `clap_params_flush` callback and inline `CLAP_EVENT_PARAM_VALUE` processing in `mamba_process()`.

---

## 2. State (Presets)

State save/load is provided via the `CLAP_EXT_STATE` extension using `clap_istream` / `clap_ostream`.

### Binary Format

| Offset | Size     | Field       | Description                     |
|--------|----------|-------------|---------------------------------|
| 0      | 4 bytes  | `version`   | `int32_t`, currently `1`        |
| 4      | 4 bytes  | `bypass`    | `float`                         |
| 8      | 4 bytes  | `wetDry`    | `float`                         |
| 12     | 4 bytes  | `intensity` | `float`                         |
| **16** |          | **Total**   |                                 |

Total: **16 bytes** (4 bytes header + 12 bytes payload).

### MambaPresetData Struct

Defined at `MambaClapPlugin.cpp:180`, packed (no padding):

```cpp
#pragma pack(push, 1)
struct MambaPresetData {
    float bypass;
    float wetDry;
    float intensity;
};
#pragma pack(pop)
```

### Versioning Strategy

- Version `1` is the current format.
- On load (`mamba_state_load`): the version is read first. If it exceeds `1`, the load fails (`return false`).
- Forward compatibility: a future version can extend the struct and increment the version number. Loaders reject versions they do not understand.
- On save (`mamba_state_save`): always writes version `1`.

### State Loading Side Effects

When state is loaded:

1. The three `std::atomic<float>` fields are updated via `memory_order_release`.
2. `currentBypassValue` is set to the loaded bypass value, synchronising the crossfade ramp state.
3. If an engine instance exists, `engine->resetState()` is called to clear SSM/conv internal states.

---

## 3. ARA Archive State

The ARA `MambaDocumentController` persists the same three parameters via `HostArchiveReader` / `HostArchiveWriter`.

### Data Format

Raw 12-byte struct, no version header:

```cpp
struct ArchivedParams {
    float bypass;
    float wetDry;
    float intensity;
};
```

### Key Differences from CLAP State

| Aspect               | CLAP State                       | ARA Archive                      |
|----------------------|----------------------------------|----------------------------------|
| Header               | `int32_t` version (`1`)          | None                             |
| Validation           | Version check (`<= 1`)           | Size check (`>= sizeof(params)`) |
| Total Size           | 16 bytes                         | 12 bytes                         |
| Atomic Store         | `memory_order_release`           | `memory_order_release`           |
| Engine Reset         | Yes (`engine->resetState()`)     | No                               |

The ARA archive stores into the same `MambaPluginState` atomics (bypass, wetDry, intensity) that the CLAP side uses. On restore, `doRestoreObjectsFromArchive` reads exactly 12 bytes — any larger payload is tolerated (size is a minimum check), which provides limited forward compatibility.

---

## 4. Model Architecture

### Hyperparameters

| Parameter    | Value | Description                            |
|-------------|-------|----------------------------------------|
| `d_model`   | 64    | Model dimension (hidden size per layer)|
| `n_layers`  | 4     | Number of stacked Mamba layers         |
| `d_state`   | 16    | SSM state dimension (N)                |
| `conv_width`| 4     | Causal depthwise convolution width     |
| `dt_rank`   | 16    | Rank of the dt projection             |

Total parameters: **~52 K** (random weights at current export).

### Architecture Overview

```
audio_chunk [B, seq_len, 1]
    |
InputProjection (Linear 1 -> d_model)
    |
MambaLayer 0 ── conv1d (depthwise, width=conv_width) + S6 selective scan
    |
MambaLayer 1
    |
MambaLayer 2
    |
MambaLayer 3
    |
LayerNorm (d_model)
    |
OutputProjection (Linear d_model -> 1)
    |
processed_audio [B, seq_len, 1]
```

Each `MambaLayer` (defined in `export_mamba_onnx.py:110`) consists of:

1. **Input projection** (`Linear 2*d_model`): splits into SSM path and gate path.
2. **Causal Conv1d** (`Conv1d d_model -> d_model, groups=d_model`): depthwise convolution with `padding = conv_width - 1`. State is prepended to each block to maintain correct causal context across chunks.
3. **SiLU activation**: `conv_out * SiLU(gate)`.
4. **S6 SSM Core** (`MambaSSMCore`): projects to `dt`, `B`, `C` via `x_proj` (Linear `d_model -> dt_rank + 2*d_state`), then `dt_proj` (Linear `dt_rank -> d_model` with Softplus). `A` is the negative-exponent of a learned `log_A_real` parameter.
5. **Selective scan** (`SelectiveScanStep`): unrolled loop over `seq_len` using discretised `dA`, `dB` matrices.
6. **Output projection** (`Linear d_model -> d_model`).

### SSM Discretisation

- `A = -exp(log_A_real)` — always negative-definite, ensures stability.
- `dA = exp(dt * A)` — zero-order hold discretisation.
- `dB = dt * B` — Euler discretisation.
- State update: `h' = dA * h + dB * x`.
- Output: `y = C * h' + D * x`.

### ONNX Input Tensors (9 inputs)

| Input Name       | Shape                     | Description                         |
|------------------|---------------------------|-------------------------------------|
| `audio_chunk`    | `[1, seq_len, 1]`         | Mono audio samples, seq_len dynamic |
| `conv0`          | `[1, 64, 4]`              | Layer 0 conv state                  |
| `conv1`          | `[1, 64, 4]`              | Layer 1 conv state                  |
| `conv2`          | `[1, 64, 4]`              | Layer 2 conv state                  |
| `conv3`          | `[1, 64, 4]`              | Layer 3 conv state                  |
| `ssm0`           | `[1, 64, 16]`             | Layer 0 SSM state                   |
| `ssm1`           | `[1, 64, 16]`             | Layer 1 SSM state                   |
| `ssm2`           | `[1, 64, 16]`             | Layer 2 SSM state                   |
| `ssm3`           | `[1, 64, 16]`             | Layer 3 SSM state                   |

### ONNX Output Tensors (3 outputs)

| Output Name        | Shape                 | Description                              |
|--------------------|-----------------------|------------------------------------------|
| `processed_audio`  | `[1, seq_len, 1]`     | Processed mono audio, seq_len dynamic    |
| `new_conv_state`   | `[1, 4, 64, 4]`       | Stacked conv states for all 4 layers     |
| `new_ssm_state`    | `[1, 4, 64, 16]`      | Stacked SSM states for all 4 layers      |

### Dynamic Axes

Only the sequence length dimension is dynamic:

```python
dynamic_axes={
    'audio_chunk': {1: 'seq_len'},
    'processed_audio': {1: 'seq_len'},
}
```

State tensors (`conv{i}`, `ssm{i}`, `new_conv_state`, `new_ssm_state`) have **fixed shapes** — no dynamic axes. This ensures ONNX Runtime can pre-allocate state buffers.

### State Management in C++

The `MambaInferenceEngine` (`MambaInferenceEngine.hpp`) manages per-layer state as `std::vector<float>` arrays:

- `convStateIn_[kMaxLayers]` — each `[d_model * conv_width]` = 256 floats.
- `ssmStateIn_[kMaxLayers]` — each `[d_model * d_state]` = 1024 floats.
- `convStateOutStack_` — `[nLayers * d_model * conv_width]` = 1024 floats.
- `ssmStateOutStack_` — `[nLayers * d_model * d_state]` = 4096 floats.

On each `processBlock` call, per-layer states are unstacked from output tensors back into `convStateIn_[i]` / `ssmStateIn_[i]` for the next block. State reset is deferred via an `std::atomic<bool> resetRequested_` flag, cleared at the start of the next `processBlock`.

---

## 5. Model Export

### How to Run

```bash
python3 export_mamba_onnx.py
```

The script produces `models/mamba_rt.onnx` and `models/mamba_rt.onnx.data` in the project root.

### Dependencies

| Package      | Version / Notes                    |
|-------------|------------------------------------|
| `torch`     | Any version with `torch.jit.trace` |
| `onnx`      | For ONNX export                    |
| `onnxruntime`| For self-verification              |
| `numpy`     | For verification tensor creation   |

### Export Parameters

| Parameter     | Default | Description                            |
|---------------|---------|----------------------------------------|
| `d_model`     | 64      | Model dimension                        |
| `d_state`     | 16      | SSM state dimension                    |
| `n_layers`    | 4       | Number of Mamba layers                 |
| `conv_width`  | 4       | Conv1d kernel width                    |
| `dt_rank`     | 16      | dt projection rank                     |
| `seq_len`     | 256     | Static sequence length for trace       |
| `opset_version` | 18   | ONNX opset version                     |

### What It Produces

| File                 | Description                               |
|----------------------|-------------------------------------------|
| `models/mamba_rt.onnx`     | ONNX graph (model definition + small constants) |
| `models/mamba_rt.onnx.data`| External data sidecar (large weight tensors)    |

The external data format is ONNX's standard external tensor storage. Both files must be deployed together — the `.onnx` file references the `.onnx.data` file by relative path at load time.

### Self-Verification

The export script runs three verification stages automatically:

**Stage 1 — Shape Validation:**

```python
assert out_audio.shape == (1, seq_len, 1)
assert out_conv.shape == (1, n_layers, d_model, conv_width)
assert out_ssm.shape == (1, n_layers, d_model, d_state)
```

**Stage 2 — Numeric Sanity:**

```
assert not np.allclose(out_audio, 0.0)   # non-zero output
assert np.all(np.isfinite(out_audio))      # no NaN or Inf
```

**Stage 3 — State Continuity:**

Processes two consecutive 32-sample chunks, passing the output states of chunk 1 as input states to chunk 2. Verifies that chunk 2 produces non-zero, finite output (proving state is correctly propagated).

### Export Mechanism

Uses `torch.jit.trace` (not `dynamo`). The per-timestep selective scan loop is unrolled as Python control flow during tracing — ONNX Runtime executes the loop iterations as repeated sub-graph calls.

```python
torch.onnx.export(
    model,
    all_args,
    file_path,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=[...],
    output_names=[...],
    dynamic_axes={...},
    dynamo=False,
)
```

### Re-Export Conditions

Re-export is required when:

- Model architecture changes (d_model, n_layers, d_state, conv_width, dt_rank).
- The model is trained (replaces random weights with learned weights).
- ONNX opset version changes.

Re-export is NOT required for runtime configuration changes (sample rate, block size) — those are handled entirely in C++ and the ONNX model processes raw floating-point samples.

---

## 6. Deployment

### File Layout

Plugin binaries resolve the model relative to their own location at runtime:

```
<install-prefix>/
  plugins/
    MambaSuiteClap.clap       # CLAP plugin binary
    MambaSuiteARA.so          # ARA library
    libonnxruntime.so.1.17.1  # ONNX Runtime shared library
    models/
      mamba_rt.onnx           # ONNX model graph
      mamba_rt.onnx.data      # ONNX external weight data (sidecar)
```

### Path Resolution via dladdr

In `clap_create_plugin` (`MambaClapPlugin.cpp:441`):

```cpp
Dl_info info;
if (dladdr((void*)clap_create_plugin, &info) && info.dli_fname) {
    state->modelPath = resolveModelPath(info.dli_fname, "mamba_rt.onnx");
}
```

`dladdr` returns the filesystem path of the shared object containing `clap_create_plugin` (i.e., the `.clap` binary). `resolveModelPath` strips the filename and appends `models/mamba_rt.onnx`. The same resolution applies to `MambaSuiteARA.so` via its own entry point.

### Sidecar File Requirement

The `.onnx.data` file must be present alongside the `.onnx` file. ONNX Runtime's `Ort::Session` loads external data automatically if the sidecar is in the same directory. If missing, session creation fails with an error indicating missing external data.

### Model Path on First Init

The `mamba_init` function (`MambaClapPlugin.cpp:231`) checks:

1. `modelPath` is not empty.
2. The file exists via `access(F_OK)`.

If either check fails, `init()` returns `false` and the plugin does not activate. A fallback path is set during `mamba_activate` if `modelPath` is empty:

```cpp
if (state->modelPath.empty()) {
    state->modelPath = resolveModelPath(nullptr, "mamba_rt.onnx");
}
```
