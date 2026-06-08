# CLARA Python Pipeline Analysis — Gaps & Issues Report

**Date:** 2026-06-07
**Files analyzed:** 8 Python files, 4 ONNX models, 1 manifest

---

## 1. Model Architecture

| Parameter | Value |
|-----------|-------|
| `d_model` | 64 |
| `d_state` | 16 |
| `n_layers` | 4 |
| `conv_width` | 4 |
| `dt_rank` | 16 |
| `cond_dim` (COND_DIM) | 8 |
| Total parameters | ~dozens of thousands (not counted in export for the sequential model specifically) |

**Architecture:** Input projection `(1+8→64)` → 4× MambaLayer → LayerNorm → Output projection `(64→1)`. Each MambaLayer: FiLM modulation (cond→gamma,beta) → SiLU-gated linear projection → causal Conv1d (kernel=4, depthwise, groups=d_model) → Selective SSM scan (S6) → output projection.

**Gaps:**
- **G1.1:** Two divergent model classes across files: `MambaRTModel` (export), `MambaRTModelParallel` (training), `MambaRTModelSequential` (export), and `STFTMambaModel` (train_clara_now). These are four separate implementations of essentially the same layers, with subtle differences in how they're composed. `export_mamba_onnx.py` uses manual `layer0-layer3` attributes; `train_clara_model.py` uses `nn.ModuleList`. Weight transfer between them works but is fragile.
- **G1.2:** The `STFTMambaModel` in `train_clara_now.py` is a *completely different architecture* — processes STFT frames at ~86 fps (n_fft=1024, hop=256) with residual blocks and `norm_before_ssm`. It cannot share weights with the raw-audio models. This is confusing — two competing training scripts (train_clara_now.py vs the yet-to-exist train_clara.py) for different model architectures.
- **G1.3:** No unified `MambaConfig` dataclass shared across all files. Export script hardcodes constants at module level. Training scripts re-type them.

## 2. ONNX Export Correctness

**Export method:** `torch.onnx.export()` with `dynamo=False`, opset 18, `do_constant_folding=True`.

**Dynamic axes:** Only `audio_chunk` and `processed_audio` have dynamic `seq_len`. State tensors are fixed-shape `[1, 64, 4]` and `[1, 64, 16]`. Batch dimension is NOT dynamic — the export is hardcoded to batch=1.

**Gaps:**
- **G2.1:** The SSM scan uses a Python `for` loop (`for t in range(seq_len)`) which gets unrolled into the ONNX graph. This works for tracing but means the ONNX graph size is proportional to the export `seq_len`. At `seq_len=256`, the graph is large. At `seq_len=1024`, it would be massive. This violates the comment "seq_len is dynamic via ONNX dynamic_axes" — the loop is static, the dynamic axis is only for the tensor shapes.
- **G2.2:** `train_clara_export.py` sets `dynamic_axes` for `batch` dimension but NOT `seq_len` — inconsistent with `export_mamba_onnx.py` which sets `seq_len` but NOT `batch`.
- **G2.3:** The `dummy_audio` uses `torch.randn` not `torch.zeros` — this is fine for tracing but means the traced graph may contain different constant-folded values than a zeros-traced version. Non-deterministic export.
- **G2.4:** No check that the exported ONNX graph operations are all supported by ONNX Runtime v1.17.1 (which is what the C++ plugin expects). Opset 18 is modern; custom ops that don't exist in v1.17.1 would crash at runtime.
- **G2.5:** NPU model (`mamba_rt_npu.onnx`, 3.6 MB) embeds weights internally (no `.data` sidecar), while `mamba_rt.onnx` (1.1 MB) uses external data. Different export parameters, no documentation explaining the difference.

## 3. Model Verification

**Checks performed:**
- `preflight_check()`: Python version, torch/onnx/onnxruntime availability, opset compatibility, dry-run forward pass
- `verify_onnx_model()`: Shape validation, non-zero output, finite output, output variance
- `validate_onnx_vs_pytorch()`: Cosine similarity (>0.999), max abs diff (<1e-4)
- State continuity: Two chunks back-to-back with state carryover
- `verify_latency.py`: Impulse response delay measurement

**Gaps:**
- **G3.1:** Verification only uses `cond = zeros(1, 8)`. Never tests with non-zero conditioning values. The FiLM layers are untested.
- **G3.2:** No test for NaN propagation — if conv_state or ssm_state contain NaN, does the model propagate or silence it?
- **G3.3:** No test for extreme input values (e.g., ±10.0 audio) — could cause SSM explosion.
- **G3.4:** Cosine similarity threshold of 0.999 is too permissive for models with random weights. A broken model that outputs zeros could still pass this if the reference also outputs near-zeros.
- **G3.5:** `verify_latency.py` hardcodes state shapes `(1,64,4)` and `(1,64,16)` — would break if model dimensions change.
- **G3.6:** No gradient-based verification (e.g., finite differences between ONNX and PyTorch gradients).

## 4. State Handling

**Initialization:** All states are zero-initialized `torch.zeros(batch, d_model, conv_width)` and `torch.zeros(batch, d_model, d_state)`.

**State update in export model:** Per-layer state extraction via `torch.stack` into `[B, 4, D, W]` and `[B, 4, D, N]` tensors.

**Gaps:**
- **G4.1:** `MambaRTModel` (export) has hardcoded 4 layers — `self.layer0` through `self.layer3` — with manual argument unpacking in `forward()`. Adding a 5th layer requires code changes in 4+ places.
- **G4.2:** The `MambaRTModel` passes `cond` to both `self.input_proj(audio_chunk, cond)` and separately to each `self.layer*(x, conv*, ssm*, cond)`. The conditioning is applied twice: once at input, once per layer. This is intentional (double conditioning) but undocumented.
- **G4.3:** In `train_clara_model.py`, the `MambaRTModelSequential` and `MambaRTModelParallel` use `nn.ModuleList` with a loop — properly generic. But `export_mamba_onnx.py`'s `MambaRTModel` does not. These should be consolidated.
- **G4.4:** No test for state overflow. What happens after 10,000 sequential blocks? Does SSM state drift or explode?

## 5. Conditioning Vector (COND_DIM=8)

**Wiring:**
1. `InputProjection`: `cat([audio, cond_expanded], dim=-1)` → projects `1+8=9` channels to `d_model=64`
2. `CondFiLM` per layer: `cond → Linear(8, 128) → chunk(γ, β)` where `γ, β ∈ [B, 64]`
3. Applied as: `x = x * (1 + γ) + β`

**Gaps:**
- **G5.1:** FiLM projections are zero-initialized (`weight.zero_()`, `bias.zero_()`), so conditioning has NO EFFECT until trained. This is intentional for identity-phase training but means the exported random model ignores cond entirely.
- **G5.2:** The cond vector is applied identically to ALL timesteps in a chunk (unsqueeze + broadcast). There's no time-varying conditioning. For real-time audio effects that change parameters mid-chunk, this is insufficient.
- **G5.3:** Cond is applied both at input projection AND per-layer FiLM. This is double application. Should be one or the other, or at least documented as a design choice.
- **G5.4:** No validation that the C++ plugin's RT-safe conditioning path (atomic pointer swap for cond vector) actually works with non-zero cond values. The verification always uses zeros.
- **G5.5:** The `manifest.json` has `"cond_shape": ["batch", "cond_dim"]` but this uses symbolic names, not actual dimension values. The C++ engine needs to know cond_dim=8, which it gets from `manifest.json` model section.

## 6. Training Compatibility

**Checkpoint loading:** `export_mamba_onnx.py` loads checkpoints via `--checkpoint` flag, supports dicts with `model`, `state_dict`, or bare keys. Handles missing keys gracefully with `strict=False`.

**Gaps:**
- **G6.1:** The export script tries to cross-validate by importing from `train_clara_model` (line 689), but the two model architectures have DIFFERENT parameter names because they use different class structures. Weight transfer silently drops keys.
- **G6.2:** Missing training modules: `verify_training_pipeline.py` imports from `train_clara_config`, `train_clara_loss`, `train_clara_dsp`, and `train_clara_dataset` — NONE of these files exist, despite being listed as "✅ done" in `TODO.md`.
- **G6.3:** `__train_laptop_ref.py` imports from `model_laptop` which also doesn't exist.
- **G6.4:** `train_clara_now.py` (STFT-based training) is fully self-contained but uses a different model architecture than `export_mamba_onnx.py` exports. A model trained with `train_clara_now.py` cannot be exported with `export_mamba_onnx.py` without significant adapter code.
- **G6.5:** No documented workflow for: train → checkpoint → export → verify → deploy. Each step is independent.

## 7. Input/Output Shapes

**Documented in manifest.json:**

| Name | Shape | Notes |
|------|-------|-------|
| `audio_chunk` | `[batch, seq_len, 1]` | Dynamic seq_len |
| `conv{0..3}` | `[batch, d_model, conv_width]` = `[1, 64, 4]` | Fixed |
| `ssm{0..3}` | `[batch, d_model, d_state]` = `[1, 64, 16]` | Fixed |
| `cond` | `[batch, cond_dim]` = `[1, 8]` | Fixed |
| `processed_audio` | `[batch, seq_len, 1]` | Dynamic seq_len |
| `new_conv_state` | `[batch, n_layers, d_model, conv_width]` | Stacked |
| `new_ssm_state` | `[batch, n_layers, d_model, d_state]` | Stacked |

**Gaps:**
- **G7.1:** The manifest says `"num_inputs": 10` but earlier the formula says `1 + 2 * n_layers + 1 = 10` ✓. However, the old model backup is named `mamba_rt_9input_backup.onnx`, suggesting an earlier version had 9 inputs (likely before `cond` was added). The C++ engine must handle both possibilities, but the code only handles 10.
- **G7.2:** The manifest uses symbolic names for shapes (`"batch"`, `"seq_len"`, `"d_model"`) instead of numeric values. The C++ code must resolve these from the model config section.
- **G7.3:** No explicit shape validation in `export_mamba_onnx.py` that checks the traced graph's input/output shapes with `onnx.checker`. `verify_training_pipeline.py` does call `onnx.checker.check_model()` but the standalone export script does not.

## 8. Error Handling

**What's handled:**
- Pre-flight: missing dependencies → `sys.exit()`
- Pre-flight: wrong torch version → `sys.exit()`
- Pre-flight: opset too high → `sys.exit()`
- Pre-flight: dry-run forward pass fails → `sys.exit()`
- Verification: NaN/Inf in output → assertion
- Verification: all-zeros output → assertion
- Training: NaN/Inf loss → skip step
- Signal handling: SIGINT/SIGTERM/SIGBREAK → graceful save

**Gaps:**
- **G8.1:** If `torch.onnx.export()` fails partway, a partial/corrupt `.onnx` file may be left on disk. No cleanup of partial output.
- **G8.2:** If the `.onnx.data` sidecar write fails, the `.onnx` file is already written and references a missing file.
- **G8.3:** No retry logic for export failures.
- **G8.4:** `write_manifest()` has a bug at line 492: `onnx_version: onnx.__version__ if 'onnx' in dir() else "unknown"` — `onnx` is imported in `preflight_check()` but not at module scope where `write_manifest` is defined. Result: manifest always writes `"unknown"` for onnx_version. Confirmed in the actual `manifest.json`.
- **G8.5:** No timeout on ONNX verification inference. An infinite loop in the ONNX graph would hang indefinitely.

## 9. Dependencies

**Required:** `torch`, `onnx`, `onnxruntime`, `numpy`

**Required for training:** `soundfile`, `torchaudio`

**Gaps:**
- **G9.1:** No `requirements.txt` or `pyproject.toml` with pinned versions. Known working: torch 2.11.0+rocm7.13.0 (from manifest), but this is ROCm-specific.
- **G9.2:** No version constraints. `torch >= 2.0` is checked in preflight, but `onnx` and `onnxruntime` have no version checks beyond "is it installed?".
- **G9.3:** `soundfile` depends on system `libsndfile`. On Windows, this requires extra installation steps not documented.
- **G9.4:** `train_clara_now.py` imports `torchaudio` but doesn't check its version.

## 10. Model Quality & Provenance

**Current state:** ALL models in `models/` directory have RANDOM weights (no checkpoint was loaded during export). The export script prints explicit warnings about this.

**Files present:**
| File | Size | Content |
|------|------|---------|
| `mamba_rt.onnx` | 1.1 MB | External data, opset 18 |
| `mamba_rt.onnx.data` | 12 MB | External weight data |
| `mamba_rt_npu.onnx` | 3.6 MB | Self-contained, opset 18 |
| `mamba_rt_9input_backup.onnx` | 3.6 MB | Self-contained, backup |
| `mamba_rt_9input_backup.onnx.data` | 12 MB | External data (unused by backup?) |
| `manifest.json` | 1.4 KB | Metadata |

**Gaps:**
- **G10.1:** No seed is set before random weight initialization. `torch.manual_seed()` is never called. Every export produces different random weights.
- **G10.2:** No weight provenance tracking. The manifest records timestamp and torch version but not: random seed, parameter histogram, parameter count per layer, weight statistics (mean/std/min/max).
- **G10.3:** No checksum verification of the `.onnx.data` sidecar against the `.onnx` reference. If the data file is corrupted or from a different export, there's no detection.
- **G10.4:** The `mamba_rt_npu.onnx` model has weights embedded (3.6 MB, no sidecar) but no documentation explaining why, when it was exported, or how it differs from `mamba_rt.onnx`.
- **G10.5:** The `mamba_rt_9input_backup` appears to be a pre-conditioning backup (despite having 10 inputs — name is misleading). No documentation of what it backs up from or when.

---

## Summary of Critical Issues

| ID | Severity | Issue |
|----|----------|-------|
| G6.2 | **BLOCKER** | 4 required Python modules don't exist (`train_clara_config`, `train_clara_loss`, `train_clara_dsp`, `train_clara_dataset`) |
| G6.4 | **HIGH** | Training architecture (`STFTMambaModel`) incompatible with export architecture (`MambaRTModel`) |
| G1.1 | **HIGH** | 3 separate model implementations with code duplication |
| G8.4 | **HIGH** | Manifest `onnx_version` always "unknown" — `'onnx' in dir()` bug |
| G2.1 | **MEDIUM** | SSM Python for-loop unrolled into ONNX graph — graph size ∝ seq_len |
| G2.2 | **MEDIUM** | Inconsistent `dynamic_axes` between export scripts |
| G5.1 | **MEDIUM** | FiLM initialized to zero — conditioning is dead until training |
| G10.1 | **MEDIUM** | No deterministic seed → unreproducible exports |
| G9.1 | **MEDIUM** | No `requirements.txt` |
| G3.1 | **LOW** | Verification never tests non-zero conditioning |
| G3.6 | **LOW** | No gradient-based ONNX verification |
| G7.1 | **LOW** | 9-input backup naming doesn't match actual 10-input model |
