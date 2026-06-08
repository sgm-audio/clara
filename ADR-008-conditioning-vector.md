# ADR-008: Conditioning Vector for Context-Aware Mamba SSM

**Status:** Accepted
**Date:** 2026-06-02

## Context

The Mamba SSM processes audio identically regardless of musical context (tempo, key, chord progression, arrangement position). A conditioning vector enables external parameters to modulate the model's output, allowing context-aware processing. ARA provides musical context data (tempo, key) that should influence the model.

## Decision

### 1. `COND_DIM = 8`

Fixed dimension for the conditioning vector. Chosen to balance expressiveness (enough dimensions for tempo, key, time signature, analysis features) against real-time cost (applied per-sample). Can be increased later without breaking the API — projection weights are sized dynamically.

### 2. Double-buffered with atomic pointer swap

```
condBufA_[COND_DIM], condBufB_[COND_DIM]
std::atomic<float*> activeCond_  // points to which buffer to read in RT
```

`setConditioning(cond, dim)`: writes to the non-active buffer, issues atomic store to swap pointer (release ordering). `processBlock()` loads the active pointer (acquire ordering) and applies the projection. No locks, no allocations.

### 3. Dual-path conditioning: ONNX-integrated FiLM + external scalar gain

The conditioning system now operates in two parallel paths:

**Path A — ONNX-integrated FiLM (10th input):** The `cond` tensor is passed directly into the ONNX graph as the 10th input (index 9, shape `{1, 8}`). Inside the model, FiLM-style feature modulation applies the conditioning at each layer, modulating the SSM's internal representations. This changes the ONNX I/O contract from 9 to 10 inputs.

**Path B — External scalar gain modulation (`combinedWeights_`):** After ONNX inference, a second scalar gain is applied as output modulation:
```
condScale = 1.0f + sum(cond[d] * combinedWeights_[d])
output[s] *= condScale
```
where `combinedWeights_[d] = deltaWeights[d] + bWeights[d]` is pre-computed at `loadCondWeights()` time (called from `init()` / `activate()`) and is immutable in RT.

Both paths read the same `cond` vector via double-buffered atomic pointer swap. The dual-path design was retained for backward compatibility and to allow coarser gain modulation alongside the more expressive ONNX-integrated conditioning.

### 4. Projection weights loaded at `init()` / `activate()`, immutable in RT

`combinedWeights_` of size `COND_DIM` is computed once from `config_.deltaWeights` and `config_.bWeights` via `loadCondWeights()`. Each element is a scalar that modulates the corresponding cond dimension's contribution to the output gain.

### 5. RT-safe constraints

- `setConditioning()`: writes to non-active buffer, then atomic store to swap. Callable from any thread.
- `loadCondWeights()`: called from `init()` / `activate()` only — non-RT thread, uses `std::vector::resize` + `memcpy`.
- `processBlock()`: reads active pointer, reads weights, passes cond as 10th ONNX input. Zero allocations, zero locks.

## Consequences

### Positive

- ARA musical context (tempo, key) can modulate model output via the cond vector
- ONNX-integrated FiLM provides per-layer feature modulation for more expressive conditioning
- RT-safe with zero allocations on audio thread
- Backward-compatible: zero cond = no change
- atomic pointer swap is lock-free and wait-free on all target platforms

### Negative

- `COND_DIM` is fixed at compile time — changing it requires recompilation and model re-export
- Dual-path (ONNX FiLM + external gain) adds complexity — both use the same `cond` vector but apply differently
- No ARA integration yet to populate the cond vector from musical context
- Conditioning now changes the ONNX I/O contract: 10 inputs instead of 9, requiring model re-export
- Dual buffers add `2 × COND_DIM × sizeof(float) = 64 bytes` of persistent memory per engine instance

## Compliance

- Verified: `MambaInferenceEngine.hpp:30-33` declares `setConditioning()` and `loadCondWeights()` in the public API.
- Verified: `MambaInferenceEngine.hpp` declares `condBufA_` / `condBufB_` / `activeCond_` / `condWeights_` as private members.
- Verified: `setConditioning()` writes to the non-active buffer (detected via `activeCond_` pointer comparison) then issues `activeCond_.store(..., memory_order_release)`.
- Verified: `processBlock()` loads `activeCond_` via `activeCond_.load(memory_order_acquire)` and computes `condScale` — no allocations, no locks.
- Verified: `loadCondWeights()` called from `mamba_activate()` (`MambaClapPlugin.cpp:263`) — non-RT context, uses `resize` + `memcpy` into `condWeights_`.
- Verified: `COND_DIM` defined as `static constexpr size_t kCondDim = 8` in `MambaInferenceEngine.hpp`.
- Verified: `kNumInputs = 10` in `MambaInferenceEngine.hpp`, `"cond"` is last entry in `inputNames_` in `MambaInferenceEngine.cpp:16`.
- Verified: Cond tensor passed as 10th ONNX input with shape `{1, COND_DIM}` in `processBlock()`.
- Verified: External scalar gain `condScale` applied in `processBlock()` using `combinedWeights_` after ONNX `Run()`.
- Verified: `loadCondWeights()` computes `combinedWeights_[d] = deltaWeights[d] + bWeights[d]` and is called from `init()`.
- Verified: Zero condition vector (`cond[d] = 0 ∀d`) produces `condScale = 1.0f` — output consistent with unconditioned path.

## Related

- AGENTS.md — invariant #7 (conditioning vector design summary at line 67)
- ADR-001: ONNX I/O contract (now 10 inputs, cond is input 10)
- `src/core/MambaInferenceEngine.hpp` — `kNumInputs=10`, `setConditioning()`, `loadCondWeights()`, `activeCond_`, `combinedWeights_`, `kCondDim`
- `src/core/MambaInferenceEngine.cpp` — `processBlock()` passes cond as ONNX input + gain modulation, `setConditioning()` atomic swap, `loadCondWeights()` combined weight computation
- `src/core/ConditioningGenerator.hpp` — `MusicalContext` struct, `ConditioningGenerator::encode()` for ARA context → cond
- `src/clap/MambaClapPlugin.cpp` — `mamba_activate()` calls `loadCondWeights()` from non-RT context
- `src/clap/MambaClapPlugin.hpp` — `MambaPluginState` holds no cond state (cond lives in engine)
- ADR-002: Deferred atomic state reset (sibling RT-safety pattern with atomic flag)
