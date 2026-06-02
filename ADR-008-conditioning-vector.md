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

### 3. Projection weights loaded at `activate()`, immutable in RT

A `std::vector<float> condWeights_` of size `COND_DIM * dModel` is loaded once per engine instance in `activate()`. Each element is a scalar that modulates the corresponding cond dimension's contribution to the output gain.

### 4. External application — does not change ONNX I/O contract

The conditioning is applied as output gain modulation in `processBlock()` after the ONNX inference run:

```
condScale = 1.0f + sum(cond[d] * weights[d])  // scalar modulation
output[s] *= condScale
```

This means:
- The ONNX model I/O stays at 9 inputs / 3 outputs (no model re-export required)
- Zero condition vector → output unchanged
- When model training is ready, conditioning can be folded into the ONNX graph

### 5. RT-safe constraints

- `setConditioning()`: writes to non-active buffer, then atomic store to swap. Callable from any thread.
- `loadCondWeights()`: called from `activate()` only — non-RT thread, uses `std::vector::resize` + `memcpy`.
- `processBlock()`: reads active pointer, reads weights. Zero allocations, zero locks.

## Consequences

### Positive

- ARA musical context (tempo, key) can modulate model output via the cond vector
- No ONNX model re-export needed
- RT-safe with zero allocations on audio thread
- Backward-compatible: zero cond = no change
- atomic pointer swap is lock-free and wait-free on all target platforms

### Negative

- `COND_DIM` is fixed at compile time — changing it requires recompilation
- Scalar gain modulation is coarse — full conditioning in the ONNX graph would be more expressive
- No ARA integration yet to populate the cond vector
- Dual buffers add `2 × COND_DIM × sizeof(float) = 64 bytes` of persistent memory per engine instance

## Compliance

- Verified: `MambaInferenceEngine.hpp:30-33` declares `setConditioning()` and `loadCondWeights()` in the public API.
- Verified: `MambaInferenceEngine.hpp` declares `condBufA_` / `condBufB_` / `activeCond_` / `condWeights_` as private members.
- Verified: `setConditioning()` writes to the non-active buffer (detected via `activeCond_` pointer comparison) then issues `activeCond_.store(..., memory_order_release)`.
- Verified: `processBlock()` loads `activeCond_` via `activeCond_.load(memory_order_acquire)` and computes `condScale` — no allocations, no locks.
- Verified: `loadCondWeights()` called from `mamba_activate()` (`MambaClapPlugin.cpp:263`) — non-RT context, uses `resize` + `memcpy` into `condWeights_`.
- Verified: `COND_DIM` defined as `static constexpr size_t kCondDim = 8` in `MambaInferenceEngine.hpp`.
- Verified: The ONNX model I/O contract (9 inputs / 3 outputs) is unchanged — verified by comparing input/output name arrays before and after conditioning implementation.
- Verified: Zero condition vector (`cond[d] = 0 ∀d`) produces `condScale = 1.0f` — output bit-identical to unconditioned path.

## Related

- AGENTS.md — invariant #7 (conditioning vector design summary at line 67)
- ADR-001: ONNX I/O contract (conditioning is external, doesn't change 9/3 contract)
- `src/core/MambaInferenceEngine.hpp` — `setConditioning()`, `loadCondWeights()`, `activeCond_`, `condWeights_`, `kCondDim`
- `src/core/MambaInferenceEngine.cpp` — `processBlock()` gain modulation, `setConditioning()` atomic swap, `loadCondWeights()` weight copy
- `src/clap/MambaClapPlugin.cpp:263` — `mamba_activate()` calls `loadCondWeights()` from non-RT context
- `src/clap/MambaClapPlugin.hpp` — `MambaPluginState` holds no cond state (cond lives in engine)
- ADR-002: Deferred atomic state reset (sibling RT-safety pattern with atomic flag)
