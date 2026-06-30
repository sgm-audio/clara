# ADR-002: Deferred Atomic State Reset for SSM Recurrent State

**Status:** Accepted (implemented)
**Date:** 2026-05-30

## Context

The Mamba SSM maintains recurrent state across audio blocks: 4 layers × 2 state types = 8 buffers per `MambaInferenceEngine` instance:

- **Conv state** (`convStateIn_[0..3]`): One `std::vector<float>` per layer, sized `dModel × convWidth` = 64 × 4 = 256 floats.
- **SSM state** (`ssmStateIn_[0..3]`): One `std::vector<float>` per layer, sized `dModel × dState` = 64 × 16 = 1024 floats.

Total state: 4 × (256 + 1024) = 5120 floats = ~20 KB per instance.

This state accumulates across `processBlock()` calls — the output states from one block become the input states for the next. When the DAW seeks transport, bypasses the plugin, or loads a new preset, this state must be **zeroed** to prevent stale values from corrupting output.

The challenge: `resetState()` can be called from **any thread**:

| Caller | Thread | Scenario |
|--------|--------|----------|
| `mamba_reset()` (CLAP) | Audio thread | DAW transport seek — host calls the plugin's reset callback |
| `mamba_state_load()` (CLAP) | GUI thread (typically) | Preset load — host sets plugin state from a saved chunk |
| `MambaPlaybackRenderer` constructor (ARA) | Non-RT thread | ARA playback renderer creation (only happens once, not during RT) |

Directly zeroing 8 buffers from a foreign thread races with `processBlock()` on the audio thread, which reads those same buffers. The operation is not atomic — zeroing 5120 floats takes measurable time.

### Options Considered

| Option | Description | Problem |
|--------|-------------|---------|
| **Mutex** | `std::mutex` in both `resetState()` and `processBlock()` | RT-safety violation — priority inversion risk if `processBlock()` contends with a lower-priority thread. Worst case: X-run while waiting for a GUI thread to finish zeroing buffers. |
| **SPSC command queue** | Push a "reset" command from any thread. Audio thread pops commands at start of each `processBlock()`. | Correct but heavyweight for a single boolean signal. Requires a queue implementation, dynamic allocation for commands, or a ring buffer. Adds a dependency (or code) for what is fundamentally a one-bit message. |
| **Double-buffered state** | Maintain two complete state sets. Atomically swap a pointer to a fresh zeroed set on reset. | 8 buffers × 2 copies = ~40 KB. Memory overhead is modest, but the swap requires managing object lifetimes — the old buffer set must be retained until the audio thread is done with it. Requires either RCU semantics or deferred deletion, adding complexity disproportionate to the problem. |
| **Immediate zero with atomic fence** | `resetState()` writes zeros to all 8 buffers, then issues an atomic fence. | A thread writing to memory owned by another thread's concurrent reader is undefined behavior regardless of fences. The reader (`processBlock()`) may be in the middle of iterating a buffer when the writer modifies it. |
| **Deferred atomic flag** (chosen) | Single `std::atomic<bool>` flag. `resetState()` sets it; `processBlock()` reads-and-clears it at entry, zeroes buffers itself on the audio thread. | Zero contention on the hot path. No locks. No allocations. Single bit of shared state. The only approach that is both RT-safe and trivially correct. |

## Decision

### 1. A Single `std::atomic<bool> resetRequested_`

Declared in `MambaInferenceEngine.hpp:38` as a member, initialized to `false` in the member initializer list (value-initialization via `{}`).

### 2. `resetState()`: Release-Store

```cpp
// MambaInferenceEngine.cpp:46-51
void MambaInferenceEngine::resetState() {
    resetRequested_.store(true, std::memory_order_release);
}
```

**Memory ordering rationale (`memory_order_release`):**

The store is a thread-safe "note to self" that state needs clearing. The release ordering ensures that **all prior writes by the calling thread** (e.g., the GUI thread's writes to parameter atomics, or the audio thread's own writes to state buffers from a previous `processBlock()` call) are visible to the audio thread before it observes the flag as `true`. Without this, the audio thread could see the flag as `true` but miss the updated parameter values that triggered the reset (e.g., a new bypass state).

Specifically, the release store establishes a **happens-before** relationship with the acquire load in `processBlock()`.

### 3. `processBlock()`: Acquire-Release Exchange

```cpp
// MambaInferenceEngine.cpp:58-63
if (resetRequested_.exchange(false, std::memory_order_acq_rel)) {
    for (size_t i = 0; i < config_.nLayers && i < kMaxLayers; ++i) {
        std::fill(convStateIn_[i].begin(), convStateIn_[i].end(), 0.0f);
        std::fill(ssmStateIn_[i].begin(), ssmStateIn_[i].end(), 0.0f);
    }
}
```

**Why `exchange` instead of `load` + separate `store`:**

A `load(acquire)` + conditional `store(release)` sequence has a TOCTOU race: between the load and the store, another thread could also call `resetState()`, and the second reset would be lost because the store unconditionally clears the flag. With `exchange`, the read-and-clear is atomic — if the flag was `true`, we atomically claim it, clear it, and zero the buffers. Any concurrent `resetState()` call after the exchange will set the flag again for the next block.

**Memory ordering rationale (`memory_order_acq_rel`):**

- **Acquire**: Ensures that the state buffer zeroing (`std::fill` calls) that follows the exchange does not reorder before the flag read. Without this, the compiler or CPU could speculatively zero buffers before checking the flag, producing a spurious reset.
- **Release**: Ensures that the zeroed state buffers are visible to any thread that calls `resetState()` after this exchange. Practically, it pairs with the release store in `resetState()` — if a `resetState()` call on another thread happens-after this exchange, its release store will see the zeroed buffers.
- **Combined (acq_rel)**: The exchange both reads and writes the flag atomically. The acquire half synchronizes with prior `resetState()` stores; the release half synchronizes with subsequent `resetState()` stores. `memory_order_acq_rel` is the correct choice for an atomic RMW that must both observe and publish.

### 4. Zeroing Mechanism: `std::fill`

The state buffers are zeroed with `std::fill(begin(), end(), 0.0f)` rather than `std::memset`. On optimized builds, both compile to identical machine code (`rep stosd` or vectorized stores). On debug builds, `std::fill` generates an inline loop — acceptable since zeroing only happens on reset, not per-block.

Note: The zeroing happens on the **audio thread**, but only when a reset was requested. This is a bounded-time operation (~5120 floats = 20 KB, ~1μs at typical memcpy bandwidth). It does not affect per-block RT guarantees because it's amortized over the reset interval.

### 5. Initial State

`init()` zeroes all state buffers via `resize(..., 0.0f)` (`MambaInferenceEngine.cpp:31-34`). The first `processBlock()` call after `init()` sees clean state without consulting the flag. The deferred reset only matters for resets that happen during active processing.

### 6. Call-Site Threading Analysis

**`mamba_reset()` (CLAP transport seek, `MambaClapPlugin.cpp:282`):**

Called from the audio thread during transport state changes. The audio thread calls `engine->resetState()` which stores `true` to the atomic. On the **next** `processBlock()` call (also on the audio thread), the exchange will see `true` and zero the buffers. This means the current block (if one is in progress) still runs with the old state — state reset is always at a block boundary, never mid-block. This is by design: resetting mid-block would cause a discontinuity (pop) in the audio output.

**`mamba_state_load()` (CLAP preset load, `MambaClapPlugin.cpp:216`):**

Called from a non-RT thread (GUI/main thread) when the host loads a plugin preset. After restoring bypass/wetDry/intensity from the saved state, it calls `engine->resetState()` to clear the SSM state for the new preset. The store is safe because it's just writing to an atomic bool. The actual zeroing happens on the audio thread at the next `processBlock()` call.

**ARA PlaybackRenderer constructor (`MambaAraController.cpp:32-45`):**

The constructor initializes a private `MambaInferenceEngine` for use by the ARA PlaybackRenderer. It does not call `resetState()` in the constructor — state is initialized to zero by `init()`. ARA has no `renderSamples()` method (ARA 2.0), so this engine instance is a stub for future use (offline analysis).

### 7. Flag Lifetime and Edge Cases

**`processBlock()` never called after `resetState()`:**

If the plugin is bypassed and then unloaded, `resetState()` may have set the flag but `processBlock()` never clears it. The flag is a member of `MambaInferenceEngine`, which is destroyed when the plugin is deactivated. The flag's state at destruction is irrelevant — no concurrent access is possible during destruction (the audio thread is stopped by the DAW before the plugin object is freed).

**`resetState()` called multiple times before `processBlock()`:**

The flag is boolean, not a counter. Three `resetState()` calls followed by one `processBlock()` call zero buffers once. This is correct — resetting three times is idempotent to resetting once. There is no hidden state to count.

**`processBlock()` called before `init()`:**

The early return guard (`if (!initialized_ || !session_) return;` at line 54) handles this. If the engine was not initialized, `processBlock()` is a no-op. The flag is never checked because no buffers exist to zero.

**`kMaxLayers` vs `config_.nLayers`:**

The code uses both: `kMaxLayers = 4` is a compile-time constant, and `config_.nLayers` defaults to 4. The loop guard `i < config_.nLayers && i < kMaxLayers` is a safety fence — if someone changes `nLayers` in Config without updating `kMaxLayers`, the loop truncates to the smaller value. Currently both are 4 and must stay in sync with the exported model.

## Consequences

### Positive
- **Zero contention on the hot path** — the exchange is the only atomic op in the RT path that's on a shared location, and it resolves to a single `lock xchg` instruction on x86.
- **No locks, no allocations** in either path.
- **Thread-safe without external synchronization** — no SPSC queue, no mutex, no double-buffer lifecycle management.
- **~20 KB memory overhead** for the deferred zeroing (the buffers already exist — the flag adds 1 byte).
- **Predictable behavior**: state always resets at a block boundary, never mid-block.

### Negative
- **State reset is delayed by up to one block.** At maxBlockSize=256 @ 48kHz, ~5.3ms between `resetState()` call and actual zeroing. Acceptable — transport seeks already have audible latency, and preset loads happen outside the RT path.
- **If `processBlock()` is called less frequently than `resetState()`, multiple resets collapse into one.** This is correct (reset is idempotent) but could surprise a developer expecting each reset to have an observable effect.
- **Non-obvious to new readers** — the split between "set flag" (`resetState()`) and "act on flag" (`processBlock()`) across two functions looks like a bug at first glance. The comment in the source documents the intent.

## Compliance

- Verified: `resetState()` (`MambaInferenceEngine.cpp:46-51`) is a single `store(release)` — no locks, no allocs.
- Verified: `processBlock()` (`MambaInferenceEngine.cpp:58-63`) uses `exchange(acq_rel)` — no locks, no allocs.
- Verified: All 8 state buffers are zeroed when the exchange returns `true`.
- Verified: `init()` zeroes all state buffers via `resize(..., 0.0f)`, providing clean initial state without relying on the flag.
- Verified: CLAP plugin calls `resetState()` from `mamba_reset()` (transport seek, `MambaClapPlugin.cpp:282`) and `mamba_state_load()` (preset load, `MambaClapPlugin.cpp:216`).
- Verified: No other thread accesses the state buffers during zeroing — the audio thread owns them exclusively.
- Verified: The `exchange` is an unconditional RMW regardless of the flag's value — this is the cheapest atomic RMW on x86 (single `lock` instruction) and cheaper than a load + branch for the uncontended case due to lack of branch misprediction.

## Related

- `src/core/MambaInferenceEngine.hpp:38` — `resetRequested_` member declaration
- `src/core/MambaInferenceEngine.cpp:46-63` — `resetState()` + deferred clear in `processBlock()`
- `src/core/MambaInferenceEngine.cpp:30-34` — `init()` zeroes state buffers via `resize`
- `src/clap/MambaClapPlugin.cpp:216` — reset on state load
- `src/clap/MambaClapPlugin.cpp:282` — reset on transport seek
- ADR-001: Shared ONNX Inference Engine (MambaCore context)
