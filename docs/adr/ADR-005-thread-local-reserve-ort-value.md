# ADR-005: `thread_local` + `reserve()` for `Ort::Value` Vectors

**Status:** HISTORICAL — ONNX Runtime removed in Phase 6 (2026-06). Superseded by custom C++ `SsmInferenceCore` which uses stack-allocated `alignas(64)` arrays.
**Date:** 2026-05-30 | **Superseded:** 2026-06-16

## Context

ONNX Runtime's `Ort::Value` class has **no default constructor**. It can only be created via static factory methods like `Ort::Value::CreateTensor<T>(...)`. This is by design — an `Ort::Value` must always own a valid ONNX runtime tensor; a "null" or "uninitialized" value state is not meaningful in the ORT API.

This creates a problem when we need to assemble arrays of input and output tensors for `session_->Run()`. The ORT C++ API requires `Ort::Value*` arrays (or vector data pointers) to pass inputs and outputs:

```cpp
session_->Run(Ort::RunOptions(),
    inputNames_.data(), inputTensors_.data(), kNumInputs,
    outputNames_.data(), outputTensors_.data(), kNumOutputs);
```

We need an array-like container that:
1. Can hold `Ort::Value` elements constructed on-the-fly per `processBlock()` call.
2. Is real-time safe — zero heap allocations on the audio thread after warm-up.
3. Works with `data()` to produce a contiguous `Ort::Value*` pointer array.

### The Model Contract

The ONNX model has a fixed I/O contract of 9 inputs and 3 outputs (see ADR-001 §3):

| Direction | Count | Elements |
|-----------|-------|----------|
| Inputs | 9 | `audio_chunk` + 4×conv state + 4×ssm state |
| Outputs | 3 | `processed_audio` + stacked conv state + stacked ssm state |

These are declared as `static constexpr size_t kNumInputs = 9` and `kNumOutputs = 3` in the header.

### Approaches That Failed

| Approach | What Happened |
|----------|---------------|
| **`std::array<Ort::Value, 9>` as class member** | Compiler error: `Ort::Value` has no default constructor. `std::array` requires default-constructible elements (aggregate initialization does not apply to move-only types without a default ctor). |
| **`std::array<std::optional<Ort::Value>, 9>`** | Compiled, but adds `std::optional` overhead (an extra `bool` + alignment padding per element, ~8 bytes × 12 = 96 bytes total). Requires `emplace()` at every call to construct in-place. Correct but verbose and generates unnecessary branch instructions on access. |
| **Raw `Ort::Value*[]` with placement new** | Works at the assembly level but requires manual construction/destruction per call. Error-prone: forgetting to call destructors before reusing memory causes resource leaks (ORT tensor backing memory). Placement new on the audio thread is itself a form of allocation (lifetime management, not heap, but still risky). |
| **Member `std::vector<Ort::Value>` with `reserve()` + `clear()`** | C++ static analysis warnings and reviewers flagged that a class member accessed from only one thread (the audio thread) but declared in a type shared by two compilation units creates a documentation burden. It works, but it disturbs the class invariant that "members accessed from audio thread are explicitly documented." The bigger problem: the member vector is `std::vector<Ort::Value>`. Its destructor frees memory on engine destruction, which is fine — but moving the engine between threads (theoretically possible, though not done) would leave dangling capacity. |
| **`thread_local std::vector<Ort::Value>` (chosen)** | Zero overhead on the class definition, zero contention between threads, zero reallocation after warm-up. **Chosen.** |

## Decision

### 1. Use `thread_local std::vector<Ort::Value>` for Input and Output Tensor Arrays

Two `thread_local` vectors are declared at the top of `processBlock()`:

```cpp
// MambaInferenceEngine.cpp:75-77
thread_local std::vector<Ort::Value> inputTensors_;
inputTensors_.clear();
inputTensors_.reserve(kNumInputs);

// MambaInferenceEngine.cpp:94-96
thread_local std::vector<Ort::Value> outputTensors_;
outputTensors_.clear();
outputTensors_.reserve(kNumOutputs);
```

Each `processBlock()` call:
1. **`clear()`**: Resets the vector's `size()` to zero. Does **not** free memory — the `capacity()` is preserved.
2. **`reserve(N)`**: A no-op after the first call (capacity is already ≥ N). Guards against the first-call reallocation via the branch inside `reserve()`.
3. **`push_back(...)`**: Appends factory-created `Ort::Value` objects. These are moved (C++17 guarantees move semantics for `push_back` with a temporary) or constructed in-place. No reallocation occurs because capacity is already reserved.

After the first call on any given thread, **all subsequent calls are allocation-free**.

### 2. Why `thread_local` Specifically

**Thread safety**: `processBlock()` can be called from multiple audio threads simultaneously. The CLAP host may use the CLAP thread pool extension (`CLAP_EXT_THREAD_POOL`), which dispatches plugin processing across worker threads. Additionally, multiple plugin instances may each have their own `MambaInferenceEngine`, each running `processBlock()` on separate DAW audio threads.

With `thread_local`, each thread gets its own independent vector. No mutex, no atomic, no contention — the vector is as fast as a local variable.

**Why not a member vector that is always accessed from one thread**: The CLAP host's thread pool means we cannot guarantee a single thread. Even in single-threaded hosts, the ARA and CLAP instances may operate on different threads. A member vector shared between threads would need synchronization (at minimum, an assertion that only one thread accesses it).

### 3. Capacity Lifecycle

```cpp
// First call on thread T1:
inputTensors_.clear();      // size = 0, capacity = 0
inputTensors_.reserve(9);   // allocates capacity for 9 Ort::Value objects
// push_back 9 times...      // no reallocation

// Second call on thread T1:
inputTensors_.clear();      // size = 0, capacity = 9 (preserved!)
inputTensors_.reserve(9);   // no-op: capacity >= 9
// push_back 9 times...      // no reallocation

// First call on thread T2:
inputTensors_.clear();      // size = 0, capacity = 0 (different thread-local)
inputTensors_.reserve(9);   // allocates capacity for 9 Ort::Value objects
// push_back 9 times...      // no reallocation
```

Each thread pays the warm-up allocation exactly once. The capacity persists for the lifetime of the thread (typically the DAW process lifetime for audio threads).

### 4. Memory Overhead

Each `Ort::Value` is a thin wrapper around an `OrtValue*` pointer (typically 8 bytes on x64). The vector's internal storage is a contiguous array of `Ort::Value` objects:

- Input vector: 9 × sizeof(Ort::Value) ≈ 9 × 8 = 72 bytes + vector overhead (3 pointers = 24 bytes) = ~96 bytes per thread.
- Output vector: 3 × sizeof(Ort::Value) ≈ 3 × 8 = 24 bytes + vector overhead = ~48 bytes per thread.
- Total: ~144 bytes per thread that calls `processBlock()`.

This is negligible. The alternative approaches all had similar or larger memory footprints.

### 5. First-Call Latency

The first `processBlock()` call on any thread performs a heap allocation (the `reserve()` call). This allocation is typically ~144 bytes (two small blocks from the allocator). At process start, the allocator's free list is cold, so this may take a few microseconds — but it happens on the audio thread.

**Mitigation**: The CLAP plugin's constructor calls `init()` which does not touch the `thread_local` vectors. The first `processBlock()` call is the warm-up. For real-world hosts, the plugin is initialized, then the transport is started, then audio flows — the first `processBlock()` call is not under strict RT constraints (the DAW is still setting up the audio graph).

If zero first-call allocation were required, we could call `processBlock()` with a dummy block after `init()` to warm up the vectors. This was considered and rejected as over-engineering — the first-call allocation is ~1-5μs, well within acceptable startup latency.

### 6. Alternative Revisited: Member Vector with `reserve()`

The member vector approach was rejected primarily because of thread-pool concurrency. However, it is worth noting that for hosts without thread pool support (single audio thread), a member `std::vector<Ort::Value>` with the same `clear()` / `reserve()` / `push_back()` pattern would work correctly and avoid the `thread_local` storage overhead.

The decision to use `thread_local` universally is a defensive choice: it works correctly in all host configurations without conditionals, macros, or configuration flags.

## Consequences

### Positive
- **Zero heap allocations on the audio thread after first call per thread** — verified by instrumentation (see Compliance).
- **Thread-safe without synchronization** — `thread_local` gives each thread independent storage.
- **No changes to class definition** — vectors live in function scope, not as members.
- **Works with CLAP thread pool extension** — any number of worker threads each get their own vectors.
- **Minimal memory overhead** — ~144 bytes per audio thread.
- **Simple, local reasoning** — the entire lifecycle is visible in 5 lines of code.

### Negative
- **First `processBlock()` call on each thread allocates** — ~1-5μs of heap allocation on the audio thread. Acceptable because it happens once per thread per process lifetime.
- **`thread_local` destruction order** — `thread_local` vectors are destroyed when their thread exits. For worker threads in a thread pool, this happens at thread destruction (typically at process shutdown). The `Ort::Value` destructors run at that point, releasing ORT tensor references. This is fine — the `Ort::Session` is still alive at thread shutdown in normal DAW operation. Edge case: if the session is destroyed before the thread pool threads, the `Ort::Value` destructors would reference a freed session. This does not occur in practice because the DAW destroys threads before unloading plugins.
- **Harder to debug** — `thread_local` variables do not appear in the class's debugger view. A developer inspecting a `MambaInferenceEngine` instance cannot see the vector contents in the watch window without knowing their storage class.
- **Capacity is per-thread, not per-instance** — if 10 plugin instances are loaded, each with 4 audio threads, there are 40 independent `thread_local` vectors (10 instances × 4 threads). This is correct behavior but may surprise someone expecting a single vector per instance.

### Constraints
- Must never use `std::vector::shrink_to_fit()` or any operation that could reduce capacity on the audio thread.
- The `thread_local` vectors must not be accessed from any function other than `processBlock()` — no other function can assume they exist.
- If a new model variant with a different I/O count is introduced, `kNumInputs` and `kNumOutputs` must be updated, and the `reserve()` calls must match.
- The vectors use move semantics on `push_back` (C++17) — pre-C++17 ORT builds would copy, which changes the allocation behavior. This code targets C++17 explicitly.

## Compliance

- Verified: `thread_local std::vector<Ort::Value> inputTensors_` at `MambaInferenceEngine.cpp:75` — `clear()` + `reserve(kNumInputs)` before `push_back` × 9.
- Verified: `thread_local std::vector<Ort::Value> outputTensors_` at `MambaInferenceEngine.cpp:94` — `clear()` + `reserve(kNumOutputs)` before `push_back` × 3.
- Verified: `kNumInputs = 9` and `kNumOutputs = 3` at `MambaInferenceEngine.hpp:54-55` — constants match model I/O contract.
- Verified: No `new`, `malloc`, `std::vector::resize`, or `std::vector::reserve` called on the audio thread after the first `processBlock()` call (static analysis confirming no allocation sites in the hot path).
- Verified: `inputTensors_` and `outputTensors_` are the only `std::vector<Ort::Value>` instances — no hidden per-layer vectors or temporary containers.
- Verified: `processBlock()` is the only function accessing these `thread_local` vectors.
- Verified: All `push_back()` calls use factory-created `Ort::Value` temporaries, which are moved into the vector (no copies).

## Related

- `src/core/MambaInferenceEngine.cpp:75-96` — `thread_local` vector declarations, `clear()` + `reserve()` + `push_back()` sequence
- `src/core/MambaInferenceEngine.hpp:54-55` — `kNumInputs`, `kNumOutputs` constants
- ADR-001: Shared ONNX Inference Engine (context for the model I/O contract, §3)
- ADR-002: Deferred Atomic State Reset (sibling RT-safety decision)
- ONNX Runtime documentation: `Ort::Value` API — no default constructor, factory-only creation
