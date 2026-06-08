# CLARA — Agent Prompt System

> MambaSuite: C++17 CLAP + ARA2 plugin using ONNX Runtime
> Status: most production-ready of all projects, builds clean, needs trained model + integration polish

---

## Master Prompt (paste at start of every session)

```
You are working on CLARA — Mamba Audio Suite, a C++17 CLAP + ARA2 plugin
using ONNX Runtime to run a Mamba SSM for context-aware audio processing.

REQUIRED READING BEFORE TOUCHING ANYTHING:
- AGENTS.md
- ADR-001 through ADR-007
- TODO_PROGRESS.md
- The README

INVARIANTS — never violate:
1. No allocations on audio thread (RT-safe). All buffers pre-allocated in init().
2. No exceptions across C boundaries. Catch in C++ land, return error code.
3. No locks in process() callback. Only atomic operations and SPSC ring buffers.
4. thread_local + reserve() pattern for any std::vector used in RT path.
5. ONNX model I/O contract: 9 inputs (audio_chunk + 4 conv state + 4 ssm state)
   → 3 outputs (processed_audio + stacked conv state + stacked ssm state).
   Never change this without updating export_mamba_onnx.py in lockstep.
6. Click-free bypass via 50ms linear crossfade — never instantaneous.
7. Conditioning vector cond[COND_DIM] is double-buffered, atomic pointer swap.
   Projection weights loaded at activate(), immutable in RT.

WORKING MODE:
- Plan mode: Audit for stubs, placeholders, TODOs, unimplemented branches.
  Output itemized list with file:line refs and severity. Do not modify.
- Build mode: Take items from audit, execute in dependency order. Commit per
  logical unit. Run build after every change. Reject changes that break build.

DELIVERABLES per session:
1. Diff-style summary of what changed
2. Decision debt — anything skipped, deferred, or stubbed
3. Updated TODO_PROGRESS.md

If you hit an architectural decision not covered by an existing ADR:
STOP, draft a new ADR, ask for review before proceeding.

If you find yourself wanting to refactor something not directly related to
the current task: DO NOT. Open an issue, move on.
```

---

## Task Backlog (execute in dependency order)

### CLARA-01: ARAPlug.h wrapper completeness audit
```
PLAN MODE ONLY for this task.

Read MambaAraController.cpp. For each ARA::PlugIn::* class used, verify
the wrapper exists in the fetched ARA SDK at pinned commit 1ca0b0aa.

For each class, document:
- Is the wrapper present?
- Are all methods used by MambaAraController implemented in the wrapper?
- Are there comments in the SDK suggesting "not yet implemented" or TODO?

Output: ara_wrapper_gap_analysis.md with one section per class, severity
(blocker / annoying / cosmetic), and recommended path:
  (a) keep using C++ wrapper, file is fine
  (b) downgrade to raw C API for this subsystem only
  (c) backport missing wrapper from ARA reference impl

Do not modify any code. This task is diagnostic only.
```

### CLARA-02: CLAP validator integration
```
Add a CTest target that runs clap-validator against build/plugins/MambaSuiteClap.clap.

Steps:
1. Fetch clap-validator via FetchContent OR detect local install via find_program
2. Configure CTest target that runs after build
3. Target: cmake --build . --target test runs clap-validator
4. Required categories: audio-ports, params, state, latency
5. Build fails if validator fails

Files to create/modify:
- CMakeLists.txt (add CTest enable + target)
- tests/clap_validator.cmake (the test logic)
- A README section under "## Testing" documenting the command

Acceptance: green PASS on all required CLAP categories. Document any
SKIPPED or KNOWN-FAILURE results with rationale in clap_validator.cmake.
```

### CLARA-03: Github Actions CI
```
Create .github/workflows/build.yml with matrix:
- OS: ubuntu-22.04, ubuntu-24.04
- Build type: Release, Debug
- ARA: ON, OFF
- Total: 8 configurations

Steps per matrix entry:
1. Checkout
2. apt-get install build-essential cmake clang
3. cmake configure with appropriate flags
4. cmake --build --parallel
5. Run clap-validator (CLARA-02)
6. Upload .clap and .so as build artifacts

Add a separate matrix entry for clang-tidy lint pass (Ubuntu 24.04 only,
informational, not blocking).

Cache the FetchContent _deps/ directory between runs (saves ~5 min per build).

Acceptance: 8 green builds on first push to main. Artifact downloads work.
```

### CLARA-04: Real conditioning bridge per ADR-006
```
Currently the ARA conditioning path may be stubbed. Implement properly.

Required structure (from ADR-006):
- ConditioningFrame struct, 16-byte aligned (one cache line on x86)
- Double-buffered: two ConditioningFrame slots, atomic pointer to active one
- ARA thread writes to back buffer, atomic-stores pointer to make it front
- Audio thread reads via atomic-load, applies projections in process()
- Projection weights (1x1 matrices for delta and B) loaded at activate(),
  immutable thereafter

Implementation:
1. Define ConditioningFrame in src/core/ConditioningFrame.h
2. Implement double-buffer in MambaInferenceEngine
3. Wire ARA controller to write
4. Wire audio thread to read

Tests:
- Unit test: concurrent writer + reader, no torn reads (1M iterations)
- Unit test: projection produces deterministic output for known cond vector
- Integration test: conditioned vs unconditioned output on same input,
  must produce reproducible diff with seed
- Sanitizer pass: clean under ThreadSanitizer

Out of scope: actually generating conditioning vectors from musical context.
That's TASK CLARA-09. Right now we just need the plumbing solid.
```

### CLARA-05: State serialization round-trip test per ADR-007
```
Write a test in tests/state_roundtrip.cpp:

1. Create plugin instance, configure params
2. Process a known input (440 Hz sine, 2 seconds, 44.1kHz)
3. Capture h_state mid-stream via the seqlock snapshot path
4. clap_state.save → byte buffer
5. Create fresh plugin instance
6. clap_state.load with that buffer
7. Process the continuation of the input
8. Assert output samples match the original within RT determinism epsilon (1e-5)

Edge cases:
- Same version save/load: must roundtrip cleanly
- Schema-version mismatch: must gracefully reset h_state to zero,
  emit warning via NRT diagnostic channel, never crash
- Truncated buffer: must reject and reset, never crash
- Garbage bytes: must reject and reset, never crash

Acceptance: all 4 scenarios pass under ASAN + TSAN.
```

### CLARA-06: Latency reporting verification test
```
Plugin reports 4 samples latency (conv_width). Write a test that proves it.

Test sequence:
1. Create plugin at each sample rate: 44.1kHz, 48kHz, 96kHz, 192kHz
2. Feed impulse: [0, 0, 0, ..., 1.0, 0, 0, ...] (impulse at sample 1000)
3. Process for 10000 samples
4. Find sample index of first nonzero output above noise floor (>1e-6)
5. Compute measured_latency = output_index - 1000
6. Assert |measured_latency - reported_latency| <= 1 at all sample rates

Reported latency may scale with sample rate (the 4-sample figure is at 44.1kHz).
Verify this matches reality, not just the constant 4.

If they don't match, DAW timing compensation will be wrong in every host.
This is high-priority for any actual user-facing release.
```

### CLARA-07: Cross-platform build matrix
```
Add full platform support beyond Linux x64.

macOS:
- Universal binary: arm64 + x86_64
- Bundle structure for .clap and .component
- Code signing pre-flight (warn if certificate missing, don't fail build)
- ONNX Runtime: use the macOS-universal release tarball from microsoft/onnxruntime

Windows:
- x64 only (no arm64 yet, save for later)
- MSVC primary, clang-cl alternate (matrix entry)
- .clap goes to %COMMONPROGRAMFILES%\CLAP at install
- ONNX Runtime: use the windows-x64 release tarball
- Static linking of MSVC runtime to avoid DLL dependencies

CMake changes:
- Detect platform, fetch correct ONNX Runtime variant
- Conditional source files for platform-specific bits
- Generate appropriate install rules per platform

Acceptance:
- Local build on each of: Ubuntu, macOS, Windows
- CI matrix updated to include all three
- One full platform working > all three half-working
```

### CLARA-08: ONNX export pipeline hardening
```
The Python export script (export_mamba_onnx.py) is currently working but
has implicit assumptions. Harden it.

Checks to add at start of export:
1. Assert mamba_ssm is installed and importable (no fallback path)
2. Assert torch version matches a known-good range
3. Assert opset 18 is available
4. Dry-run a single forward pass and verify output shape

Output validation:
1. Run exported ONNX model under onnxruntime on test input
2. Run PyTorch model on identical input
3. Cosine similarity must be > 0.999 for processed_audio output
4. Max abs diff < 1e-4 for state outputs

Manifest:
- Write models/manifest.json next to mamba_rt.onnx with:
  - torch version
  - mamba_ssm version
  - opset
  - input/output shape contract
  - SHA256 of the .onnx and .onnx.data files
  - Export timestamp

Plugin should refuse to load any model whose manifest doesn't match the
in-source expected schema version.

Acceptance: export is idempotent, validation passes, manifest is correct.
```

### CLARA-09: Conditioning vector generation from musical context
```
DEPENDS ON CLARA-04 (plumbing must be solid first).

This is the part that turns "infrastructure works" into "actually useful."

Conditioning vector cond[COND_DIM] should encode:
- Tempo (1 dim, normalized log-tempo)
- Key (12 dims, one-hot or chroma-style soft encoding)
- Mode (2 dims, major/minor soft mix)
- Time signature numerator (3 dims, 3/4 vs 4/4 vs 6/8 soft mix)
- Section type if available (verse/chorus/bridge — 4 dims soft mix)
- Reserved space for future (remaining dims, zero-initialized)

ARA thread responsibilities:
- Pull tempo from ARA musical context
- Pull key/mode from ARA tonal context if available, else estimate
  via krumhansl-schmuckler on the audio source
- Update conditioning frame when context changes
- Trigger atomic swap

Test:
- Feed two identical audio inputs through the same model with different
  conditioning vectors. Output must differ in a way that's traceable to
  the projection weights, not random noise.
- Feed audio + identical context vs audio + zeroed context. Both should
  produce valid audio output (no clicks, no NaN).
```

### CLARA-10: User-facing UI polish
```
The plugin currently has a minimal/placeholder UI. Replace with:

Required controls:
- Bypass toggle (large, top-left, click-free engage/disengage)
- Wet/dry knob (0-100%)
- Intensity knob (0-100%, drives the conditioning strength)
- Model selector dropdown (when multiple .onnx files in models/ dir)
- Status indicator: "Ready" / "Loading model..." / "Error: {message}"

Optional but high-value:
- Real-time gain reduction meter (vertical, dB-scale)
- Conditioning vector visualization (small grid showing active cond dims)
- "About" / version popup with model + plugin version

Constraints:
- Use whatever UI framework CLARA already has (JUCE or custom — match existing)
- Resizable: minimum 400x300, maximum 1200x800
- Honor DAW dark/light theme if API exposes it

Out of scope for this task: actual GUI scripting, presets, A/B comparison,
visualization of inference output. Those are V2.
```

---

## Decision Debt Log (review during each session)

- ARA C++ wrapper completeness — CLARA-01 must verify before depending on it
- Model versioning — currently no schema version in conditioning frame or weight file. CLARA-08 introduces it; everything depending on weights must respect it
- macOS code signing — CLARA-07 must produce signed builds for any user-facing release
- iOS / mobile — explicitly out of scope, never auto-add

---

## Done = Shipping?

CLARA is "done" for first user release when:
1. CLAP validator passes 100% (CLARA-02)
2. All 3 platforms build green in CI (CLARA-07)
3. State save/load tests pass (CLARA-05)
4. Latency is verified correct (CLARA-06)
5. ONNX manifest enforcement works (CLARA-08)
6. Conditioning bridge actually conditions (CLARA-04 + CLARA-09)
7. UI is usable by a non-developer (CLARA-10)
8. A trained model is dropped in (separate, depends on training run)

Until then: do not release publicly. OK for closed beta.
