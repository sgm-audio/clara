# ARA Host Bridge for Conditioning (CLARA-09 / P1 #15)

## Problem

The CLAPA plugin's conditioning vector (`cond[COND_DIM=8]`) is currently set to default/neutral values from `ConditioningGenerator::defaultContext()`. It encodes tempo, key, mode, time signature, and section type for the ONNX model's CondFiLM layers, but these values never reflect the actual musical context from the DAW.

The ARA 2.0 protocol provides real-time musical context (tempo, key signature, time signature, chord changes) through the host's content access controller, but this data is never extracted or pushed into the conditioning pipeline.

## Goal

Wire ARA's `MusicalContext` `→` ARA DocumentController `→` MambaInferenceEngine `→` ONNX CondFiLM, so the SSM model receives real musical context during audio processing.

---

## 1. Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  DAW Host                                                           │
│  ┌──────────────┐    ┌──────────────────────┐    ┌───────────────┐ │
│  │ MusicalContext│───>│ ContentAccessController│──>│ ARA Factory   │ │
│  └──────────────┘    └──────────────────────┘    └───────┬───────┘ │
└──────────────────────────────────────────────────────────┼─────────┘
                                                           │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ARA Plug-In Side                                                   │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐  │
│  │ MambaDocumentController│───>│ doUpdateMusicalContextContent()│  │
│  └──────────────────────┘    └──────────────┬───────────────────┘  │
│                                             │                       │
│         ┌───────────────────────────────────┘                       │
│         ▼                                                           │
│  ┌──────────────────┐    ┌──────────────────────────────────────┐  │
│  │ ARA Content Reader│───>│ Extract: tempo, key, mode, time sig │  │
│  └──────────────────┘    └──────────────┬───────────────────────┘  │
│                                         │                           │
│         ┌───────────────────────────────┘                           │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ ConditioningGenerator::encode(araCtx, cond, COND_DIM)         │  │
│  │ cond[0] = tempoBpm / 200.0                                    │  │
│  │ cond[1] = keyPitchClass / 12.0                                │  │
│  │ cond[2] = mode (0=major, 1=minor, 0.5=unknown)               │  │
│  │ cond[3] = timeSigNumerator / 8.0                              │  │
│  │ cond[4] = sectionType / 6.0                                   │  │
│  │ cond[5..7] = reserved (zero)                                  │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│         ┌───────────────────┘                                       │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ MambaInferenceEngine::setConditioning(cond, COND_DIM)         │  │
│  │ → atomic double-buffer swap (RT-safe)                         │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
└─────────────────────────────┼─────────────────────────────────────┘
                              │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Audio Thread (RT-safe)                                              │
│  ┌──────────────────────┐    ┌────────────────────────────────────┐  │
│  │ processBlock() reads │───>│ cond tensor fed as 10th ONNX input│  │
│  │ activeCond_ pointer  │    │ → CondFiLM applies per-layer      │  │
│  └──────────────────────┘    │   modulation to conv/ssm dynamics │  │
│                              └────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Extracting Musical Context from ARA API

### 2.1 Content Access Controller

The ARA host provides `getDocumentContentAccessController()` on the `ARADocumentControllerHostInstance`. This controller has a `createReaderForContentType()` method that returns a content reader for `ARAContentType::kMusicalContext`.

```cpp
// In doUpdateMusicalContextContent():
auto* hostCtrl = getDocumentControllerHostInstance();
if (!hostCtrl) return;

auto* contentAccess = hostCtrl->getDocumentContentAccessController();
if (!contentAccess) return;

// Create reader for musical context
auto* reader = contentAccess->createReaderForContentType(
    ARAContentType::kMusicalContext,
    musicalContext,
    &range);
if (!reader) return;

// Iterate events
uint32_t numEvents = reader->getEventCount();
for (uint32_t i = 0; i < numEvents; ++i) {
    auto* event = reader->getEvent(i);
    if (!event) continue;
    // event->type determines what data is available
    // event->data points to the event payload
}
reader->destroyReader();
```

### 2.2 ARA Musical Context Events

ARA 2.0 defines these content events for `kMusicalContext`:

| Event Type | Payload | Fields |
|-----------|---------|--------|
| `kARAEventParameter` | `ARAEventParameterContent` | `parameterId` (tempo, time sig, key sig, etc.) |
| `kARAEventTempo` | `ARAEventTempoContent` | `beatsPerMinute` (double) |
| `kARAEventTimeSignature` | `ARAEventTimeSignatureContent` | `numerator`, `denominator` (uint32) |
| `kARAEventKeySignature` | `ARAEventKeySignatureContent` | `key` (enum), `scale` (major/minor) |
| `kARAEventChord` | `ARAEventChordContent` | `root`, `type`, `additions`, `alterations` |

Note: Not all hosts implement all event types. The reader may return zero events. A robust implementation must handle this gracefully.

### 2.3 Encoding to cond Vector

After extracting context values, use `ConditioningGenerator::encode()`:

```cpp
ConditioningGenerator::MusicalContext ctx;
ctx.tempoBpm = extractedTempo;              // default: 120.0
ctx.keyPitchClass = extractedKey;           // default: 0 (C)
ctx.mode = (scale == kARAKeyScaleMinor) ? 1 : 0;
ctx.timeSigNumerator = extractedNumerator;  // default: 4
ctx.sectionType = 0;                        // default: 0 (verse)

float cond[COND_DIM];
ConditioningGenerator::encode(ctx, cond, COND_DIM);
engine->setConditioning(cond, COND_DIM);
```

---

## 3. Thread Safety

### 3.1 Call Path Analysis

| Step | Thread | RT-Safe? | Notes |
|------|--------|----------|-------|
| `doUpdateMusicalContextContent()` | Main (non-RT) | N/A | Called by ARA host on main thread |
| ARA content reader creation | Main | N/A | Heap allocs OK |
| `ConditioningGenerator::encode()` | Main | N/A | Simple math, no allocs |
| `setConditioning(cond, 8)` | Main | **Yes** | Atomic double-buffer swap; non-blocking |
| `processBlock()` reads `activeCond_` | Audio | **Yes** | Atomic load, no locks |
| Cond tensor fed to ONNX | Audio | **Yes** | Thread-local Ort::Values |

### 3.2 Condition Vector Update Protocol

```
Main Thread:                              Audio Thread:
  setConditioning(cond, 8)                  processBlock():
    ↓ atomic_store(activeIdx)                  ↓ atomic_load(activeIdx)
  (returns immediately)                      reads cond from active buffer
                                             feeds to ONNX model
```

The double-buffer ensures:
- Writer always writes to the dormant buffer before flipping the atomic index
- Reader always reads the active buffer without locking
- No torn reads: the buffer switch is a single atomic store of a small integer

### 3.3 Atomic Memory Order Guarantees

```cpp
// Writer (main thread):
dormantBuf_.fill(src, dim);                     // non-atomic writes
activeIdx_.store(1 - activeIdx_, std::memory_order_release);  // publish

// Reader (audio thread):
int idx = activeIdx_.load(std::memory_order_acquire);  // acquire
const float* data = (idx == 0) ? condBufA_.data : condBufB_.data;
// Use data — guaranteed to see the fill() writes
```

---

## 4. When to Trigger Analysis

| Trigger | ARA Callback | Action |
|---------|-------------|--------|
| Musical context changes | `doUpdateMusicalContextContent()` | Extract context, encode, `setConditioning()` |
| Audio source content changes | `doUpdateAudioSourceContent()` | Set `contentDirty_`, defer full re-analysis |
| Audio source samples enabled | `didEnableAudioSourceSamplesAccess()` | Trigger initial analysis pass |
| Transport position change (future) | ARA transport access | Re-evaluate current section context |
| Chord change (future) | `kARAEventChord` in reader | Update section/effect dims |

### Prioritized Implementation Sequence

1. **Phase 1**: `doUpdateMusicalContextContent()` → extract tempo/key/mode/time-sig → `setConditioning()` ↑
2. **Phase 2**: Add section-type inference from audio source regions
3. **Phase 3**: Add chord-aware conditioning via `kARAEventChord`

---

## 5. Fallback Behavior

When MusicalContext is unavailable (pre-ARA projects, non-ARA hosts, or ContentAccessController returns no events):

```cpp
ConditioningGenerator::MusicalContext fallback;
fallback.tempoBpm = 120.0f;
fallback.keyPitchClass = 0;   // C
fallback.mode = 0;            // major
fallback.timeSigNumerator = 4;
// remaining fields default to 0

// This produces cond = [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
// → neutral musical context
```

This matches the current `ConditioningGenerator::defaultContext()` behavior.

---

## 6. Interface Between ARA Controller and CLAP Plugin

### Current Architecture Problem

The ARA DocumentController (`MambaDocumentController`) and the CLAP plugin (`MambaPluginState`) are separate objects with no communication channel. The ARA controller has an engine pointer (`araEngine_`) that points to the `MambaPlaybackRenderer`'s engine, but the CLAP plugin's engine is a separate instance.

### Proposed Architecture

There are two design options:

#### Option A: Shared Engine (Recommended)

The CLAP plugin owns a shared inference engine, and both the CLAP process() and ARA analysis read from it. The ARA controller pushes conditioning updates to this shared engine.

```cpp
// MambaDocumentController gets a pointer to the shared engine:
void MambaDocumentController::setSharedEngine(
    MambaInferenceEngine* engine) noexcept
{
    sharedEngine_ = engine;
}

// In doUpdateMusicalContextContent():
if (sharedEngine_) {
    // ...extract context, encode...
    sharedEngine_->setConditioning(cond, COND_DIM);
}
```

**Pros**: Single model instance, no sync needed, conditioning always reflects latest context.
**Cons**: Requires wiring during plugin initialization; ARA controller needs CLAP plugin reference.

#### Option B: SPSC Queue Bridge

The ARA controller pushes cond updates to the CLAP plugin's SPSC deferred task queue, which the CLAP plugin's `mamba_on_main_thread` processes.

```cpp
// ARA controller pushes cond update:
struct CondUpdateTask {
    float cond[COND_DIM];
};
// Enqueued to CLAP plugin's SPSC queue

// CLAP plugin's on_main_thread pops and calls setConditioning:
state->engine->setConditioning(task.cond, COND_DIM);
```

**Pros**: Decoupled, follows existing deferred task pattern.
**Cons**: Higher latency (queue turn-around), more complex, redundant indirection.

---

**Recommendation: Option A (Shared Engine)** — simpler, lower latency, fewer moving parts. The ARA controller already has a `setAraEngine()` pattern; extend it to accept the shared engine reference from the CLAP plugin.

### Wiring in CLAP `mamba_activate()`:

```cpp
// In clap_create_plugin():
// The CLAP plugin stores a pointer to the ARA DocumentController
// (set externally by the DAW when both plugins are loaded)

// In mamba_activate():
if (state->araController) {
    state->araController->setSharedEngine(state->engine.get());
}
```

Note: The DAW loads the CLAP and ARA plugins separately. The mechanism to discover the ARA controller from the CLAP plugin is host-dependent. In Bitwig and REAPER, both share the same `clap_plugin` instance. In Studio One, they may be separate. This wiring requires host-specific handling and is deferred until a host integration test is available.

---

## 7. Integration with SPSCQueue Deferred Tasks

The existing `AnalysisQueueDrain` deferred task (currently a no-op) should be used for computationally expensive analysis that cannot run in the ARA callback:

| Task | Route | Reason |
|------|-------|--------|
| Lightweight cond update (tempo, key, mode) | Direct `setConditioning()` from ARA callback | RT-safe atomic swap, no allocations |
| Heavy analysis (audio feature extraction, beat tracking) | SPSC queue `AnalysisQueueDrain` | ARA callback is non-RT but should return quickly; offload to main thread |

---

## 8. Testing Strategy

### Unit Tests
- Mock ARA MusicalContext → verify cond vector encoding
- Verify fallback/neutral values when context unavailable
- Thread-safety test: concurrent cond writes from ARA callback while audio thread reads

### Integration Tests
- Create full ARA project with known tempo/key/time-sig
- Verify cond vector is updated when context changes
- Verify conditioning flows through ONNX FiLM layers

### Test Hooks
```cpp
// Add to MambaDocumentController for testing:
void MambaDocumentController::simulateMusicalContextUpdate(
    float tempoBpm, int keyPitchClass, int mode, int timeSigNum)
{
    ConditioningGenerator::MusicalContext ctx;
    ctx.tempoBpm = tempoBpm;
    ctx.keyPitchClass = keyPitchClass;
    ctx.mode = mode;
    ctx.timeSigNumerator = timeSigNum;
    float cond[COND_DIM];
    ConditioningGenerator::encode(ctx, cond, COND_DIM);
    if (sharedEngine_) {
        sharedEngine_->setConditioning(cond, COND_DIM);
    }
}
```

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Host doesn't provide ContentAccessController | No cond update → default values | Already handled: defaultContext() gives neutral cond |
| ARA content reader returns no events | Empty extraction | Handle zero-event case gracefully |
| Host delivers context on wrong thread | Data race | setConditioning() is fully RT-safe (atomic double-buffer) |
| Frequent context updates (every bar) | Excessive cond swaps | Atomic double-buffer handles rapid updates; compare-and-skip if unchanged |
| CLAP and ARA plugins are separate instances | Shared engine not available | Fall back to independent ARA engine with neutral cond; host bridge wiring is P2 |

---

## 10. Implementation Roadmap

1. ✅ Phase 0: `setConditioning()` and atomic double-buffer in engine
2. ✅ Phase 1: `ConditioningGenerator` with encode/decode/neutral/defaultContext
3. ✅ Phase 2: Default context wired in `mamba_activate()` via CLAP
4. ⬜ Phase 3: ARA content reader in `doUpdateMusicalContextContent()` (this design)
5. ⬜ Phase 4: Shared engine wiring between CLAP and ARA
6. ⬜ Phase 5: Section-type inference from audio regions
7. ⬜ Phase 6: Chord-aware conditioning
