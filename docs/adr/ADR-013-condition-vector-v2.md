# ADR-013: ConditionVector V2 — d_cond=128 with CREPE, Semantic, Cross-Track

**Status:** Accepted  
**Date:** 2026-07-02  
**Replaces:** ADR-008 (COND_DIM=8, external gain modulation), ADR-010 (dual-path ONNX conditioning)

## Context

The original conditioning scheme (ADR-008, ADR-010) used `COND_DIM=8` with slots for DAW tempo/key/signature and external post-inference gain modulation. PerTrackProcessor uses a much richer conditioning pipeline: CREPE pitch tracking, SemanticEncoder text/audio embeddings, CrossTrackAggregator for multi-stem mixing, and loudness features. These feed into a CondMLP that produces per-sample FiLM conditioning at `COND_DIM=128`.

## Decision

**Conditioning uses `COND_DIM=128` with four source channels combined via CondMLP.**

### Conditioning Slot Map

| Slots | Source | Dimension | Description |
|-------|--------|-----------|-------------|
| 0-31 | F0Conditioner | 32 | CREPE 360-bin → Linear(d_f0=32) pitch projection |
| 32-159 | SemanticEncoder | 128 | Waveform → conv stem → Mamba3MIMOLayer → projection |
| 160-223 | CrossTrackAggregator | 64 | Causal MHA over K other stems in mix |
| 224-227 | Loudness | 4 | RMS, peak, crest factor, LUFS estimate |
| 228-255 | Reserved | 28 | Future use (DAW context, user knobs, CLAP encoder) |

### CondMLP Architecture

```
[f0_proj(32) + sem_emb(128) + cross_attn(64) + loudness(4) = 228 dims]
    ↓
Linear(228, 256) → SiLU → Linear(256, 128) = d_cond
    ↓
Per-layer FiLM: cond → film_proj_w → γ(128), β(128) per d_model channel
Per-head Δ modulation: cond → dt_film_proj_w → γ(4), β(4) per head
```

### Double-Buffered Thread Safety

The condition vector is double-buffered in C++: a non-RT thread writes to one buffer, then atomically swaps a pointer. The audio thread reads from the swapped buffer via acquire. This prevents torn reads and ensures RT-safe conditioning updates.

```cpp
// C++ pattern
struct ConditionProcessor {
    alignas(64) float cond_[2][COND_DIM];  // double-buffer
    std::atomic<size_t> read_idx_{0};       // audio thread reads this
    void update(const float* new_cond);      // non-RT thread swaps
    const float* get() const;                // audio thread reads
};
```

## Consequences

- **No backward compatibility**: ADR-008's `COND_DIM=8` and external gain modulation are replaced.
- **Slot allocation is not backward compatible**: The slot map changed from DAW(0-7)/CLAP(8-39)/knobs(40-71)/reserved(72-127) to the new layout above.
- **DAW context slots are reserved (not removed)**: Slots 228-255 provide room for future ARA-driven tempo/key/signature conditioning.
- **CLAP text/audio encoder conditioning**: When implemented (Phase 2/3), will occupy reserved slots or extend the semantic encoder input.
- **C++ engine must implement CondMLP**: `ConditionProcessor` currently uses the old slot map. It needs the new 228→128 CondMLP path.
