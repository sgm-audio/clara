# CLARA Latency Architecture

## Overview

CLARA reports a fixed latency of 4 samples via the CLAP latency extension. This document explains where that latency comes from, how it's measured, and how to verify it.

## Source of Latency

### Causal Convolution (conv_width=4)

The Mamba SSM block contains a causal 1D convolution with kernel width `conv_width=4` (see `export_mamba_onnx.py`). A causal convolution only looks at past samples, padding the left side with `conv_width - 1 = 3` zeros.

The first output sample that depends on real (non-zero-padded) input is at index `conv_width - 1 = 3`.

However, the latency of a causal conv is typically the full kernel width (4 samples) because:
- Sample 0 of output depends on [0, 0, 0, input[0]] (3 zeros + 1 real input)
- The "valid" output (fully determined by real input) starts at sample 3
- But the convolution is sliding — the output at sample N depends on inputs N-3 through N

### Reported Latency

```
static uint32_t clap_latency_get(const clap_plugin_t*) {
    return 4;  // conv_width samples
}
```

The DAW uses this to advance the timeline, ensuring the processed output aligns with other tracks.

## Verification

### Test: tests/test_latency.cpp

Feeds an impulse at a known position through the MambaInferenceEngine and measures the output delay.

### Script: scripts/verify_latency.py

Runs the same verification at the ONNX level using onnxruntime directly.

## Sample Rate Independence

The latency is reported as a constant 4 samples regardless of sample rate. This means:
- 4 samples @ 44.1kHz = ~0.09ms
- 4 samples @ 48kHz = ~0.08ms  
- 4 samples @ 96kHz = ~0.04ms
- 4 samples @ 192kHz = ~0.02ms

The causal convolution operates on sample indices, not time. The kernel width is fixed at 4 samples regardless of sample rate.

## Potential Issues

If the ONNX model's causal conv implementation uses different padding (e.g., reflection padding, symmetric padding, or no padding), the actual latency could differ from the reported 4 samples. The impulse-response test verifies this.

## Related

- ADR-001: Shared ONNX Inference Engine (conv_width=4 rationale at line 24)
- ADR-003: Click-Free Bypass (crossfade adds no additional latency)
- `src/clap/MambaClapPlugin.cpp:157-163` — latency extension
- `tests/test_latency.cpp` — verification test
