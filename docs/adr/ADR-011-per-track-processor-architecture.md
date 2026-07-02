# ADR-011: PerTrackProcessor Architecture (R=8, Spectral, Complex Poles)

**Status:** Accepted  
**Date:** 2026-07-02  
**Replaces:** ADR-001 (ONNX inference engine), ADR-009 (sub-block processing)

## Context

The original CLARA architecture used a time-domain Mamba-3 SSM with R=4 MIMO, trapezoidal real-valued SSM, RoPE angles, and no spectral processing (direct per-sample audio). The mamba3 external drive contained `PerTrackProcessor` — a spectral-domain model with R=8 structured features, complex-pole SSM, CREPE f0 conditioning, and cross-track attention. After audit and comparison (`docs/ARCHITECTURE_COMPARISON.md`), PerTrackProcessor was adopted as the canonical CLARA architecture.

## Decision

**CLARA uses PerTrackProcessor (R=8, spectral-domain, complex poles) as canonical architecture.**

### Architecture Overview

```
Input waveform (44.1kHz, stereo)
    ↓
STFT Frontend (n_fft=2048, hop_length=512, Hann window)
    ├── Magnitude → mel filterbank (n_mels=128)
    └── Waveform → R8FeatureExtractor (8 band energies + transient + centroid)
    ↓
R8FeatureExtractor: 8 soft triangular band filters
    sub_bass (0-60Hz), bass (60-250Hz), low_mid (250-500Hz),
    mid (500-2000Hz), high_mid (2000-4000Hz), air (4000-8000Hz),
    transient_energy, spectral_centroid
    ↓
SemanticEncoder: conv stem → 2× Mamba3MIMOLayer → projection → d_sem=128
F0Conditioner: CREPE 360-bin → Linear(d_f0=32)
CrossTrackAggregator: causal MHA over K other stems → d_cross=64
CondMLP: [d_sem + d_f0 + d_cross + loudness(4)] → d_cond=128
    ↓
6× Mamba3MIMOLayer (gradient checkpointed)
    PerBandSSM: R=8, d_model=128, d_state=32, nheads=4
    Complex-pole trapezoidal SSM (no RoPE)
    FiLM conditioning from CondMLP
    ↓
Decoder: d_model → 2×F_bins (complex STFT) → ISTFT → waveform
```

### Key Dimensions

| Parameter | Value | Notes |
|-----------|-------|-------|
| d_model | 128 | Embedding dimension |
| d_inner | 256 | 2× d_model expansion |
| d_state | 32 | SSM state dimension |
| nheads | 4 | Parallel heads |
| headdim | 64 | d_inner / nheads |
| R (mimo_rank) | 8 | Spectral bands + transient + centroid |
| n_layers | 6 | SSM layers |
| n_fft | 2048 | STFT window |
| hop_length | 512 | ~11.6ms @ 44.1kHz |
| n_mels | 128 | Mel filterbank |
| cond_dim | 128 | FiLM conditioning input |
| d_semantic | 128 | Semantic encoder output |
| d_f0 | 32 | CREPE projection |
| d_cross | 64 | Cross-track attention output |
| Total params | ~4.26M | ~3.9M mamba core + 360K conditioning/encoder |

### Complex-Pole SSM

Unlike the legacy R=4 time-domain model (trapezoidal with real dd_A), PerTrackProcessor uses complex-pole trapezoidal discretization:

```
A = exp(log_a_real + i * a_imag)  # complex diagonal
B = B_proj(x)                     # input-dependent
C = C_proj(x)                     # input-dependent
h_t = A * h_{t-1} + B * x_t      # complex state update
y_t = C * h_t.real + D * x_t      # real output
```

No RoPE rotation is used — the complex poles provide inherent frequency selectivity.

### Latency

The STFT frontend introduces n_fft/2 = 1024 samples of latency (~23.2ms @ 44.1kHz). The CLAP plugin reports this via `clap_plugin_latency`. This replaces the previous zero-latency claim from the time-domain R=4 architecture.

## Consequences

- **C++ engine must be updated** (H3 in `docs/TODO.md`): `SsmInferenceCore` currently implements R=4 time-domain with RoPE. It needs STFT/ISTFT wrappers, complex-pole SSM, R8 feature extraction, and v3 weight format support.
- **Training is spectral-domain**: Uses STFT frontend, operates on frames, not samples.
- **Higher latency**: Zero-latency is no longer a selling point. Tradeoff for higher-quality processing.
- **Model.bin must be v3 format**: 18 arrays/layer with complex pole parameters.
