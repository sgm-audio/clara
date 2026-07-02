# ADR-012: PerTrackProcessor Weight Format V3

**Status:** Accepted  
**Date:** 2026-07-02  
**Replaces:** ADR-001 §3 (ONNX model export), ADR-010 §1 (ONNX-integrated weights)

## Context

The legacy C++ engine loads `model.bin` in Version 2 format (13 float32 arrays per layer, R=4, real-valued SSM). PerTrackProcessor requires Version 3 with complex pole parameters, R8 extractor weights, and STFT frontend configuration. The training pipeline (`train_clara.py`) already exports V3 format.

## Decision

**Flat binary weight files use Version 3 format (18 arrays per layer + frontend config).**

### V3 Header Layout (96 bytes)

| Offset | Field | Type | Notes |
|--------|-------|------|-------|
| 0 | magic | uint32 | `0x4D424D41` ("MBMA") |
| 4 | version | uint32 | `3` |
| 8 | d_state | uint32 | 32 |
| 12 | d_model | uint32 | 128 |
| 16 | d_inner | uint32 | 256 |
| 20 | nheads | uint32 | 4 |
| 24 | headdim | uint32 | 64 |
| 28 | mimo_rank | uint32 | 8 |
| 32 | n_layers | uint32 | 6 |
| 36 | cond_dim | uint32 | 128 |
| 40 | in_proj_dim | uint32 | PerTrackProcessor in_proj output dim |
| 44 | weight_data_size | uint32 | Bytes after header |
| 48 | n_fft | uint32 | 2048 (STFT) |
| 52 | hop_length | uint32 | 512 |
| 56 | n_mels | uint32 | 128 |
| 60 | r8_bands | uint32 | 8 |
| 64 | n_semantic | uint32 | 128 |
| 68 | d_f0 | uint32 | 32 |
| 72 | d_cross | uint32 | 64 |
| 76-95 | reserved | uint32[5] | Future use |

### Per-Layer Arrays (18 arrays, all float32)

| # | Array | Shape | Notes |
|---|-------|-------|-------|
| 0 | in_proj_w | [in_proj_dim, d_model] | Fused projection |
| 1 | dt_bias | [nheads] | Δ bias |
| 2 | B_bias | [nheads, mimo_rank, d_state] | B projection bias |
| 3 | C_bias | [nheads, mimo_rank, d_state] | C projection bias |
| 4 | B_norm_weight | [d_state] | RMSNorm γ for B |
| 5 | C_norm_weight | [d_state] | RMSNorm γ for C |
| 6 | mimo_x | [nheads, mimo_rank, headdim] | Per-band input mix |
| 7 | mimo_z | [nheads, mimo_rank, headdim] | Per-band gate mix |
| 8 | mimo_o | [nheads, mimo_rank, headdim] | Per-band output mix |
| 9 | D_proj | [nheads] | Skip connection weight |
| 10 | out_proj_w | [d_model, d_inner] | Output projection |
| 11 | film_proj_w | [2*d_model, cond_dim] | FiLM γ,β per channel |
| 12 | dt_film_proj_w | [2*nheads, cond_dim] | FiLM for Δ |
| 13 | log_a_real | [nheads, mimo_rank] | Complex pole real part |
| 14 | a_imag | [nheads, mimo_rank] | Complex pole imag part |
| 15 | r8_band_w | [r8_bands, n_mels] | R8 triangular filter weights |
| 16 | r8_transient_w | [1, headdim] | Transient energy extractor |
| 17 | r8_centroid_w | [1, headdim] | Spectral centroid extractor |

### Frontend Weights Block (after all layers)

| Array | Shape | Notes |
|-------|-------|-------|
| mel_filterbank | [n_mels, n_fft//2+1] | Mel filterbank matrix |
| semantic_conv_w | [semantic conv params] | SemanticEncoder conv stem |
| semantic_proj_w | [d_semantic, ...] | SemanticEncoder output projection |
| f0_proj_w | [d_f0, 360] | CREPE bin projection |
| cross_attn_w | [d_cross, ...] | CrossTrackAggregator weights |
| cond_mlp_w | [cond_dim, ...] | CondMLP weights |

### Weight Data Size

```
Per layer: sum of all 18 array element counts × 4 bytes
6 layers × per_layer + frontend block = total weight_data_size
```

## Consequences

- **Not backward compatible with V2**: Old `model.bin` files will fail `validate_header()` with version mismatch.
- **C++ engine must implement V3 loader**: `WeightLoader::setup_layer_pointers()` needs 18-array layout and frontend block.
- **`validate_export.py` already expects V3**: No change needed.
- **File size**: ~8-12MB (vs 6.5MB for V2 R=4 model) due to R8 expansion and frontend weights.
