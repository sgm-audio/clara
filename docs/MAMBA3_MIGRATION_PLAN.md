# CLARA Mamba-1 → Mamba-3 Migration Plan

**Status:** Historical — Phases 0-6 Complete. PerTrackProcessor adopted July 2026.
**Date:** June 15, 2026 | **Updated:** July 2, 2026
**Note:** This document describes the ORIGINAL Mamba-3 R=4 time-domain architecture (~1.66M params). As of July 2026, CLARA has adopted the PerTrackProcessor (R=8, spectral-domain, ~4.26M params) as the canonical architecture. See ADR-011, ADR-012, ADR-013. This document is retained for historical reference of the original migration specifications.

---

## A. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     DAW Host (Audio Thread)                       │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              CLAP Plugin (one per model)                   │    │
│  │  e.g. "Mamba Saturator.clap", "Mamba Compressor.clap"    │    │
│  │                                                           │    │
│  │  ┌────────┐  ┌──────────────────┐  ┌────────┐           │    │
│  │  │ Audio  │→ │ SSMInferenceCore │→ │ Audio  │           │    │
│  │  │ In     │  │  (6 layers,      │  │ Out    │           │    │
│  │  │        │  │   MIMO rank 4)   │  │        │           │    │
│  │  └────────┘  └────────┬─────────┘  └────────┘           │    │
│  │                        │                                 │    │
│  │  ┌─────────────────────▼────────────────────────────┐    │    │
│  │  │         Conditioning Vector (128 floats)          │    │    │
│  │  │  ┌──────┬────────┬────────┬──────────────────┐    │    │
│  │  │  │ 0-7  │  8-39  │ 40-71  │   72-127         │    │    │
│  │  │  │ DAW  │  CLAP  │  User  │   Reserved        │    │    │
│  │  │  │ ctx  │  embed │ knobs  │   (future)        │    │    │
│  │  │  └──────┴────────┴────────┴──────────────────┘    │    │
│  │  └────────────────────────────────────────────────────┘    │
│  │                                                           │    │
│  │  Cond vector updated by (atomic pointer swap, RT-safe):   │    │
│  │  • ARA (DAW context) — real-time, on tempo/section change │    │
│  │  • CLAP text encoder — on user prompt (main thread)       │    │
│  │  • CLAP audio encoder — on transport stop (main thread)   │    │
│  │  • Preset system — instant, precomputed embeddings         │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │         ARA Extension (MambaSuiteARA.so)                  │    │
│  │  • Reads: tempo, key, time signature, section markers     │    │
│  │  • Writes: cond vector slots 0-7 in real-time             │    │
│  │  • Persists: bypass/wetDry/intensity in archive           │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     Main Thread (Non-RT)                         │
│                                                                   │
│  ┌──────────────┐    ┌───────────────────┐    ┌──────────────┐  │
│  │ User types   │───→│ CLAP Text Encoder  │───→│ 32-float    │  │
│  │ "gritty vox" │    │ (ONNX, 300MB)      │    │ embedding    │  │
│  └──────────────┘    └───────────────────┘    └──────┬───────┘  │
│                                                      │          │
│  ┌──────────────┐    ┌───────────────────┐          │          │
│  │ Transport    │───→│ CLAP Audio Encoder│───→ 32-float         │
│  │ STOP event   │    │ (ONNX, 300MB)     │    embedding         │
│  └──────────────┘    └───────────────────┘    └──────┬───────┘  │
│                                                      ▼          │
│                                              Atomic swap into   │
│                                              cond vector slots  │
│                                              8-39 (RT-safe)     │
└──────────────────────────────────────────────────────────────────┘

Runtime: Shared libmamba_core.so (C++ engine, loaded by all plugins).
Each plugin loads its own model.bin at init (dedicated weights per task).
CLAP encoders are shared via libmamba_clap.so (loaded once, used by all).
```

---

## B. Model Configuration — Mamba-3 with MIMO R=4

### Dimensions (compile-time constants)

| Constant | Value | Rationale |
|---|---|---|
| `MAX_D_STATE` | 32 | SSM state dimension |
| `MAX_D_MODEL` | 128 | Embedding dimension |
| `MAX_D_INNER` | 256 | 2× d_model expansion |
| `MAX_HEADDIM` | 64 | Per-head dimension |
| `MAX_NHEADS` | 4 | Parallel heads |
| `MAX_MIMO_RANK` | **4** | 4 slots per head — sub/bass/mids/highs |
| `MAX_NUM_BC_HEADS` | 1 | B/C groups |
| `MAX_NUM_ROPE_ANGLES` | 16 | = d_state / 2 |
| `MAX_CHUNK_SIZE` | 64 | Process block size (matches engine) |
| `MAX_BLOCK_SIZE` | 512 | Max samples per process() call |
| `MAX_LAYERS` | 6 | Number of SSM layers |
| `COND_DIM` | 128 | DAW(8) + CLAP(32) + user knobs(32) + reserved(56) |
| `SSM_STATE_NORM_INTERVAL` | 4096 | Periodic state re-normalization |

### Parameter count

| Component | Params |
|---|---|
| Input projection | 129 × 128 = 16,512 |
| Per MambaLayer (×6): | |
| — in_proj (fused) | 128 × 419 = 53,632 |
| — dt_bias | 4 |
| — B_bias | 4 × 4 × 32 = 512 |
| — C_bias | 4 × 4 × 32 = 512 |
| — B_norm_weight | 32 |
| — C_norm_weight | 32 |
| — mimo_x | 4 × 4 × 64 = 1,024 |
| — mimo_z | 4 × 4 × 64 = 1,024 |
| — mimo_o | 4 × 4 × 64 = 1,024 |
| — D_proj | 4 |
| — out_proj | 256 × 128 = 32,768 |
| — film_proj | 128 × 256 = 32,768 |
| — dt_film_proj | 128 × 8 = 1,024 |
| Subtotal/layer | ~123,356 |
| 6 layers | ~740,136 |
| LayerNorm | 256 |
| Output projection | 128 × 1 = 128 |
| **TOTAL** | **~757,032** |

### State size per layer (with R=4)

| State array | Shape | Floats |
|---|---|---|
| ssm_real | [4, 64, 32] | 8,192 |
| ssm_imag | [4, 64, 32] | 8,192 |
| k_state | [4, 4, 32] | 512 |
| angle_state | [4, 16] | 64 |
| **Per layer** | | **16,960** |
| **× 6 layers** | | **101,760** (~400KB) |

### Compute cost

| Operation | MACs |
|---|---|
| Fused in_proj matvec | 128 × 419 = 53,632 |
| out_proj | 256 × 128 = 32,768 |
| SSM recurrence (R=4) | 4 × 4 × 64 × 32 = 32,768 |
| FiLM projection | 128 × 256 = 32,768 |
| MIMO coupling | 4 × (4 × 64) × 3 = 3,072 |
| RoPE rotation | 4 × 64 × 16 = 4,096 |
| **Total per layer** | **~159,104** |
| **× 6 layers** | **~954,624** |

Real-world estimate with 3× overhead: ~2.87 µs/sample. At 44.1kHz that's **~12.6% of one core**. Still fine for a channel strip.

---

## C. C++ Engine Architecture — Post-ONNX

### What was removed

| File | Replacement | Reason |
|---|---|---|
| `src/core/MambaInferenceEngine.h` | `src/core/SsmInferenceCore.h` | ONNX Runtime → custom engine |
| `src/core/MambaInferenceEngine.cpp` | `src/core/SsmInferenceCore.cpp` | Ort::Session → SSMBlock array |
| `train_clara_export.py` | `scripts/export_mamba_weights.py` | ONNX → flat binary |
| `export_mamba_onnx.py` | _(deleted)_ | Incompatible architecture |
| `train_clara_parallel.py` | `train_clara.py` | Mamba-1 → Mamba-3 |
| `models/*.onnx` | `models/*.bin` | Weight format change |
| `libonnxruntime.so` | _(not shipped)_ | Dependency eliminated |

### New file structure

```
src/
├── core/
│   ├── MambaConfig.h              # Dimension constants (compile-time)
│   ├── MambaConfig.cpp            # Validation + serialization
│   ├── SsmBlock.h                 # SSMState, SSMWeights, SSMBlock class
│   ├── SsmBlock.cpp               # 8-step per-sample + block processing
│   ├── SsmInferenceCore.h         # Multi-layer wrapper, state management
│   ├── SsmInferenceCore.cpp       # Forward, reset, cond update
│   ├── WeightLoader.h             # Flat binary version 2 loader
│   ├── WeightLoader.cpp           # mmap + pointer arithmetic + validation
│   ├── ConditionProcessor.h       # FiLM conditioning math
│   ├── ConditionProcessor.cpp     # Per-layer γ,β from cond vector
│   ├── RingBuffer.h               # SPSC ring buffer (from common/)
│   └── DoubleBuffer.h             # Atomic-swap cond vector (from common/)
├── clap/
│   ├── MambaClapPlugin.h          # Updated: SSMInferenceCore member
│   ├── MambaClapPlugin.cpp        # Updated: init/process/reset
│   ├── ClapLatency.cpp            # Returns 0 samples
│   └── clap_entry.cpp             # Unchanged
├── ara/
│   ├── MambaAraController.cpp     # 128-dim cond, transport stop analysis
│   └── MambaAraPlaybackRenderer.cpp # Wire cond dimensions
└── common/
    ├── ConditioningGenerator.hpp  # Expand 8→128 dims
    └── Platform.hpp               # Update model path → .bin
```

### Core inference — 8-step pipeline (`SsmBlock::process_sample()`)

```
Step 0: FiLM conditioning
  If conditioning_active_:
    gamma, beta = film_proj @ cond
    u_film[ch] = input * (1 + gamma) + beta
  Else:
    u_film[ch] = input

Step 1: Fused in_proj
  in_proj_out = in_proj_w @ u_film     # single matvec → 419 floats
  Split: z(256), x(64), B(4×4×32), C(4×4×32),
         dd_dt(4), dd_A(4), trap(4), angles(16)

Step 2: A and Δ preprocessing
  foreach head h:
    DT_t[h] = softplus(dd_dt[h] + dt_bias[h] +
                       dt_film_γ·cond + dt_film_β·cond)
    A_t[h]  = clamp(-softplus(dd_A[h]), min=-A_floor)

Step 3: RMSNorm on B and C
  inv_rms = 1/sqrt(mean(B²) + 1e-6)
  B_normed = inv_rms * B_norm_weight + B_bias
  C_normed = inv_rms * C_norm_weight + C_bias

Step 4: RoPE rotation on B and C
  angle_state[h] += angles[h]
  foreach pair in B, C:
    B_rot = B * cos(angle) + B_pair * sin(angle)
    C_rot = C * cos(angle) - C_pair * sin(angle)

Step 5: Multi-head MIMO SSM recurrence (R=4)
  foreach head h, slot r:
    half_a = A_t[h] * DT_t[h] * 0.5
    A_bar = (1 + half_a) / (1 - half_a)
    B_bar_scale = DT_t[h] / (1 - half_a)
    h_state[h,r] = A_bar * h_state[h,r] +
                   B_bar_scale * B_normed[h,r] * x_slot[h,r]
    y_head[h] += C_normed[h,r] · h_state[h,r]

Step 6: MIMO z-gating
  z_gated = z * sigmoid(z)              # SiLU
  y_out[hdim] = z_gated[hdim] * y_head[h]

Step 7: Output projection + D skip + residual
  out = out_proj @ y_out +
        D_proj[h] * x[h,hdim] +
        u_film_residual

Step 8: Periodic state normalization (every 4096 samples)
  If any |h_state[i]| > 1.0:
    scale = 1/max_abs; h_state[i] *= scale
```

### Block processing

`SsmBlock::process_block(samples, blockSize)` processes up to `MAX_CHUNK_SIZE=64` samples. The per-sample loop is unrolled with `#pragma unroll(4)`. All temp arrays stack-allocated with `alignas(64)`.

---

## D. Weight Format (Version 2)

```
File header (64 bytes):
  magic:          0x4D424D41 ("MBMA")
  version:        2
  d_state:        32
  d_model:        128
  d_inner:        256
  nheads:         4
  headdim:        64
  mimo_rank:      4
  n_layers:       6
  cond_dim:       128
  reserved:       [0, 0, 0, 0]
  weight_data_size: <total bytes of all arrays>

Per layer (× n_layers), 13 float32 arrays:
   0. in_proj_w      [419 × 128]
   1. dt_bias        [4]
   2. B_bias         [4 × 4 × 32]
   3. C_bias         [4 × 4 × 32]
   4. B_norm_weight  [32]
   5. C_norm_weight  [32]
   6. mimo_x         [4 × 4 × 64]
   7. mimo_z         [4 × 4 × 64]
   8. mimo_o         [4 × 4 × 64]
   9. D_proj         [4]
  10. out_proj_w     [256 × 128]
  11. film_proj_w    [256 × 128]
  12. dt_film_proj_w [8 × 128]

Total file size: ~3MB (757K params × 4 bytes + overhead)
```

---

## E. Training Pipeline — Multi-Stage Curriculum

### Stage 0: Self-Supervised Pretraining (SSL)

| Aspect | Detail |
|---|---|
| **Data** | FMA Full (100GB) + Slakh2100 Redux + FSD50K + AudioSet subset |
| **Task** | Masked audio modeling — mask 500ms chunks, reconstruct |
| **Loss** | STFT-L1 + Mel-L1 + adversarial (spectrogram discriminator) |
| **CLAP head** | Contrastive audio-text alignment head (distill from LAION CLAP) |
| **Hardware** | MI300X via AMD AI Developer Cloud ($1.99/hr) |
| **Duration** | ~100K steps (2-4h with Triton kernels, 8-12h with reference) |
| **Cost** | ~$8-24 |
| **Output** | Model that understands audio structure + noise characteristics |

### Stage 1: Supervised Effect Training (dedicated per plugin)

Each plugin gets its own dedicated training run. No multi-tasking compromise.

| Plugin | Data (paired) | Loss | Steps | Est. cost |
|---|---|---|---|---|
| **Mamba Saturator** | Clean → tape/tube/digital sat + CLAP captions | STFT-L1 + Mel-L1 + SI-SDR + harmonic distortion loss | 100K | ~$8 |
| **Mamba Compressor** | Clean → compressed (various ratios/attack/release) | Same + RMS envelope matching + crest factor loss | 100K | ~$8 |
| **Mamba Denoiser** | Noisy → clean + CLAP captions | Same + CDPAM perceptual loss | 150K | ~$12 |
| **Mamba Transparent** | Clean → clean (identity) | Same (strict SI-SDR focus) | 50K | ~$4 |
| **Mamba Remixer** | Input → target style pairs + CLAP captions | Style loss + content loss + adversarial | 200K | ~$16 |

**Total Stage 1 cost: ~$48-60**

### Stage 2: Adversarial Fine-Tuning

| Aspect | Detail |
|---|---|
| **Each plugin** | Takes its Stage 1 model + adds spectrogram discriminator |
| **Loss** | Stage 1 losses + adversarial loss |
| **Effect** | Eliminates digital artifacts. Output sounds "real." |
| **Duration** | 20K steps per plugin (~$2 each) |

### Stage 3: RLHF / Preference Optimization

| Aspect | Detail |
|---|---|
| **When** | Post-MVP, once you have user data |
| **What** | A/B comparisons of model outputs, you or beta testers rate |
| **Method** | DPO (simpler than PPO, no reward model needed) |
| **Effect** | Learns subjective quality — warmth, punch, clarity |

### Training environment

```
MI300X, ROCm 6.x, PyTorch 2.7+rocm
Triton Mamba-3 SISO kernels expected to work
Fallback: pure-PyTorch reference forward pass
$100 credits → ~50 hours → all of Stage 0 + Stage 1 within budget
```

---

## F. CLAP Integration — Option C (Full Quality, Hybrid)

### What ships in every plugin bundle

| File | Size | Purpose |
|---|---|---|
| `Mamba<Effect>.clap` | ~2MB | Plugin binary |
| `MambaSuiteARA.so` | ~500KB | ARA extension |
| `libmamba_core.so` | ~200KB | Shared C++ engine |
| `libmamba_clap.so` | ~600MB | CLAP text + audio encoders (shared) |
| `model.bin` | ~3MB | Trained weights |
| `presets.bin` | ~200KB | 200+ precomputed CLAP embeddings |
| **Total per plugin** | **~805MB** (CLAP lib shared across plugins) |

### Three CLAP paths

**Path 1: Precomputed presets (instant, zero main-thread cost)**
```
Plugin init → loads presets.bin → array of {name, embed[32], params[4]}
User selects preset "Worn Vinyl"
  → Copy embed[32] → cond slots 8-39 (atomic swap)
  → Copy params → plugin state (atomic swap)
  → Instant. No main thread time.
```

**Path 2: Live text prompting (5-20ms, main thread only)**
```
User types "make this sound gritty"
  → Main thread: clap_text_encoder → Ort::Session::Run()
  → 32-float embedding → atomic swap → cond slots 8-39
  → User expects brief delay — they pressed Enter
```

**Path 3: Transport-stop audio analysis (10-50ms, main thread only)**
```
User stops transport
  → ARA notifies: transportState = stopped
  → Main thread: read last 5s from ring buffer
  → clap_audio_encoder → 32-float embedding
  → Atomic swap → cond slots 8-39
  → Plugin "listens" between takes, adapts on next play
```

### Standalone audio encoder product

The `clap_audio_encoder.onnx` can be packaged separately:
- "CLAP Audio Analyzer" — drag-drop audio files → export CLAP embeddings
- Low-cost or free tool to drive plugin adoption
- Users can analyze reference tracks and match their vibe
- Potential: batch analysis, reference track matching, automatic preset selection

---

## G. Plugin Integration — Dedicated Per-Task Plugins

### Plugin architecture

```
User installs "Mamba Saturator"
  → /usr/lib/clap/MambaSaturator.clap
  → /usr/lib/clap/MambaSuiteARA.so
  → /usr/lib/libmamba_core.so (shared engine)
  → /usr/lib/libmamba_clap.so (shared CLAP encoders)
  → /usr/share/mamba/MambaSaturator/model.bin
  → /usr/share/mamba/MambaSaturator/presets.bin

User installs "Mamba Compressor"
  → /usr/lib/clap/MambaCompressor.clap
  → /usr/lib/clap/MambaSuiteARA.so (same ARA lib)
  → (shared libmamba_core.so and libmamba_clap.so already installed)
  → /usr/share/mamba/MambaCompressor/model.bin
  → /usr/share/mamba/MambaCompressor/presets.bin
```

### What stays the same across all plugins

- CLAP extensions: params, state, audio-ports (stereo), note-ports, tail, thread-check
- ARA: DocumentController + PlaybackRenderer + ARAGetFactory
- 3 atomics: bypass (stepped), wet/dry, intensity
- Click-free bypass (50ms linear crossfade, per-sample ramp)
- State save/restore: bypass, wetDry, intensity (3 floats)
- Cond vector update mechanism (double-buffered atomic swap)
- Lock-free primitives (SPSCQueue, DoubleBuffer)

### What changes per plugin

| Aspect | Per-plugin variation |
|---|---|
| **model.bin** | Trained for specific task (saturation, compression, denoising, etc.) |
| **UI** | Focused controls matching the task (saturation = drive + tone + mix, compressor = threshold + ratio + attack + release + mix) |
| **CLAP plugin ID** | Unique per plugin (e.g., "com.mamba.saturator", "com.mamba.compressor") |
| **ARA capabilities** | Same ARA extension, but cond vector interpretation differs per model |
| **Presets** | Task-specific presets and CLAP embeddings |

---

## H. Product Roadmap — The "Whoa, Dude" Plugin Suite

### Phase 1 — Foundation Plugin (MVP)

**Plugin: Mamba Transparent**
- Core: Identity / clean boost
- Train on: clean → clean (STFT-L1 + Mel-L1 + SI-SDR)
- Why first: It's the simplest task, proves the engine works, establishes zero-latency transparency
- CLAP: "transparent," "clean," "airy," "natural"
- Sells as: Utility knife — perfect null with dry, zero latency, phase-coherent

### Phase 2 — Core Smart Effects

**Plugin: Mamba Saturator**
- Adaptive saturation that evolves with input
- MIMO R=4: sub(clean) + bass(warmth) + mids(harmonics) + highs(air)
- CLAP: "worn vinyl," "tape warmth," "tube saturation," "gritty," "aggressive"
- Training data: paired clean→saturated at various drive levels + CLAP captions
- Unique: Frequency-aware saturation that doesn't muddy the lows when saturating mids

**Plugin: Mamba Compressor**
- Program-dependent with lookahead-free prediction
- SSM state predicts dynamics — no lookahead latency
- MIMO R=4: sub(gentle) + bass(normal) + mids(active) + highs(gentle)
- CLAP: "70s glue," "EDM pumping," "transparent leveling," "aggressive"
- Training data: paired clean→compressed at various ratios/times + CLAP captions
- Unique: Spectral compression per-band with semantic ducking

**Plugin: Mamba Denoiser**
- Context-aware noise reduction
- Identifies "musical noise" vs. signal
- CLAP: "clean vocal," "vinyl crackle preserve," "tape hiss removal"
- Training data: paired noisy→clean + CLAP captions
- Unique: Doesn't kill reverb tails or vinyl character you want to keep

### Phase 3 — Wild Creative Tools

**Plugin: Mamba Remixer**
- Semantic style transfer
- CLAP: "turn this acoustic guitar into Boards of Canada hazy synth"
- ARA reads session context → style-appropriate processing
- Training data: paired original→stylized + CLAP captions
- Unique: One plugin that can completely transform a track's character

**Plugin: Mamba Reverb / Spatial Sculptor**
- Adaptive spatial processing
- CLAP: "intimate vocal club," "epic orchestral hall"
- Mamba scans input scene → applies adaptive early reflections + tail modulation
- Training data: dry→various spatial configurations + CLAP captions
- Unique: Reverb that has character — saturation, compression, modulation in the wet path

**Plugin: Mamba Generator**
- Generative layerer / infinite variations
- Mamba audio generation strengths
- CLAP: "add gritty riser fitting the energy," "extend break with evolving textures"
- Training data: generative audio modeling (masked prediction from context)
- Unique: Generates matching layers that blend with existing audio

### Phase 4 — Advanced / Niche

**Plugin: Mamba Session Listener**
- Multi-track ARA-aware processing
- Sees multiple ARA-connected tracks simultaneously
- Auto-EQ clashes, dynamic balance, master bus processing respecting arrangement
- CLAP: "make everything gel like reference track X"
- Unique: Mix decisions informed by full session context

**Plugin: Mamba Stem Separator**
- Mamba-based source separation + per-stem enhancement
- Separate: vocals, drums, bass, other
- Per-stem: denoise vocals, saturate drums, compress bass to kick
- Training data: stem-separated datasets (Slakh2100, MUSDB18)

**Plugin: Mamba Emulations**
- Train on real hardware chains (Neve → LA-2A → tape)
- Model learns the transfer function
- CLAP: controllability ("more saturation" = more nonlinearity)
- Training data: swept sine through hardware chain + music samples
- Unique: Hardware emulation with semantic control, not just IR matching

**Plugin: Mamba Playground**
- Training / fine-tuning sandbox
- Record examples + text descriptions → fine-tune LoRA adapter
- Ships with tools, not just models
- Target: Power users and pro studios who want custom "personalities"
- Training data: user-provided before/after pairs

### Phase 5 — Live Performance

**Plugin: Mamba Live**
- Low-latency inference at its absolute limit
- ARA reads genre/mood shifts → morphs processing in real-time
- MIDI/CC mapping for on-the-fly spectrum tweaking
- CLAP: vibe shifting based on playing style
- Unique: Clean jazz piano → saturated lo-fi when you switch to beats, seamlessly

---

## I. MIMO Rank Analysis

| Rank | Slots/head | Total streams/layer | What each slot learns | SSM MACs | Total MACs (6 layers) | CPU µs/sample | Core % @44.1kHz |
|---|---|---|---|---|---|---|---|
| **1** (SISO) | 1 | 4 | One feature per head | 8,192 | ~786K | ~2.3 µs | ~10% |
| **2** | 2 | 8 | Transient + tonal pair | 16,384 | ~850K | ~2.5 µs | ~11% |
| **3** | 3 | 12 | Transient + tonal + noise | 24,576 | ~904K | ~2.7 µs | ~12% |
| **4** (MVP) | 4 | 16 | Sub + bass + mids + highs | 32,768 | ~955K | ~2.9 µs | **~13%** |
| **8** (research) | 8 | 32 | 8 frequency bands | 65,536 | ~1.2M | ~3.6 µs | ~16% |

### Rank collapse risk

| Rank | Risk of collapse | Mitigation | Verdict |
|---|---|---|---|
| 2 | Very low | Emergent specialization (proven) | Standard |
| 3 | Low | Slightly harder to differentiate 3 slots | Worth it |
| **4** | **Low-Medium** | **Use band-weighted loss to encourage frequency specialization** | **MVP** |
| 8 | High | Needs sparse regularization + band-split frontend | Roadmap |

**Recommendation:** R=4 for MVP with band-weighted loss (weight different frequency regions differently in the loss function). R=8 is genuine research territory — put it in the roadmap.

---

## J. Migration Phases — Detailed Timeline

### Phase 0: Configuration & Types (2-3 days)
Files created: `MambaConfig.h`, `MambaConfig.cpp`
CMake: remove ONNX Runtime dep, add new core lib targets

### Phase 1: Python Reference Forward (3-5 days) — DO THIS FIRST
File created: `reference_forward.py`
Bit-for-bit PyTorch Mamba-3 with R=4, full 8-step pipeline. Gold standard for C++ tests.

### Phase 2: Weight Export Pipeline (1-2 days)
Files created: `scripts/export_mamba_weights.py`, `scripts/validate_export.py`

### Phase 3: C++ SsmBlock Implementation (5-7 days)
Files created: `SsmBlock.h/.cpp`, `WeightLoader.h/.cpp`
Critical: bit-exact match with Python reference

### Phase 4: Multi-Layer Engine (2-3 days)
Files created: `SsmInferenceCore.h/.cpp`, `ConditionProcessor.h/.cpp`

### Phase 5: CLAP Plugin Adaptation (2-3 days)
Files modified: `MambaClapPlugin.h/.cpp`, `ClapLatency.cpp`, `Platform.hpp`, `ConditioningGenerator.hpp`

### Phase 6: ARA Bridge Update (1 day)
Files modified: `MambaAraController.cpp`, `MambaAraPlaybackRenderer.cpp`

### Phase 7: CLAP Encoder Integration (2-3 days)
Load CLAP text + audio encoders, preset system, transport-stop analysis

### Phase 8: Training Pipeline (5-7 days)
File created: `train_clara.py`

### Phase 9: Testing & Verification (3-5 days)
Bit-exact Python↔C++, state continuity, stability, extreme values, real-time safety, benchmark

### Phase 10: Training Runs (ongoing)
Stage 0 (SSL) → Stage 1 (per-plugin supervised) → Stage 2 (adversarial) → Stage 3 (RLHF later)

**Total dev time: ~25-30 working days** before training begins.

---

## K. Files Changed — Complete Manifest

### New files
```
src/core/MambaConfig.h
src/core/MambaConfig.cpp
src/core/SsmBlock.h
src/core/SsmBlock.cpp
src/core/SsmInferenceCore.h
src/core/SsmInferenceCore.cpp
src/core/WeightLoader.h
src/core/WeightLoader.cpp
src/core/ConditionProcessor.h
src/core/ConditionProcessor.cpp
reference_forward.py
scripts/export_mamba_weights.py
scripts/validate_export.py
train_clara.py
```

### Modified files
```
src/clap/MambaClapPlugin.h        (replace engine type)
src/clap/MambaClapPlugin.cpp      (init/process/reset for new engine)
src/clap/ClapLatency.cpp          (0 samples)
src/core/Platform.hpp             (.bin path resolution)
src/common/ConditioningGenerator.hpp (8→128 dims)
src/ara/MambaAraController.cpp    (128-dim cond, transport stop)
src/ara/MambaAraPlaybackRenderer.cpp (wire cond)
CMakeLists.txt                    (remove ONNX, add new libs)
.github/workflows/ci.yml          (update for new build/test)
```

### Deleted files
```
src/core/MambaInferenceEngine.h   (ONNX wrapper)
src/core/MambaInferenceEngine.cpp (ONNX wrapper)
train_clara_parallel.py           (Mamba-1 model)
train_clara_export.py             (ONNX export)
export_mamba_onnx.py              (incompatible)
train_clara_now.py                (was already deprecated)
models/*.onnx                     (no longer needed)
libonnxruntime.so.1.17.1          (no longer shipped)
```

### Unchanged files
```
All CLAP extension files (clap_entry.cpp, audio_ports, params, state, tail, thread_check)
All ARA factory/entry stubs (ARAGetFactory, MambaAraFactory, MambaAraDocumentController)
Lock-free primitives (SPSCQueue, DoubleBuffer)
SIMD helpers (Platform.hpp SIMD paths)
Arena allocator
Build system structure
Coding conventions (Wall, Werror, fno-rtti, visibility=hidden)
CI/CD structure
AGENTS.md (updated with reference to this file)
ADR-*.md (architectural decisions remain valid)
```

---

## L. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Python↔C++ divergence | High | Critical — silent quality loss | Write reference_forward.py BEFORE C++. Bit-exact CI check. |
| Triton kernels fail on ROCm | Medium | High — slower training | Pure-PyTorch fallback. Test on MI300X early. |
| MIMO R=4 rank collapse | Low-Medium | Reduced MIMO benefit | Band-weighted loss. Monitor mimo_x singular values. |
| CPU real-time > 13% | Low | Degraded UX | Can reduce to 4 layers. Profile early. |
| CLAP encoder ~20ms latency | Low | Brief text prompt delay | Main thread only. User expects it. |
| Training data insufficient | Medium | Model quality | FMA + Slakh2100 + FSD50K + AudioSet = comprehensive. |
| MI300X credits run out | Low | Can't finish training | $100 = ~50h. All Stage 0+1 fits. Apply for more. |
| Memory bandwidth bottleneck | Low | Higher latency than computed | Profile L1/L2 cache behavior. Optimize state layout for cache lines. |

---

## M. Plugin Directory Structure (Install Layout)

```
/usr/lib/
├── clap/
│   ├── MambaSaturator.clap
│   ├── MambaCompressor.clap
│   ├── MambaDenoiser.clap
│   ├── MambaTransparent.clap
│   └── ... (one per model)
├── libmamba_core.so       (shared C++ engine)
├── libmamba_clap.so       (shared CLAP text + audio encoders)
└── libmamba_ara.so        (shared ARA extension — same lib, every plugin references it)

/usr/share/mamba/
├── MambaSaturator/
│   ├── model.bin
│   └── presets.bin
├── MambaCompressor/
│   ├── model.bin
│   └── presets.bin
├── MambaDenoiser/
│   ├── model.bin
│   └── presets.bin
├── MambaTransparent/
│   ├── model.bin
│   └── presets.bin
└── ... (one per plugin)
```

---

## N. Decision Log

| Date | Decision | Rationale | Status |
|---|---|---|---|
| 2026-06-15 | Remove ONNX Runtime | Custom C++ engine supports full Mamba-3 features. No opset limits. No dependency. | CONFIRMED |
| 2026-06-15 | Multi-head MIMO R=4 | Sub/bass/mids/highs per head. +7% compute for massive cross-frequency benefit. | CONFIRMED |
| 2026-06-15 | d_model=128, 6 layers | ~750K params. 13% core @ 44.1kHz. Real-time safe on modern CPUs. | CONFIRMED |
| 2026-06-15 | Dedicated models per plugin | One model per task. No compromise. Each plugin has focused UI. | CONFIRMED |
| 2026-06-15 | CLAP Option C (full quality) | Full text + audio encoders shipped. Precomputed presets. Transport-stop analysis. | CONFIRMED |
| 2026-06-15 | CLAP analysis on transport stop | Plugin "listens" between takes. 10-50ms on main thread. | CONFIRMED |
| 2026-06-15 | Flat binary weight format | Version 2, 13 arrays per layer, mmap'd at init. | CONFIRMED |
| 2026-06-15 | Python reference FIRST | Bit-for-bit gold standard before writing any C++. Prevents EmT-style divergence. | CONFIRMED |
| 2026-06-15 | Multi-stage training (SSL → Supervised → Adversarial → RLHF) | Each stage unlocks more quality. RLHF later. | CONFIRMED |
| 2026-06-15 | Train on MI300X (AMD AI Developer Cloud) | $1.99/hr, $100 credits = ~50 hours. All Stage 0+1 fits in budget. | CONFIRMED |
| 2026-06-15 | Ship all presets, no download-on-demand | Instant loading. User experience matters. | CONFIRMED |
| 2026-06-15 | Standalone CLAP Audio Analyzer | Separate product. Drives adoption. Can be free/low-cost. | CONFIRMED |
