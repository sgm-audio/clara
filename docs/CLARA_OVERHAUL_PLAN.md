# CLARA Mamba-3 Architecture & Training Overhaul Plan

**Last updated:** 2026-06-22  
**Status:** Plan — awaiting user direction  
**Source references:** `mamba3/model.py`, `mamba3/train.py`, `mamba3 BAZZITE/train.py`, `MAMBA3_TRAINING_GUIDE.md`, `MAMBA3_OPTIMIZED_TRAINING.md`, `MambaDAW/`, `Mamba Based/`

---

## Table of Contents

1. [Architectural Decisions](#1-architectural-decisions)
   - 1.1 [Spectral Features — R=* Options](#11-spectral-features--r-options)
   - 1.2 [Complex Poles vs RoPE](#12-complex-poles-vs-rope)
   - 1.3 [Signal Domain: Spectral vs Time](#13-signal-domain-spectral-vs-time)
   - 1.4 [Model Size: d_model, nheads, n_layers](#14-model-size-d_model-nheads-n_layers)
   - 1.5 [Chunk Size & clip_seconds](#15-chunk-size--clip_seconds)
   - 1.6 [dt Initialization & Clamping](#16-dt-initialization--clamping)
2. [Phase 1: Training Infrastructure](#2-phase-1-training-infrastructure)
3. [Phase 2: Python Model Architecture](#3-phase-2-python-model-architecture)
4. [Phase 3: C++ Engine Rewrite](#4-phase-3-c-engine-rewrite)
5. [Phase 4: Data Pipeline](#5-phase-4-data-pipeline)
6. [Phase 5: Loss Functions](#6-phase-5-loss-functions)
7. [Phase 6: Phase Strategy & Training Loop](#7-phase-6-phase-strategy--training-loop)
8. [Phase 7: Export & Validation](#8-phase-7-export--validation)
9. [Dependency Graph & Timeline](#9-dependency-graph--timeline)
10. [Risk Register](#10-risk-register)
11. [File Manifest](#11-file-manifest)

---

## 1. Architectural Decisions

### 1.1 Spectral Features — R=* Options

**What R means:** R is the number of structured spectral features extracted per STFT frame and fed into the MIMO (Multiple-Input Multiple-Output) SSM layer. Each head has R MIMO slots — the B and C projection matrices have shape `[nheads, R, d_state]` and the input to each head's recurrence is R-dimensional.

**Current state (CLARA):** R = mimo_rank = 4. Features are the time-domain embedding channels directly — no spectral structure, no frequency selectivity. The 4 MIMO slots learn whatever the gradient pushes them toward, with no inductive bias.

**Reference (mamba3):** R = 8 structured spectral features:
| Slot | Feature | Hz range | Description |
|------|---------|----------|-------------|
| 0 | sub_bass | 20-80 Hz | Deep low end |
| 1 | bass | 80-250 Hz | Bass fundamentals |
| 2 | low_mid | 250-800 Hz | Low harmonics, body |
| 3 | mid | 800-2500 Hz | Vocals, instrument fundamentals |
| 4 | high_mid | 2500-8000 Hz | Presence, attack |
| 5 | air | 8000-20000 Hz | Brilliance, air |
| 6 | transient | (computed) | Onset density via 1D conv on envelope |
| 7 | centroid | (computed) | Spectral balance (log-normalized) |

---

#### Option R-4: Minimal spectral (mimo_rank=4, keep current C++ engine sizes)

**Features:**
| Slot | Feature | Method |
|------|---------|--------|
| 0 | low band energy | 20-500 Hz learned soft band |
| 1 | mid band energy | 500-4000 Hz learned soft band |
| 2 | high band energy | 4000-20000 Hz learned soft band |
| 3 | transient+centroid | combined onset density + spectral centroid |

**Pro:** Fits existing C++ engine's `MAX_MIMO_RANK=4`. Zero array size changes in C++. Minimal weight format change.
**Con:** 3 frequency bands is coarse — can't learn sub-bass vs bass separation. Spectral centroid + transient in one slot forces the head to unentangle them. May limit saturation harmonic modeling.

**Dim change:** `B_bias`/`C_bias` stay [4, 4, 32] = 512 each. No C++ array size changes.
**Total params impact:** ~0 (same as current)

---

#### Option R-6: Extended musical bands (requires C++ MAX_MIMO_RANK=6)

**Features:**
| Slot | Feature | Hz range |
|------|---------|----------|
| 0 | sub_bass | 20-80 Hz |
| 1 | bass | 80-250 Hz |
| 2 | low_mid | 250-800 Hz |
| 3 | mid | 800-2500 Hz |
| 4 | high_mid+air | 2500-20000 Hz |
| 5 | transient+centroid | computed |

**Pro:** Separates sub-bass from bass (critical for saturation/compression). Follows musical frequency division. More expressive than R=4 but fewer slots than R=8.
**Con:** Still conflates high_mid+air (ok — they behave similarly under saturation). Still conflates transient+centroid. Requires C++ engine change (MAX_MIMO_RANK=6).

**Dim change:** `B_bias`/`C_bias` expand from [4,4,32] to [4,6,32] = 768 each (+256 each).
**C++ impact:** All `MAX_MIMO_RANK` arrays grow by 1.5×. Stack state stays roughly same.

---

#### Option R-8: Full mamba3 match (recommended — proven architecture)

**Features:** All 8 slots as listed above. Exact match to mamba3's `R8FeatureExtractor`.

**Pro:** Proven in mamba3's 100K-step training run. 6 frequency bands give musically meaningful frequency resolution. Separate transient and centroid lets the model independently track onsets and spectral balance. Each plugin (Saturator, Compressor, Denoiser) can use different subsets without retraining the extractor.
**Con:** Largest C++ engine change — all `MAX_MIMO_RANK` arrays grow 2×. More weights (B_bias alone goes from 512 to 1024 per layer).

**Dim change:** `B_bias`/`C_bias` expand from [4,4,32] to [4,8,32] = 1024 each (+512 each).
**Param count impact:** +6,144 per layer × 6 layers = +36,864 params (total: ~1.7M vs 1.66M).
**C++ impact:** `MAX_MIMO_RANK=8`. All `W_B_BIAS_ELEMS`, `W_C_BIAS_ELEMS`, `W_MIMO_X_ELEMS`, `W_MIMO_Z_ELEMS`, `W_MIMO_O_ELEMS` grow 2×.

---

#### Option R-10: Future-proof (band + learnable)

Everything in R=8 plus:
| Slot | Feature |
|------|---------|
| 8 | learnable feature A (data-driven) |
| 9 | learnable feature B (data-driven) |

**Pro:** Extra capacity for domain-specific features (e.g., stereo width, harmonic density for saturation).
**Con:** Untested. No reference implementation. C++ `MAX_MIMO_RANK=10` is a large change.

---

### 1.2 Complex Poles vs RoPE

**Current (CLARA):** Real-valued A with RoPE rotation of B and C vectors.
```
A = -softplus(dd_A)          # real, negative = stable decay
B_normed → RoPE(angles)      # frequency selectivity via rotation
C_normed → RoPE(angles)      # same rotation
```

**Proposed (mamba3):** Complex-valued poles with trapezoidal discretization.
```
A = -exp(log_a_real) + j * A_imag    # complex = decay + frequency
A_d = (1 + dt*A/2) / (1 - dt*A/2)    # trapezoidal → complex
B_d = dt / (1 - dt*A/2)              # complex scaling
h[t] = A_d * h[t-1] + B_d * update   # complex recurrence
y = C_readout(cat(real(h), imag(h))) # real readout
```

**Why complex poles are better for audio:**
| Aspect | Real + RoPE | Complex poles |
|--------|-------------|---------------|
| Frequency selectivity | Indirect (RoPE rotates B/C, not A) | Direct (A_imag = oscillator frequency) |
| Bandpass behavior | No — A controls only decay | Yes — each head = resonant bandpass |
| Transient response | Exponential decay only | Damped sinusoid (correct for audio) |
| Parameter count | angles=16 floats per head | A_imag=1 float per head |
| C++ complexity | Moderate (trig on every sample) | Higher (complex division per sample) |

**Param impact:** Removing RoPE angles (nheads × num_rope_angles = 64) reduces in_proj_dim from 1612 to 1552. Adding log_a_real and A_imag adds 8 floats per layer. Net: in_proj_dim shrinks by 56.

**C++ impact:** `SsmBlock` must handle `std::complex<float>` for state. State memory doubles (real + imag).

---

### 1.3 Signal Domain: Spectral vs Time

**Current (CLARA):** Time domain. Audio sample → scalar → x[d_model][0] → Mamba → x[d_model][0] → audio sample.
```
audio[T] → embed → Mamba stack (T steps, time domain) → output[T]
```

**Proposed (mamba3):** Spectral domain. Audio → STFT → features → Mamba (frame-wise) → decode → ISTFT.
```
audio[L] → STFT → mag_sq[T,F]
    → R* features[T,R] → input_proj[T,d_model]
    → Mamba stack (T frames, spectral domain)
    → decoder[T,2*F] → complex STFT → ISTFT → audio[L]
```

**Why spectral domain:**
| Aspect | Time domain | Spectral domain |
|--------|------------|----------------|
| Sample rate | 44,100 steps/second | ~86 frames/second (hop=512) |
| SSM steps per second of audio | 44,100 | 86 (512× fewer) |
| Harmonic structure | Must be learned from scratch | Built into STFT representation |
| Phase handling | Implicit | Explicit (complex STFT) |
| Real-time feasibility | Hard (0-latency constraint) | Easier (lookahead via FFT) |
| Frequency resolution | None (scalar per step) | Full (n_fft/2+1 bins) |

**Key trade-off:** Time domain is 0-latency (no STFT frame overlap). Spectral domain introduces latency equal to `n_fft` samples (~46ms at 2048 FFT). For a mixing plugin (not live monitoring) this is acceptable.

**Consequence for C++ engine:**
- Must add STFT (`torch.stft` equivalent) and ISTFT (`torch.istft` equivalent)
- Must add `R8FeatureExtractor` (learnable band filterbank)
- Must add decoder (project d_model → 2*F_bins → complex STFT)
- Must add overlap-add synthesis buffer
- Latency becomes `n_fft` samples (~46ms at 44.1kHz with n_fft=2048)

---

### 1.4 Model Size: d_model, nheads, n_layers

**Current (CLARA):** d_model=128, nheads=4, headdim=64, n_layers=6 → 1,662,384 params.

**mamba3 reference:** d_model=256, nheads=16, headdim=16, n_layers=6 → ~3.9M params (full model), ~2.1M (Mamba core only).

| Dimension | CLARA (current) | CLARA (proposed) | mamba3 ref | Rationale |
|-----------|----------------|-------------------|------------|-----------|
| d_model | 128 | **128** | 256 | Keep 128 for C++ compatibility. Can scale later. |
| nheads | 4 | **4** | 16 | 4 heads are enough for 4-band R-4. If R=8, consider 8 heads. |
| headdim | 64 | **64** | 16 | headdim = d_inner/nheads = 256/4 = 64. Stays fixed. |
| n_layers | 6 | **6** | 6 | Depth is adequate. |
| d_inner | 256 | **256** | — | 2× d_model. Stay at 256. |

**Recommendation:** Keep current dimensions for Phase 2-3. Scale d_model to 256 and nheads to 8 in a future iteration after the first successful training run.

---

### 1.5 Chunk Size & clip_seconds

**Current (CLARA):** chunk_size=64 (SSM sub-block), clip_seconds=0.5 (22,050 samples).

**mamba3 reference:** chunk_size=128, clip_seconds=4.0 (176,400 samples).

| Parameter | CLARA (current) | CLARA (proposed) | Why |
|-----------|----------------|-------------------|-----|
| chunk_size | 64 | **64** | Match C++ engine's `kSubBlockSize`. Keep for now. |
| clip_seconds | 0.5 | **4.0** | 176K samples gives enough temporal context. With hop=512, that's ~345 frames — 3 orders of magnitude fewer SSM steps than time-domain. |

**Impact of clip_seconds=4.0:**
- Dataset chunks are 176,400 samples long
- With hop=512: T = 1 + 176400/512 = **345 frames** per clip
- Each frame has R spectral features → Mamba stack processes 345 steps
- Compare to time-domain: 176,400 SSM steps — spectral domain is **511× fewer steps**
- This is why spectral domain makes 4s clips practical

---

### 1.6 dt Initialization & Clamping

**Current (CLARA):** `dt_bias = zeros(nheads)` → `softplus(0) ≈ 0.693`. No clamping.

**mamba3 reference:** `dt_bias = full((nheads,), -4.0)` → `softplus(-4) ≈ 0.018`. Clamped to `[0.001, 0.1]`.

| | CLARA (current) | mamba3 | Proposed |
|---|---|---|---|
| dt_bias init | 0.0 | -4.0 | **-4.0** |
| Softplus output | ~0.693 | ~0.018 | **~0.018** |
| dt range | [0, ∞) | [0.001, 0.1] | **[0.001, 0.1]** |

**Rationale:** Slow initial Δ prevents pole explosion in early training. Complex poles with large dt produce A_d with |A_d| > 1 (unstable oscillation). Clamping prevents this.

---

## 2. Phase 1: Training Infrastructure

**Zero model changes. Deliverable: training script runs on 860M with current C++ engine.**

### 2.1 ROCm Environment — `train_mamba3.py`

```python
import os
# Order matters: set before torch import
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.3")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1103")
os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF",
    "garbage_collection_threshold:0.8,max_split_size_mb:256")
os.environ.setdefault("MIOPEN_FIND_MODE", "5")
```

These must be at the very top of the file, before any `import torch`.

### 2.2 CLI Flags — Complete Redesign

```python
parser = argparse.ArgumentParser(description="CLARA Mamba-3 Training Pipeline")

# === Data + I/O ===
parser.add_argument('--data', required=True,
    help='Path to MUSDB18-HQ training root (stem directory structure)')
parser.add_argument('--save', default='./checkpoints',
    help='Checkpoint directory')
parser.add_argument('--resume',
    help='Resume from checkpoint path')

# === Training Schedule ===
parser.add_argument('--steps', type=int, default=100000)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--lr', type=float, default=3e-4,
    help='Initial learning rate (mamba3 ref: 3e-4)')
parser.add_argument('--weight-decay', type=float, default=1e-4,
    help='Weight decay for non-bias/norm params (mamba3 ref: 1e-4)')
parser.add_argument('--clip-grad-norm', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=2000)
parser.add_argument('--amp', action='store_true',
    help='Enable bf16 AMP (saves ~50% VRAM, opt-in for stability)')

# === DataLoader ===
parser.add_argument('--num-workers', type=int, default=0,
    help='DataLoader workers. 0 safest for iGPU shared RAM.')
parser.add_argument('--cache-limit', type=int, default=16,
    help='In-memory cache limit in tracks')

# === Validation + Checkpointing ===
parser.add_argument('--val-split', type=float, default=0.1)
parser.add_argument('--val-every', type=int, default=2000)
parser.add_argument('--ckpt-every', type=int, default=1000)
parser.add_argument('--log-every', type=int, default=50)
parser.add_argument('--early-stop-patience', type=int, default=10)
parser.add_argument('--device', default='cuda')

# === Model Architecture (future — for experimentation) ===
parser.add_argument('--d-model', type=int, default=128,
    help='Model dimension (128=C++ compatible, 256=experimental)')
parser.add_argument('--d-state', type=int, default=32)
parser.add_argument('--n-layers', type=int, default=6)

# === Export ===
parser.add_argument('--export',
    help='Export checkpoint to model.bin')
parser.add_argument('--output', default='models/model_trained.bin')
```

**Key changes from current:**
- `--data` → required (no synthetic fallback)
- `--lr` → 3e-4 (was 1e-3)
- `--weight-decay` → 1e-4 (was 1e-5)
- `--amp` → new flag (opt-in bf16)
- `--grad-accum` → removed (not needed with spectal domain + AMP)
- `--clip-seconds` → removed (hardcoded to 4.0 in dataset)
- `--chunk-size` → removed (hardcoded to 64)
- `--scan` → removed (always parallel for training, sequential for validation)
- `--precomputed` / `--cache` → removed (replaced by StemDataset with on-the-fly STFT)
- `--d-model`, `--d-state`, `--n-layers` → added for experimentation

### 2.3 AMP Implementation

```python
def train(args):
    amp = args.amp and device.type == 'cuda'
    autocast_dtype = torch.bfloat16 if amp else torch.float32
    
    # ...
    
    while step < n_steps:
        for batch in dataloader:
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=amp):
                # Forward pass — includes STFT, R* features, Mamba, decoder
                out = model(target_wave, target_crepe, other_embs=None)
                loss_dict = task_loss(out['output_stft'], target_wave, model)
            
            # SSM recurrence is inside model.forward()
            # Mamba3MIMOLayer disables autocast internally for fp32 complex math
```

**Inside Mamba3MIMOLayer.forward():**
```python
with torch.autocast(device_type=x.device.type, enabled=False):
    # fp32 for complex trapezoidal discretization + SSM scan
    dt_fp32 = dt.float()
    x_fp32 = x.float()
    A_d, B_d = self._discretize_trapezoidal(
        self.log_a_real.float(), self.a_imag.float(), dt_fp32
    )
    out = self._ssm_scan_parallel(x_fp32, A_d, B_d)
out = out.to(x.dtype)  # back to bf16 for rest of computation
```

This pattern ensures complex pole arithmetic (precision-sensitive) stays in fp32 while everything else (STFT, projections, decoder) benefits from bf16 speed.

### 2.4 Gradient Checkpointing

```python
for layer in model.mamba_layers:
    layer.gradient_checkpointing = True

# In Mamba3MIMOLayer.forward():
def forward(self, x, x_r8, cond=None):
    if self.gradient_checkpointing and self.training:
        return torch.utils.checkpoint.checkpoint(
            self._forward_impl, x, x_r8, cond,
            use_reentrant=False
        )
    return self._forward_impl(x, x_r8, cond)
```

### 2.5 Optimizer

```python
decay_params = [p for n, p in model.named_parameters()
                if 'bias' not in n and 'norm' not in n and p.requires_grad]
nodecay_params = [p for n, p in model.named_parameters()
                  if ('bias' in n or 'norm' in n) and p.requires_grad]
optimizer = torch.optim.AdamW([
    {'params': decay_params,   'weight_decay': args.weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0},
], lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

def lr_lambda(step):
    warmup = args.warmup
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, args.steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

Key: `zero_grad(set_to_none=True)` for 10-15% VRAM savings.

### 2.6 NaN/Inf Protection

```python
loss = loss_dict['total']
if torch.isnan(loss) or torch.isinf(loss):
    print(f"[step {step}] NaN/Inf loss — skipping batch", flush=True)
    optimizer.zero_grad(set_to_none=True)
    continue
```

### 2.7 Signal Handler (unchanged from current)

First Ctrl+C → save interrupt checkpoint. Second Ctrl+C → hard exit.

### 2.8 Decoupled Validation & Checkpointing

```python
# Independent checks — NOT nested
if step > 0 and step % val_every == 0:
    val_loss = validate(current_phase)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(step, loss.item(), is_best=True)
        plateau_count = 0
    else:
        plateau_count += 1
        if plateau_count >= early_stop_patience:
            print(f"  Early stopping — no improvement for {early_stop_patience} val checks")
            save_checkpoint(step, loss.item(), name="final_earlystop.pt")
            return model

if step > 0 and step % ckpt_every == 0:
    save_checkpoint(step, loss.item())
```

---

## 3. Phase 2: Python Model Architecture

### 3.1 Complex Pole SSM — Mamba3MIMOLayer

See §1.2 for the math. Implementation details:

**New parameters per layer:**
```python
# Complex poles (per-head)
self.log_a_real = nn.Parameter(torch.zeros(cfg.n_heads))  # → -exp() = decay
self.a_imag = nn.Parameter(torch.linspace(0, math.pi, cfg.n_heads))  # frequency in radians

# Δ initialization (slow start)
self.dt_bias = nn.Parameter(torch.full((cfg.n_heads,), -4.0))

# Δ clamping
self.dt_min = cfg.dt_min
self.dt_max = cfg.dt_max
```

**Removed parameters:**
```python
# REMOVED: dt_bias init was zeros (now -4.0)
# REMOVED: num_rope_angles, angle_state
# REMOVED: apply_rope() method
# REPLACED B_bias/C_bias init: was zeros, now zeros (no RoPE needed)
```

**New discretize method:**
```python
def _discretize_trapezoidal(self, a_real, a_imag, dt):
    """
    Complex trapezoidal discretization.
    a_real: (H,) — log-space decay (will be negated)
    a_imag: (H,) — oscillator frequencies
    dt: (B, T, H) — per-step step sizes
    
    Returns:
      A_d: (B, T, H) complex — state transition
      B_d_scale: (B, T, H) complex — input scaling
    """
    a_cont = torch.complex(-torch.exp(a_real), a_imag)  # (H,) complex
    
    half_dt_a = torch.complex(
        dt * (-torch.exp(a_real)) / 2,
        dt * a_imag / 2
    )  # (B, T, H) complex
    
    A_d = (1 + half_dt_a) / (1 - half_dt_a)       # |A_d| ≤ 1
    B_d_scale = dt / (1 - half_dt_a)               # complex scaling
    return A_d, B_d_scale
```

**New in_proj slicing (no RoPE angles):**
```python
# in_proj_dim = 1552 (was 1612)
proj = self.in_proj(x)  # [B, T, 1552]
idx = 0
z      = proj[..., :idx + cfg.d_inner];                 idx += cfg.d_inner       # 256
x_h    = proj[..., idx:idx + cfg.nheads * cfg.headdim]; idx += cfg.nheads * cfg.headdim  # 256
B_raw  = proj[..., idx:idx + cfg.nheads * cfg.mimo_rank * cfg.d_state]; idx += ...  # 512
C_raw  = proj[..., idx:idx + cfg.nheads * cfg.mimo_rank * cfg.d_state]; idx += ...  # 512
dd_dt  = proj[..., idx:idx + cfg.nheads];                idx += cfg.nheads         # 4
dd_A_real = proj[..., idx:idx + cfg.nheads];             idx += cfg.nheads         # 4
dd_A_imag = proj[..., idx:idx + cfg.nheads];             idx += cfg.nheads         # 4
trap   = proj[..., idx:idx + cfg.nheads];                idx += cfg.nheads         # 4
```

**New SSM scan (complex parallel):**
```python
def _ssm_scan_parallel(self, x, A_d, B_d):
    """
    Parallel associative scan with complex state.
    x: (B, T, R) — MIMO input features
    A_d: (B, T, H) complex
    B_d: (B, T, H) complex
    
    Returns: (B, T, H, head_dim) — output sequence
    """
    B_batch, T, R = x.shape
    
    # MIMO B: R → N complex per head
    bc = self.B_proj(x)  # (B, T, N*2)
    B_real, B_imag = bc.chunk(2, dim=-1)
    B_vec = torch.complex(B_real, B_imag)  # (B, T, N) complex
    
    # u_t = B_d_t * B_vec_t
    a = A_d.unsqueeze(-1)              # (B, T, H, 1)
    b_d = B_d.unsqueeze(-1)            # (B, T, H, 1)
    b_vec = B_vec.unsqueeze(2)         # (B, T, 1, N)
    u = b_d * b_vec                    # (B, T, H, N) complex
    
    # Parallel associative scan (O(log T) steps)
    h = self._parallel_associative_scan(a, u)  # (B, T, H, N) complex
    
    # Readout: cat real+imag → C projection
    h_cat = torch.cat([h.real, h.imag], dim=-1)  # (B, T, H, N*2)
    out = self.C_proj(h_cat)                      # (B, T, H, head_dim)
    return out
```

**Sequential scan (for C++ bit-exact verification):**
```python
def _ssm_scan_sequential(self, x, A_d, B_d):
    """O(T) sequential scan matching C++ engine step-by-step."""
    B, T, R = x.shape
    H, N = self.H, self.N
    
    bc = self.B_proj(x)
    B_real, B_imag = bc.chunk(2, dim=-1)
    B_vec = torch.complex(B_real, B_imag)
    
    h = torch.zeros(B, H, N, dtype=torch.complex64, device=x.device)
    outputs = []
    for t in range(T):
        A_d_t = A_d[:, t, :]  # (B, H)
        B_d_t = B_d[:, t, :]  # (B, H)
        B_input = B_d_t.unsqueeze(-1) * B_vec[:, t, :].unsqueeze(1)  # (B, H, N)
        h = A_d_t.unsqueeze(-1) * h + B_input
        h_cat = torch.cat([h.real, h.imag], dim=-1)
        y_t = self.C_proj(h_cat)
        outputs.append(y_t)
    return torch.stack(outputs, dim=1)
```

### 3.2 R* Spectral Feature Extractor

For each R option (§1.1), the extractor architecture changes. This assumes R=8 (recommended):

```python
class R8FeatureExtractor(nn.Module):
    """
    Extracts R=8 structured spectral features per frame:
    [sub_bass, bass, low_mid, mid, high_mid, air, transient, centroid]
    
    Band boundaries are learnable (initialized from musical Hz ranges).
    Learnable = soft triangular filterbank with trainable center freqs.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        sr = cfg.sample_rate
        n_fft = cfg.n_fft  # 2048
        
        # Frequency axis
        self.register_buffer("freqs", torch.linspace(0, sr/2, n_fft//2 + 1))
        
        # Initialize band centers (Hz) at geometric mean of boundaries
        # For R=6 band pairs (sub_ass, bass, low_mid, mid, high_mid, air)
        band_pairs = [(20,80), (80,250), (250,800), (800,2500), (2500,8000), (8000,20000)]
        centers_hz = [math.sqrt(lo*hi) for (lo,hi) in band_pairs]
        
        # Store in log-Hz for unconstrained optimization
        self.log_centers = nn.Parameter(torch.log(torch.tensor(centers_hz)))
        self.log_bandwidths = nn.Parameter(torch.full((6,), math.log(0.5)))  # ~1/3 octave
        
        # Transient detector: 1D conv on waveform envelope
        self.transient_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        
        # Normalization
        self.log_max_freq = math.log(sr / 2)
        
        # Project R=8 → d_model
        self.input_proj = nn.Linear(8, cfg.d_model)
    
    def _soft_band_energy(self, mag_sq):
        """Soft triangular filterbank. mag_sq: (B, T, F) → (B, T, 6) log-energies."""
        freqs = self.freqs.view(1, 1, -1, 1)
        centers = torch.exp(self.log_centers).view(1, 1, 1, -1)
        bws = torch.exp(self.log_bandwidths).view(1, 1, 1, -1)
        
        log_ratio = torch.log(freqs.clamp(min=1.0) / centers)
        weights = F.relu(1.0 - log_ratio.abs() / bws)
        weights = weights / (weights.sum(dim=2, keepdim=True).clamp(min=1e-8))
        
        energies = (mag_sq.unsqueeze(-1) * weights).sum(dim=2)
        return torch.log1p(energies)
    
    def _transient_density(self, waveform, T):
        """Per-frame onset density. waveform: (B, L), T: target frames."""
        envelope = waveform.abs().unsqueeze(1)  # (B, 1, L)
        diff = F.relu(self.transient_conv(envelope))
        pooled = F.adaptive_avg_pool1d(diff, T)
        return pooled.transpose(1, 2)
    
    def _spectral_centroid(self, mag_sq):
        """Normalized log spectral centroid."""
        freqs = self.freqs.clamp(min=1.0)
        weights = mag_sq / (mag_sq.sum(dim=-1, keepdim=True).clamp(min=1e-8))
        centroid_hz = (weights * freqs).sum(dim=-1, keepdim=True)
        return torch.log(centroid_hz.clamp(min=1.0)) / self.log_max_freq
    
    def forward(self, waveform, mag_sq):
        """
        waveform: (B, L)
        mag_sq: (B, T, F) — |STFT|^2
        Returns: (B, T, d_model) — embedded features
        """
        band_energy = self._soft_band_energy(mag_sq)        # (B, T, 6)
        T = mag_sq.shape[1]
        transient   = self._transient_density(waveform, T)   # (B, T, 1)
        centroid    = self._spectral_centroid(mag_sq)        # (B, T, 1)
        
        r8 = torch.cat([band_energy, transient, centroid], dim=-1)  # (B, T, 8)
        return self.input_proj(r8)  # (B, T, d_model)
    
    def extract_r8_raw(self, waveform, mag_sq):
        """Return raw R=8 features (before d_model projection) for MIMO input."""
        band_energy = self._soft_band_energy(mag_sq)
        T = mag_sq.shape[1]
        transient = self._transient_density(waveform, T)
        centroid = self._spectral_centroid(mag_sq)
        return torch.cat([band_energy, transient, centroid], dim=-1)  # (B, T, 8)
```

### 3.3 Semantic Encoder

Contrastive pretraining module (Phase 1):

```python
class SemanticEncoder(nn.Module):
    """
    Lightweight contrastive semantic encoder.
    Conv stem → 2 Mamba-3 layers → mean pool → projection.
    """
    def __init__(self, cfg):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv1d(cfg.n_mels, cfg.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=3, padding=1),
            nn.LayerNorm(cfg.d_model),
        )
        self.mamba_layers = nn.ModuleList([
            Mamba3MIMOLayer(cfg) for _ in range(2)
        ])
        self.proj_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
    
    def forward(self, mel, r8, return_sequence=False):
        """
        mel: (B, T, n_mels)
        r8: (B, T, R) — raw MIMO features
        """
        x = mel.transpose(1, 2)       # (B, n_mels, T)
        x = self.conv_stem[0](x)      # Conv1d
        x = self.conv_stem[1](x)      # GELU
        x = self.conv_stem[2](x)      # Conv1d
        x = x.transpose(1, 2)         # (B, T, d_model)
        x = self.conv_stem[3](x)      # LayerNorm
        
        for layer in self.mamba_layers:
            x = layer(x, r8)          # No conditioning in encoder
        
        if return_sequence:
            return x                  # (B, T, d_model)
        
        pooled = x.mean(dim=1)
        return F.normalize(self.proj_head(pooled), dim=-1)
```

### 3.4 F0 Conditioner (CREPE)

```python
class F0Conditioner(nn.Module):
    """Projects CREPE 360-D f0 probability vector → cond_dim."""
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(360, 128),
            nn.GELU(),
            nn.Linear(128, cfg.d_f0),   # 32-dim
            nn.LayerNorm(cfg.d_f0),
        )
    
    def forward(self, crepe_probs):  # (B, T, 360)
        return self.proj(crepe_probs)  # (B, T, 32)
```

### 3.5 Cross-Track Aggregator

```python
class CrossTrackAggregator(nn.Module):
    """
    Causal multi-head attention over track dimension (not time).
    Each timestep independently attends over K other tracks' features.
    """
    def __init__(self, cfg):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_semantic,  # 128
            num_heads=4,
            batch_first=True,
        )
        self.proj = nn.Linear(cfg.d_semantic, cfg.d_cross)  # → 64-dim
        self.norm = nn.LayerNorm(cfg.d_cross)
    
    def forward(self, target_emb, other_embs):
        """
        target_emb: (B, T, D)
        other_embs: (B, K, T, D) — K other tracks
        Returns: (B, T, d_cross)
        """
        B, T, D = target_emb.shape
        K = other_embs.shape[1]
        
        q = target_emb.reshape(B*T, 1, D)
        kv = other_embs.permute(0, 2, 1, 3).reshape(B*T, K, D)
        
        ctx, _ = self.attn(q, kv, kv)
        ctx = ctx.squeeze(1).reshape(B, T, D)
        return self.norm(self.proj(ctx))
```

### 3.6 Mamba3MIMOLayer — Full forward() Signature

```python
def forward(self, x, x_r8, cond=None):
    """
    x: (B, T, d_model) — track embedding (from R* extractor)
    x_r8: (B, T, R) — raw structured features (MIMO input)
    cond: (B, T, d_cond) or None — FiLM conditioning
    Returns: (B, T, d_model)
    """
    # 1. FiLM conditioning
    if cond is not None:
        gb = self.film_proj(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        x = x * (1 + gamma) + beta
    
    # 2. Δ projection + FiLM on Δ
    dt_raw = self.dt_proj(x)
    if cond is not None:
        dt_gb = self.dt_film_proj(cond)
        dt_g, dt_b = dt_gb.chunk(2, dim=-1)
        dt_raw = dt_raw * (1 + dt_g) + dt_b
    
    # 3. Softplus + clamp
    dt = F.softplus(dt_raw + self.dt_bias)
    dt = dt.clamp(min=self.dt_min, max=self.dt_max)
    
    # 4. Trapezoidal discretization + MIMO SSM scan (fp32 forced)
    with torch.autocast(device_type=x.device.type, enabled=False):
        dt_fp32 = dt.float()
        x_r8_fp32 = x_r8.float()
        A_d, B_d_scale = self._discretize_trapezoidal(
            self.log_a_real.float(), self.a_imag.float(), dt_fp32
        )
        out = self._ssm_scan_parallel(x_r8_fp32, A_d, B_d_scale)
    
    out = out.to(x.dtype)
    out = out.reshape(B, T, -1)  # flatten heads
    
    # 5. Output projection + residual
    out = self.out_proj(out)
    return self.norm(out + x)
```

### 3.7 Full Model: PerTrackProcessor

```python
class PerTrackProcessor(nn.Module):
    """
    Complete per-track processor:
    STFT → R* features → semantic enc → f0 cond → Mamba stack → decoder → ISTFT
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Feature extraction
        self.r8_extractor = R8FeatureExtractor(cfg)
        self.f0_cond = F0Conditioner(cfg)
        self.semantic_enc = SemanticEncoder(cfg)
        self.cross_track = CrossTrackAggregator(cfg)
        
        # Conditioning MLP
        cond_in = cfg.d_semantic + cfg.d_f0 + cfg.d_cross + 4  # + loudness(4)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, cfg.cond_dim),
            nn.GELU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
            nn.LayerNorm(cfg.cond_dim),
        )
        
        # Mamba stack
        self.mamba_layers = nn.ModuleList([
            Mamba3MIMOLayer(cfg) for _ in range(cfg.n_layers)
        ])
        
        # Decoder: d_model → complex STFT bins
        F_bins = cfg.n_fft // 2 + 1
        self.decoder = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Linear(cfg.d_model * 2, 2 * F_bins),  # real + imag
        )
        
        # Mel projection (learnable, non-negative init)
        self.mel_proj = nn.Linear(F_bins, cfg.n_mels, bias=False)
        with torch.no_grad():
            self.mel_proj.weight.abs_()
        
        # Pre-register STFT window
        self.register_buffer("stft_window", torch.hann_window(cfg.n_fft))
    
    def _compute_stft(self, waveform):
        stft = torch.stft(waveform, n_fft=self.cfg.n_fft,
                          hop_length=self.cfg.hop_length,
                          window=self.stft_window,
                          return_complex=True, normalized=False)
        stft = stft.transpose(1, 2)  # (B, T, F)
        return stft.abs().pow(2), stft
    
    def _compute_mel(self, mag_sq):
        projected = self.mel_proj(mag_sq).clamp(min=0.0)
        return torch.log1p(projected)
    
    def _loudness_features(self, waveform, T):
        """4-dim: [rms, peak, crest_factor, lufs_approx] per frame."""
        # ... per-frame loudness computation ...
        pass  # Same as mamba3 reference
    
    def forward(self, waveform, crepe_probs, other_embs=None):
        """
        waveform: (B, L)
        crepe_probs: (B, T, 360)
        other_embs: (B, K, T, d_semantic) or None
        Returns dict with 'output_stft' (complex), 'track_emb', 'r8_raw'
        """
        mag_sq, stft_in = self._compute_stft(waveform)
        T = mag_sq.shape[1]
        
        # Features
        x = self.r8_extractor(waveform, mag_sq)             # (B, T, d_model)
        r8_raw = self.r8_extractor.extract_r8_raw(waveform, mag_sq)  # (B, T, R)
        mel = self._compute_mel(mag_sq)                     # (B, T, n_mels)
        
        # Semantic embedding (per-frame)
        sem_emb = self.semantic_enc(mel, r8_raw, return_sequence=True)
        
        # F0 conditioning
        f0_emb = self.f0_cond(crepe_probs)
        
        # Cross-track context
        if other_embs is not None:
            cross_ctx = self.cross_track(sem_emb, other_embs)
        else:
            cross_ctx = torch.zeros(waveform.shape[0], T, self.cfg.d_cross, device=waveform.device)
        
        # Build conditioning vector
        loudness = self._loudness_features(waveform, T)
        cond_in = torch.cat([sem_emb, f0_emb, cross_ctx, loudness], dim=-1)
        cond = self.cond_mlp(cond_in)  # (B, T, cond_dim)
        
        # Mamba stack
        for layer in self.mamba_layers:
            x = layer(x, r8_raw, cond=cond)
        
        # Decode to STFT
        stft_pred = self.decoder(x)
        F_bins = self.cfg.n_fft // 2 + 1
        stft_real = stft_pred[..., :F_bins].float()
        stft_imag = stft_pred[..., F_bins:].float()
        stft_out = torch.complex(stft_real, stft_imag)
        
        return {
            'output_stft': stft_out,
            'track_emb': sem_emb,
            'r8_raw': r8_raw,
        }
    
    def waveform_from_stft(self, stft):
        """ISTFT reconstruction."""
        return torch.istft(stft.transpose(1, 2),
                          n_fft=self.cfg.n_fft,
                          hop_length=self.cfg.hop_length,
                          window=self.stft_window)
```

---

## 4. Phase 3: C++ Engine Rewrite

### 4.1 MambaConfig.h — Changes

```cpp
// REMOVED
// constexpr size_t MAX_NUM_ROPE_ANGLES = 16;
// constexpr size_t MAX_NUM_BC_HEADS = 1;

// ADDED
constexpr size_t MAX_MIMO_RANK = 8;  // Was 4. Must match R option.
constexpr float DT_MIN = 0.001f;
constexpr float DT_MAX = 0.1f;

// CHANGED: in_proj_dim
// Old: 1612 (with RoPE)
// New: 1552 (complex poles, no RoPE, dd_A split into real+imag)
constexpr size_t MAX_IN_PROJ_DIM = MAX_D_INNER
    + MAX_NHEADS * MAX_HEADDIM
    + MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE  // B
    + MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE  // C
    + MAX_NHEADS * 3  // dd_dt + dd_A_real + dd_A_imag
    + MAX_NHEADS;     // trap

// NEW weight version
constexpr uint32_t WEIGHT_VERSION = 3;

// NEW per-layer arrays (15 instead of 13)
constexpr size_t W_LOG_A_REAL_ELEMS = MAX_NHEADS;
constexpr size_t W_A_IMAG_ELEMS = MAX_NHEADS;
constexpr size_t W_B_BIAS_ELEMS = MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE;  // Grows with MIMO_RANK
constexpr size_t W_C_BIAS_ELEMS = W_B_BIAS_ELEMS;
constexpr size_t W_MIMO_X_ELEMS = MAX_NHEADS * MAX_MIMO_RANK * MAX_HEADDIM;
constexpr size_t W_MIMO_Z_ELEMS = W_MIMO_X_ELEMS;
constexpr size_t W_MIMO_O_ELEMS = W_MIMO_X_ELEMS;

// State now complex (real + imag arrays, same shape)
constexpr size_t SSM_REAL_FLOATS = MAX_NHEADS * MAX_HEADDIM * MAX_D_STATE;  // 8192
constexpr size_t SSM_IMAG_FLOATS = MAX_NHEADS * MAX_HEADDIM * MAX_D_STATE;  // 8192
// REMOVED: ANGLE_STATE_FLOATS, K_STATE_FLOATS
constexpr size_t STATE_FLOATS_PER_LAYER = SSM_REAL_FLOATS + SSM_IMAG_FLOATS;  // 16384
```

### 4.2 SsmBlock.hpp — New Interface

```cpp
class SsmBlock {
public:
    // Called on [Main] thread
    bool init(const float* weightData, size_t layerIndex);
    void resetState();
    
    // Called on [Audio] thread
    void processSample(const float* input,  // [d_model]
                       const float* cond,   // [COND_DIM]
                       float* output);      // [d_model]
    void processBlock(const float* input,   // [MAX_BLOCK_SIZE][d_model]
                      const float* cond,    // [COND_DIM]
                      float* output,        // [MAX_BLOCK_SIZE][d_model]
                      size_t numSamples);

private:
    // Weights (15 arrays per layer, all float32)
    // Same base as v2 but: no RoPE angles, added log_a_real, a_imag
    
    // Complex state (replaces real+angle state from v2)
    alignas(64) float stateReal_[MAX_NHEADS * MAX_HEADDIM * MAX_D_STATE];
    alignas(64) float stateImag_[MAX_NHEADS * MAX_HEADDIM * MAX_D_STATE];
    
    // Complex pole parameters (loaded from weights)
    float logAReal_[MAX_NHEADS];
    float AImag_[MAX_NHEADS];
    
    // Methods
    void stepComplexSsm(const float* inProjOutput, float* output);
    void discretizeTrapezoidal(float dt, int head,
                               float& A_real, float& A_imag,
                               float& B_scale_real, float& B_scale_imag);
};
```

### 4.3 stepComplexSsm() — Core Logic

```cpp
void SsmBlock::stepComplexSsm(const float* inProjOutput, float* output) {
    // 1. Parse in_proj output (1552 floats)
    // in_proj order: z(256) + x(256) + B(512) + C(512) + dd_dt(4) + dd_A_real(4) + dd_A_imag(4) + trap(4)
    const float* z   = inProj;                // z
    const float* x_h = inProj + d_inner;       // x
    const float* B_r = x_h + nheads * headdim; // B_raw
    const float* C_r = B_r + nheads * mimoRank * d_state; // C_raw
    const float* dd_dt = C_r + nheads * mimoRank * d_state;
    const float* dd_A_real = dd_dt + nheads;
    const float* dd_A_imag = dd_A_real + nheads;
    const float* trap_data = dd_A_imag + nheads;
    // (trap_data unused in complex pole version — trapezoidal is now the only method)
    
    // 2. Per-head processing
    for (int h = 0; h < nheads; ++h) {
        // dt = softplus(dd_dt[h] + dtBias[h])
        float dt = softplus(dd_dt[h] + dtBias_[h]);
        dt = std::clamp(dt, DT_MIN, DT_MAX);
        
        // Complex trapezoidal discretization
        // A = -exp(logAReal[h]) + j*AImag[h]
        float A_real = -std::exp(logAReal_[h]);
        float A_imag = AImag_[h] + dd_A_imag[h];  // input-modulated frequency
        
        // A_d = (1 + dt*(A_r+jA_i)/2) / (1 - dt*(A_r+jA_i)/2)
        float half_dt_real = dt * A_real * 0.5f;
        float half_dt_imag = dt * A_imag * 0.5f;
        
        float denom_real = 1.0f - half_dt_real;
        float denom_imag = -half_dt_imag;
        float denom_mag = denom_real * denom_real + denom_imag * denom_imag;
        
        // A_d
        float A_d_real = ((1.0f + half_dt_real) * denom_real - half_dt_imag * denom_imag) / denom_mag;
        float A_d_imag = ((1.0f + half_dt_real) * denom_imag + half_dt_imag * denom_real) / denom_mag;
        
        // B_d = dt / (1 - dt*A/2)
        float B_d_real = dt * denom_real / denom_mag;
        float B_d_imag = dt * (-denom_imag) / denom_mag;  // dt * conj(1 - dt*A/2) / |...|^2
        // Wait: B_d = dt / (1 - half_dt_a) = dt * conj(1 - half_dt_a) / |1 - half_dt_a|^2
        // conj(1 - half_dt_a) = 1 - half_dt_real + j*half_dt_imag
        B_d_imag = dt * half_dt_imag / denom_mag;  // correct
        B_d_real = dt * (1.0f - half_dt_real) / denom_mag;
        
        // 3. SSM recurrence: h[t] = A_d * h[t-1] + B_d * input
        int state_idx = h * headdim * d_state;
        int mimo_slot = 0;  // Which MIMO slot does the current input activate?
        float mimo_input_real = 0.0f, mimo_input_imag = 0.0f;
        
        // MIMO: sum over R slots of B_proj[h][r][s] * x_r8[r]
        // (B is projected from x_h via MIMO weights, not done here)
        
        // For each d_state component:
        for (int s = 0; s < d_state; ++s) {
            float old_real = stateReal_[state_idx + s];
            float old_imag = stateImag_[state_idx + s];
            
            // B_d * input_contribution
            float input_real = /* B_proj_contribution per slot and state */;
            float input_imag = 0.0f;  // B projection output is real → complex
            
            stateReal_[state_idx + s] = A_d_real * old_real - A_d_imag * old_imag
                                       + B_d_real * input_real - B_d_imag * input_imag;
            stateImag_[state_idx + s] = A_d_real * old_imag + A_d_imag * old_real
                                       + B_d_real * input_imag + B_d_imag * input_real;
        }
        
        // 4. Readout: C_normed[h][r][s] * state[h][d][s] summed over s,r
        // Then z-gate, output projection, D skip, residual
        // (Matches existing z-gate + output logic)
    }
}
```

Note: The SSM recurrence shown above is the per-sample inner loop. For the parallel scan in training (Python), the recurrence is vectorized. For C++ inference, the sequential per-sample loop is correct and matches `_ssm_scan_sequential()` exactly.

### 4.4 WeightLoader — Version 3

```cpp
struct WeightHeader {
    uint32_t magic;          // 0x4D424D41 ("MBMA")
    uint32_t version;        // 3
    uint32_t d_state;
    uint32_t d_model;
    uint32_t d_inner;
    uint32_t nheads;
    uint32_t headdim;
    uint32_t mimo_rank;      // NOW 8 (was 4)
    uint32_t n_layers;
    uint32_t cond_dim;
    uint32_t in_proj_dim;    // 1552 (was 1612)
    uint32_t weight_data_size;
    uint32_t reserved[3];    // pad to 64 bytes
};

// Per-layer array layout (15 arrays, v3):
// [0]  in_proj_w        [in_proj_dim * d_model]           // 1552*128 = 198,656
// [1]  dt_bias          [nheads]                          // 4
// [2]  B_bias           [nheads * mimo_rank * d_state]    // With R=8: 4*8*32 = 1024
// [3]  C_bias           [nheads * mimo_rank * d_state]    // 1024
// [4]  B_norm_weight    [d_state]                         // 32
// [5]  C_norm_weight    [d_state]                         // 32
// [6]  mimo_x           [nheads * mimo_rank * headdim]    // With R=8: 4*8*64 = 2048
// [7]  mimo_z           [nheads * mimo_rank * headdim]    // 2048
// [8]  mimo_o           [nheads * mimo_rank * headdim]    // 2048
// [9]  D_proj           [nheads]                          // 4
// [10] out_proj_w       [d_model * d_inner]               // 128*256 = 32,768
// [11] film_proj_w      [2*d_model * cond_dim]            // 256*128 = 32,768
// [12] dt_film_proj_w   [2*nheads * cond_dim]             // 8*128 = 1024
// [13] log_a_real       [nheads]                          // 4 — NEW
// [14] a_imag           [nheads]                          // 4 — NEW
```

---

## 5. Phase 4: Data Pipeline

### 5.1 StemDataset

```python
class StemDataset(Dataset):
    """
    Multi-stem dataset for MUSDB18-HQ.
    Returns (target_stem, mixture, other_stems, target_crepe, other_crepe).
    
    Directory layout:
      root/
        track_name/
          bass.wav          + bass.crepe.npy + bass.pitched.npy
          drums.wav         + drums.crepe.npy + drums.pitched.npy
          vocals.wav        + vocals.crepe.npy + vocals.pitched.npy
          other.wav         + other.crepe.npy + other.pitched.npy
          mixture.wav
    """
    
    STEM_NAMES = ['bass', 'drums', 'vocals', 'other']
    
    def __init__(self, root, cfg, clip_seconds=4.0, augment=True,
                 cache_limit=16, is_val=False):
        self.root = Path(root)
        self.cfg = cfg
        self.clip_len = int(clip_seconds * cfg.sample_rate)
        self.augment = augment and not is_val
        self.is_val = is_val
        
        # Discover tracks
        self.tracks = sorted([
            d for d in self.root.iterdir()
            if d.is_dir() and all((d / f"{s}.wav").exists() for s in self.STEM_NAMES)
        ])
        assert len(self.tracks) > 0, f"No complete tracks found in {root}"
        
        # LRU cache
        self.cache_limit = cache_limit
        self.cache_queue = []
        self.cache = {}  # {track_dir: {'stems': {}, 'crepe': {}}}
    
    def __len__(self):
        return len(self.tracks) * len(self.STEM_NAMES) * 4
    
    def _load_stem(self, track_dir, stem_name):
        """Load and cache a stem WAV file."""
        # LRU cache logic (same as mamba3 reference)
        # Uses soundfile to avoid torchcodec/ffmpeg issues
        pass
    
    def _load_crepe(self, track_dir, stem_name):
        """Load pre-extracted CREPE f0 .npy file."""
        crepe_path = track_dir / f"{stem_name}.crepe.npy"
        pitched_path = track_dir / f"{stem_name}.pitched.npy"
        
        if not crepe_path.exists():
            return torch.zeros(T, 360, dtype=torch.float32), 0.0
        
        probs = np.load(str(crepe_path)).astype(np.float32)
        pitched = float(np.load(str(pitched_path)))
        return torch.from_numpy(probs), pitched
    
    def _crop_with_crepe(self, wav, crepe):
        """Sync-crop waveform and CREPE to clip_len."""
        # See mamba3 reference for implementation
        # Converts sample offset to frame offset using hop_length
        pass
    
    def _pitch_shift(self, wav, crepe, semitones):
        """Pitch-shift waveform and roll CREPE 360-D vector by matching bins."""
        # Resample waveform: new_freq = int(44100 * 2^(semitones/12))
        # Roll CREPE: bins_shift = int(round(5 * semitones))  # 20 cents/bin
        pass
    
    def __getitem__(self, idx):
        """Returns dict with target, other_stems, target_crepe, other_crepe, etc."""
        # Same logic as mamba3 StemDataset
        # 1. Select track, stem
        # 2. Load wave + CREPE
        # 3. Sync-crop
        # 4. Load other stems
        # 5. Apply augmentations (pitch shift ALL stems together, then gain/polarity individually)
        pass


def collate_valid(batch):
    """Filter out invalid (missing stem) batches."""
    batch = [b for b in batch if b.get('valid', True)]
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)
```

### 5.2 CREPE f0 Extraction Script

Create `scripts/extract_crepe.py` (ported from mamba3 reference):
```bash
python extract_crepe.py --data ~/datasets/musdb18hq/train --device cuda --model full
```

Uses `torchcrepe` to extract 360-D pitch probability vectors at 100 Hz frame rate. Saves `.crepe.npy` (probability distribution) and `.pitched.npy` (binary: is this stem pitched?).

### 5.3 DataLoader Setup (in train())

```python
full_dataset = StemDataset(args.data, cfg, clip_seconds=4.0, augment=True, cache_limit=args.cache_limit)
val_dataset = StemDataset(args.data, cfg, clip_seconds=4.0, augment=False, is_val=True, cache_limit=args.cache_limit)

# Deterministic train/val split by TRACK (no leakage)
n_tracks = len(full_dataset.tracks)
n_val = max(1, int(n_tracks * args.val_split))
random.Random(42).shuffle(full_dataset.tracks)
val_tracks = full_dataset.tracks[-n_val:]
train_tracks = full_dataset.tracks[:-n_val]
full_dataset.tracks = train_tracks
val_dataset.tracks = val_tracks

train_loader = DataLoader(
    full_dataset, batch_size=args.batch, shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,            # ALWAYS
    collate_fn=collate_valid,   # FILTER INVALID
    drop_last=True,             # STABLE BATCH SIZES
)
val_loader = DataLoader(
    val_dataset, batch_size=args.batch, shuffle=False,
    num_workers=0,              # NO FORK FOR VAL
    pin_memory=True,
    collate_fn=collate_valid,
    drop_last=False,            # TEST EVERY VAL SAMPLE
)
```

---

## 6. Phase 5: Loss Functions

### 6.1 NT-Xent Contrastive Loss (Phase 1)

```python
class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross-entropy for contrastive learning."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.T = temperature
    
    def forward(self, z1, z2):
        """z1, z2: (B, d) — L2-normalized embeddings."""
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)       # (2B, d)
        sim = torch.mm(z, z.T) / self.T       # (2B, 2B)
        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels + B, labels])
        mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))
        return F.cross_entropy(sim, labels)
```

### 6.2 Multi-Resolution STFT Loss (4 scales)

```python
class SingleResolutionSTFTLoss(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
    
    def forward(self, pred, target):
        stft_pred = torch.stft(pred, self.n_fft, hop_length=self.hop_length,
                               win_length=self.win_length, window=self.window,
                               return_complex=True).abs().clamp(min=1e-8)
        stft_tgt = torch.stft(target, self.n_fft, hop_length=self.hop_length,
                              win_length=self.win_length, window=self.window,
                              return_complex=True).abs().clamp(min=1e-8)
        # Spectral convergence (normalized Frobenius norm)
        sc_loss = (torch.norm(stft_tgt - stft_pred, p='fro')
                   / torch.norm(stft_tgt, p='fro').clamp(min=1e-8))
        # Log-magnitude L1
        lm_loss = F.l1_loss(torch.log(stft_tgt), torch.log(stft_pred))
        return sc_loss, lm_loss

class MultiResolutionSTFTLoss(nn.Module):
    """4 scales: 512/1024/2048/4096."""
    def __init__(self, fft_sizes=[512, 1024, 2048, 4096],
                 hop_sizes=[128, 256, 512, 1024],
                 win_lengths=[512, 1024, 2048, 4096]):
        super().__init__()
        self.losses = nn.ModuleList([
            SingleResolutionSTFTLoss(n, h, w)
            for n, h, w in zip(fft_sizes, hop_sizes, win_lengths)
        ])
    
    def forward(self, pred, target):
        sc_total = 0.0
        lm_total = 0.0
        for loss in self.losses:
            sc, lm = loss(pred, target)
            sc_total += sc
            lm_total += lm
        return sc_total + lm_total
```

### 6.3 Mel-L1 Loss (3 scales)

```python
class MelL1Loss(nn.Module):
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.mel_transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=n, hop_length=n//4, n_mels=m,
            )
            for n, m in [(512, 64), (1024, 80), (2048, 128)]
        ])
    
    def forward(self, pred, target):
        loss = 0.0
        for mel_tf in self.mel_transforms:
            p = torch.log1p(mel_tf(pred))
            t = torch.log1p(mel_tf(target))
            loss += F.l1_loss(p, t)
        return loss / len(self.mel_transforms)
```

### 6.4 SI-SDR (ramped)

```python
def si_sdr_loss(pred, target):
    """SI-SDR in dB. pred/target: (B, L)"""
    eps = 1e-8
    target_energy = (target ** 2).sum(dim=-1, keepdim=True)
    dot = (pred * target).sum(dim=-1, keepdim=True)
    proj = dot / (target_energy + eps) * target
    noise = pred - proj
    si_sdr = 10 * torch.log10(
        (proj ** 2).sum(-1) / ((noise ** 2).sum(-1) + eps) + eps
    )
    return -si_sdr.mean()
```

### 6.5 Combined Loss (mamba3 formula)

```python
class TrackProcessorLoss(nn.Module):
    """Multi-objective loss matching mamba3 schedule."""
    
    def __init__(self, cfg):
        super().__init__()
        self.mr_stft = MultiResolutionSTFTLoss()
        self.mel_l1 = MelL1Loss(cfg.sample_rate)
        self.global_step = 0
    
    def _si_sdr_weight(self):
        """Ramp 0→0.1 between steps 5k and 15k."""
        if self.global_step < 5000:
            return 0.0
        elif self.global_step > 15000:
            return 0.1
        else:
            return 0.1 * (self.global_step - 5000) / 10000
    
    def forward(self, pred_stft, target_wave, model):
        """
        pred_stft: (B, T, F) complex — model output
        target_wave: (B, L) — target waveform
        """
        pred_wave = model.waveform_from_stft(pred_stft)
        
        L = min(pred_wave.shape[-1], target_wave.shape[-1])
        pred_wave = pred_wave[..., :L]
        target_wave = target_wave[..., :L]
        
        losses = {}
        losses['mr_stft'] = self.mr_stft(pred_wave, target_wave)
        losses['mel_l1'] = self.mel_l1(pred_wave, target_wave)
        
        si_w = self._si_sdr_weight()
        if si_w > 0:
            losses['si_sdr'] = si_sdr_loss(pred_wave, target_wave) * si_w
        else:
            losses['si_sdr'] = torch.tensor(0.0, device=pred_wave.device)
        
        losses['total'] = (1.0 * losses['mr_stft']
                           + 0.5 * losses['mel_l1']
                           + losses['si_sdr'])
        
        self.global_step += 1
        return losses
```

---

## 7. Phase 6: Phase Strategy & Training Loop

### 7.1 Phase Schedule

```
Phase 1 (0-20%):  Contrastive semantic encoder pretraining
  - Model: semantic_enc only (freeze mamba_layers + decoder)
  - Loss: NT-Xent only
  - Data: augmented pairs of same stem (pitch shift + gain + noise + time shift)
  - Learning rate: 3e-4 → cosine to 3e-5
  - Validation: NT-Xent loss on val set augmented pairs

Phase 2 (20-50%): Single-track reconstruction
  - Model: full PerTrackProcessor
  - Loss: MR-STFT + Mel-L1 + ramped SI-SDR
  - Data: target stem + CREPE f0 (no cross-track)
  - Conditioning: f0 only (cross_track_ctx = zeros)

Phase 3 (50-100%): Cross-track conditioned
  - Model: full PerTrackProcessor
  - Loss: MR-STFT + Mel-L1 + SI-SDR
  - Data: target stem + CREPE f0 + OTHER stems
  - Conditioning: f0 + cross_track_ctx from other stems
  - Per-step: process all K other stems through semantic_enc to get embeddings
```

### 7.2 Augmentation for Contrastive (Phase 1)

```python
def augment_contrastive(wav, crepe):
    """Augment waveform+CREPE for contrastive pair generation."""
    # 1. Pitch shift ±2 semitones (50% chance)
    if random.random() < 0.5:
        semitones = random.uniform(-2.0, 2.0)
        wav, crepe = pitch_shift(wav, crepe, semitones)
    
    # 2. Random gain ±4 dB
    gain_db = random.uniform(-4, 4)
    wav = wav * (10 ** (gain_db / 20))
    
    # 3. Random polarity flip (50% chance)
    if random.random() < 0.5:
        wav = -wav
    
    # 4. Random circular time shift
    shift = random.randint(0, wav.shape[-1] // 4)
    wav = torch.roll(wav, shift, dims=-1)
    crepe = torch.roll(crepe, shift // cfg.hop_length, dims=0)
    
    # 5. Additive noise
    noise_level = random.uniform(0, 0.005)
    wav = wav + noise_level * torch.randn_like(wav)
    
    return wav.clamp(-1.0, 1.0), crepe
```

### 7.3 Training Loop Pseudocode

```python
while step < n_steps:
    for batch in dataloader:
        if batch is None:
            continue  # collate_valid returned nothing
        
        target = batch['target'].to(device)
        target_crepe = batch['target_crepe'].to(device)
        other_stems = batch['other_stems'].to(device)
        other_crepe = batch['other_crepe'].to(device)
        T_frames = 1 + target.shape[-1] // cfg.hop_length
        
        current_phase = (1 if step < PHASE_1_END
                         else 2 if step < PHASE_2_END
                         else 3)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=device, dtype=bf16, enabled=amp):
            if current_phase == 1:
                # Contrastive — only semantic encoder
                aug1 = augment_contrastive(target, target_crepe)
                aug2 = augment_contrastive(target, target_crepe)
                z1 = encode_contrastive(model, aug1[0], aug1[1])
                z2 = encode_contrastive(model, aug2[0], aug2[1])
                loss = ntxent(z1, z2)
            
            elif current_phase == 2:
                # Single-track reconstruction
                out = model(target, target_crepe, other_embs=None)
                loss_dict = task_loss(out['output_stft'], target, model)
                loss = loss_dict['total']
            
            else:  # Phase 3
                # Cross-track: process other stems → embeddings → attend
                K = other_stems.shape[1]
                embs = []
                for k in range(K):
                    stem_k = other_stems[:, k, :]
                    crepe_k = other_crepe[:, k, :, :]
                    out_k = model(stem_k, crepe_k, other_embs=None)
                    embs.append(out_k['track_emb'])
                other_embs = torch.stack(embs, dim=1)
                out = model(target, target_crepe, other_embs=other_embs)
                loss_dict = task_loss(out['output_stft'], target, model)
                loss = loss_dict['total']
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[step {step}] NaN/Inf — skipping")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
        scheduler.step()
        
        # Logging (windowed average)
        if step % log_every == 0:
            # ... print loss components, step/s, lr ...
        
        # Validation (decoupled)
        if step > 0 and step % val_every == 0:
            val_loss = validate_phase(model, val_loader, current_phase)
            # ... early stopping logic ...
        
        # Checkpoint (decoupled)
        if step > 0 and step % ckpt_every == 0:
            save_checkpoint(step, loss.item())
        
        step += 1
```

### 7.4 Phase-Aware Validation

```python
@torch.no_grad()
def validate_phase(model, val_loader, phase):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    for batch in val_loader:
        if batch is None:
            continue
        
        target = batch['target'].to(device)
        target_crepe = batch['target_crepe'].to(device)
        
        with torch.autocast(device_type=device, dtype=bf16, enabled=amp):
            if phase == 1:
                z1 = encode_contrastive(model, target, target_crepe)
                z2 = encode_contrastive(model,
                      augment_contrastive(target, target_crepe)[0],
                      augment_contrastive(target, target_crepe)[1])
                vl = ntxent(z1, z2)
            elif phase == 2:
                out = model(target, target_crepe, other_embs=None)
                vl = task_loss(out['output_stft'], target, model)['total']
            else:
                other_stems = batch['other_stems'].to(device)
                other_crepe = batch['other_crepe'].to(device)
                K = other_stems.shape[1]
                embs = []
                for k in range(K):
                    out_k = model(other_stems[:,k,:], other_crepe[:,k,:,:], other_embs=None)
                    embs.append(out_k['track_emb'])
                other_embs = torch.stack(embs, dim=1)
                out = model(target, target_crepe, other_embs=other_embs)
                vl = task_loss(out['output_stft'], target, model)['total']
        
        if not (torch.isnan(vl) or torch.isinf(vl)):
            total_loss += vl.item()
            n_batches += 1
    
    model.train()
    return total_loss / max(1, n_batches)
```

---

## 8. Phase 7: Export & Validation

### 8.1 Export Script — `scripts/export_mamba_weights.py`

Designed for multiple plugins (Saturator, Compressor, Denoiser):
```bash
# Usage
python scripts/export_mamba_weights.py \
    --checkpoint checkpoints/best.pt \
    --output models/MambaSaturator/model.bin \
    --verify

python scripts/export_mamba_weights.py \
    --checkpoint checkpoints/compressor_best.pt \
    --output models/MambaCompressor/model.bin \
    --verify
```

**Features:**
- Full v3 header validation
- Per-array integrity check (non-NaN, non-Inf, expected shapes)
- SHA256 checksum embedded in reserved header bytes
- `--verify` mode: read back and validate entire file
- Support for both FP32 export (C++ engine) and optional INT8 quantization (future)

### 8.2 Weight Format v3 (15 arrays per layer)

```
Header (64 bytes):
  [0] magic      uint32  = 0x4D424D41 ("MBMA")
  [1] version    uint32  = 3
  [2] d_state    uint32  = 32
  [3] d_model    uint32  = 128
  [4] d_inner    uint32  = 256
  [5] nheads     uint32  = 4
  [6] headdim    uint32  = 64
  [7] mimo_rank  uint32  = 4|6|8 (depends on R option)
  [8] n_layers   uint32  = 6
  [9] cond_dim   uint32  = 128
  [10] in_proj_dim uint32 = 1552
  [11] weight_data_size uint32
  [12] sha256_prefix uint32  (first 4 bytes of SHA256 of weight data)
  [13] reserved  uint32  = 0
  
Per layer (repeated n_layers times):
  in_proj_w   [in_proj_dim * d_model]   float32
  dt_bias     [nheads]                   float32
  B_bias      [nheads*R*d_state]        float32
  C_bias      [nheads*R*d_state]        float32
  B_norm      [d_state]                  float32
  C_norm      [d_state]                  float32
  mimo_x      [nheads*R*headdim]        float32
  mimo_z      [nheads*R*headdim]        float32
  mimo_o      [nheads*R*headdim]        float32
  D_proj      [nheads]                   float32
  out_proj_w  [d_model*d_inner]         float32
  film_proj_w [2*d_model*cond_dim]      float32
  dt_film_proj_w [2*nheads*cond_dim]    float32
  log_a_real  [nheads]                   float32  ← NEW
  a_imag      [nheads]                   float32  ← NEW
```

### 8.3 C++ Engine Tests

```cpp
// tests/test_mamba3.cpp — updated tests:

TEST(test_bit_exact, complex_pole) {
    // Load exported v3 weights
    // Run stepComplexSsm() on known input
    // Compare output to Python _ssm_scan_sequential()
    // Tolerance: 0.1 (complex arithmetic has higher cross-library error)
}
```

### 8.4 Validation Pipeline

```bash
# 1. Export
python scripts/export_mamba_weights.py \
    --checkpoint checkpoints/best.pt \
    --output models/MambaTransparent/model.bin

# 2. Verify integrity
python scripts/export_mamba_weights.py \
    --verify models/MambaTransparent/model.bin

# 3. C++ load + bit-exact
cmake -B build -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build -R mamba3 --output-on-failure

# 4. Full integration (CLAP plugin)
ctest --test-dir build --output-on-failure
```

---

## 9. Dependency Graph & Timeline

```
Phase 1: Training Infrastructure [~2-3 hours]
  ├── 1.1 ROCm environment       — 5 mins (edit train_mamba3.py)
  ├── 1.2 CLI flags              — 15 mins
  ├── 1.3 AMP implementation     — 30 mins
  ├── 1.4 Gradient checkpointing — 15 mins
  ├── 1.5 zero_grad(set_to_none) — 5 mins
  ├── 1.6 DataLoader             — 30 mins
  ├── 1.7 NaN/Inf handling       — 5 mins
  ├── 1.8 Decoupled val/ckpt     — 10 mins
  └── 1.9 Phase-aware validation — 20 mins
  ✅ Test: CPU smoke run

Phase 2: Python Model Architecture [~4-6 hours]
  ├── 2.1 Complex pole SSM       — 60 mins
  │   ├── Mamba3Config update
  │   ├── New in_proj slicing
  │   ├── _discretize_trapezoidal
  │   ├── _ssm_scan_parallel (complex)
  │   └── _ssm_scan_sequential (complex)
  ├── 2.2 R* Feature Extractor   — 45 mins
  ├── 2.3 Semantic Encoder       — 30 mins
  ├── 2.4 F0 Conditioner         — 10 mins
  ├── 2.5 Cross-Track Aggregator — 20 mins
  ├── 2.6 PerTrackProcessor      — 45 mins
  │   ├── STFT/mel/loudness
  │   ├── Full forward pipeline
  │   └── ISTFT decoder
  └── 2.7 Loss functions         — 30 mins
      ├── NT-Xent
      ├── MR-STFT (4-scale)
      ├── Mel-L1 (3-scale)
      └── TrackProcessorLoss
  ✅ Test: Forward pass match with _ssm_scan_sequential

Phase 3: C++ Engine [~6-8 hours]
  ├── 3.1 MambaConfig.h          — 15 mins
  ├── 3.2 SsmBlock.hpp           — 30 mins
  ├── 3.3 stepComplexSsm()       — 120 mins
  ├── 3.4 Weight v3 loader       — 45 mins
  ├── 3.5 STFT/ISTFT (if needed) — 60 mins
  └── 3.6 Tests                  — 45 mins
  ✅ Test: ctest 6/6 pass

Phase 4: Data Pipeline [~60 mins]
  ├── 4.1 StemDataset            — 30 mins
  ├── 4.2 CREPE extraction       — 15 mins (script)
  ├── 4.3 Augmentation           — 15 mins
  └── 4.4 collate_valid          — 5 mins
  ✅ Test: Dataset loads + CREPE alignment matches

Phase 5: Training Launch [~6-12 hours real time]
  ├── 5.1 CREPE extraction (full dataset) — ~2 hours GPU
  ├── 5.2 Phase 1 (contrastive)          — 11-28 hours
  ├── 5.3 Phase 2 (single-track)         — 25-50 hours
  └── 5.4 Phase 3 (cross-track)          — 56-111 hours
  ✅ Test: Export → C++ load → plugin validation

Total wall clock (860M): ~3-7 days continuous training
Total engineering time: ~12-16 hours
```

---

## 10. Risk Register

| # | Risk | Likelihood | Severity | Mitigation |
|---|------|-----------|----------|------------|
| R1 | Complex pole A_d has |A_d| > 1 (unstable) | Medium | High | fp32 SSM core, dt clamping, periodic state renormalization |
| R2 | bf16 AMP introduces NaN in complex arithmetic | Medium | High | Disable autocast inside SSM recurrence. Verify no NaN in first 1000 steps. |
| R3 | CREPE extraction fails on GPU on Bazzite | Low | Medium | Fall back to CPU extraction (slower but works). Already happened in user's Bazzite log. |
| R4 | R=8 increases C++ state memory 2× | Low | Low | 16,384 floats per layer × 6 = 98,304 floats = 384KB. Stack-allocated. Fine. |
| R5 | Python parallel scan diverges from C++ sequential | Medium | High | Bit-exact test with _ssm_scan_sequential. Tolerance 0.1 for complex division floating-point variation. |
| R6 | Training takes >7 days | Medium | Low | Acceptable for local training. Can reduce model size (--d-model) for faster iteration. |
| R7 | STFT/ISTFT latency in C++ engine | Low | High | 46ms is acceptable for mixing plugins. Not for live monitoring. Document clearly. |
| R8 | Gradient checkpointing makes training 25% slower | Certain | Low | Acceptable trade-off. Enables 2× larger batch. Net throughput similar. |

---

## 11. File Manifest

| File | Phase | Action | Summary |
|------|-------|--------|---------|
| `train_mamba3.py` | 1,2 | Rewrite | ROCm env, CLI redesign, AMP, DataLoader, training loop, phase strategy |
| `dataset.py` | 4 | Rewrite | StemDataset with CREPE, pitch-shift, cache; remove CachedMelDataset/AudioDataset |
| `loss.py` | 2,5 | Split from train | NT-Xent, MR-STFT(4), Mel-L1(3), SI-SDR, TrackProcessorLoss |
| `scripts/extract_crepe.py` | 4 | Create | torchcrepe extraction with resume support |
| `scripts/precompute_features.py` | 4 | Remove | No longer needed (on-the-fly STFT in StemDataset) |
| `scripts/export_mamba_weights.py` | 7 | Create | v3 export + verify (replaces train_clara export) |
| `scripts/validate_export.py` | 7 | Update | v3 header validation |
| `scripts/dump_reference_output.py` | 7 | Update | Complex pole reference outputs |
| `scripts/setup_cloud.sh` | 1 | Archive | Rewrite for local-only training or remove |
| `src/core/MambaConfig.h` | 3 | Rewrite | R=4/6/8, remove RoPE, add poles, dt_min/max, v3 format |
| `src/core/SsmBlock.hpp` | 3 | Rewrite | Complex state, log_a_real, a_imag, no angle_state |
| `src/core/SsmBlock.cpp` | 3 | Rewrite | stepComplexSsm(), new discretize, new in_proj slicing |
| `src/core/WeightHeader.h` | 3 | Update | V3 struct with 15 arrays |
| `src/core/WeightLoader.cpp` | 3 | Update | V3 parsing |
| `src/core/SsmInferenceCore.*` | 3 | Update | New block processing |
| `src/core/MambaInferenceEngine.*` | 3 | Archive | Dead code (old ONNX pipeline) |
| `src/clap/MambaClapPlugin.*` | 3 | Update | New engine interface |
| `src/ara/MambaAraController.*` | 3 | Update | New engine interface |
| `tests/test_mamba3.cpp` | 7 | Update | Complex pole tests |
| `docs/TRAINING.md` | 1 | Rewrite | New flags, environment, dataset, phase strategy |
| `AGENTS.md` | 1 | Update | New gaps, config |
| `CLARA_OVERHAUL_PLAN.md` | — | This file | Keep as reference |
