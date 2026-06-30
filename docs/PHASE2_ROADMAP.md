# Phase 2 Roadmap — Conditioning & Multi-Stem Training

**Date:** 2026-06-20
**Status:** Planning — Phase 0 (identity training) is current focus. Phase 2 items documented here for reference.

---

## Overview

Phase 2 adds conditioning (f0, DAW context, user knobs) and multi-stem training to Clara's Mamba-3 model. This is the bridge between "identity autoencoder" (Phase 0) and "per-effect specialist" (Phase 3+).

```
Phase 0: Identity training (current)
    ↓
Phase 2: Conditioning + multi-stem
    ↓
Phase 3: Per-effect supervised training
    ↓
Phase 4: Adversarial fine-tuning
    ↓
Phase 5: RLHF (post-MVP)
```

---

## Phase 2 Work Items

### 2.1 CREPE f0 Extraction (Infrastructure — Phase 1)

**Status:** Scripts created, not wired to model.

| File | Status | Purpose |
|------|--------|---------|
| `scripts/extract_crepe.py` | ✅ Created | Offline f0 extraction using `torchcrepe` `tiny` model |
| `scripts/extract_f0.sh` | ✅ Created | Batch wrapper for extraction |

**Key parameters:**
- Model: `torchcrepe` `tiny` (speed over accuracy for preprocessing)
- Hop length: **256** (matches Clara's frame rate, NOT mamba3 drive's 512)
- Sample rate: 44,100 Hz
- Output per WAV: `f0.npy`, `confidence.npy`, `periodicity.npy`

**Usage:**
```bash
# Extract f0 for all WAVs in a directory
bash scripts/extract_f0.sh ~/datasets/musdb18hq/train

# Or individually
python scripts/extract_crepe.py --input track.wav --output track_f0.npz
```

**Not wired yet:** f0 arrays sit alongside WAVs. Phase 2.3 will load them into the conditioning vector.

---

### 2.2 StemDataset Class (Infrastructure — Phase 1)

**Status:** Planned, not yet created.

**Purpose:** Load multi-stem audio (MUSDB18-HQ has drums/bass/other/vocals stems) with bounded FIFO cache for memory efficiency.

**Design (from mamba3 drive concept):**
```python
class StemDataset(Dataset):
    """Multi-stem loader with bounded FIFO cache.
    
    Loads stems on demand, caches last N stems in memory.
    Returns (mixture, stems_dict) pairs for supervised training.
    """
    def __init__(self, root, chunk_samples=22050, cache_size=32):
        self.root = Path(root)
        self.chunk_samples = chunk_samples
        self.cache = OrderedDict()  # bounded FIFO
        self.cache_size = cache_size
        # Discover all tracks with stems
        self.tracks = self._find_tracks()
    
    def __getitem__(self, idx):
        track = self.tracks[idx]
        if track not in self.cache:
            self._load_and_cache(track)
        return self.cache[track]
    
    def _load_and_cache(self, track):
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)  # evict oldest
        stems = {
            'drums': load_wav(track / 'drums.wav'),
            'bass': load_wav(track / 'bass.wav'),
            'other': load_wav(track / 'other.wav'),
            'vocals': load_wav(track / 'vocals.wav'),
        }
        mixture = sum(stems.values())
        self.cache[track] = (mixture, stems)
```

**MUSDB18-HQ stem structure:**
```
musdb18hq/train/
  ├── A Classic Education - NightOwl/
  │   ├── drums.wav
  │   ├── bass.wav
  │   ├── other.wav
  │   └── vocals.wav
  ├── ...
```

**Not wired to model yet:** Returns stems, but model doesn't use them in Phase 0.

---

### 2.3 f0 Conditioning Integration (Phase 2 — Model Changes)

**Status:** Design only.

**What changes:**
1. `Mamba3Model` gains `f0_proj`: `nn.Linear(1, cond_dim)` — projects scalar f0 to conditioning vector
2. Training loop loads pre-extracted `f0.npy` alongside audio
3. f0 features go into `cond[40-71]` (reserved slots in C++ engine)
4. Conditioning vector: `[DAW(8), CLAP(32), f0(32), knobs(32), reserved(24)]`

**C++ engine already supports this:**
```cpp
// MambaConfig.h
constexpr size_t COND_KNOB_OFFSET  = 40;
constexpr size_t COND_KNOB_SIZE    = 32;
constexpr size_t COND_RESERVED_OFFSET = 72;
constexpr size_t COND_RESERVED_SIZE   = 56;
```

Slots 40-71 are available for f0 features. No C++ changes needed.

---

### 2.4 SI-SDR Ramping Loss (Phase 2 — Loss Changes)

**Status:** Design from mamba3 drive, adoption confirmed by user.

**Current loss (Phase 0):**
```python
loss = 0.5 * MSE(pred, target) + 0.5 * MultiResolutionSTFT(pred, target)
```

**Phase 2 loss with SI-SDR ramping:**
```python
# SI-SDR weight ramps from 0 to 0.1 over training
if global_step < 5000:
    si_sdr_weight = 0.0
elif global_step < 15000:
    si_sdr_weight = 0.1 * (global_step - 5000) / 10000
else:
    si_sdr_weight = 0.1

loss = 0.5 * MSE(pred, target) \
     + 0.5 * MultiResolutionSTFT(pred, target) \
     + si_sdr_weight * SI_SDR(pred, target)
```

**SI-SDR implementation:**
```python
class SI_SDR(nn.Module):
    """Scale-Invariant Signal-to-Distortion Ratio."""
    def forward(self, pred, target):
        # Project pred onto target
        dot = (pred * target).sum(dim=-1, keepdim=True)
        s_target = dot * target / (target.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
        e_noise = pred - s_target
        si_sdr = 10 * torch.log10(s_target.pow(2).sum() / (e_noise.pow(2).sum() + 1e-8))
        return -si_sdr  # minimize negative SI-SDR
```

---

### 2.5 Multi-Head MIMO Slot Specialization (Phase 2 — Architecture)

**Status:** Design only. MIMO R=4 already supports 4 frequency slots per head.

**Concept:** The 4 MIMO slots per head learn to specialize:
- Slot 0: Sub-bass (< 100 Hz)
- Slot 1: Bass (100-300 Hz)
- Slot 2: Mids (300 Hz - 4 kHz)
- Slot 3: Highs (> 4 kHz)

**Training signal:** f0 conditioning helps the model route energy to appropriate slots. No explicit slot assignment — the model learns this from data.

---

### 2.6 CLAP Text/Audio Encoders (Phase 2/3 — Future)

**Status:** Design only. Significant infrastructure needed.

**Concept:**
- User types "gritty vox" → CLAP text encoder → 32-dim embedding → `cond[8-39]`
- Transport stops → CLAP audio encoder → 32-dim embedding → `cond[8-39]`
- Preset system: precomputed embeddings for instant loading

**This is Phase 2/3 scope** — requires shipping CLAP models with the plugin (~300MB each), main-thread inference, and atomic conditioning updates.

---

## Phase 2 Dependency Graph

```
2.1 CREPE extraction (infrastructure)
    ↓
2.3 f0 conditioning integration (model change)
    ↓
2.4 SI-SDR ramping loss (loss change)
    ↓
2.5 MIMO slot specialization (emergent from conditioning)
    
2.2 StemDataset (infrastructure)
    ↓
    (used by Phase 3 per-effect training)

2.6 CLAP encoders (independent, Phase 2/3)
```

---

## Phase 2 Training Commands (Draft)

**With f0 conditioning:**
```bash
# Step 1: Extract f0 for all training data
bash scripts/extract_f0.sh ~/datasets/musdb18hq/train

# Step 2: Train with f0 conditioning
python train_mamba3.py \
  --data ~/datasets/musdb18hq/train \
  --device cuda \
  --steps 150000 \
  --batch 4 \
  --grad-accum 4 \
  --f0-conditioning \
  --f0-weight 0.5
```

**With multi-stem data:**
```bash
# Using StemDataset (Phase 2.2)
python train_mamba3.py \
  --data ~/datasets/musdb18hq/train \
  --device cuda \
  --steps 150000 \
  --multi-stem \
  --stem-weight 0.3
```

---

## Phase 2 vs Phase 0 Comparison

| Aspect | Phase 0 (Current) | Phase 2 (Planned) |
|--------|-------------------|-------------------|
| Task | Identity (passthrough) | Effect-specific |
| Conditioning | Random vectors | f0 + DAW context |
| Loss | MSE + STFT | MSE + STFT + SI-SDR |
| Dataset | Mixture tracks | Stems (drums/bass/other/vocals) |
| Steps | 100K | 150K+ |
| Output | Base checkpoint | Per-effect model |
| C++ engine | Unchanged | Unchanged (slots 40-71 used) |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| f0 extraction too slow | Training bottleneck | Pre-extract once, cache as `.npy` |
| StemDataset OOM | Can't load all stems | Bounded FIFO cache (configurable) |
| SI-SDR NaN | Training crash | Ramp schedule avoids early instability |
| MIMO slots don't specialize | No frequency routing | Emergent from data; no explicit supervision needed |
| CLAP model too large | Plugin bloat | Phase 2/3; use quantized models if needed |

---

## Next Steps

1. **Complete Phase 0** — Identity training on MI300X
2. **Create `StemDataset`** — Phase 1 infrastructure
3. **Port CREPE extraction** — Phase 1 infrastructure
4. **Add SI-SDR ramping** — Phase 2 loss improvement
5. **Wire f0 conditioning** — Phase 2 model change
6. **Design CLAP encoder integration** — Phase 2/3 planning
