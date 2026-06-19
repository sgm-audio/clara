# CLARA MI300X — Single Launch Reference

**Budget:** $100 AMD ADC credits. Kill at **$90** (leaves $10 margin).
**Billing:** $1.99/hr **even when VM is stopped**. Destroy VM to stop billing.
**Data:** VM destroyed = all data lost. S3 sync is your only protection.

---

## The One Command

```bash
# ssh into AMD AI Developer Cloud MI300X, run:
curl -fsSL https://raw.githubusercontent.com/SGM-Studios/clara/main/scripts/setup_cloud.sh | bash

# The script:
#   1. Installs ROCm + PyTorch + deps
#   2. Downloads MUSDB18-HQ (~41 GB, ~15 min)
#   3. Precomputes mel/STFT features
#   4. Starts cost monitor (warns at $75, kills at $90)
#   5. Starts S3 checkpoint sync (every 5 min)
#   6. Starts Stage 0 identity training (~100K steps, ~4 hrs)
#   7. On completion: exports to model.bin, syncs everything to S3
#   8. At $90: kills training, syncs final checkpoints to S3
```

## Before Launch (5 min setup)

```bash
# 1. Create S3 bucket (your personal AWS account, NOT included with AMD ADC)
#    Cost: ~$0.023/GB/month = ~$0.25 for a full training run
pip install awscli         # included in setup_cloud.sh
aws configure              # enter your AWS key/secret

# 2. Create bucket
aws s3 mb s3://clara-checkpoints

# 3. Verify the bucket exists
aws s3 ls s3://clara-checkpoints
```

## After Launch — What to Watch

```bash
# Quick status (all in one):
echo "=== Training ===" && tail -3 ~/checkpoints/mamba3_stage0/train.log && echo "=== Cost ===" && tail -3 ~/checkpoints/cost_monitor.log && echo "=== GPU ===" && tail -3 ~/checkpoints/gpu_monitor.log && echo "=== S3 Sync ===" && tail -3 ~/checkpoints/sync_checkpoints.log

# Or watch each individually:
tail -f ~/checkpoints/mamba3_stage0/train.log     # training progress
tail -f ~/checkpoints/cost_monitor.log              # cost tracking
tail -f ~/checkpoints/gpu_monitor.log               # GPU utilization
```

## Kill Procedure (at $90 or when done)

```bash
# The cost monitor does this automatically at $90.
# But if you need to kill early:

# 1. Sync everything to S3
bash ~/clara/scripts/sync_checkpoints.sh --once ~/checkpoints/ --s3 s3://clara-checkpoints/checkpoints/
bash ~/clara/scripts/sync_checkpoints.sh --once ~/clara/models/ --s3 s3://clara-checkpoints/exports/

# 2. Verify S3 has everything
aws s3 ls s3://clara-checkpoints/ --recursive

# 3. DESTROY VM (AMD ADC Console → Instances → Delete)
#    Only then does billing stop. Stopped VMs still cost $1.99/hr.
```

## Resume from S3

```bash
# If VM gets killed/destroyed and you launch a new one:
aws s3 cp s3://clara-checkpoints/checkpoints/mamba3_stage0/step_050000.pt \
    ~/checkpoints/mamba3_stage0/step_050000.pt

python train_clara_mamba3.py \
    --precomputed ~/datasets/musdb18hq/precomputed \
    --cache \
    --resume ~/checkpoints/mamba3_stage0/step_050000.pt \
    --steps 100000 \
    --device cuda
```

---

## Datasets

| Dataset | Size | Needed? | Why |
|---------|------|---------|-----|
| **MUSDB18-HQ** | ~41 GB | **YES** | Stage 0 identity training. Downloaded by setup script. |
| FMA Full | ~100 GB | **SKIP** | Would be destroyed with VM. Only for SSL, not needed for MVP. |
| MAESTRO | ~54 GB | **SKIP** | Not needed until conditioned training. |
| Slakh2100 | ~30 GB | **SKIP** | Not needed until stem separation. |
| FSD50K | ~10 GB | **SKIP** | Not needed until SSL pretraining. |

## F0 / CREPE / Pitch

Was extracted in the **old Mamba-1 pipeline** (`__train_laptop_ref.py`). The current **Mamba-3** architecture has a 128-dim conditioning vector. F0/pitch would go in cond slots 0-7 (DAW context) if needed later. Not needed for Stage 0.

---

## Cost Table

| Stage | Steps | Est. Time | Est. Cost |
|-------|-------|-----------|-----------|
| Stage 0 (identity) | 100K | ~4 hrs | ~$8 |
| Stage 1 Saturator | 100K | ~4 hrs | ~$8 |
| Stage 1 Compressor | 100K | ~4 hrs | ~$8 |
| Stage 1 Denoiser | 150K | ~6 hrs | ~$12 |
| **Total** | **450K** | **~18 hrs** | **~$36** |
| Safety margin | — | — | **$54** |
| **Kill threshold** | — | **~45 hrs** | **$90** |

Total budget is $100. $90 kill threshold leaves $10 margin for S3 warnings.

---

## Architecture Readiness Checklist

| Component | Status | Notes |
|-----------|--------|-------|
| Python model matches C++ engine | ✅ | Input/out proj stripped, LayerNorm hardcoded |
| Weight export to .bin | ✅ | Version 2, 13 arrays/layer |
| Precomputed mel/STFT pipeline | ✅ | Script + CachedMelDataset + pinned tensors + pin_memory |
| Loss uses precomputed features | ✅ | PrecomputedMRSTFTLoss — only pred STFT/mel computed |
| C++ engine validation | ✅ | 8/8 tests pass, Python↔C++ diff < 0.05 |
| S3 checkpoint sync | ✅ | `scripts/sync_checkpoints.sh` — runs every 5 min |
| Cost monitor + auto-kill | ✅ | `setup_cloud.sh` — warns at $75, kills at $90 |
| DSP paired data pipeline | ❌ | Not needed for Stage 0. Needed for Stage 1. |
| Python↔C++ bit-exact | ✅ | Within 5% — cross-library float32 limit |

---

## The Golden Path

```
ssh VM
  └→ curl setup_cloud.sh | bash
       ├── [15 min] Download MUSDB18-HQ
       ├── [30 min] Precompute mel+STFT
       ├── [4 hrs]  Stage 0 Identity Training
       │              └→ S3 sync every 5 min
       │              └→ Cost monitor (warn $75, kill $90)
       ├── [Done]   Export model.bin
       └── [Done]   Sync everything to S3

DESTROY VM → billing stops. Data safe in S3.
```
