# CLARA — Next Session Instructions

## State at Session End (2026-06-04)

### Current Status: 81/95 TODO items done, 14 remaining

**Phase 1 training (identity autoencoder) is READY TO LAUNCH** — all source files verified. The first training attempt revealed two ROCm-specific issues that are now fixed.

---

## Known Issues Fixed This Session

### 1. CUDA OOM — Parallel Scan Memory

**Problem:** `clip_seconds=2.0` produces L=88200 samples at 44.1kHz. Parallel scan materializes `[B, L, D, N]` = `[4, 88200, 64, 16]` per layer = ~1.35 GB/layer × 4 layers = OOM on 12 GB GPU.

**Fix Added:** `--clip-seconds` CLI argument on both `train_clara.py` and `train_clara_launcher.py`. Use `--clip-seconds 0.5` (L=22050) for first run. This drops per-layer state to ~0.17 GB.

**Status:** ✅ Fixed but untested in a full run.

### 2. HIPFFT_PARSE_ERROR — MR-STFT on ROCm

**Problem:** `torch.stft()` crashes on ROCm 7.13 with `HIPFFT_PARSE_ERROR` when computing multi-resolution STFT loss.

**Fix Added:** CPU fallback in `MultiResolutionSTFTLoss.forward()` — if CUDA device, moves tensors to CPU for STFT compute. Catches `RuntimeError` and skips that FFT size gracefully.

**Status:** ✅ Fixed but CPU STFT is slow. For next session, consider:
- Option A: Verify if the CPU fallback works acceptably (may add ~2-5 min per 500 steps)
- Option B: Set `--stft-weight 0 --l1-weight 1.0` to skip STFT loss entirely for Phase 1
- Option C: Install `torchaudio` and use `torchaudio.functional.spectrogram` as alternative

### 3. Launcher Runtime Bug

**Problem:** `_phase_num()` function defined at line 120 but called at line 65 — `NameError` crash.

**Fix:** Hoisted function definition to top of file.

### 4. Launcher Hardcoded `num_workers=2`

**Fix:** Changed default to 4, added `--num-workers` CLI flag.

### 5. `make_zero_states()` Arg Order Bug

**Problem:** Function signature is `make_zero_states(batch_size, d_model, d_state, n_layers)` but was called with positional args `(B, d_model, n_layers, d_state)`. With d_state=16, n_layers=4, this swapped silently — n_layers=4 worked by accident if d_state was also 4, but would break if they diverged.

**Fix:** Callers now use keyword args.

---

## Launch Checklist (Next Session)

### Step 1: Short test run (5-10 min)
```bash
python train_clara_launcher.py --phase identity --clip-seconds 0.5 --steps 500 --no-cache
```
This skips RAM cache (20GB takes minutes to load) and runs 500 steps. Verify:
- Loss decreases from initial (expect ~0.02 → ~0.005)
- GPU memory stable (~6-8 GB used)
- Checkpoint saved at step 500

### Step 2: If STFT CPU fallback is too slow
```bash
python train_clara.py --data <path> --phase identity --clip-seconds 0.5 --steps 5000 --stft-weight 0 --l1-weight 1
```
But `--stft-weight` is NOT a CLI arg yet — would need to add to `train_clara.py`.

### Step 3: Full Phase 1 run (~20 min)
```bash
python train_clara_launcher.py --phase identity --clip-seconds 0.5
```

### Step 4: Export and test in C++
```bash
python train_clara_export.py --checkpoint checkpoints_clara/checkpoint_50000.pt --output models/mamba_rt.onnx
# Rebuild C++ and run tests
cmake --build build --parallel
cd build && ctest --output-on-failure
```

---

## Remaining Tasks

### P0 (1 remaining)
- `TODO.md` says P0 has 4/5 done but tracking shows `#` item may be P0 #3 status

### P1 (1 remaining)
- **#16**: Training pipeline written and verified — needs actual launch on 860M

### P4 (2 remaining)
- **#44**: SPSC two-thread test already done (test_spsc_two_thread at test_clap_plugin.cpp:434)
- **#45**: CLAP-level conditioning tests deferred until ARA host bridge (CLARA-09)

### P5 — Next Actions (11 remaining)
| # | Action | Depends On | Effort |
|---|--------|-----------|--------|
| N2 | Launch Phase 1 training | Nothing (ready now) | ~20 min |
| N3 | Export Phase 1 → ONNX → C++ verify | N2 complete | ~10 min |
| N4 | Phase 1B with conditioning | N2 complete | ~30 min |
| N5 | Phase 2: saturation training | N2, N4, synthetic data only | ~40 min |
| N6 | Phase 3: compression training | N5 | ~40 min |
| N7 | Phase 4: denoising training | Needs DNS download (~20 GB) | ~60 min |
| N8 | Phase 5: multi-effect training | N6 | ~60 min |
| N9 | ARA host bridge (CLARA-09) | Independent | ~1 week |
| N10 | MAESTRO v3 download | Before N5,N6 | ~5 GB |
| N11 | DNS Challenge download | Before N7 | ~20 GB |
| N12 | ONNX dynamic seq_len | Independent, low urgency | ~1 day |

---

## Architecture Reference

### Model: 76,865 params
- d_model=64, d_state=16, n_layers=4, conv_width=4, dt_rank=16, cond_dim=8
- Parallel scan for training (fast, GPU-memory hungry)
- Sequential scan for export (slow, memory-efficient)
- CondFiLM: per-layer FiLM modulation via learned projection from cond[8] → [gamma, beta] per channel

### ONNX I/O (10 inputs, 3 outputs)
Inputs: `audio_chunk`, `conv0`-`conv3` (4 conv states), `ssm0`-`ssm3` (4 SSM states), `cond`
Outputs: `processed_audio`, `new_conv_state`, `new_ssm_state`
Sequence length: 64 samples (sub-block processing in C++)

### C++ Engine
- `kSubBlockSize=64` — processes any host block in 64-sample chunks
- `kNumInputs=10` — includes cond tensor
- `modelReloadPending` atomic flag for RT-safe engine swap
- No external gain modulation (cond folded into ONNX via FiLM)

### Dataset: MUSDB18-HQ at `C:\Dev\mamba3\mamba3\datasets\musdb18hq\train`
- 500 WAV files (100 tracks × 5 stems: mixture, bass, drums, vocals, other)
- Each stem ~40 MB at 44.1 kHz stereo
- 3.5-5 min per track → ~200-300 chunks of 0.5s each per file
- RAM cache: 500 files, ~20 GB (takes ~30-60s to load)

---

## Next Session Quick Start

```bash
cd "C:\Dev\Audio Development\0 CLARA"

# 1. Quick 500-step test
python train_clara_launcher.py --phase identity --clip-seconds 0.5 --steps 500 --no-cache

# 2. If that works, full Phase 1 (50K steps ~20 min)
python train_clara_launcher.py --phase identity --clip-seconds 0.5

# 3. Export and verify in C++
python train_clara_export.py --checkpoint checkpoints_clara/phase1_identity_best.pt
```

## TO-DO BEFORE LAUNCH

- [ ] #1: Train complete run (e.g. python train_clara_launcher.py --phase identity)
- [ ] #2: Or consider: temp fix for stft hipFFT?
- [ ] #3: Do not proceed to training yet. Save all steps to a file for next session.

**Session was closed with training pipeline ready to launch but not yet started on 860M iGPU.**
