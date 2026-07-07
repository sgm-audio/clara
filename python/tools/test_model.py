#!/usr/bin/env python3
"""Smoke test for CLARA's PerTrackProcessor (R=8 spectral-domain).

Verifies:
  - ROCm environment and GPU visibility
  - PerTrackProcessor builds correctly
  - Forward pass produces finite STFT output
  - bf16 AMP stability
  - Forward + backward gradients are finite

Usage:
  python python/tools/test_model.py --device cuda
  python python/tools/test_model.py --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from train_clara import Mamba3Config, PerTrackProcessor

# ── Helpers ──────────────────────────────────────────────────────────────

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"
SKIP = "\033[90m–\033[0m"


def vram_mb() -> float:
    return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0


def peak_vram_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def check(cond: bool, msg: str, fatal: bool = False) -> bool:
    print(f"  {PASS if cond else FAIL} {msg}")
    if not cond and fatal:
        raise AssertionError(f"Fatal: {msg}")
    return cond


def section(title: str):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


def run_section(label: str, fn):
    section(label)
    try:
        fn()
        return True
    except AssertionError as e:
        print(f"  {FAIL} {e}")
        return False
    except Exception:
        traceback.print_exc()
        return False


# ── Section 0: ROCm / HSA env ──────────────────────────────────────────


def test_rocm_env():
    cuda_ok = torch.cuda.is_available()
    check(cuda_ok, f"torch.cuda.is_available() → {cuda_ok}")
    if not cuda_ok:
        print(f"  {WARN} CPU only")
        return

    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"    HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION', '')}")
    print(f"    PYTORCH_ROCM_ARCH={os.environ.get('PYTORCH_ROCM_ARCH', '')}")

    # Tiny SGEMM smoke test
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    c = (a @ b).mean()
    check(not torch.isnan(c), "64×64 SGEMM: no NaN")


# ── Section 1: Model build ──────────────────────────────────────────────


def test_build(cfg: Mamba3Config):
    t0 = time.time()
    model = PerTrackProcessor(cfg)
    build_s = time.time() - t0
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  Config: d_model={cfg.d_model}, d_state={cfg.d_state}, "
        f"nheads={cfg.nheads}, n_layers={cfg.n_layers}, R={cfg.R}"
    )
    print(f"  Params: {n:,}")
    print(f"  Build: {build_s:.2f}s | VRAM: {vram_mb():.0f} MB")
    check(n > 0, "has trainable params")
    check(build_s < 30, "build < 30s")


# ── Section 2: Forward pass ────────────────────────────────────────────


def test_forward(dev: torch.device, cfg: Mamba3Config):
    sr = cfg.sample_rate
    clip_len = sr * 1
    B = 1
    model = PerTrackProcessor(cfg).to(dev).eval()
    waveform = torch.randn(B, clip_len, device=dev)
    T = 1 + clip_len // cfg.hop_length
    crepe = torch.rand(B, T, 360, device=dev)

    reset_peak()
    t0 = time.time()
    with torch.no_grad():
        out = model(waveform, crepe, other_embs=None)
    elapsed = time.time() - t0
    stft = out["output_stft"]

    print(f"  STFT out: {stft.shape} | time: {elapsed:.3f}s | VRAM: {vram_mb():.0f} MB")
    check(stft.shape[0] == B and stft.shape[1] == T, "shape ok")
    check(bool(torch.isfinite(stft).all().item()), "no NaN/Inf")


# ── Section 3: bf16 AMP ───────────────────────────────────────────────


def test_amp(dev: torch.device, cfg: Mamba3Config):
    if dev.type == "cpu" or not hasattr(torch, "bfloat16"):
        print(f"  {SKIP} CPU or no bf16")
        return

    sr = cfg.sample_rate
    clip_len = sr
    B = 1
    m_fp32 = PerTrackProcessor(cfg).to(dev).eval()
    m_bf16 = PerTrackProcessor(cfg).to(dev).eval()
    m_bf16.load_state_dict(m_fp32.state_dict())

    waveform = torch.randn(B, clip_len, device=dev)
    T = 1 + clip_len // cfg.hop_length
    crepe = torch.rand(B, T, 360, device=dev)

    with torch.no_grad():
        fp_out = m_fp32(waveform, crepe, other_embs=None)["output_stft"].float()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bf_out = m_bf16(waveform, crepe, other_embs=None)["output_stft"].float()

    has_nan = torch.isnan(bf_out).any().item()
    has_inf = torch.isinf(bf_out).any().item()
    mean_err = (fp_out - bf_out).abs().mean().item()
    max_err = (fp_out - bf_out).abs().max().item()

    check(not has_nan and not has_inf, "bf16 no NaN/Inf")
    check(mean_err < 0.5, f"bf16 vs fp32 mean Δ: {mean_err:.4f}")


# ── Section 4: Forward + backward ─────────────────────────────────────


def test_backward(dev: torch.device, cfg: Mamba3Config):
    clip_len = cfg.sample_rate // 2  # 0.5s for speed
    B = 1
    model = PerTrackProcessor(cfg).to(dev).train()
    waveform = torch.randn(B, clip_len, device=dev)
    T = 1 + clip_len // cfg.hop_length
    crepe = torch.rand(B, T, 360, device=dev)
    target = torch.randn(B, clip_len, device=dev)

    model.zero_grad(set_to_none=True)
    reset_peak()
    t0 = time.time()

    out = model(waveform, crepe, other_embs=None)
    stft_pred = out["output_stft"]
    est_wav = torch.istft(
        stft_pred.permute(0, 2, 1),
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window=torch.hann_window(cfg.n_fft, device=dev),
        length=clip_len,
    )
    loss = F.l1_loss(est_wav, target)
    loss.backward()
    elapsed = time.time() - t0

    print(f"  loss={loss.item():.4f} | time: {elapsed:.1f}s | peak VRAM: {peak_vram_mb():.0f} MB")
    check(bool(torch.isfinite(loss).item()), "loss finite", fatal=True)

    n_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_zero = sum(1 for p in model.parameters() if p.grad is None or p.grad.abs().sum() == 0)
    print(f"  grads: {n_grads} nonzero, {n_zero} zero")
    check(n_grads > 0, "some nonzero grads")


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="PerTrackProcessor smoke test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args()

    cfg = Mamba3Config()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skip = set(args.skip)
    results = {}

    sections = [
        ("0", "ROCm / HSA env", test_rocm_env),
        ("1", "Model build", lambda: test_build(cfg)),
        ("2", "Forward pass", lambda: test_forward(dev, cfg)),
        ("3", "bf16 AMP", lambda: test_amp(dev, cfg)),
        ("4", "Forward + backward", lambda: test_backward(dev, cfg)),
    ]

    for idx, label, fn in sections:
        if idx in skip:
            print(f"\n{SKIP} [{idx}] {label} — skipped")
            continue
        results[f"[{idx}] {label}"] = run_section(f"[{idx}] {label}", fn)

    section("SUMMARY")
    passed = sum(results.values())
    total = len(results)
    for name, ok in results.items():
        print(f"  {PASS if ok else FAIL} {name}")
    print(f"\n  {passed}/{total} passed")
    if passed == total:
        print(f"\n  {PASS} Ready to train:")
        print(
            "     HSA_OVERRIDE_GFX_VERSION=11.0.0 HIP_VISIBLE_DEVICES=0 ROC_VISIBLE_DEVICES=0 python train_clara.py --data ~/datasets/musdb18hq/train --device cuda"
        )
    else:
        print(f"\n  {FAIL} Fix failures first")


if __name__ == "__main__":
    main()
