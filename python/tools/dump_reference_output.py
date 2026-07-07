#!/usr/bin/env python3
"""Dump PerTrackProcessor forward pass outputs for validation.

Generates three files:
  - model_ref.bin:    Mamba-3 v3 flat binary weights
  - input_ref.bin:   Audio samples (float32) with header (uint32 N)
  - output_ref.bin:  Predicted STFT (float32 complex) with header (uint32 N)

NOTE: PerTrackProcessor is a spectral-domain model (STFT frontend → Mamba-3 →
ISTFT output). The C++ engine (SsmInferenceCore) is a time-domain model
(R=4, per-sample). These are fundamentally different architectures — bit-exact
Python↔C++ comparison is NOT possible. The C++ engine must first be updated
to support R=8 complex-pole SSM before this can be used.

This script validates the Python training pipeline export is working correctly.

Usage:
    python python/tools/dump_reference_output.py --output-dir models/
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from train_clara import Mamba3Config, PerTrackProcessor, export_weights

# ── Constants ───────────────────────────────────────────────────────────

SAMPLE_RATE = 44100
NUM_SAMPLES = 2048  # ~46ms at 44.1kHz
COND_DIM = 128


def generate_sine(n_samples: int, freq: float = 440.0) -> torch.Tensor:
    """1-second sine wave at given frequency."""
    t = torch.arange(n_samples, dtype=torch.float32) / SAMPLE_RATE
    return torch.sin(2.0 * np.pi * freq * t) * 0.5


def write_bin(filepath: str, data: np.ndarray):
    """Write array as binary: uint32 N followed by N float32 samples."""
    flat = data.reshape(-1).astype(np.float32)
    with open(filepath, "wb") as f:
        f.write(struct.pack("<I", len(flat)))
        f.write(flat.tobytes())
    print(f"  Wrote {filepath} ({len(flat)} floats, {len(flat)*4:,} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Dump PerTrackProcessor reference outputs for v3 export"
    )
    parser.add_argument(
        "--output-dir", default="models", help="Output directory for reference files"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    # ── Create model with random weights ──
    print(f"Creating PerTrackProcessor (seed={args.seed})...")
    cfg = Mamba3Config()
    model = PerTrackProcessor(cfg)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  R={cfg.R}, d_model={cfg.d_model}, n_layers={cfg.n_layers}")

    # ── Export weights ──
    ckpt_path = output_dir / "_reference_checkpoint.pt"
    torch.save(
        {
            "step": 0,
            "model": model.state_dict(),
            "cfg": cfg.__dict__,
        },
        ckpt_path,
    )

    bin_path = output_dir / "model_ref.bin"
    export_weights(str(ckpt_path), str(bin_path))
    os.remove(ckpt_path)

    # ── Generate test audio ──
    audio = generate_sine(NUM_SAMPLES)
    print(f"  Test audio: {NUM_SAMPLES} samples, 440Hz sine")

    # Conditioning vector (fake — just needs to be the right shape)
    cond = torch.zeros(COND_DIM)
    cond[0] = 0.5
    cond[40] = 0.3
    print(f"  Conditioning: cond[0]={cond[0].item()}, cond[40]={cond[40].item()}")

    # ── Fake CREPE probabilities ──
    T_frames = 1 + NUM_SAMPLES // cfg.hop_length
    crepe_probs = torch.rand(1, T_frames, 360)
    crepe_probs = crepe_probs / crepe_probs.sum(dim=-1, keepdim=True)  # normalize

    # ── Run PerTrackProcessor forward ──
    print("Running PerTrackProcessor forward pass...")
    model.eval()
    with torch.no_grad():
        # Add batch dim
        audio_b = audio.unsqueeze(0)  # [1, NUM_SAMPLES]
        crepe_b = crepe_probs  # [1, T, 360]

        out = model(audio_b, crepe_b, other_embs=None)
        stft_out = out["output_stft"]  # [1, T, F] complex

    print(f"  Output STFT: {stft_out.shape}, finite={torch.isfinite(stft_out).all().item()}")

    # ── Write reference files ──
    write_bin(str(output_dir / "input_ref.bin"), audio.numpy())
    write_bin(str(output_dir / "output_ref.bin"), stft_out[0].numpy())

    # Write metadata
    meta = output_dir / "ref_metadata.txt"
    with open(meta, "w") as f:
        f.write(f"sample_rate={SAMPLE_RATE}\n")
        f.write(f"num_samples={NUM_SAMPLES}\n")
        f.write(f"T_frames={T_frames}\n")
        f.write(f"n_fft={cfg.n_fft}\n")
        f.write(f"hop_length={cfg.hop_length}\n")
        f.write("version=3 (PerTrackProcessor R=8)\n")
    print(f"  Wrote {meta}")

    print(f"\nDone. Reference files in {output_dir}/")
    print("\nNOTE: C++ engine (R=4, time-domain) cannot bit-compare with this.")
    print("      H3 (bit-exact Python↔C++) requires C++ engine R=8 update first.")


if __name__ == "__main__":
    main()
