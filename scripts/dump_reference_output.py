#!/usr/bin/env python3
"""
Dump reference forward pass outputs for Python↔C++ bit-exact validation.

Generates three files consumed by test_mamba3.cpp's test_bit_exact():
  - model_ref.bin:   Mamba-3 flat binary weights (Version 2)
  - input_ref.bin:   Audio samples (float32) with header (uint32 N)
  - output_ref.bin:  Python model output (float32) with header (uint32 N)

Both Python and C++ run the same weights on the same input with the same
conditioning vector. CTest compares the outputs sample-by-sample.

Usage:
    python scripts/dump_reference_output.py --output-dir models/
    mkdir -p /tmp/ref && python scripts/dump_reference_output.py --output-dir /tmp/ref
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from train_clara_mamba3 import Mamba3Model, Mamba3Config, export_weights


# ── Constants (must match C++ test) ─────────────────────────────────

SINE_FREQ     = 440.0
SAMPLE_RATE   = 44100
NUM_SAMPLES   = 256
COND_DIM      = 128


def generate_sine(n_samples: int, freq: float, sr: float) -> torch.Tensor:
    """Generate a 1kHz sine wave."""
    t = torch.arange(n_samples, dtype=torch.float32) / sr
    return (torch.sin(2.0 * np.pi * freq * t) * 0.5).float()


def run_python_model(model: Mamba3Model, audio: torch.Tensor,
                     cond: torch.Tensor) -> torch.Tensor:
    """Run the Python model with per-sample stateful processing matching C++.

    The C++ engine processes samples one at a time with state carried across
    calls. We match this by calling model on one sample at a time instead of
    using the batched forward pass (which resets state per call).

    This is the bit-exact gold standard. If pre-sample and batched forward
    differ, the pre-sample path is the reference (it matches the C++ engine).
    """
    B = 1
    T = audio.shape[0]
    cfg = model.cfg
    device = next(model.parameters()).device

    # Pad cond to batch dimension
    cond_b = cond.unsqueeze(0)  # [1, COND_DIM]

    # Per-layer state (persistent across samples, matches C++ SSMState)
    layer_states = []
    for _ in range(cfg.n_layers):
        layer_states.append({
            'ssm_real': torch.zeros(1, cfg.nheads, cfg.headdim, cfg.d_state, device=device),
            'angle_state': torch.zeros(1, cfg.nheads, cfg.num_rope_angles, device=device),
        })

    output = torch.zeros(T, device=device)

    for t in range(T):
        # Embed scalar → D_MODEL (first channel, match C++ x[0] = input[i])
        x = torch.zeros(1, 1, cfg.d_model, device=device)
        x[0, 0, 0] = audio[t]

        # Forward through all layers, carrying state (reset_state=False
        # keeps SSM state across samples, matching C++ engine behavior)
        layer_idx = 0
        for layer in model.layers:
            state = layer_states[layer_idx]
            x = layer.forward_batch(x, state, cond_b, reset_state=False)
            layer_idx += 1
            # LayerNorm (fixed gamma=1, beta=0)
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + 1e-5)

        # First D_MODEL channel → scalar (match C++ output[i] = x[0])
        output[t] = x[0, 0, 0]

    return output


def write_bin(filepath: str, data: torch.Tensor):
    """Write tensor as binary: uint32 N followed by N float32 samples."""
    arr = data.detach().cpu().numpy().astype(np.float32).reshape(-1)
    with open(filepath, 'wb') as f:
        f.write(struct.pack('<I', len(arr)))
        f.write(arr.tobytes())
    print(f"  Wrote {filepath} ({len(arr)} samples, {len(arr)*4:,} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Dump reference forward pass for Python↔C++ validation")
    parser.add_argument('--output-dir', default='models',
                        help='Output directory for reference files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (must match export in test_mamba3.cpp)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    # ── Create model with random weights ──
    print(f"Creating Mamba-3 model (seed={args.seed})...")
    cfg = Mamba3Config()
    model = Mamba3Model(cfg)
    print(f"  Parameters: {model.num_params():,}")

    # ── Checkpoint and export weights ──
    # Save checkpoint in the format export_weights expects
    ckpt_path = output_dir / '_reference_checkpoint.pt'
    torch.save({
        'step': 0,
        'model_state_dict': model.state_dict(),
        'cfg': cfg.__dict__,
    }, ckpt_path)

    bin_path = output_dir / 'model_ref.bin'
    export_weights(str(ckpt_path), str(bin_path))
    os.remove(ckpt_path)

    # ── Generate test audio ──
    audio = generate_sine(NUM_SAMPLES, freq=SINE_FREQ, sr=SAMPLE_RATE)
    print(f"  Test audio: {NUM_SAMPLES} samples, {SINE_FREQ}Hz sine")

    # ── Conditioning vector (match C++ test) ──
    # The test_bit_exact in test_mamba3.cpp sets:
    #   cond[0] = 0.5f;   // DAW context
    #   cond[40] = 0.3f;  // user knob
    cond = torch.zeros(COND_DIM)
    cond[0] = 0.5
    cond[40] = 0.3
    print(f"  Conditioning: cond[0]={cond[0].item()}, cond[40]={cond[40].item()}")

    # ── Run Python forward pass (per-sample, matching C++) ──
    print("Running Python forward pass (per-sample, stateful)...")
    model.eval()
    with torch.no_grad():
        output = run_python_model(model, audio, cond)

    print(f"  Output: std={output.std().item():.6f}, "
          f"max_abs={output.abs().max().item():.6f}, "
          f"finite={torch.isfinite(output).all().item()}")

    # ── Write reference files ──
    write_bin(str(output_dir / 'input_ref.bin'), audio)
    write_bin(str(output_dir / 'output_ref.bin'), output)

    print(f"\nDone. Reference files in {output_dir}/")
    print(f"  model_ref.bin   — flat binary weights")
    print(f"  input_ref.bin   — {NUM_SAMPLES} audio samples (float32)")
    print(f"  output_ref.bin  — {NUM_SAMPLES} Python output (float32)")
    print(f"\nRun C++ test: ctest --test-dir build -R test_mamba3")


if __name__ == '__main__':
    main()
