#!/usr/bin/env python3
"""
Dump Mamba-3 reference outputs for C++ bit-exact comparison.

Generates:
  1. models/model_ref.bin    — seeded weights (matching C++ WeightLoader format)
  2. models/input_ref.bin    — test input signal (256 samples of sine)
  3. models/output_ref.bin   — expected output from Python reference

Usage:
    python scripts/dump_reference_output.py --seed 42
    # Then C++ test loads all three .bin files and compares output
"""

import struct
import numpy as np
from pathlib import Path

# Add repo root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from reference_forward import (
    D_MODEL, D_STATE, D_INNER, NHEADS, HEADDIM, MIMO_RANK,
    N_LAYERS, COND_DIM, NUM_ROPE_ANGLES, IN_PROJ_DIM,
    ReferenceMamba3Engine,
)


def generate_test_data(seed: int = 42, output_dir: str = "models"):
    """Generate reference model, test input, and expected output."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Generate model.bin with fixed seed
    print("Generating reference weights...")
    import scripts.export_mamba_weights as exporter
    exporter.export_weights(str(out / "model_ref.bin"), seed=seed)

    # 2. Generate test input: 256 samples of 440 Hz sine at 44.1 kHz
    print("Generating test input...")
    N = 256
    sr = 44100.0
    freq = 440.0
    t = np.arange(N) / sr
    input_signal = np.sin(2.0 * np.pi * freq * t).astype(np.float32)

    with open(out / "input_ref.bin", "wb") as f:
        f.write(struct.pack('<I', N))  # sample count
        f.write(input_signal.tobytes())

    # 3. Generate conditioning vector
    cond = np.zeros(COND_DIM, dtype=np.float32)
    cond[0] = 0.5  # tempo slot: moderate
    cond[1] = 0.0  # key: C
    cond[40] = 0.3  # intensity knob

    # 4. Run Python reference forward pass
    print("Running reference forward pass...")
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    engine = ReferenceMamba3Engine(device='cpu')
    audio_tensor = torch.from_numpy(input_signal.copy())
    cond_tensor = torch.from_numpy(cond.copy())

    output_tensor = engine.process_block(audio_tensor, cond_tensor)
    output_signal = output_tensor.numpy().astype(np.float32)

    with open(out / "output_ref.bin", "wb") as f:
        f.write(struct.pack('<I', N))
        f.write(output_signal.tobytes())

    # 5. Print summary
    print(f"\n=== Reference data generated ===")
    print(f"  {out / 'model_ref.bin'}:  {Path(out / 'model_ref.bin').stat().st_size:,} bytes")
    print(f"  {out / 'input_ref.bin'}:  {Path(out / 'input_ref.bin').stat().st_size:,} bytes")
    print(f"  {out / 'output_ref.bin'}: {Path(out / 'output_ref.bin').stat().st_size:,} bytes")
    print(f"\n  Input stats:  min={input_signal.min():.4f} max={input_signal.max():.4f}")
    print(f"  Output stats: min={output_signal.min():.4f} max={output_signal.max():.4f} std={output_signal.std():.4f}")
    print(f"  Output finite: {np.all(np.isfinite(output_signal))}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Dump Mamba-3 reference outputs")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', default='models', help='Output directory')
    args = parser.parse_args()
    generate_test_data(args.seed, args.output_dir)
