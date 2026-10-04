#!/usr/bin/env python3
"""Generate Python↔C++ reference fixtures for the R=4 time-domain engine.

Produces, next to the model file:
  - input_ref.bin:  uint32 N + N float32 audio samples (440 Hz sine, 512 samples)
  - output_ref.bin: uint32 N + N float32 engine outputs (first d_model channel)
  - ref_meta.txt:   model sha256, conditioning vector description, sample rate

The reference is computed by python/reference_forward.py, which implements the
same 8-step Mamba-3 pipeline as src/core/SsmBlock.cpp. tests/test_mamba3.cpp
loads these files and compares its own output against them.

Why this exists: the previous fixture generator (dump_reference_output.py, now a
shim around this script) produced a reference from the R=8 spectral
PerTrackProcessor while the C++ engine implements the R=4 time-domain model.
Those are different architectures, so the comparison could never pass — its own
docstring said so. This script generates a reference from the engine's actual
model, so the test is meaningful.

The weight file is parsed here with the same layout the C++ WeightLoader uses
(64-byte header + 13 arrays per layer), which doubles as a check that the export
format and the loader agree.

Usage:
    python python/tools/dump_engine_reference.py                       # models/model.bin
    python python/tools/dump_engine_reference.py --model models/model.bin --samples 512
    python python/tools/dump_engine_reference.py --seed 42 --force-model  # regenerate weights
"""

import argparse
import hashlib
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from export_mamba_weights import (  # noqa: E402
    BYTES_PER_LAYER,
    FLOATS_PER_LAYER,
    export_weights,
)
from reference_forward import (  # noqa: E402
    COND_DIM,
    D_INNER,
    D_MODEL,
    D_STATE,
    HEADDIM,
    IN_PROJ_DIM,
    MIMO_RANK,
    N_LAYERS,
    NHEADS,
    NUM_ROPE_ANGLES,
    ReferenceMamba3Engine,
    SSMWeights,
)

SAMPLE_RATE = 44100
DEFAULT_SAMPLES = 512
DEFAULT_FREQ = 440.0

# Conditioning vector the C++ test sets (test_mamba3.cpp test_python_reference)
COND_VALUES = {0: 0.5, 40: 0.3}

# Per-layer array shapes in file order — must match WeightLoader::setup_layer_pointers()
LAYER_ARRAY_SHAPES = [
    (IN_PROJ_DIM, D_MODEL),
    (NHEADS,),
    (NHEADS, MIMO_RANK, D_STATE),
    (NHEADS, MIMO_RANK, D_STATE),
    (D_STATE,),
    (D_STATE,),
    (NHEADS, MIMO_RANK, HEADDIM),
    (NHEADS, MIMO_RANK, HEADDIM),
    (NHEADS, MIMO_RANK, HEADDIM),
    (NHEADS,),
    (D_MODEL, D_INNER),
    (2 * D_MODEL, COND_DIM),
    (2 * NHEADS, COND_DIM),
]


def load_model(path: Path) -> list[SSMWeights]:
    """Parse a v2 flat binary weight file into SSMWeights per layer."""
    raw = path.read_bytes()
    if len(raw) < 64:
        raise ValueError(f"{path}: too small to be a weight file ({len(raw)} bytes)")

    (
        magic,
        version,
        d_state,
        d_model,
        d_inner,
        nheads,
        headdim,
        mimo_rank,
        n_layers,
        cond_dim,
        in_proj_dim,
        weight_data_size,
    ) = struct.unpack_from("<12I", raw, 0)

    if magic != 0x4D424D41:
        raise ValueError(
            f"{path}: bad magic 0x{magic:08X} (expected 0x4D424D41 'MBMA'). "
            "If this file contains 'version https://git-lfs.github.com/spec/v1' "
            "it is an unresolved Git LFS pointer, not a model."
        )
    if version != 2:
        raise ValueError(f"{path}: unsupported weight version {version} (expected 2)")
    if (d_state, d_model, d_inner, nheads, headdim, mimo_rank, n_layers, in_proj_dim) != (
        D_STATE,
        D_MODEL,
        D_INNER,
        NHEADS,
        HEADDIM,
        MIMO_RANK,
        N_LAYERS,
        IN_PROJ_DIM,
    ):
        raise ValueError(f"{path}: dimensions do not match the R=4 time-domain engine")
    if cond_dim != COND_DIM:
        raise ValueError(f"{path}: cond_dim {cond_dim} != {COND_DIM}")
    if 64 + weight_data_size != len(raw):
        raise ValueError(
            f"{path}: header says {weight_data_size} weight bytes (+64 header) "
            f"but file is {len(raw)} bytes"
        )

    flat = np.frombuffer(raw, dtype="<f4", offset=64)
    expected_floats = N_LAYERS * FLOATS_PER_LAYER
    if flat.size != expected_floats:
        raise ValueError(f"{path}: expected {expected_floats} floats, found {flat.size}")

    layers = []
    offset = 0
    for _ in range(N_LAYERS):
        arrays = []
        for shape in LAYER_ARRAY_SHAPES:
            count = int(np.prod(shape))
            arrays.append(torch.from_numpy(flat[offset : offset + count].reshape(shape).copy()))
            offset += count
        layers.append(
            SSMWeights(
                in_proj_w=arrays[0],
                dt_bias=arrays[1],
                B_bias=arrays[2],
                C_bias=arrays[3],
                B_norm_weight=arrays[4],
                C_norm_weight=arrays[5],
                mimo_x=arrays[6],
                mimo_z=arrays[7],
                mimo_o=arrays[8],
                D_proj=arrays[9],
                out_proj_w=arrays[10],
                film_proj_w=arrays[11],
                dt_film_proj_w=arrays[12],
            )
        )
    assert offset == expected_floats
    _ = (BYTES_PER_LAYER,)  # documented relationship; asserted via file size above
    return layers


def write_bin(filepath: Path, data: np.ndarray) -> None:
    flat = np.ascontiguousarray(data, dtype="<f4").reshape(-1)
    with open(filepath, "wb") as f:
        f.write(struct.pack("<I", flat.size))
        f.write(flat.tobytes())
    print(f"  wrote {filepath} ({flat.size} floats, {flat.size * 4:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="models/model.bin", help="Weight file to reference")
    parser.add_argument("--output-dir", default=None, help="Defaults to the model's directory")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=42, help="Seed if the model must be generated")
    parser.add_argument(
        "--force-model",
        action="store_true",
        help="Regenerate the weight file even if it exists (overwrites; dev only)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.force_model or not model_path.exists():
        print(f"Generating weights (seed={args.seed}) → {model_path}")
        export_weights(str(model_path), args.seed)
    elif model_path.read_bytes()[:8] == b"version ":
        raise SystemExit(
            f"{model_path} is a Git LFS pointer, not a model. "
            "This repository stores weights directly in Git since 2026-10-04; "
            "re-clone or run with --force-model to regenerate fixtures."
        )

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    print(f"Model: {model_path} ({model_path.stat().st_size:,} bytes)")
    print(f"  sha256: {digest}")

    layers = load_model(model_path)

    engine = ReferenceMamba3Engine(device="cpu")
    engine.weights = layers  # use the shipped weights, not fresh random ones

    cond = torch.zeros(COND_DIM)
    for idx, value in COND_VALUES.items():
        cond[idx] = value
    print("  conditioning: " + ", ".join(f"cond[{i}]={v}" for i, v in COND_VALUES.items()))

    t = torch.arange(args.samples, dtype=torch.float32) / SAMPLE_RATE
    audio = torch.sin(2.0 * torch.pi * DEFAULT_FREQ * t) * 0.5

    engine.reset_states()
    with torch.no_grad():
        out = engine.process_block(audio, cond)

    print(f"  input:  {args.samples} samples, {DEFAULT_FREQ} Hz sine")
    print(
        f"  output: std={out.std().item():.6f} max={out.abs().max().item():.6f} "
        f"finite={torch.isfinite(out).all().item()}"
    )

    write_bin(output_dir / "input_ref.bin", audio.numpy())
    write_bin(output_dir / "output_ref.bin", out.numpy())

    meta = output_dir / "ref_meta.txt"
    with open(meta, "w") as f:
        f.write("reference=python/reference_forward.py (R=4 time-domain, 8-step pipeline)\n")
        f.write(f"model={model_path.name}\n")
        f.write(f"model_sha256={digest}\n")
        f.write(f"sample_rate={SAMPLE_RATE}\n")
        f.write(f"samples={args.samples}\n")
        f.write(f"conditioning={COND_VALUES}\n")
        f.write(f"output_std={out.std().item():.8f}\n")
    print(f"  wrote {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
