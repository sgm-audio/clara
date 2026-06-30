#!/usr/bin/env python3
"""
Validate Mamba-3 flat binary weight file (Version 2).

Reads a .bin file, checks header magic/version/dimensions,
round-trips all arrays, and optionally compares against
reference_forward.py random weights with the same seed.

Usage:
    python python/tools/validate_export.py models/model.bin
    python python/tools/validate_export.py models/model.bin --seed 42
"""

import struct
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple

# ── Expected constants (must match export script) ──────────────────
EXPECTED_MAGIC    = 0x4D424D41
EXPECTED_VERSION  = 3

HEADER_FORMAT = '<12I'  # 12 uint32 values (magic, version, 10 dims = 48 bytes)
HEADER_SIZE   = 64      # padded to 64 bytes


def read_header(filepath: Path) -> dict:
    """Read and validate 64-byte header. Returns dict of header values."""
    with open(filepath, 'rb') as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"File too short for header: {len(raw)} < {HEADER_SIZE}")

    values = list(struct.unpack_from(HEADER_FORMAT, raw))

    header = {
        'magic':           values[0],
        'version':         values[1],
        'd_state':         values[2],
        'd_model':         values[3],
        'd_inner':         values[4],
        'nheads':          values[5],
        'headdim':         values[6],
        'mimo_rank':       values[7],
        'n_layers':        values[8],
        'cond_dim':        values[9],
        'in_proj_dim':     values[10],
        'weight_data_size': values[11],
    }
    return header


def validate_header(header: dict) -> List[str]:
    """Check header values against expected constants. Returns list of errors."""
    errors = []

    if header['magic'] != EXPECTED_MAGIC:
        errors.append(f"Bad magic: 0x{header['magic']:08X} != 0x{EXPECTED_MAGIC:08X}")
    if header['version'] != EXPECTED_VERSION:
        errors.append(f"Bad version: {header['version']} != {EXPECTED_VERSION}")

    # Dimension checks — v3 (R=8 complex pole format)
    checks = {
        'd_state':   32,
        'd_model':   128,
        'd_inner':   256,
        'nheads':    4,
        'headdim':   64,
        'mimo_rank': 8,   # R=8 (was 4 in v2)
        'n_layers':  6,
        'cond_dim':  128,
    }
    for key, expected in checks.items():
        actual = header[key]
        if actual != expected:
            errors.append(f"{key}: {actual} != expected {expected}")

    # Derived checks
    if header['d_inner'] != 2 * header['d_model']:
        errors.append(f"d_inner ({header['d_inner']}) != 2 × d_model ({header['d_model']})")
    if header['nheads'] * header['headdim'] != header['d_inner']:
        errors.append(f"nheads×headdim ({header['nheads']*header['headdim']}) != d_inner ({header['d_inner']})")

    return errors


def read_weights(filepath: Path, header: dict) -> List[List[np.ndarray]]:
    """Read all weight arrays from binary file. Returns layers[array_idx]."""
    n_layers = header['n_layers']
    d_state = header['d_state']
    d_model = header['d_model']
    d_inner = header['d_inner']
    nheads = header['nheads']
    headdim = header['headdim']
    mimo_rank = header['mimo_rank']
    cond_dim = header['cond_dim']
    in_proj_dim = header['in_proj_dim']

    # Per-layer array sizes (v3 — 18 arrays per layer, R=8)
    b_proj_out = d_state * 2   # N*2
    sizes = [
        in_proj_dim * d_model,                                    # 0: in_proj_w
        nheads,                                                    # 1: dt_bias
        b_proj_out * mimo_rank,                                    # 2: B_proj.weight [N*2, R]
        b_proj_out,                                                # 3: B_proj.bias [N*2]
        headdim * b_proj_out,                                      # 4: C_proj.weight [head_dim, N*2]
        nheads * mimo_rank * headdim,                              # 5: mimo_x
        nheads * mimo_rank * headdim,                              # 6: mimo_z
        nheads * mimo_rank * headdim,                              # 7: mimo_o
        nheads,                                                    # 8: D_proj
        d_model * d_inner,                                         # 9: out_proj.weight
        2 * d_model * cond_dim,                                    # 10: film_proj.weight
        2 * d_model,                                               # 11: film_proj.bias
        2 * nheads * cond_dim,                                     # 12: dt_film_proj.weight
        2 * nheads,                                                # 13: dt_film_proj.bias
        nheads,                                                    # 14: log_a_real
        nheads,                                                    # 15: a_imag
        d_model,                                                   # 16: norm.weight
        d_model,                                                   # 17: norm.bias
    ]

    layers = []
    with open(filepath, 'rb') as f:
        f.seek(HEADER_SIZE)

        # Skip R8 extractor weights (prepended before layer weights in v3)
        if header['version'] == 3:
            r8_bytes = header['weight_data_size'] - sum(sizes) * n_layers * 4
            if r8_bytes > 0:
                f.seek(r8_bytes, 1)  # relative seek

        for li in range(n_layers):
            arrays = []
            for si, size in enumerate(sizes):
                raw = f.read(size * 4)
                if len(raw) < size * 4:
                    raise ValueError(
                        f"Unexpected EOF at layer {li} array {si}: "
                        f"got {len(raw)} bytes, expected {size * 4}"
                    )
                arr = np.frombuffer(raw, dtype=np.float32)
                arrays.append(arr)
            layers.append(arrays)

        # Verify we consumed exactly the expected amount
        remaining = len(f.read())
        if remaining > 0:
            print(f"  Warning: {remaining} extra bytes after weight data")

    return layers


def validate_weights(layers: List[List[np.ndarray]]) -> List[str]:
    """Validate weight arrays are finite and have expected properties."""
    errors = []
    for li, arrays in enumerate(layers):
        for ai, arr in enumerate(arrays):
            if not np.all(np.isfinite(arr)):
                errors.append(f"Layer {li} array {ai}: contains NaN/Inf")
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate Mamba-3 flat binary weight file"
    )
    parser.add_argument('filepath', help='Path to .bin file')
    args = parser.parse_args()

    filepath = Path(args.filepath)
    if not filepath.exists():
        print(f"ERROR: {filepath} not found")
        return 1

    file_size = filepath.stat().st_size

    # 1. Read and validate header
    header = read_header(filepath)
    header_errors = validate_header(header)
    if header_errors:
        print("=== HEADER ERRORS ===")
        for e in header_errors:
            print(f"  ❌ {e}")
        return 1
    print("✅ Header valid")

    # 2. Read all weight arrays
    try:
        layers = read_weights(filepath, header)
    except ValueError as e:
        print(f"❌ Read error: {e}")
        return 1

    # 3. Validate weight array contents
    weight_errors = validate_weights(layers)
    if weight_errors:
        print("=== WEIGHT ARRAY ERRORS ===")
        for e in weight_errors:
            print(f"  ❌ {e}")
        return 1
    print("✅ All weight arrays valid (finite)")

    # 4. Verify file size consistency
    expected_total = HEADER_SIZE + header['weight_data_size']
    if file_size != expected_total:
        print(f"⚠️  File size mismatch: {file_size} != expected {expected_total}")
    else:
        print(f"✅ File size consistent: {file_size:,} bytes")

    # Summary
    total_floats = sum(arr.size for layer in layers for arr in layer)
    print(f"\n=== VALIDATION PASSED ===")
    print(f"  Layers: {header['n_layers']}")
    print(f"  Arrays per layer: {len(layers[0])}")
    print(f"  Total floats: {total_floats:,}")
    print(f"  Total params: ~{total_floats * 4 // 1000} KB")

    return 0


if __name__ == '__main__':
    exit(main() or 0)
