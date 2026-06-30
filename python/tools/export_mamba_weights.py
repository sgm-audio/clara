#!/usr/bin/env python3
"""
Export Mamba-3 weights to flat binary format (Version 2).

Reads a reference Mamba-3 model (from reference_forward.py SSMWeights format),
writes a single .bin file with 64-byte header + 13 arrays per layer × n_layers.

Format matches src/core/MambaConfig.h constants:
  Header: magic("MBMA"), version(2), dims[9], reserved[5], weight_data_size
  Per layer: 13 float32 arrays in order

Usage:
    python python/tools/export_mamba_weights.py --output models/model.bin
    python python/tools/export_mamba_weights.py --output models/model.bin --seed 42
"""

import struct
import argparse
import numpy as np
from pathlib import Path


# ── Constants (must match src/core/MambaConfig.h) ──────────────────

D_STATE       = 32
D_MODEL       = 128
D_INNER       = 256
NHEADS        = 4
HEADDIM       = D_INNER // NHEADS       # 64
MIMO_RANK     = 4
NUM_ROPE_ANGLES = D_STATE // 2          # 16
N_LAYERS      = 6
COND_DIM      = 128

IN_PROJ_DIM = (D_INNER
               + NHEADS * HEADDIM
               + NHEADS * MIMO_RANK * D_STATE   # B
               + NHEADS * MIMO_RANK * D_STATE   # C
               + NHEADS                          # dd_dt
               + NHEADS                          # dd_A
               + NHEADS                          # trap
               + NHEADS * NUM_ROPE_ANGLES)       # angles

WEIGHT_MAGIC   = 0x4D424D41  # "MBMA"
WEIGHT_VERSION = 2

# Per-layer array element counts (float32)
W_IN_PROJ_ELEMS       = IN_PROJ_DIM * D_MODEL             # 1612 × 128 = 206336
W_DT_BIAS_ELEMS       = NHEADS                            # 4
W_B_BIAS_ELEMS        = NHEADS * MIMO_RANK * D_STATE      # 512
W_C_BIAS_ELEMS        = W_B_BIAS_ELEMS                    # 512
W_B_NORM_ELEMS        = D_STATE                           # 32
W_C_NORM_ELEMS        = D_STATE                           # 32
W_MIMO_X_ELEMS        = NHEADS * MIMO_RANK * HEADDIM      # 1024
W_MIMO_Z_ELEMS        = W_MIMO_X_ELEMS                    # 1024
W_MIMO_O_ELEMS        = W_MIMO_X_ELEMS                    # 1024
W_D_PROJ_ELEMS        = NHEADS                            # 4
W_OUT_PROJ_ELEMS      = D_MODEL * D_INNER                 # 32768
W_FILM_PROJ_ELEMS     = 2 * D_MODEL * COND_DIM            # 32768
W_DT_FILM_PROJ_ELEMS  = 2 * NHEADS * COND_DIM             # 1024

FLOATS_PER_LAYER = (W_IN_PROJ_ELEMS + W_DT_BIAS_ELEMS + W_B_BIAS_ELEMS
                    + W_C_BIAS_ELEMS + W_B_NORM_ELEMS + W_C_NORM_ELEMS
                    + W_MIMO_X_ELEMS + W_MIMO_Z_ELEMS + W_MIMO_O_ELEMS
                    + W_D_PROJ_ELEMS + W_OUT_PROJ_ELEMS + W_FILM_PROJ_ELEMS
                    + W_DT_FILM_PROJ_ELEMS)

BYTES_PER_LAYER = FLOATS_PER_LAYER * 4


def generate_random_weights(seed: int = 42):
    """Generate random Mamba-3 weights matching reference_forward.py SSMWeights layout."""
    rng = np.random.default_rng(seed)
    layers = []
    for _ in range(N_LAYERS):
        layer = []

        # Xavier-uniform-ish init
        def rand_xavier(*shape):
            fan_in = shape[-1] if len(shape) >= 2 else shape[0]
            bound = np.sqrt(6.0 / max(fan_in, 1))
            return (rng.random(shape, dtype=np.float32) * 2 - 1) * bound

        # 0. in_proj_w [IN_PROJ_DIM, D_MODEL]
        layer.append(rand_xavier(IN_PROJ_DIM, D_MODEL).reshape(-1))

        # 1. dt_bias [NHEADS]
        layer.append(np.zeros(NHEADS, dtype=np.float32))

        # 2. B_bias [NHEADS, MIMO_RANK, D_STATE]
        layer.append(rand_xavier(NHEADS, MIMO_RANK, D_STATE).reshape(-1))

        # 3. C_bias
        layer.append(rand_xavier(NHEADS, MIMO_RANK, D_STATE).reshape(-1))

        # 4. B_norm_weight [D_STATE]
        layer.append(np.ones(D_STATE, dtype=np.float32))

        # 5. C_norm_weight [D_STATE]
        layer.append(np.ones(D_STATE, dtype=np.float32))

        # 6. mimo_x [NHEADS, MIMO_RANK, HEADDIM]
        layer.append(rand_xavier(NHEADS, MIMO_RANK, HEADDIM).reshape(-1))

        # 7. mimo_z
        layer.append(rand_xavier(NHEADS, MIMO_RANK, HEADDIM).reshape(-1))

        # 8. mimo_o
        layer.append(rand_xavier(NHEADS, MIMO_RANK, HEADDIM).reshape(-1))

        # 9. D_proj [NHEADS]
        layer.append(np.zeros(NHEADS, dtype=np.float32))

        # 10. out_proj_w [D_MODEL, D_INNER]
        layer.append(rand_xavier(D_MODEL, D_INNER).reshape(-1))

        # 11. film_proj_w [2 * D_MODEL, COND_DIM]
        layer.append(rand_xavier(2 * D_MODEL, COND_DIM).reshape(-1))

        # 12. dt_film_proj_w [2 * NHEADS, COND_DIM]
        layer.append(rand_xavier(2 * NHEADS, COND_DIM).reshape(-1))

        layers.append(layer)
    return layers


def write_header(f, weight_data_size: int):
    """Write 64-byte file header."""
    header_values = [
        WEIGHT_MAGIC,
        WEIGHT_VERSION,
        D_STATE,
        D_MODEL,
        D_INNER,
        NHEADS,
        HEADDIM,
        MIMO_RANK,
        N_LAYERS,
        COND_DIM,
        IN_PROJ_DIM,
        weight_data_size,
    ]
    # Write as 12 uint32 values (48 bytes)
    for v in header_values:
        f.write(struct.pack('<I', v))
    # Pad to 64 bytes with zeros
    f.write(b'\x00' * (64 - len(header_values) * 4))


def export_weights(output_path: str, seed: int = 42):
    """Export random Mamba-3 weights to flat binary file."""
    layers = generate_random_weights(seed)
    weight_data_size = N_LAYERS * BYTES_PER_LAYER

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        write_header(f, weight_data_size)

        for layer_idx, arrays in enumerate(layers):
            for arr_idx, arr in enumerate(arrays):
                arr_flat = np.ascontiguousarray(arr, dtype=np.float32)
                f.write(arr_flat.tobytes())

    file_size = Path(output_path).stat().st_size
    expected_size = 64 + weight_data_size
    assert file_size == expected_size, (
        f"Size mismatch: wrote {file_size}, expected {expected_size}"
    )

    print(f"Exported {output_path}")
    print(f"  Layers: {N_LAYERS}")
    print(f"  Arrays per layer: {len(layers[0])}")
    print(f"  Floats per layer: {FLOATS_PER_LAYER:,}")
    print(f"  Total weight bytes: {weight_data_size:,}")
    print(f"  File size: {file_size:,} bytes")


def main():
    parser = argparse.ArgumentParser(
        description="Export Mamba-3 weights to flat binary (Version 2)"
    )
    parser.add_argument('--output', default='models/model.bin',
                        help='Output .bin file path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for weight generation')
    args = parser.parse_args()

    export_weights(args.output, args.seed)


if __name__ == '__main__':
    main()
