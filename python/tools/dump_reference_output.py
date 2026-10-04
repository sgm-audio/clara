#!/usr/bin/env python3
"""DEPRECATED — use dump_engine_reference.py instead.

This script used to generate the Python↔C++ comparison fixtures from
`PerTrackProcessor` (R=8, spectral domain). The C++ engine implements the R=4
time-domain model, so the fixtures it produced were never comparable — its own
docstring noted that "bit-exact Python↔C++ comparison is NOT possible". The
"bit-exact" CTest target could therefore never pass.

python/tools/dump_engine_reference.py now generates fixtures from the model the
engine actually runs. This shim forwards to it so existing invocations keep
working.

Usage (same as the new script):
    python3 python/tools/dump_reference_output.py --model models/model.bin
"""

import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Deprecated shim — see module docstring")
    parser.add_argument("--model", default="models/model.bin")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=512)
    args, _unknown = parser.parse_known_args()

    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dump_engine_reference.py")
    print("NOTE: dump_reference_output.py is deprecated; running dump_engine_reference.py instead.")
    cmd = [
        sys.executable,
        target,
        "--model",
        args.model,
        "--seed",
        str(args.seed),
        "--samples",
        str(args.samples),
    ]
    if args.output_dir:
        cmd += ["--output-dir", args.output_dir]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
