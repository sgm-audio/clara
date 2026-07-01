#!/usr/bin/env python3
"""filter_audio_quality.py — Scan audio files and filter by quality metrics.

Output:
  pass_list.txt   — newline-delimited list of files that passed all checks
  reject_report.csv — CSV with file path and rejection reasons

Thresholds are tunable. Defaults target identity-training-quality audio:
  - Peak < -0.1 dBFS   (no clipping)
  - RMS  > -40 dB       (not silent / near-silent)
  - DC offset < 1e-4    (no DC bias)
  - Clip ratio < 0.1%   (no hard clipping)
  - Duration > 1.0 sec  (not corrupted / single-sample files)

Usage:
    python scripts/filter_audio_quality.py ~/datasets/fma/
    python scripts/filter_audio_quality.py ~/datasets/fma/ --min-rms-db -30
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf


def analyze_file(path: Path) -> dict:
    """Return quality metrics dict. Contains 'error' key + pass=False on failure."""
    metrics: dict = {
        "path": str(path),
        "error": None,
        "pass": True,
        "sr": 0,
        "duration": 0.0,
        "peak_db": 0.0,
        "rms_db": 0.0,
        "dc_offset": 0.0,
        "clip_ratio": 0.0,
    }

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:
        metrics["error"] = f"load: {e}"
        metrics["pass"] = False
        return metrics

    # Convert to mono
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]

    n = len(data)
    if n == 0:
        metrics["error"] = "zero-length"
        metrics["pass"] = False
        return metrics

    metrics["sr"] = int(sr)
    metrics["duration"] = n / max(sr, 1)

    peak = float(np.max(np.abs(data)))
    metrics["peak_db"] = 20.0 * np.log10(max(peak, 1e-12))

    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    metrics["rms_db"] = 20.0 * np.log10(max(rms, 1e-12))

    metrics["dc_offset"] = float(np.mean(data))

    clip_count = int(np.sum(np.abs(data) > 0.999))
    metrics["clip_ratio"] = clip_count / max(n, 1)

    return metrics


CSV_HEADER = ["file", "reason", "peak_db", "rms_db", "dc_offset", "clip_ratio", "duration_s"]


def write_reject_row(writer, metrics: dict, reasons: str) -> None:
    writer.writerow(
        [
            metrics["path"],
            reasons,
            f"{metrics['peak_db']:.2f}",
            f"{metrics['rms_db']:.2f}",
            f"{metrics['dc_offset']:.6f}",
            f"{metrics['clip_ratio']:.4f}",
            f"{metrics['duration']:.1f}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter audio files by quality — output pass list for training"
    )
    parser.add_argument("input_dir", help="Directory to scan (recursive)")
    parser.add_argument(
        "--output",
        default="pass_list.txt",
        help="Output file with passing file paths (default: pass_list.txt)",
    )
    parser.add_argument(
        "--reject-log",
        default="reject_report.csv",
        help="CSV report of rejected files (default: reject_report.csv)",
    )
    parser.add_argument(
        "--max-peak-db", type=float, default=-0.1, help="Reject if peak > this dBFS (default: -0.1)"
    )
    parser.add_argument(
        "--min-rms-db", type=float, default=-40.0, help="Reject if RMS < this dB (default: -40)"
    )
    parser.add_argument(
        "--max-dc-offset",
        type=float,
        default=1e-4,
        help="Reject if abs(DC offset) > this (default: 1e-4)",
    )
    parser.add_argument(
        "--max-clip-ratio",
        type=float,
        default=0.001,
        help="Reject if clipped sample ratio > this (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Reject if shorter than this in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=[".wav", ".flac", ".WAV", ".FLAC"],
        help="File extensions to scan (default: .wav .flac)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file progress output")
    args = parser.parse_args()

    root = Path(args.input_dir)
    if not root.is_dir():
        print(f"ERROR: Directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all audio files (recursive, deduplicated)
    seen = set()
    files: list[Path] = []
    for ext in args.ext:
        for f in root.rglob(f"*{ext}"):
            if f.is_file() and f.resolve() not in seen:
                seen.add(f.resolve())
                files.append(f)
    files.sort()

    total = len(files)
    if total == 0:
        print("ERROR: No audio files found", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {total} audio files from {args.input_dir}")
    print(
        f"Thresholds: peak<{args.max_peak_db}dBFS  RMS>{args.min_rms_db}dB  "
        f"DC<{args.max_dc_offset}  clip<{args.max_clip_ratio}  dur>{args.min_duration}s"
    )
    print()

    passed: list[str] = []
    reject_reasons: list[str] = []  # for summary counter

    with open(args.reject_log, "w", newline="") as rf:
        writer = csv.writer(rf)
        writer.writerow(CSV_HEADER)

        for i, path in enumerate(files):
            if not args.quiet and i % max(1, total // 20) == 0:
                print(f"  {i}/{total}...", end="\r", flush=True)

            m = analyze_file(path)

            # Unreadable / corrupted
            if m.get("error"):
                reject_reasons.append("load_error")
                write_reject_row(writer, m, f"load_error: {m['error']}")
                continue

            reasons: list[str] = []
            if m["peak_db"] > args.max_peak_db:
                reasons.append(f"peak={m['peak_db']:.1f}dB")
            if m["rms_db"] < args.min_rms_db:
                reasons.append(f"rms={m['rms_db']:.1f}dB")
            if abs(m["dc_offset"]) > args.max_dc_offset:
                reasons.append(f"dc={m['dc_offset']:.6f}")
            if m["clip_ratio"] > args.max_clip_ratio:
                reasons.append(f"clip={m['clip_ratio']:.4f}")
            if m["duration"] < args.min_duration:
                reasons.append(f"dur={m['duration']:.1f}s")

            if reasons:
                reason_str = "; ".join(reasons)
                reject_reasons.append(reason_str)
                write_reject_row(writer, m, reason_str)
            else:
                passed.append(str(path.resolve()))

    if not args.quiet:
        print(f"  {total}/{total}    ")

    n_pass = len(passed)
    n_reject = len(reject_reasons)
    print(
        f"\nResults: {n_pass} passed, {n_reject} rejected "
        f"({100 * n_pass / max(total, 1):.1f}% pass rate)"
    )

    # Write pass list
    with open(args.output, "w") as f:
        f.write("\n".join(passed) + "\n")
    print(f"Pass list written to: {args.output}")
    print(f"Reject report:        {args.reject_log}")

    # Reject summary by category
    if reject_reasons:
        # Group by first reason keyword
        keywords: Counter = Counter()
        for r in reject_reasons:
            keyword = r.split(";")[0].split("=")[0].split(":")[0].strip()
            keywords[keyword] += 1
        print("\nReject breakdown:")
        for kw, count in keywords.most_common():
            print(f"  {kw}: {count}")


if __name__ == "__main__":
    main()
