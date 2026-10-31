#!/usr/bin/env python3
"""prep_dataset.py — Build unified training dataset via symlink farm.

Accepts multiple sources (directories or pass-list .txt files), creates
a flat directory of uniquely-named symlinks for training consumption.

Sources:
  - Directory:  recursively finds *.wav + *.flac files
  - .txt file:  reads as pass list (one absolute path per line)

Output: symlinks at <output_dir>/ named as {label}_{index:06d}.{ext}
  Example: fma_000042.flac → /data/fma/track_42.flac

Usage:
    python scripts/prep_dataset.py \
        --source musdb ~/datasets/musdb18hq/train \
        --source fma pass_list.txt \
        --output ~/datasets/clara_phase1
"""
import argparse
import os
import sys
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".WAV", ".FLAC"}


def collect_files(source_label: str, source_path: str) -> list[tuple[str, Path]]:
    """Return list of (label, resolved_path) tuples from a source.

    source_path may be:
      - A directory: recursively find all WAV/FLAC files
      - A .txt file: read as pass list (one path per line)
    """
    src = Path(source_path).expanduser().resolve()
    files: list[tuple[str, Path]] = []

    if src.is_file() and src.suffix == ".txt":
        # Pass list — one absolute path per line
        print(f"  [{source_label}] Reading pass list: {src}")
        with open(src) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = Path(line).expanduser().resolve()
                if p.is_file() and p.suffix in AUDIO_EXTS:
                    files.append((source_label, p))
        print(f"  [{source_label}] {len(files)} valid files from pass list")

    elif src.is_dir():
        # Directory — recursive find
        print(f"  [{source_label}] Scanning directory: {src}")
        seen = set()
        for ext in AUDIO_EXTS:
            for f in src.rglob(f"*{ext}"):
                if f.is_file():
                    resolved = f.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        files.append((source_label, resolved))
        print(f"  [{source_label}] {len(files)} files found")

    else:
        print(
            f"  [{source_label}] WARNING: {src} is not a directory or .txt file — skipping",
            file=sys.stderr,
        )

    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified training dataset via symlink farm")
    parser.add_argument(
        "--source",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        dest="sources",
        required=True,
        help="Add a source: LABEL (short name) and PATH (dir or pass-list.txt). "
        "Repeat for multiple sources.",
    )
    parser.add_argument("--output", required=True, help="Output directory for symlink farm")
    parser.add_argument(
        "--dry-run", action="store_true", help="Collect files but do not create symlinks"
    )
    args = parser.parse_args()

    # Collect all files from all sources
    all_files: list[tuple[str, Path]] = []
    for label, src_path in args.sources:
        files = collect_files(label, src_path)
        all_files.extend(files)

    if not all_files:
        print("ERROR: No files collected from any source", file=sys.stderr)
        sys.exit(1)

    # Sort for deterministic naming
    all_files.sort(key=lambda x: (x[0], str(x[1])))

    # Count per source
    from collections import Counter

    source_counts: Counter = Counter()
    for label, _ in all_files:
        source_counts[label] += 1
    print(f"\nTotal files: {len(all_files)}")
    for label, count in source_counts.most_common():
        print(f"  {label}: {count}")

    # Estimate disk usage (symlinks are tiny, but report source size)
    total_size_bytes = 0
    size_errors = 0
    for _, path in all_files:
        try:
            total_size_bytes += path.stat().st_size
        except OSError:
            size_errors += 1
    if size_errors:
        print(f"  ({size_errors} files could not be stat'd for size)")
    total_gb = total_size_bytes / (1024**3)
    print(f"  Estimated source size: {total_gb:.1f} GB")

    if args.dry_run:
        print("\n[Dry run — no symlinks created]")
        return

    # Create output directory
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Counter per source for unique naming
    counters: Counter = Counter()

    created = 0
    errors = 0
    for label, src_path in all_files:
        idx = counters[label]
        counters[label] += 1
        ext = src_path.suffix.lower()  # .wav, .flac
        name = f"{label}_{idx:06d}{ext}"
        dst = out / name

        # Remove existing symlink or stale file
        if dst.exists() or dst.is_symlink():
            dst.unlink()

        try:
            os.symlink(src_path, dst)
            created += 1
        except OSError as e:
            print(f"  ERROR: symlink {name} → {src_path}: {e}", file=sys.stderr)
            errors += 1
            if errors > 10:
                print("  Too many errors — aborting", file=sys.stderr)
                sys.exit(1)

    print(f"\nSymlinks created: {created}")
    if errors:
        print(f"Errors: {errors}")
    print(f"Output directory: {out}")
    print("\nTrain command:")
    print(f"  python train_clara.py --data {out} --device cuda")


if __name__ == "__main__":
    main()
