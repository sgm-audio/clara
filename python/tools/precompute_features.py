#!/usr/bin/env python3
"""Precompute audio features for CLARA training — run once, use forever.

Computes and saves mel spectrograms + STFT magnitudes for all audio files.
The training dataset loads precomputed features from disk (or RAM cache),
eliminating all CPU-side DSP computation during training.

Usage (one-time preprocessing):
    python python/tools/precompute_features.py \
        --input ~/datasets/musdb18hq/train \
        --output ~/datasets/musdb18hq/precomputed \
        --chunk-seconds 0.5 \
        --sample-rate 44100 \
        --n-fft 1024 \
        --hop-length 256 \
        --n-mels 64 \
        --workers 8

Output structure:
    output/
    ├── metadata.json          # dataset stats, feature shapes
    ├── train/
    │   ├── track_001_chunk_000.npz
    │   ├── track_001_chunk_001.npz
    │   └── ...
    └── (same for 'test' if --test-input provided)

Each .npz contains:
    mel:     [n_mel_frames, n_mels]    float32  — log-mel spectrogram
    stft:    [n_fft//2+1, n_stft_frames] float32 — STFT magnitude
    audio:   [chunk_samples]            float32  — raw audio (target)
    sr:      int                        — sample rate
    offset:  int                        — start sample in original file
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# ── Feature computation ──────────────────────────────────────────────


def compute_mel(audio: np.ndarray, sr: int, n_fft: int, hop_length: int, n_mels: int) -> np.ndarray:
    """Compute log-mel spectrogram. Returns [n_frames, n_mels]."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0.0,
        f_max=sr / 2.0,
        norm="slaney",
        mel_scale="htk",
    )
    with torch.no_grad():
        t = torch.from_numpy(audio).float()
        if t.dim() == 1:
            t = t.unsqueeze(0)
        mel = mel_transform(t)  # [1, n_mels, n_frames]
        mel = mel.log1p()  # log(1 + mel), same as training loss
    return mel.squeeze(0).T.numpy()  # [n_frames, n_mels]


def compute_stft(audio: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Compute STFT magnitude. Returns [n_fft//2+1, n_frames]."""
    window = torch.hann_window(n_fft)
    with torch.no_grad():
        t = torch.from_numpy(audio).float()
        if t.dim() == 1:
            t = t.unsqueeze(0)
        stft = torch.stft(t, n_fft, hop_length=hop_length, window=window, return_complex=True)
        mag = stft.abs()  # [1, n_fft//2+1, n_frames]
    return mag.squeeze(0).numpy()  # [n_fft//2+1, n_frames]


def process_file(args) -> dict:
    """Process one audio file, returns list of chunk dicts."""
    (
        filepath,
        output_dir,
        chunk_seconds,
        sr,
        n_fft,
        hop_length,
        n_mels,
        file_index,
        total_files,
    ) = args

    try:
        audio, file_sr = sf.read(str(filepath), dtype="float32", always_2d=True)
    except Exception as e:
        return {"file": str(filepath), "error": str(e), "chunks": []}

    # Mix to mono
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)

    # Resample if needed
    if file_sr != sr:
        t = torch.from_numpy(audio).float().unsqueeze(0)
        audio = torchaudio.functional.resample(t, file_sr, sr).squeeze(0).numpy()

    # Normalize to prevent clipping
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9

    chunk_samples = int(chunk_seconds * sr)
    n_chunks = max(1, len(audio) // chunk_samples)

    chunks = []
    for ci in range(n_chunks):
        start = ci * chunk_samples
        end = start + chunk_samples
        chunk_audio = audio[start:end]

        # Pad if final chunk is short
        if len(chunk_audio) < chunk_samples:
            chunk_audio = np.pad(chunk_audio, (0, chunk_samples - len(chunk_audio)))

        # Compute features
        mel = compute_mel(chunk_audio, sr, n_fft, hop_length, n_mels)
        stft = compute_stft(chunk_audio, n_fft, hop_length)

        # Save chunk
        stem = Path(filepath).stem
        out_name = f"file_{file_index:04d}_{stem}_c{ci:04d}.npz"
        out_path = Path(output_dir) / out_name

        np.savez_compressed(
            out_path,
            mel=mel.astype(np.float32),
            stft=stft.astype(np.float32),
            audio=chunk_audio,
            sr=sr,
            offset=start,
        )

        chunks.append(
            {
                "file": str(filepath),
                "chunk_index": ci,
                "out_path": str(out_path),
                "mel_shape": mel.shape,
                "stft_shape": stft.shape,
                "audio_shape": chunk_audio.shape,
            }
        )

    return {
        "file": str(filepath),
        "file_index": file_index,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Precompute audio features for CLARA training")
    parser.add_argument("--input", required=True, help="Input audio directory (train)")
    parser.add_argument("--test-input", help="Optional test audio directory")
    parser.add_argument("--output", required=True, help="Output precomputed directory")
    parser.add_argument(
        "--chunk-seconds", type=float, default=0.5, help="Chunk duration in seconds (default: 0.5)"
    )
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for preprocessing")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have precomputed output",
    )
    args = parser.parse_args()

    sr = args.sample_rate
    n_fft = args.n_fft
    hop_length = args.hop_length
    n_mels = args.n_mels
    chunk_seconds = args.chunk_seconds
    chunk_samples = int(chunk_seconds * sr)

    print("CLARA Feature Precomputation")
    print(f"{'='*50}")
    print(f"Input:        {args.input}")
    print(f"Output:       {args.output}")
    print(f"Chunk:        {chunk_seconds}s = {chunk_samples} samples")
    print(f"SR:           {sr}")
    print(f"STFT:         n_fft={n_fft}, hop={hop_length}")
    print(f"Mel:          n_mels={n_mels}")
    print(f"Workers:      {args.workers}")
    print()

    # Find audio files
    def find_audio(root):
        return sorted(
            [f for f in Path(root).rglob("*.wav") if f.is_file()]
            + [f for f in Path(root).rglob("*.flac") if f.is_file()]
        )

    train_files = find_audio(args.input)
    print(f"Train files:  {len(train_files)}")
    test_files = find_audio(args.test_input) if args.test_input else []
    print(f"Test files:   {len(test_files)}")
    if not train_files:
        print("ERROR: No audio files found")
        return 1

    # Create output dirs
    train_out = Path(args.output) / "train"
    test_out = Path(args.output) / "test" if test_files else None
    train_out.mkdir(parents=True, exist_ok=True)
    if test_out:
        test_out.mkdir(parents=True, exist_ok=True)

    # Build work items
    def make_work(files, output_dir):
        items = []
        for fi, fp in enumerate(files):
            if args.skip_existing:
                # Check if we already processed this file (rough check)
                stem = fp.stem
                existing = list(output_dir.glob(f"*{stem}*.npz"))
                if existing:
                    continue
            items.append(
                (fp, str(output_dir), chunk_seconds, sr, n_fft, hop_length, n_mels, fi, len(files))
            )
        return items

    train_work = make_work(train_files, train_out)
    test_work = make_work(test_files, test_out) if test_files else []
    all_work = train_work + test_work

    print(f"Files to process: {len(all_work)}")
    print(f"Output dir: {args.output}")
    print()

    if not all_work:
        print("All files already precomputed (use --skip-existing to skip)")
        # Still write metadata
        write_metadata(
            args.output,
            train_out,
            test_out,
            train_files,
            test_files,
            chunk_seconds,
            sr,
            n_fft,
            hop_length,
            n_mels,
            chunk_samples,
        )
        return 0

    # Process
    t0 = time.time()
    processed = 0
    errors = 0
    total_chunks = 0

    # Estimate storage
    # Per chunk: mel(201*64*4=51KB) + stft(513*201*4=412KB) + audio(2048*4=8KB) ≈ 471KB
    est_mb_per_chunk = (
        chunk_samples * 4  # audio
        + (n_fft // 2 + 1) * (chunk_samples // hop_length + 1) * 4  # STFT
        + (chunk_samples // hop_length + 1) * n_mels * 4
    ) / 1e6
    est_total_mb = est_mb_per_chunk * len(all_work) * 10  # ~10 chunks/file
    print(f"Estimated output: ~{est_total_mb:.0f} MB")
    print()

    print("Processing...", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file, w): w for w in all_work}

        for future in as_completed(futures):
            result = future.result()
            if result.get("error"):
                print(f"\n  ERROR {result['file']}: {result['error']}")
                errors += 1
            else:
                processed += 1
                total_chunks += result["total_chunks"]
                if processed % 20 == 0 or processed == len(all_work):
                    elapsed = time.time() - t0
                    rate = processed / max(elapsed, 0.01)
                    print(
                        f"\r  [{processed}/{len(all_work)}] "
                        f"{total_chunks} chunks, "
                        f"{rate:.1f} file/s, "
                        f"{(time.time()-t0)/60:.1f}m elapsed",
                        end="",
                        flush=True,
                    )

    print()
    print(f"\nDone: {processed} files, {total_chunks} chunks " f"in {(time.time()-t0)/60:.1f} min")
    if errors:
        print(f"Errors: {errors}")

    # Write metadata
    write_metadata(
        args.output,
        train_out,
        test_out,
        train_files,
        test_files,
        chunk_seconds,
        sr,
        n_fft,
        hop_length,
        n_mels,
        chunk_samples,
    )

    # Print RAM cache estimate
    # Per chunk in RAM: mel + stft + audio (audio is already in RAM if cached)
    mb_per_chunk = (
        chunk_samples * 4
        + (n_fft // 2 + 1) * (chunk_samples // hop_length + 1) * 4
        + (chunk_samples // hop_length + 1) * n_mels * 4
    ) / 1e6
    print(
        f"\nRAM cache estimate: {total_chunks * mb_per_chunk:.0f} MB "
        f"for all chunks ({total_chunks} total)"
    )
    print(f"  Or ~{mb_per_chunk * 10:.0f} MB per file (~10 chunks/file)")
    print(f"\nPrecomputed features at: {args.output}")
    print(f"Use with: python train_clara.py --precomputed {args.output} --cache")


def write_metadata(
    output_dir,
    train_out,
    test_out,
    train_files,
    test_files,
    chunk_seconds,
    sr,
    n_fft,
    hop_length,
    n_mels,
    chunk_samples,
):
    """Write dataset metadata JSON."""

    # Count chunks
    def count_chunks(d):
        if not d or not d.exists():
            return 0
        return len(list(d.glob("*.npz")))

    metadata = {
        "version": 1,
        "chunk_seconds": chunk_seconds,
        "chunk_samples": chunk_samples,
        "sample_rate": sr,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "train_files": len(train_files),
        "train_chunks": count_chunks(train_out),
        "test_files": len(test_files),
        "test_chunks": count_chunks(test_out),
    }
    with open(Path(output_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written: {output_dir}/metadata.json")


if __name__ == "__main__":
    main()
