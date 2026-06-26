#!/usr/bin/env python
"""Convert any audio to DAC 24 kHz mono WAV (float32). (multiprocessing-capable)

Examples:
    python preprocess.py --in_dir /raw/opensinger --out_dir data/train/public/opensinger
    # long full songs (GV/MSSV): chunk-split + parallel:
    python preprocess.py --in_dir data/raw/gv_ex --out_dir data/train/public/gv \
        --flatten --segment_sec 30 --jobs 64

- recursively scans the input folder (wav/flac/mp3/m4a/ogg).
- normalizes to 24 kHz, mono, float32 WAV (44k etc. are resampled).
- drops files shorter than 1 s.
- with --segment_sec, splits long files into chunks (training still draws a further random
  excerpt from each via DAC's AudioDataset).
- --jobs N runs N worker processes in parallel.
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".opus"}


def process_one(p_str, in_dir, out_dir, sr, min_sec, trim_db, flatten, segment_sec, seen_suffix):
    """Process a single file. Returns (kept, skipped)."""
    p = Path(p_str)
    try:
        wav, _ = librosa.load(str(p), sr=sr, mono=True)
    except Exception:
        return (0, 1)

    if trim_db is not None:
        wav, _ = librosa.effects.trim(wav, top_db=trim_db)
    if len(wav) / sr < min_sec:
        return (0, 1)

    wav = wav.astype(np.float32)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak > 1.0:
        wav = wav / peak

    if flatten:
        # build a unique stem from the relative path to avoid filename clashes in parallel
        rel = p.relative_to(in_dir).with_suffix("")
        stem = str(rel).replace(os.sep, "_")
        base = Path(out_dir) / stem
    else:
        rel = p.relative_to(in_dir).with_suffix("")
        base = Path(out_dir) / rel
        base.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    if segment_sec is not None and len(wav) / sr > segment_sec:
        seg = int(segment_sec * sr)
        n_chunks = len(wav) // seg
        for ci in range(n_chunks):
            chunk = wav[ci * seg:(ci + 1) * seg]
            sf.write(str(base.parent / f"{base.name}_seg{ci:03d}.wav"), chunk, sr, subtype="FLOAT")
            kept += 1
        tail = wav[n_chunks * seg:]
        if len(tail) / sr >= min_sec:
            sf.write(str(base.parent / f"{base.name}_seg{n_chunks:03d}.wav"), tail, sr, subtype="FLOAT")
            kept += 1
    else:
        sf.write(str(base.parent / f"{base.name}.wav"), wav, sr, subtype="FLOAT")
        kept += 1
    return (kept, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--min_sec", type=float, default=1.0)
    ap.add_argument("--trim_db", type=float, default=None)
    ap.add_argument("--flatten", action="store_true")
    ap.add_argument("--segment_sec", type=float, default=None,
                    help="split long files into chunks of this length in seconds (for full songs like GV/MSSV)")
    ap.add_argument("--jobs", type=int, default=1, help="number of parallel worker processes")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [str(p) for p in in_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    if not files:
        print(f"[!] No audio files found under {in_dir}.")
        return
    print(f"[i] {len(files)} files -> {out_dir} (jobs={args.jobs}, segment={args.segment_sec})")

    worker = partial(process_one, in_dir=in_dir, out_dir=out_dir, sr=args.sr,
                     min_sec=args.min_sec, trim_db=args.trim_db, flatten=args.flatten,
                     segment_sec=args.segment_sec, seen_suffix=None)

    kept = skipped = 0
    if args.jobs <= 1:
        for f in tqdm(files):
            k, s = worker(f); kept += k; skipped += s
    else:
        with Pool(args.jobs) as pool:
            for k, s in tqdm(pool.imap_unordered(worker, files, chunksize=8), total=len(files)):
                kept += k; skipped += s

    print(f"[done] wrote {kept} wav, skipped {skipped} -> {out_dir}")


if __name__ == "__main__":
    main()
