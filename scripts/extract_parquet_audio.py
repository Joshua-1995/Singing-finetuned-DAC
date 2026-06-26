#!/usr/bin/env python
"""Extract audio from HF datasets parquet (audio struct<bytes,path>) to 24 kHz mono wav.

For datasets where the audio is encoded inside parquet files, e.g. ACE-KiSing.

Example:
    python extract_parquet_audio.py \
        --parquet_dir data/raw/ace-kising/data \
        --out_dir data/train/public/ace_kising \
        --audio_col audio --id_col segment_id
"""
import argparse
import io
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm


def decode(b: bytes, sr: int):
    try:
        data, osr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if osr != sr:
            data = librosa.resample(data, orig_sr=osr, target_sr=sr)
        return data.astype(np.float32)
    except Exception:
        # formats soundfile cannot read (e.g. mp3) -> librosa fallback
        data, _ = librosa.load(io.BytesIO(b), sr=sr, mono=True)
        return data.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--audio_col", default="audio")
    ap.add_argument("--id_col", default=None, help="column used for the filename (else an index)")
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--min_sec", type=float, default=1.0)
    ap.add_argument("--glob", default="*.parquet")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.parquet_dir).rglob(args.glob))
    if not files:
        print(f"[!] No parquet files in {args.parquet_dir}."); return
    print(f"[i] {len(files)} parquet files -> {out_dir}")

    kept, skipped, gidx = 0, 0, 0
    for pf_path in files:
        pf = pq.ParquetFile(str(pf_path))
        cols = [args.audio_col] + ([args.id_col] if args.id_col else [])
        for batch in pf.iter_batches(batch_size=256, columns=cols):
            d = batch.to_pydict()
            audios = d[args.audio_col]
            ids = d.get(args.id_col) if args.id_col else None
            for j, a in enumerate(tqdm(audios, leave=False, desc=pf_path.name)):
                gidx += 1
                b = a.get("bytes") if isinstance(a, dict) else None
                if not b:
                    skipped += 1; continue
                try:
                    wav = decode(b, args.sr)
                except Exception as e:
                    skipped += 1; continue
                if len(wav) / args.sr < args.min_sec:
                    skipped += 1; continue
                peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
                if peak > 1.0:
                    wav = wav / peak
                name = (ids[j] if ids else f"seg_{gidx:07d}")
                name = str(name).replace("/", "_")
                sf.write(str(out_dir / f"{name}.wav"), wav, args.sr, subtype="FLOAT")
                kept += 1
        print(f"  {pf_path.name}: {kept} written so far")

    print(f"[done] extracted {kept}, skipped {skipped} -> {out_dir}")


if __name__ == "__main__":
    main()
