#!/usr/bin/env python
"""모든 오디오를 DAC 24kHz mono WAV(float32)로 변환한다. (멀티프로세싱 지원)

사용 예:
    python preprocess.py --in_dir /raw/opensinger --out_dir data/train/public/opensinger
    # 긴 곡(GV/MSSV)은 청크 분할 + 병렬:
    python preprocess.py --in_dir data/raw/gv_ex --out_dir data/train/public/gv \
        --flatten --segment_sec 30 --jobs 64

- 입력 폴더 재귀 탐색 (wav/flac/mp3/m4a/ogg).
- 24kHz, mono, float32 WAV로 통일 (44k 등은 resample).
- 1초 미만 제외.
- --segment_sec 설정 시 긴 파일을 청크로 분할 (학습 시 DAC가 추가로 random 구간을 또 뽑음).
- --jobs N 으로 N개 프로세스 병렬 처리.
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
    """파일 하나를 처리. (kept, skipped) 반환."""
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
        # 병렬에서 파일명 충돌 방지를 위해 상대경로 기반 고유 stem 생성
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
                    help="긴 파일을 이 길이(초) 청크로 분할 (GV/MSSV 같은 전곡용)")
    ap.add_argument("--jobs", type=int, default=1, help="병렬 프로세스 수")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [str(p) for p in in_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    if not files:
        print(f"[!] {in_dir} 안에서 오디오 파일을 찾지 못했습니다.")
        return
    print(f"[i] {len(files)}개 파일 -> {out_dir} (jobs={args.jobs}, segment={args.segment_sec})")

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

    print(f"[done] 생성 {kept}개 wav, 제외 {skipped}개 -> {out_dir}")


if __name__ == "__main__":
    main()
