#!/usr/bin/env python
"""Split preprocessed data into train/val.

The DAC repo uses a folder-based AudioDataset (not CSV), so we physically move (or copy)
files into the val folder. From each dataset (source directory) we randomly draw a
val_ratio fraction so every dataset is split evenly.

Example:
    # move 5% of data/train/public/opensinger to data/val/public/opensinger
    python make_split.py --root data --val_ratio 0.05 --seed 0

Default behaviour:
- find each leaf dataset directory under data/train/
- recreate the val set under data/val/ with the same relative path
"""
import argparse
import random
import shutil
from pathlib import Path


def leaf_dataset_dirs(train_root: Path):
    """Per-dataset directories under train_root that contain wavs (directly or nested).

    Layout: data/train/{public|private}/{dataset}/...  ->  {public|private}/{dataset}
    """
    dirs = set()
    for wav in train_root.rglob("*.wav"):
        rel = wav.relative_to(train_root)
        # public/opensinger/xxx.wav -> public/opensinger
        parts = rel.parts
        if len(parts) >= 2:
            dirs.add(Path(parts[0]) / parts[1])
        else:
            dirs.add(Path(parts[0]))
    return sorted(dirs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="parent dir of train/ and val/")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="copy instead of move")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.root)
    train_root = root / "train"
    val_root = root / "val"

    datasets = leaf_dataset_dirs(train_root)
    if not datasets:
        print(f"[!] No wavs under {train_root}. Run preprocess.py first.")
        return

    total_moved = 0
    for ds in datasets:
        src_dir = train_root / ds
        wavs = sorted(src_dir.rglob("*.wav"))
        if not wavs:
            continue
        n_val = max(1, int(round(len(wavs) * args.val_ratio)))
        picked = rng.sample(wavs, min(n_val, len(wavs)))
        for w in picked:
            rel = w.relative_to(train_root)
            dst = val_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if args.copy:
                shutil.copy2(w, dst)
            else:
                shutil.move(str(w), str(dst))
        total_moved += len(picked)
        print(f"  {ds}: {len(picked)} of {len(wavs)} -> val")

    print(f"[done] moved {total_moved} files to val (val_ratio={args.val_ratio})")


if __name__ == "__main__":
    main()
