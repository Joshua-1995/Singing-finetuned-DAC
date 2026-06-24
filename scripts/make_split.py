#!/usr/bin/env python
"""전처리된 데이터를 train/val로 분할한다.

DAC repo는 CSV가 아니라 '폴더 기반' AudioDataset을 쓰므로, 파일을 실제로
val 폴더로 이동(또는 복사)한다. 각 데이터셋(소스 디렉토리)에서 val_ratio 비율을
랜덤 추출해 균등하게 분리한다.

사용 예:
    # data/train/public/opensinger 의 5%를 data/val/public/opensinger 로 이동
    python make_split.py --root data --val_ratio 0.05 --seed 0

기본 동작:
- data/train/ 하위의 각 '말단' 데이터셋 디렉토리를 찾는다.
- 동일한 상대경로로 data/val/ 아래에 val 셋을 만든다.
"""
import argparse
import random
import shutil
from pathlib import Path


def leaf_dataset_dirs(train_root: Path):
    """train_root 아래에서 wav를 직접(또는 하위에) 포함하는 데이터셋 단위 디렉토리.

    구조: data/train/{public|private}/{dataset}/...  ->  {public|private}/{dataset}
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
    ap.add_argument("--root", default="data", help="train/ 과 val/ 의 상위 디렉토리")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="이동 대신 복사")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.root)
    train_root = root / "train"
    val_root = root / "val"

    datasets = leaf_dataset_dirs(train_root)
    if not datasets:
        print(f"[!] {train_root} 안에 wav가 없습니다. 먼저 preprocess.py를 실행하세요.")
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
        print(f"  {ds}: {len(wavs)}개 중 {len(picked)}개 -> val")

    print(f"[done] 총 {total_moved}개를 val로 분리 (val_ratio={args.val_ratio})")


if __name__ == "__main__":
    main()
