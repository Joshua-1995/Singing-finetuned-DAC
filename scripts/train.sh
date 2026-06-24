#!/usr/bin/env bash
# Fine-tune / continue-training launcher for Singing-finetuned-DAC.
# Self-contained: uses repo-local train.py + conf. Run from the repo root.
#
# Prerequisites:
#   1. pip install -r requirements.txt  (+ torch cu128 for Blackwell GPUs)
#   2. Apply the two third-party patches in docs/PATCHES.md (argbind, audiotools).
#   3. Put your 24kHz mono wavs under data/train/... and data/val/... :
#        python scripts/preprocess.py --in_dir <raw> --out_dir data/train/<name> --jobs 64 [--segment_sec 30]
#        python scripts/make_split.py --root data --val_ratio 0.05
#   4. To fine-tune FROM the official pretrained DAC, bootstrap the start checkpoint:
#        python scripts/bootstrap_pretrained.py --save_path runs/dac_singing_ft
#      (or train from scratch by setting `resume: false` in conf/singing_24khz.yml)
#
# Usage:  bash scripts/train.sh [extra --argbind overrides ...]
set -e

CONF=conf/singing_24khz.yml
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDNN_BENCHMARK=${CUDNN_BENCHMARK:-1}

NTRAIN=$(find data/train -name '*.wav' 2>/dev/null | wc -l)
NVAL=$(find data/val -name '*.wav' 2>/dev/null | wc -l)
echo "[i] train wav: $NTRAIN | val wav: $NVAL"
if [ "$NTRAIN" -eq 0 ]; then
  echo "[!] No wavs under data/train. Prepare data first (see header / README)."; exit 1
fi

echo "[i] launching training (TensorBoard logs -> runs/dac_singing_ft/logs)"
exec python scripts/train.py --args.load "$CONF" "$@"
