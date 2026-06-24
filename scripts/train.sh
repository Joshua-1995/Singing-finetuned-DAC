#!/usr/bin/env bash
# DAC 24kHz Singing-Voice Fine-tuning 런처
# 사용법:
#   bash scripts/train.sh                 # 기본(joint) fine-tune, pretrained에서 resume
#   bash scripts/train.sh fresh           # save_path를 비우고 처음부터 fine-tune
#
# 사전 조건:
#   - data/train/ , data/val/ 에 24kHz mono wav가 들어 있어야 함 (preprocess.py + make_split.py)
#   - runs/dac_singing_ft/latest/dac 에 pretrained generator 패키지가 있어야 함 (이미 생성됨)
set -e

REPO=/root/descript-audio-codec
PY=/root/miniconda3/envs/dac_ft/bin/python
CONF=/root/dac_singing/conf/singing_24khz.yml

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDNN_BENCHMARK=1

NTRAIN=$(find /root/dac_singing/data/train -name '*.wav' | wc -l)
NVAL=$(find /root/dac_singing/data/val -name '*.wav' | wc -l)
echo "[i] train wav: $NTRAIN | val wav: $NVAL"
if [ "$NTRAIN" -eq 0 ]; then
  echo "[!] data/train 에 wav가 없습니다. 먼저 데이터를 준비하세요:"
  echo "    python scripts/preprocess.py --in_dir <raw> --out_dir data/train/<public|private>/<name>"
  echo "    python scripts/make_split.py --root data --val_ratio 0.05"
  exit 1
fi

cd "$REPO"
echo "[i] launching training (TensorBoard logs -> runs/dac_singing_ft/logs)"
exec "$PY" scripts/train.py --args.load "$CONF" "$@"
