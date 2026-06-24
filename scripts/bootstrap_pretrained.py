#!/usr/bin/env python
"""Prepare the fine-tuning start checkpoint from the official pretrained DAC 24kHz.

The training loop resumes from a folder-package at <save_path>/<tag>/dac, but the
official release ships a single .pth. This downloads the pretrained 24kHz generator
and saves it in that folder-package format (with empty extra data, so optimizer/
tracker start fresh -> step 0). The discriminator is absent by design and will be
re-initialized during training.

Usage:
    python scripts/bootstrap_pretrained.py --save_path runs/dac_singing_ft --tag latest
"""
import argparse
from pathlib import Path

import dac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_path", default="runs/dac_singing_ft")
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--model_type", default="24khz")
    args = ap.parse_args()

    path = dac.utils.download(model_type=args.model_type)
    model = dac.DAC.load(str(path))
    out = Path(args.save_path) / args.tag
    out.parent.mkdir(parents=True, exist_ok=True)
    saved = model.save_to_folder(str(out), {})   # empty extra -> fresh optimizer/tracker
    print(f"[done] pretrained generator package -> {saved}")
    print(f"       now run: bash scripts/train.sh")


if __name__ == "__main__":
    main()
