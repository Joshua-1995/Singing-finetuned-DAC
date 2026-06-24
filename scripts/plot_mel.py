#!/usr/bin/env python
"""문서용 Mel-spectrogram 비교 그림 생성.

각 샘플에 대해 [원본 | pretrained 재합성 | fine-tuned 재합성] mel-spectrogram을
나란히 그려 PNG로 저장한다. 고음역 재현 차이를 눈으로 보여주는 용도.

사용:
  python plot_mel.py --manifest runs/eval/manifest.json \
      --pretrained pretrained --finetuned runs/dac_singing_ft/best \
      --n 4 --out docs/mel_compare.png --device cuda
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
import dac


def load_model(ckpt, device):
    if ckpt == "pretrained":
        m = dac.DAC.load(str(dac.utils.download(model_type="24khz")))
    elif Path(ckpt).is_dir():
        m, _ = dac.model.DAC.load_from_folder(folder=ckpt, map_location="cpu", package=True)
    else:
        m = dac.DAC.load(ckpt)
    return m.to(device).eval()


@torch.no_grad()
def recon(model, wav, sr, device):
    x = torch.from_numpy(wav)[None, None].to(device)
    z, *_ = model.encode(model.preprocess(x, sr))
    y = model.decode(z)
    n = min(y.shape[-1], x.shape[-1])
    return y[0, 0, :n].cpu().numpy()


def melspec(wav, sr):
    S = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, hop_length=256, n_mels=128, fmax=sr // 2)
    return librosa.power_to_db(S, ref=np.max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pretrained", default="pretrained")
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--dur", type=float, default=4.0)
    ap.add_argument("--out", default="docs/mel_compare.png")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    items = json.load(open(args.manifest))
    # 데이터셋별로 골고루 n개
    seen, picks = set(), []
    for it in items:
        if it["dataset"] not in seen:
            picks.append(it); seen.add(it["dataset"])
        if len(picks) >= args.n:
            break
    pre = load_model(args.pretrained, args.device)
    ft = load_model(args.finetuned, args.device)

    sr = 24000
    fig, axes = plt.subplots(len(picks), 3, figsize=(13, 3 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    titles = ["Original", "Pretrained DAC", "Fine-tuned (singing)"]
    for r, it in enumerate(picks):
        wav, _ = librosa.load(it["path"], sr=sr, mono=True, offset=it["offset"], duration=args.dur)
        outs = [wav, recon(pre, wav, sr, args.device), recon(ft, wav, sr, args.device)]
        vmin = -80
        for c, (w, t) in enumerate(zip(outs, titles)):
            M = melspec(w, sr)
            ax = axes[r, c]
            librosa.display.specshow(M, sr=sr, hop_length=256, x_axis="time", y_axis="mel",
                                     fmax=sr // 2, ax=ax, vmin=vmin, vmax=0, cmap="magma")
            if r == 0:
                ax.set_title(t, fontsize=12)
            ax.set_ylabel(it["dataset"] if c == 0 else "")
            if c > 0:
                ax.set_yticks([])
            ax.set_xlabel("")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[done] {len(picks)} samples -> {args.out}")


if __name__ == "__main__":
    main()
