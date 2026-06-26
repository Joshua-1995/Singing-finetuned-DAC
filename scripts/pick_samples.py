#!/usr/bin/env python
"""Per-clip before/after diff: find the samples with the biggest improvement.

Runs pretrained + fine-tuned on each manifest clip, computes per-clip mel distance
and F0 RMSE (cents) for both, ranks by improvement, prints the top samples, and
plots [Original | Pretrained | Fine-tuned] mel-spectrograms for the two highlights
(largest mel drop, largest F0 RMSE drop).

Usage:
  python pick_samples.py --manifest runs/eval/manifest.json \
      --finetuned runs/dac_singing_ft/best --out docs/highlight_samples.png --device cuda
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, librosa, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_quality import load_model, f0_metrics  # reuse
import dac.nn.loss as L
from audiotools import AudioSignal


@torch.no_grad()
def roundtrip(model, wav, sr, device):
    x = torch.from_numpy(wav)[None, None].to(device)
    z, *_ = model.encode(model.preprocess(x, sr))
    y = model.decode(z)
    n = min(y.shape[-1], x.shape[-1])
    return y[0, 0, :n].cpu().numpy(), n


def melspec(wav, sr):
    S = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, hop_length=256, n_mels=128, fmax=sr//2)
    return librosa.power_to_db(S, ref=np.max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--pretrained", default="pretrained")
    ap.add_argument("--out", default="docs/highlight_samples.png")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sr = 24000
    items = json.load(open(args.manifest))
    pre = load_model(args.pretrained, args.device)
    ft = load_model(args.finetuned, args.device)
    mel_loss = L.MelSpectrogramLoss(n_mels=[5,10,20,40,80,160,320],
        window_lengths=[32,64,128,256,512,1024,2048], mel_fmin=[0]*7, mel_fmax=[None]*7,
        pow=1.0, clamp_eps=1e-5, mag_weight=0.0).to(args.device)

    def mdist(a, b, n):
        xa = AudioSignal(torch.from_numpy(a[:n])[None,None].to(args.device), sr)
        xb = AudioSignal(torch.from_numpy(b[:n])[None,None].to(args.device), sr)
        return float(mel_loss(xb, xa))

    rows = []
    for it in items:
        wav, _ = librosa.load(it["path"], sr=sr, mono=True, offset=it["offset"], duration=it["duration"])
        if len(wav) < sr*0.5: continue
        rp, npp = roundtrip(pre, wav, sr, args.device)
        rf, nf = roundtrip(ft, wav, sr, args.device)
        n = min(len(wav), npp, nf)
        ref = wav[:n].astype(np.float64)
        try:
            f0p = f0_metrics(ref, rp[:n].astype(np.float64), sr)["f0_rmse_cents"]
            f0f = f0_metrics(ref, rf[:n].astype(np.float64), sr)["f0_rmse_cents"]
        except Exception:
            f0p = f0f = float("nan")
        rows.append({"it": it, "wav": wav,
                     "mel_pre": mdist(ref, rp, n), "mel_ft": mdist(ref, rf, n),
                     "f0_pre": f0p, "f0_ft": f0f})

    for r in rows:
        r["d_mel"] = r["mel_pre"] - r["mel_ft"]
        r["d_f0"] = (r["f0_pre"] - r["f0_ft"]) if not (np.isnan(r["f0_pre"]) or np.isnan(r["f0_ft"])) else -np.inf

    best_mel = max(rows, key=lambda r: r["d_mel"])
    best_f0  = max(rows, key=lambda r: r["d_f0"])

    print("== Top-5 by Mel distance drop ==")
    for r in sorted(rows, key=lambda r: -r["d_mel"])[:5]:
        print(f"  {r['it']['dataset']:10s} {Path(r['it']['path']).name:28s} mel {r['mel_pre']:.3f}->{r['mel_ft']:.3f} (Δ-{r['d_mel']:.3f})")
    print("== Top-5 by F0 RMSE drop (cents) ==")
    for r in sorted(rows, key=lambda r: -r["d_f0"])[:5]:
        print(f"  {r['it']['dataset']:10s} {Path(r['it']['path']).name:28s} f0 {r['f0_pre']:.1f}->{r['f0_ft']:.1f} (Δ-{r['d_f0']:.1f}c)")

    # plot the two highlights
    picks = [("Largest Mel-distance drop", best_mel), ("Largest F0-RMSE drop", best_f0)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    titles = ["Original", "Pretrained DAC", "Fine-tuned (singing)"]
    for r_i, (tag, r) in enumerate(picks):
        wav = r["wav"]
        rp,_ = roundtrip(pre, wav, sr, args.device); rf,_ = roundtrip(ft, wav, sr, args.device)
        for c,(w,t) in enumerate(zip([wav,rp,rf], titles)):
            ax = axes[r_i,c]
            librosa.display.specshow(melspec(w,sr), sr=sr, hop_length=256, x_axis="time",
                y_axis="mel", fmax=sr//2, ax=ax, vmin=-80, vmax=0, cmap="magma")
            if r_i==0: ax.set_title(t, fontsize=12)
            if c==0:
                lbl=f"{tag}\n[{r['it']['dataset']}]"
                if 'Mel' in tag: lbl+=f"\nmel {r['mel_pre']:.2f}->{r['mel_ft']:.2f}"
                else: lbl+=f"\nF0 {r['f0_pre']:.0f}->{r['f0_ft']:.0f}c"
                ax.set_ylabel(lbl, fontsize=9)
            else: ax.set_yticks([])
            ax.set_xlabel("")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\n[done] highlights -> {args.out}")


if __name__ == "__main__":
    main()
