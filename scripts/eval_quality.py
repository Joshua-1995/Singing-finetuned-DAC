#!/usr/bin/env python
"""Quantitative reconstruction-quality evaluation for singing (before/after comparison).

Measures standard codec metrics + singing-specific pitch metrics in one pass. DAC preserves
input length, so original vs reconstruction are sample-aligned, making frame-level F0/MCD
comparison valid.

Metrics:
  - mel_dist   : multi-scale mel L1 distance        (lower ↓ is better)
  - stft_dist  : multi-scale STFT distance          (↓)
  - si_sdr     : scale-invariant SDR (dB)           (higher ↑ is better)
  - pesq       : PESQ wideband (16k)                (↑, 1~4.5)
  - stoi       : STOI                               (↑, 0~1)
  - mcd        : Mel-Cepstral Distortion (dB)       (↓)
  - f0_rmse_cents : F0 RMSE (cents)                 (↓)  <- singing pitch accuracy
  - f0_corr    : log-F0 Pearson correlation         (↑)  <- vibrato/melody preservation
  - vde        : Voicing Decision Error             (↓)  <- voiced/unvoiced errors
  - gpe        : Gross Pitch Error (>20% off ratio) (↓)

Usage:
  # build a fixed eval set and measure the pretrained baseline
  python eval_quality.py --ckpt pretrained --out runs/eval/baseline.json \
      --build_manifest runs/eval/manifest.json --per_dataset 30 --device cpu
  # after training, measure on the same eval set
  python eval_quality.py --ckpt runs/dac_singing_ft/best --out runs/eval/finetuned.json \
      --manifest runs/eval/manifest.json --device cuda
  # before/after comparison table
  python eval_quality.py --compare runs/eval/baseline.json runs/eval/finetuned.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


# ----------------------------- metrics -----------------------------
def f0_metrics(ref, rec, sr, frame_period=5.0):
    """F0/voicing metrics based on pyworld Harvest."""
    import pyworld as pw
    ref = np.ascontiguousarray(ref, dtype=np.float64)
    rec = np.ascontiguousarray(rec, dtype=np.float64)
    f0r, t = pw.harvest(ref, sr, frame_period=frame_period)
    f0c, _ = pw.harvest(rec, sr, frame_period=frame_period, f0_floor=71.0)
    n = min(len(f0r), len(f0c))
    f0r, f0c = f0r[:n], f0c[:n]
    vr, vc = f0r > 0, f0c > 0
    both = vr & vc
    out = {}
    out["vde"] = float(np.mean(vr != vc)) if n else float("nan")
    if both.sum() >= 2:
        cents = 1200.0 * np.log2(f0c[both] / f0r[both])
        out["f0_rmse_cents"] = float(np.sqrt(np.mean(cents ** 2)))
        lr, lc = np.log(f0r[both]), np.log(f0c[both])
        out["f0_corr"] = float(np.corrcoef(lr, lc)[0, 1]) if lr.std() > 0 and lc.std() > 0 else float("nan")
        gpe = np.abs(f0c[both] - f0r[both]) / f0r[both] > 0.2
        out["gpe"] = float(np.mean(gpe))
    else:
        out["f0_rmse_cents"] = out["f0_corr"] = out["gpe"] = float("nan")
    return out


def si_sdr(ref, rec):
    """Standard Scale-Invariant SDR (dB). Higher is better."""
    eps = 1e-8
    ref = ref - ref.mean(); rec = rec - rec.mean()
    alpha = (rec * ref).sum() / ((ref ** 2).sum() + eps)
    target = alpha * ref
    noise = rec - target
    return float(10.0 * np.log10(((target ** 2).sum() + eps) / ((noise ** 2).sum() + eps)))


def mcd(ref, rec, sr, order=24, frame_period=5.0):
    """Mel-Cepstral Distortion (dB). WORLD envelope -> pysptk.sp2mc (standard). Lower is better."""
    import pyworld as pw
    import pysptk
    # mel-warping coefficient by sample rate
    alpha = {16000: 0.42, 22050: 0.45, 24000: 0.46, 44100: 0.53, 48000: 0.554}.get(sr, 0.46)
    def mcc(x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        f0, t = pw.harvest(x, sr, frame_period=frame_period)
        sp = pw.cheaptrick(x, f0, t, sr)
        return pysptk.sp2mc(sp, order, alpha), (f0 > 0)
    cr, vr = mcc(ref); cc, vc = mcc(rec)
    n = min(len(cr), len(cc))
    if n < 1:
        return float("nan")
    voiced = vr[:n] & vc[:n]                   # voiced frames only (standard practice)
    if voiced.sum() < 1:
        voiced = np.ones(n, dtype=bool)
    diff = cr[:n][voiced, 1:] - cc[:n][voiced, 1:]   # exclude 0th (energy) coefficient
    dist = np.sqrt(np.sum(diff ** 2, axis=1))
    return float((10.0 / math.log(10)) * math.sqrt(2) * np.mean(dist))


def pesq_wb(ref, rec, sr):
    from pesq import pesq as _pesq
    import librosa
    r = librosa.resample(ref.astype(np.float32), orig_sr=sr, target_sr=16000)
    c = librosa.resample(rec.astype(np.float32), orig_sr=sr, target_sr=16000)
    n = min(len(r), len(c))
    return float(_pesq(16000, r[:n], c[:n], "wb"))


def stoi_metric(ref, rec, sr):
    from pystoi import stoi as _stoi
    n = min(len(ref), len(rec))
    return float(_stoi(ref[:n], rec[:n], sr, extended=False))


# ----------------------------- eval loop -----------------------------
def build_manifest(val_root, per_dataset, dur, seed, out_path):
    import random, soundfile as sf
    rng = random.Random(seed)
    val_root = Path(val_root)
    items = []
    # data/val/{public,private}/{dataset}
    ds_dirs = sorted([d for d in val_root.glob("*/*") if d.is_dir()])
    for d in ds_dirs:
        wavs = sorted(d.rglob("*.wav"))
        if not wavs:
            continue
        picked = rng.sample(wavs, min(per_dataset, len(wavs)))
        for w in picked:
            info = sf.info(str(w))
            total = info.frames / info.samplerate
            off = 0.0 if total <= dur else rng.uniform(0, total - dur)
            items.append({"dataset": d.name, "path": str(w), "offset": round(off, 3),
                          "duration": round(min(dur, total), 3)})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(items, open(out_path, "w"), indent=2)
    print(f"[manifest] {len(items)} clips, {len(ds_dirs)} datasets -> {out_path}")
    return items


def load_model(ckpt, device):
    import dac
    if ckpt == "pretrained":
        m = dac.DAC.load(str(dac.utils.download(model_type="24khz")))
    elif Path(ckpt).is_dir():
        m, _ = dac.model.DAC.load_from_folder(folder=ckpt, map_location="cpu", package=True)
    else:
        m = dac.DAC.load(ckpt)
    return m.to(device).eval()


def evaluate(ckpt, items, device, sr=24000):
    import torch, librosa
    from audiotools import AudioSignal
    import dac.nn.loss as L
    mel_loss = L.MelSpectrogramLoss(n_mels=[5,10,20,40,80,160,320],
        window_lengths=[32,64,128,256,512,1024,2048], mel_fmin=[0]*7, mel_fmax=[None]*7,
        pow=1.0, clamp_eps=1e-5, mag_weight=0.0).to(device)
    stft_loss = L.MultiScaleSTFTLoss().to(device)
    model = load_model(ckpt, device)

    from collections import defaultdict
    rows = []
    for it in items:
        wav, _ = librosa.load(it["path"], sr=sr, mono=True, offset=it["offset"], duration=it["duration"])
        if len(wav) < sr * 0.5:
            continue
        x = torch.from_numpy(wav)[None, None].to(device)
        with torch.no_grad():
            xp = model.preprocess(x, sr)
            z, *_ = model.encode(xp)
            y = model.decode(z)
        n = min(y.shape[-1], x.shape[-1])
        xa = AudioSignal(x[..., :n], sr); ya = AudioSignal(y[..., :n], sr)
        ref = x[0, 0, :n].cpu().numpy().astype(np.float64)
        rec = y[0, 0, :n].cpu().numpy().astype(np.float64)
        r = {"dataset": it["dataset"]}
        try: r["mel_dist"] = float(mel_loss(ya, xa))
        except Exception: r["mel_dist"] = float("nan")
        try: r["stft_dist"] = float(stft_loss(ya, xa))
        except Exception: r["stft_dist"] = float("nan")
        try: r["si_sdr"] = si_sdr(ref, rec)        # standard SI-SDR (higher = better)
        except Exception: r["si_sdr"] = float("nan")
        for name, fn in [("pesq", pesq_wb), ("stoi", stoi_metric)]:
            try: r[name] = fn(ref, rec, sr)
            except Exception: r[name] = float("nan")
        try:
            from audiotools.metrics import quality as _q
            r["visqol"] = float(_q.visqol(ya, xa, mode="audio"))  # MOS-LQO, audio mode (48k), matches DAC paper
        except Exception:
            r["visqol"] = float("nan")
        try: r["mcd"] = mcd(ref, rec, sr)
        except Exception: r["mcd"] = float("nan")
        try: r.update(f0_metrics(ref, rec, sr))
        except Exception: r.update({"f0_rmse_cents": float("nan"), "f0_corr": float("nan"),
                                    "vde": float("nan"), "gpe": float("nan")})
        rows.append(r)
    return rows


def aggregate(rows):
    keys = [k for k in rows[0] if k != "dataset"]
    def mean(rs, k):
        vals = [r[k] for r in rs if not (isinstance(r[k], float) and math.isnan(r[k]))]
        return float(np.mean(vals)) if vals else float("nan")
    overall = {k: mean(rows, k) for k in keys}
    per_ds = {}
    for ds in sorted(set(r["dataset"] for r in rows)):
        rs = [r for r in rows if r["dataset"] == ds]
        per_ds[ds] = {"n": len(rs), **{k: mean(rs, k) for k in keys}}
    return {"overall": overall, "n": len(rows), "per_dataset": per_ds}


# ----------------------------- comparison output -----------------------------
DIRECTION = {"visqol":"↑","mel_dist":"↓","stft_dist":"↓","si_sdr":"↑","pesq":"↑","stoi":"↑","mcd":"↓",
             "f0_rmse_cents":"↓","f0_corr":"↑","vde":"↓","gpe":"↓"}

def print_compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    oa, ob = A["overall"], B["overall"]
    print(f"\n{'metric':16s} {'dir':3s} {'before':>10s} {'after':>10s} {'Δ':>10s}  result")
    print("-" * 64)
    for k in DIRECTION:
        if k not in oa: continue
        before, after = oa[k], ob[k]
        d = after - before
        good = (d < 0) if DIRECTION[k] == "↓" else (d > 0)
        mark = "improved" if good else "worse"
        print(f"{k:16s} {DIRECTION[k]:3s} {before:10.4f} {after:10.4f} {d:+10.4f}  {mark}")
    print(f"\n(n: before={A.get('n')}, after={B.get('n')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="pretrained | <best/latest folder> | <.pth>")
    ap.add_argument("--out")
    ap.add_argument("--manifest", help="use an existing manifest")
    ap.add_argument("--build_manifest", help="path to write a new manifest")
    ap.add_argument("--val_root", default="data/val")
    ap.add_argument("--per_dataset", type=int, default=30)
    ap.add_argument("--dur", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE","AFTER"))
    args = ap.parse_args()

    if args.compare:
        print_compare(*args.compare); return

    if args.build_manifest:
        items = build_manifest(args.val_root, args.per_dataset, args.dur, args.seed, args.build_manifest)
    else:
        items = json.load(open(args.manifest))
        print(f"[manifest] loaded {len(items)} clips from {args.manifest}")

    print(f"[eval] ckpt={args.ckpt} device={args.device} clips={len(items)}")
    rows = evaluate(args.ckpt, items, args.device)
    res = aggregate(rows)
    res["ckpt"] = args.ckpt
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[done] {res['n']} clips -> {args.out}")
    print("\n=== OVERALL ===")
    for k, v in res["overall"].items():
        print(f"  {k:16s} {v:.4f}  {DIRECTION.get(k,'')}")


if __name__ == "__main__":
    main()
