#!/usr/bin/env python
"""Fine-tuned DAC로 오디오를 encode->decode 재합성하고 품질 지표를 출력한다.

사용 예:
    # fine-tune 체크포인트로 검증 샘플 재합성
    python reconstruct.py --ckpt runs/dac_singing_ft/best --in data/val --out recon_out --n 8

    # pretrained와 비교 (--compare_pretrained)
    python reconstruct.py --ckpt runs/dac_singing_ft/best --in data/val --out recon_out --compare_pretrained

지표: mel L1 distance(낮을수록 좋음). 원본/재합성 wav를 out/ 에 저장하여 청취 비교.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import dac
from audiotools import AudioSignal


def load_model(ckpt: str):
    p = Path(ckpt)
    if p.is_dir():
        # trainer가 저장한 폴더 패키지 (best/ 또는 latest/)
        model, _ = dac.model.DAC.load_from_folder(folder=str(p), map_location="cpu", package=True)
    else:
        model = dac.DAC.load(str(p))
    return model.cuda().eval()


@torch.no_grad()
def roundtrip(model, sig: AudioSignal):
    sig = sig.clone().to("cuda")
    x = model.preprocess(sig.audio_data, sig.sample_rate)
    z, codes, latents, _, _ = model.encode(x)
    y = model.decode(z)
    n = min(y.shape[-1], sig.audio_data.shape[-1])
    return AudioSignal(y[..., :n].cpu(), sig.sample_rate), n


@torch.no_grad()
def mel_l1(model_loss, a: AudioSignal, b: AudioSignal):
    n = min(a.audio_data.shape[-1], b.audio_data.shape[-1])
    aa = AudioSignal(a.audio_data[..., :n].cuda(), a.sample_rate)
    bb = AudioSignal(b.audio_data[..., :n].cuda(), b.sample_rate)
    return float(model_loss(aa, bb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best/ latest/ 폴더 또는 .pth")
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", default="recon_out")
    ap.add_argument("--n", type=int, default=8, help="재합성할 샘플 수")
    ap.add_argument("--compare_pretrained", action="store_true")
    args = ap.parse_args()

    import dac.nn.loss as L
    mel_loss = L.MelSpectrogramLoss(
        n_mels=[5, 10, 20, 40, 80, 160, 320],
        window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
        mel_fmin=[0] * 7, mel_fmax=[None] * 7, pow=1.0, clamp_eps=1e-5, mag_weight=0.0).cuda()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.ckpt)
    pre = load_model(dac.utils.download(model_type="24khz")) if args.compare_pretrained else None

    wavs = sorted(Path(args.in_dir).rglob("*.wav"))[: args.n]
    if not wavs:
        print(f"[!] {args.in_dir} 에 wav가 없습니다."); return

    ft_scores, pre_scores = [], []
    for i, w in enumerate(wavs):
        sig = AudioSignal(str(w))
        if sig.sample_rate != 24000:
            sig = sig.resample(24000)
        rec, _ = roundtrip(model, sig)
        d_ft = mel_l1(mel_loss, rec, sig); ft_scores.append(d_ft)
        sig.write(str(out_dir / f"{i:02d}_orig.wav"))
        rec.write(str(out_dir / f"{i:02d}_recon_ft.wav"))
        line = f"[{i:02d}] {w.name}: mel_L1(ft)={d_ft:.4f}"
        if pre is not None:
            recp, _ = roundtrip(pre, sig)
            d_pre = mel_l1(mel_loss, recp, sig); pre_scores.append(d_pre)
            recp.write(str(out_dir / f"{i:02d}_recon_pretrained.wav"))
            line += f" | mel_L1(pretrained)={d_pre:.4f} | 개선={d_pre-d_ft:+.4f}"
        print(line)

    print(f"\n평균 mel_L1 (fine-tuned): {np.mean(ft_scores):.4f}")
    if pre_scores:
        print(f"평균 mel_L1 (pretrained): {np.mean(pre_scores):.4f}")
        print(f"평균 개선폭: {np.mean(pre_scores)-np.mean(ft_scores):+.4f}  (양수면 fine-tune이 더 좋음)")
    print(f"\n샘플 저장 -> {out_dir}/  (원본 vs 재합성 청취 비교)")


if __name__ == "__main__":
    main()
