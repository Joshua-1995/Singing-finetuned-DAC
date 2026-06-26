# Model Card — Singing-finetuned-DAC

## Model details
- **Model**: Descript Audio Codec (DAC) 24 kHz, fine-tuned for singing voice. **Architecture unchanged.**
- **Base**: [`weights_24khz_8kbps_0.0.4`](https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.4/weights_24khz.pth) (74.7 M generator, RVQ 32×1024 dim-8, hop 320 ≈ 75 Hz, 8 kbps).
- **Type**: neural audio codec (encoder–RVQ–decoder + GAN discriminator during training).
- **Developed for**: high-fidelity singing-voice reconstruction; backbone for Singing Voice Conversion.
- **License**: research / non-commercial (see below).

## Intended use
- **In scope**: encoding/decoding singing voice at 24 kHz; latent (`z`, `codes`) extraction for
  downstream singing models (SVC/SVS); research on neural codecs for singing.
- **Out of scope**: commercial deployment (non-commercial data); speech/general-audio (use the
  original DAC); languages/styles far from the training mix without verification.

## Training
- **Method**: full fine-tune (encoder+decoder+quantizer) from the pretrained generator;
  discriminator re-initialized (official release has none) and warmed up.
- **Hyper-parameters**: batch 16, 3 s segments, AdamW lr 1e-4 (β 0.8/0.99), ExpLR γ=0.999996,
  λ = {mel 15, feat 2, gen 1, vq-commit 0.25, vq-codebook 1.0}, quantizer dropout 0.5, 200 k steps.
- **Compute**: 1× RTX PRO 6000 (Blackwell), PyTorch 2.11 + CUDA 12.8.

## Training data (~472 h, 24 kHz mono, mostly monophonic singing)
| Dataset | Lang | Hours | License / source |
|---|---|---:|---|
| MSSV (Multi-Speaker Singing Voice) | KO | 228.8 | [AI-Hub Terms of Use](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105) ([#465](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=465)) — Korea-only |
| GV (Guide Vocal) | KO | 143.3 | [AI-Hub Terms of Use](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105) ([#473](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=473)) — Korea-only |
| ACE-KiSing | ZH | 30.0 | [CC BY-NC 4.0](https://huggingface.co/datasets/espnet/ace-kising-segments) |
| M4Singer | ZH | 28.2 | [CC BY-NC-SA 4.0](https://github.com/M4Singer/M4Singer) |
| HESD | KO | 14.0 | internal (not redistributed) |
| CSD | KO/EN | 4.6 | [CC BY-NC-SA 4.0](https://zenodo.org/record/4785016) |

`opencpop` excluded (CC BY-NC-ND forbids derivatives). MSSV/GV are AI-Hub (NIA, Korea)
datasets — access is restricted to Korean nationals and overseas use requires a separate
NIA agreement; not redistributed here. Attribution: *This work used datasets from "The Open
AI Dataset Project (AI-Hub, S. Korea)" (www.aihub.or.kr).*

## Evaluation
Fixed held-out set (160 clips across 6 datasets), identical clips before/after
(`scripts/eval_quality.py`). Pretrained DAC → Fine-tuned:

| Metric | Pretrained | Fine-tuned | Δ |
|---|---:|---:|---|
| Mel distance ↓ | 0.668 | 0.391 | −0.277 |
| STFT distance ↓ | 1.358 | 1.105 | −0.253 |
| SI-SDR (dB) ↑ | −9.6 | +15.6 | +25.1 |

Off-the-shelf DAC reconstructs singing waveforms poorly (SI-SDR −9.6 dB on singing vs ~16 dB
on general audio); fine-tuning restores it to the codec's native quality on the singing domain.
`scripts/eval_quality.py` additionally reports PESQ/STOI/MCD/F0 metrics.

## License
Research / non-commercial only. Weights inherit the non-commercial terms of the
CC BY-NC(-SA) and internal training data. DAC code/architecture: MIT © Descript.
