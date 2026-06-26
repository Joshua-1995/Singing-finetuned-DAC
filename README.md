# Singing-finetuned-DAC

Fine-tuned weights of the **Descript Audio Codec (DAC) 24 kHz** for **singing voice**.

> **No architecture changes.** This is the official pretrained DAC 24 kHz model,
> further trained (full fine-tune) on a large singing-voice corpus. The goal is better
> reconstruction of singing — especially **high pitch range, vibrato, and F0 fidelity** —
> which the original general-purpose DAC handles less well because it saw very little
> a-cappella singing during training.

**Research / non-commercial use only** (see [License](#license)).

---

## Why

The official DAC 24 kHz is a *universal* codec trained on a speech / music / general-audio
mix. Its singing content is tiny (≈10 h of a-cappella VocalSet + a few hours of
accompanied vocals in MUSDB18). Singing has wide pitch range and vibrato that a
general codec reconstructs imperfectly. Fine-tuning on ~470 h of (mostly monophonic)
singing adapts the encoder/decoder/quantizer to this domain.

This codec is the backbone of a Singing Voice Conversion pipeline: `wav → DAC latent
z (B, 1024, T) @ 75 Hz → (manipulate) → DAC decode → wav`, so reconstruction quality
upper-bounds the whole system.

## Results

Pretrained DAC → fine-tuned, on a fixed held-out set (160 singing clips across the 6
datasets; identical clips before/after). Metric definitions match the DAC paper's
`audiotools` implementations. Reproduce with [`scripts/eval_quality.py`](scripts/eval_quality.py).

| Metric | Pretrained DAC | Fine-tuned | Δ |
|---|---:|---:|---|
| Mel distance ↓ | 0.668 | **0.391** | −0.277 |
| STFT distance ↓ | 1.358 | **1.105** | −0.253 |
| SI-SDR (dB) ↑ | −9.6 | **+15.6** | +25.1 |

Fine-tuned = best checkpoint (val mel/loss 1.31 → 0.38). Off-the-shelf DAC reconstructs
singing waveforms poorly (SI-SDR −9.6 dB on singing, vs ~16 dB reported on general audio);
fine-tuning restores it to the codec's native quality regime on the singing domain.

## Installation

```bash
git clone https://github.com/Joshua-1995/Singing-finetuned-DAC.git
cd Singing-finetuned-DAC
pip install -r requirements.txt
```

PyTorch is **not** pinned in `requirements.txt` — install the build that matches your GPU:

```bash
# Blackwell GPUs (sm_120, e.g. RTX PRO 6000): CUDA 12.8 build
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# older GPUs: pick the matching CUDA build from https://pytorch.org
```

> **For training only** (not inference): also apply the two small third-party patches in
> [`docs/PATCHES.md`](docs/PATCHES.md) (`argbind`, `audiotools`) — needed when running the
> 2023-era DAC training code on recent PyTorch.

## Usage

Install (above) and download the weights, then:

```python
import dac
from audiotools import AudioSignal

# load fine-tuned weights (single-file export; see Releases / HF Hub link below)
model = dac.DAC.load("dac_singing_finetune_24khz.pth").eval().to("cuda")

signal = AudioSignal("song.wav").resample(24000).to_mono()
x = model.preprocess(signal.audio_data.cuda(), 24000)
z, codes, latents, _, _ = model.encode(x)   # z: (B, 1024, T) @ ~75 Hz
y = model.decode(z)                          # reconstructed waveform
```

> Weights are hosted separately (not in this Git repo) — see the
> **[Weights](#weights)** section.

## How it was fine-tuned

- **Base**: `descript-audio-codec` 24 kHz, 8 kbps (`weights_24khz_8kbps_0.0.4`),
  74.7 M-param generator, RVQ 32×1024 (dim 8), hop 320 (~75 Hz latent).
- **Strategy**: full fine-tune (encoder + decoder + quantizer) resumed from the
  pretrained generator. The official release ships no discriminator, so the
  discriminator (MPD + MRD + MSD) is **re-initialized and warmed up** from scratch.
- **Key hyper-parameters** ([`conf/singing_24khz.yml`](conf/singing_24khz.yml)):
  batch 16, 3 s segments, AdamW lr 1e-4 (betas 0.8/0.99), ExponentialLR γ=0.999996,
  loss λ = {mel 15, feat 2, gen 1, vq-commit 0.25, vq-codebook 1.0}, 200 k steps.
- **Hardware**: single NVIDIA RTX PRO 6000 (Blackwell, sm_120), PyTorch 2.11 + CUDA 12.8.

### Compatibility patches (new PyTorch ↔ older DAC toolchain)
Training the 2023-era DAC code on a Blackwell GPU (which needs CUDA 12.8+) required
three small patches — documented in [`docs/PATCHES.md`](docs/PATCHES.md):
1. `argbind` — handle PyTorch's `lr: float | Tensor` union annotation.
2. `audiotools` — `soundfile` fallback for the removed `torchaudio.info` legacy API.
3. DAC `train.py` — length-align recon/target before spectral losses when the segment
   length is an exact multiple of the hop.

## Data

~472 h total, all converted to 24 kHz mono, 5 % held out for validation. Long full-song
recordings were split into 30 s chunks. **Audio is not redistributed here** (see license).

| Dataset | Lang | Hours (train) | License | Source |
|---|---|---:|---|---|
| MSSV (Multi-Speaker Singing Voice) | Korean | 228.8 | [AI-Hub Terms of Use](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105) | [AI-Hub #465](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=465) |
| GV (Guide Vocal) | Korean | 143.3 | [AI-Hub Terms of Use](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105) | [AI-Hub #473](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=473) |
| ACE-KiSing | Chinese | 30.0 | CC BY-NC 4.0 | [HF](https://huggingface.co/datasets/espnet/ace-kising-segments) |
| M4Singer | Chinese | 28.2 | CC BY-NC-SA 4.0 | [GitHub](https://github.com/M4Singer/M4Singer) |
| HESD | Korean | 14.0 | internal | — |
| CSD | Korean/English | 4.6 | CC BY-NC-SA 4.0 | [Zenodo](https://zenodo.org/record/4785016) |

See [`scripts/DATASETS.md`](scripts/DATASETS.md) for download instructions for the
public datasets. (`opencpop` was excluded — its CC BY-NC-ND license forbids derivatives.)

### Data availability & reproducibility
- **MSSV & GV are [AI-Hub](https://www.aihub.or.kr) datasets** (built with support from
  NIA, Korea). AI-Hub access is **restricted to Korean nationals**, requires login +
  per-dataset approval, and **any overseas use/export requires a separate agreement with
  NIA**. They are therefore **not redistributed here**, and users outside Korea generally
  cannot obtain them — substitute your own singing data to reproduce.
- **HESD** is an internal corpus and is not redistributed.
- The public datasets (ACE-KiSing, M4Singer, CSD) are freely downloadable (see links above).
- **No audio from any dataset is included in this repository** (code + docs + weights only).

## Train / further fine-tune (your own data)

The full pipeline is included so you can continue fine-tuning on your own singing data.

```bash
# 0. setup
pip install -r requirements.txt
#    Blackwell (sm_120) GPUs: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
#    then apply the two third-party patches in docs/PATCHES.md (argbind, audiotools)

# 1. preprocess any audio to 24 kHz mono (resamples; --segment_sec splits long songs)
python scripts/preprocess.py --in_dir <raw> --out_dir data/train/<name> --jobs 64 [--segment_sec 30]
#    (parquet audio datasets, e.g. ACE-KiSing: scripts/extract_parquet_audio.py)

# 2. train/val split (5% per dataset)
python scripts/make_split.py --root data --val_ratio 0.05

# 3a. to fine-tune FROM the official pretrained DAC, bootstrap the start checkpoint:
python scripts/bootstrap_pretrained.py --save_path runs/dac_singing_ft
#    (or continue from OUR weights: place them at runs/dac_singing_ft/latest/dac;
#     or train from scratch by setting `resume: false` in conf/singing_24khz.yml)

# 3b. train (run from repo root)
bash scripts/train.sh
#    monitor: tensorboard --logdir runs/dac_singing_ft/logs

# 4. evaluate before/after on a fixed set
python scripts/eval_quality.py --ckpt pretrained    --out runs/eval/baseline.json  --build_manifest runs/eval/manifest.json
python scripts/eval_quality.py --ckpt runs/dac_singing_ft/best --out runs/eval/finetuned.json --manifest runs/eval/manifest.json
python scripts/eval_quality.py --compare runs/eval/baseline.json runs/eval/finetuned.json
```

`scripts/train.py` is adapted from the DAC training script (MIT); the one change plus
the two third-party patches are documented in [`docs/PATCHES.md`](docs/PATCHES.md).
Hyper-parameters live in [`conf/singing_24khz.yml`](conf/singing_24khz.yml).

## Weights

Hosted on the Hugging Face Hub (not in this Git repo): **_link TBD_**

| File | Size | Use |
|---|---:|---|
| `dac_singing_finetune_24khz.pth` | 286 MB | **Inference** — generator only; load with `dac.DAC.load(...)` |
| `dac_singing_finetune_full_ckpt.tar.gz` | 2.1 GB | **Continue training** — generator + discriminator + optimizer/scheduler/tracker |

The official DAC release ships no discriminator, so cold-start fine-tuning is unstable. We
also publish the **full checkpoint** so you can resume fine-tuning smoothly. To continue from it:

```bash
tar xzf dac_singing_finetune_full_ckpt.tar.gz   # -> best/{dac,discriminator}
mkdir -p runs/dac_singing_ft && mv best runs/dac_singing_ft/latest
bash scripts/train.sh    # resumes generator + discriminator from our checkpoint
```

## License

**Research / non-commercial use only.** The fine-tuning data includes CC BY-NC /
CC BY-NC-SA datasets and internal (non-redistributable) corpora, so the resulting
weights inherit non-commercial terms. The DAC code/architecture is MIT
(© Descript). Do not use these weights in commercial products.

## Acknowledgements

- [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) (Kumar et al., NeurIPS 2023) — base model & training code.
- Public singing datasets, with thanks to their authors:
  [ACE-KiSing](https://huggingface.co/datasets/espnet/ace-kising-segments),
  [M4Singer](https://github.com/M4Singer/M4Singer),
  [CSD](https://zenodo.org/record/4785016).
- **AI-Hub datasets** (MSSV / GV), built with support from the National Information Society
  Agency (NIA), Republic of Korea:

  > This work used datasets from **"The Open AI Dataset Project (AI-Hub, S. Korea)"** —
  > [Multi-Speaker Singing Voice (#465)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=465)
  > and [Guide Vocal (#473)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=473).
  > All data information can be accessed through **[AI-Hub](https://www.aihub.or.kr)**.

- Compute: _alpha-test program — provider name TBD_ (1× NVIDIA RTX PRO 6000, Blackwell).

## Citation

This work is a fine-tune of the **Descript Audio Codec**; please cite the original paper:

```bibtex
@inproceedings{kumar2023high,
  title     = {High-Fidelity Audio Compression with Improved {RVQGAN}},
  author    = {Kumar, Rithesh and Seetharaman, Prem and Luebs, Alejandro and
               Kumar, Ishaan and Kumar, Kundan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}
```

If you use these fine-tuned weights, please also link this repository.

### Dataset references
- **M4Singer** — Zhang et al., *M4Singer: A Multi-Style, Multi-Singer and Musical Score Provided Mandarin Singing Corpus*, NeurIPS 2022.
- **CSD** — Choi et al., *Children's Song Dataset for Singing Voice Research*, ISMIR 2020.
- **ACE-KiSing / Opencpop** — see the [ESPnet](https://github.com/espnet/espnet) singing recipes.
