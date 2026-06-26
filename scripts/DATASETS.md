# Public singing-voice datasets — download & license

Goal: collect research/non-commercial singing data. **Only audio is needed (no labels)** for
codec training. After downloading, preprocess to 24 kHz mono with:
`python scripts/preprocess.py --in_dir <downloaded_dir> --out_dir data/train/<name> [--segment_sec 30]`

> ⚠️ Most of these are **CC BY-NC** (non-commercial). Fine for research, but not for commercial use.

## Tier 1 — one-command, no human gate
```bash
cd data/raw   # any working dir

# CSD (Children's Song) — CC BY-NC-SA 4.0, ~3h, WAV 44.1k
wget -O CSD.zip "https://zenodo.org/records/4785016/files/CSD.zip?download=1"

# MUSDB18-HQ — vocals stem only. Non-commercial/academic, ~10h vocals, 22.7GB (all stems)
wget -O musdb18hq.zip "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
#   -> use only vocals.wav from each track folder

# ACE-KiSing (KiSing-v2) — CC BY-NC 4.0, ~32.5h, stored as HF parquet (audio inside)
huggingface-cli download espnet/ace-kising-segments --repo-type dataset --local-dir ./ace-kising
#   -> extract audio: python scripts/extract_parquet_audio.py --parquet_dir ace-kising/data \
#         --out_dir data/train/ace_kising --audio_col audio --id_col segment_id

# M4Singer (HF mirror) — upstream CC BY-NC-SA 4.0, ~30h, 10GB zip
huggingface-cli download --repo-type dataset umoubuton/m4singer m4_opencpop.zip --local-dir ./m4singer
#   note: the zip may also contain Opencpop (CC BY-NC-ND) — exclude it if you need derivative rights
```
CSD + MUSDB(vocals) + ACE-KiSing + M4Singer ≈ **75h** (exceeds a 50h target).

## Tier 2 — automatable via Google Drive (`pip install gdown`, may hit quota)
```bash
pip install gdown
gdown 1EofoZxvalgMjZqzUEuEdleHIZ6SHtNuK   # OpenSinger,  CC BY-NC-SA 4.0, ~50h, WAV 24k
gdown 1xC37E59EWRRFFLdG3aJkVqwtLDgtFNqW   # M4Singer (official alt), CC BY-NC-SA 4.0, ~30h
```
For large public Drive files use `gdown --fuzzy <view-url>`. (No verified HF mirror of the
original OpenSinger WAV corpus exists — only codec-processed derivatives, which are not the raw set.)

## Tier 3 — manual gate (form / email) — apply early, cannot be scripted
| Dataset | Gate | License | Hours |
|---|---|---|---|
| Opencpop | Google Form + email (forms.gle/LnsbLqE6GcExhT5U6) | CC BY-NC-ND 4.0 | ~5.2h |
| PopCS (DiffSinger) | email author (jinglinliu@zju.edu.cn) | CC BY-NC-SA 4.0 | ~5.9h |
| NUS-48E | Drive folder only (`gdown --folder <id>`) | license unverified (research) | ~2.8h |
| NHSS | signed EULA by email | research-only (signed) | ~7h |

## Notes / unverified
- Exact GB sizes for ACE-KiSing, M4Singer (beyond the 10GB zip), Opencpop, PopCS, NUS-48E, NHSS are not published.
- Native sample rates for M4Singer/PopCS/NUS-48E/NHSS are unconfirmed — preprocessing resamples to 24k anyway.
- Original KiSing v1 site was unreachable; use ACE-KiSing instead.
- `opencpop` is CC BY-NC-**ND** (no derivatives) — excluded from this project's training set.
