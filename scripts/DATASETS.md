# 공개 노래 데이터셋 — 다운로드 & 라이선스 (2026 조사)

목표: 학습 허용(연구/비상업) 라이선스의 노래 음성 ≥50시간 확보. **코덱 학습이라 오디오만 필요(라벨 불필요).**

> ⚠️ 라이선스 주의: 아래 대부분이 **CC BY-NC**(비상업) 계열입니다. 연구용 fine-tune에는 문제없지만,
> 이 코덱을 **상업 배포**할 계획이면 NC 제한이 걸립니다. 9개 중 상업 허용은 없습니다.

전처리는 모두 동일: `python scripts/preprocess.py --in_dir <받은폴더> --out_dir data/train/public/<name>`
(어떤 sample rate든 24kHz mono로 변환됨).

---

## TIER 1 — 지금 바로 단일 명령으로 자동 다운로드 (사람 개입 불필요)

```bash
cd /root/dac_singing/data/raw
PY=/root/miniconda3/envs/dac_ft/bin/python   # huggingface-cli는 이 env에 설치됨

# 1) CSD (Children's Song) — 가장 깨끗함. CC BY-NC-SA 4.0, ~3h, 1.9GB, WAV 44.1k
wget -O CSD.zip "https://zenodo.org/records/4785016/files/CSD.zip?download=1"

# 2) MUSDB18-HQ — vocals stem만 사용. 비상업/학술 전용, ~10h(vocals), 22.7GB(전체 stem)
wget -O musdb18hq.zip "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
#   → 각 트랙 폴더의 vocals.wav 만 모아서 전처리 (mixture/drums/bass/other 제외)

# 3) ACE-KiSing (KiSing-v2) — CC BY-NC 4.0, ~32.5h, WAV 48k mono  (원조 KiSing 대신 이걸 사용)
huggingface-cli download espnet/ace-kising-segments --repo-type dataset --local-dir ./ace-kising

# 4) M4Singer (HF 미러) — 업스트림 CC BY-NC-SA 4.0, ~30h, 10GB zip
huggingface-cli download --repo-type dataset umoubuton/m4singer m4_opencpop.zip --local-dir ./m4singer
#   ※ zip 이름이 m4_opencpop.zip → Opencpop이 함께 들어있을 수 있음. 풀고 내용 확인.
```
- CSD/MUSDB18-HQ는 Zenodo 정식 레코드(DOI 10.5281/zenodo.4785016, .../3338373), 직링크 안정적.
- MUSDB18-HQ만 라이선스 뉘앙스 주의("비상업/교육 전용"). vocals.wav만 필요.
- M4Singer: HF 카드는 MIT라 표기되나 **업스트림 공식은 CC BY-NC-SA 4.0(NC)** — 업스트림 기준 따를 것.

→ TIER 1만으로 CSD(3) + MUSDB vocals(10) + ACE-KiSing(32.5) + M4Singer(30) ≈ **75h** 확보 가능 (목표 50h 초과).

## TIER 2 — 자동이지만 Google Drive 경유 (`gdown` 필요, 쿼터 주의)

```bash
/root/miniconda3/envs/dac_ft/bin/pip install gdown
# OpenSinger — CC BY-NC-SA 4.0, ~50h, WAV 24k/16bit
gdown 1EofoZxvalgMjZqzUEuEdleHIZ6SHtNuK
# M4Singer (공식, HF 대안) — CC BY-NC-SA 4.0, ~30h
gdown 1xC37E59EWRRFFLdG3aJkVqwtLDgtFNqW
```
- 대용량 공개 Drive는 `gdown --fuzzy <뷰어URL>`로 바이러스검사 인터스티셜 처리.
- OpenSinger의 **원본 WAV** HF 미러는 없음(코덱 가공 파생본만 존재, 사용 금지).

## TIER 3 — 수동 승인 필요 (폼/이메일) — 스크립트 불가, 시간 걸리니 미리 신청

| Dataset | 게이트 | 위치 | License | Hours |
|---------|--------|------|---------|-------|
| Opencpop | Google Form+이메일 | forms.gle/LnsbLqE6GcExhT5U6 / zpcoftts@gmail.com | CC BY-NC-ND 4.0 | ~5.2h |
| PopCS (DiffSinger) | 저자 이메일 신청 | jinglinliu@zju.edu.cn | CC BY-NC-SA 4.0 | ~5.9h |
| NUS-48E | Drive 폴더만(폼 없음) | `gdown --folder "https://drive.google.com/drive/folders/12pP9uUl0HTVANU3IPLnumTJiRjPtVUMx"` | 공식 라이선스 미확인(연구용 가정) | ~2.8h |
| NHSS | 서명 EULA 이메일 | NUSLicence.pdf 서명 → bidisha.iitg@gmail.com, haizhou.li@nus.edu.sg | 연구 전용(서명) | ~7h |

## 권장 수집 순서 (헤드리스 자동화)
1. CSD → 2. MUSDB18-HQ(vocals) → 3. ACE-KiSing → 4. M4Singer(HF)  *(여기까지 단일 명령, ~75h)*
5. OpenSinger(+M4Singer 공식) via gdown
6. Opencpop 폼 / PopCS·NHSS 이메일은 승인 지연되니 병행 신청

## 미검증 항목 (플래그)
- GB 용량: ACE-KiSing, M4Singer(10GB zip 외), Opencpop, PopCS, NUS-48E, NHSS 미공개.
- Sample rate: M4Singer(보통 24k 추정), PopCS, NUS-48E, NHSS 미확인 — 전처리가 어차피 24k로 통일하므로 무관.
- NUS-48E 정식 라이선스 미확인(폴더 README 확인). 원조 KiSing v1 사이트 다운 → ACE-KiSing 사용.
- PopCS 공개 Drive 링크는 존재하나 신청-게이트 의도라 철회될 수 있음 → 정식 신청 권장.
