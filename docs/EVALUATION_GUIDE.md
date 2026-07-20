# PrivaText-CXR — 평가 지시서 (Evaluation Handoff Guide)

> **이 문서의 용도**: 이 저장소에서 생성한 흉부 X선 이미지의 평가 결과(`./eval/eps<N>/`)를
> **다른 채팅/세션에서 분석**하기 위한 자기완결적 안내서. 코드를 열지 않고도
> (1) 연구 파이프라인, (2) 각 지표의 의미·방향·읽는 법, (3) 결과 파일 구조,
> (4) 분석 시 해석 지침을 알 수 있도록 정리했다.

---

## 1. 연구 개요 (무엇을 만들고 왜 평가하나)

- **목표**: **이미지 단위 차등 프라이버시(DP)** 하에서, 방사선 리포트(텍스트)를 조건으로
  흉부 X선(CXR)을 생성하는 **Latent Diffusion Model(LDM)**.
- **구성**: `VAE(latent)` + `UNet diffusion`, 조건은 **BioBERT** 텍스트 임베딩.
  사전학습 LDM을 **DP-SGD + LoRA**로 미세조정 → privacy budget **epsilon(ε=1/3/5/10)** 별
  체크포인트(`ldm_lora_eps*.pt`) 생성. LoRA `alpha`, `MTDDPM` 등 변형 간 비교.
- **평가 의도(Utility 관점)**: 생성물의 **① 이미지 품질**, **② 특징공간 분포**,
  **③ 텍스트-이미지 정합**을 측정하고, **모델(ε/alpha/변형) 간 비교** + **privacy–utility tradeoff**를 정량화.
- **핵심 전제**: 학습에는 CLIP/contrastive alignment loss가 **없다**. CLIPScore·분류기 등은
  전부 **외부 사전학습 모델을 이용한 평가 전용** 지표이며 조건용 BioBERT와 다른 모델이라
  circularity(순환 참조)가 없다.

## 2. 파이프라인 (end-to-end)

```
MIMIC-CXR (image + report)
   └─ VAE 학습            (Autoencoder_train.py)              → vae_ep*.pt
   └─ LDM 사전학습        (LDM_train.py)                      → ldm_epoch*.pt
   └─ DP LoRA 미세조정    (LDM_dp_finetune.py, DP-SGD+LoRA)   → ldm_lora_eps{1,3,5,10}.pt
                              (ckpt 안에 args, epsilon_spent, lora_rank/alpha 저장)
   └─ 추론/생성           (LDM_dp_inference.py 또는 generate_and_eval.py)
                              → ./<gen>/samples/*.png + descriptions.csv + grid_all.png
   └─ 평가                (아래 §4 스크립트)                  → ./eval/eps<N>/*.json, tsne.png
```

- 생성물 폴더 규약: `./generated_eps<N>/` (또는 임의 `--gen_output_dir`) 안에
  `samples/`(개별 PNG), `grid_all.png`(n×n 미리보기), `descriptions.csv`(index,description,file).
- **`descriptions.csv`** 는 각 생성 이미지가 어떤 프롬프트로 만들어졌는지의 매핑 →
  CLIPScore가 이미지–프롬프트 페어링에 사용.

## 3. 결과 디렉토리 구조 (분석 시 읽을 파일)

```
./eval/eps<N>/
    eval_summary.json     # 모든 지표가 병합되는 메인 파일 (여기부터 보면 됨)
    fds.json              # FDS 단독 결과 (양방향 KL)
    clipscore.json        # CLIPScore(gen/real/gap) 단독 결과
    tsne.png              # real vs generated 특징공간 산점도
    tsne_embedding.npz    # t-SNE 2D 좌표 (재플롯용: emb, n_real, perplexity)
    ckpt_info.json        # 이 결과를 만든 ckpt의 학습 args, epsilon_spent, lora_rank/alpha
    _features/            # feature 캐시(real_*.npz, gen_*.npz) — 재실행 가속용, 분석엔 불필요
    eval_pairs.csv, pairs/# (LDM_dp_eval.py 사용 시) paired ssim/psnr/lpips 상세
```

- **분석은 `eval_summary.json` 하나로 대부분 가능**. `ckpt_info.json`으로 어떤 ε/설정인지 확인.
- 여러 지표 스크립트가 **같은 `--out_dir`을 쓰면** `eval_summary.json`에 **누적 병합**된다
  (`run_feature_eval`/`generate_and_eval`는 기존 키를 보존하며 갱신). 단
  **`LDM_dp_eval.py`는 덮어쓰므로, paired 지표를 먼저 돌린 뒤 feature/CLIP 지표를 나중에** 돌릴 것.

## 4. 지표 레퍼런스 (개념 · 방향 · JSON 키 · 읽는 법 · 주의)

방향 표기: **(↑)** 클수록 좋음, **(↓)** 작을수록 좋음.

### 4.1 이미지 품질 — 분포 기반 (feature 공간)

| 지표 | 방향 | 의미 | 주 backbone |
|---|---|---|---|
| **FID** | ↓ | real·gen feature를 각각 가우시안으로 보고 **Fréchet(2-Wasserstein) 거리** | `xrv`(1024, CXR도메인) / `inception`(2048) |
| **FDS (FeatureKL)** | ↓ | 같은 가우시안 사이 **KL divergence**. 비대칭이라 방향이 정보 | 동일 |

- **FID** — `eval_summary.json.fid`, `fid_backbone`. 낮을수록 생성 분포가 real 분포에 가까움.
- **FDS** — `fds.json` / summary 키:
  - `fds_gen_given_real` = D_KL(gen‖real): **hallucination**(real 밖 mode 생성)에 민감.
  - `fds_real_given_gen` = D_KL(real‖gen): **mode collapse**(real mode 누락)에 민감.
  - `fds_symmetric` = 둘의 평균.
  - `pca_dim` 이 기록됨(=PCA 축소 차원, null이면 미사용). `feature_dim`, `n_real`, `n_gen` 참고.
  - **읽는 법**: `real_given_gen ≫ gen_given_real` 이면 **mode collapse 경향**(생성 다양성 부족).
    반대면 hallucination 경향.

> **backbone 선택 의미**: `xrv`(TorchXRayVision DenseNet-121)는 CXR로 사전학습 → 임상적으로
> 더 유의미. `inception`은 자연영상 통계라 참고용.

### 4.2 이미지 품질 — 픽셀/지각 기반 (paired, `LDM_dp_eval.py`)

| 지표 | 방향 | 의미 |
|---|---|---|
| **SSIM** | ↑ | 구조 유사도(밝기·대비·구조) |
| **PSNR** | ↑ | 픽셀 MSE 기반 신호대잡음비 (dB) |
| **LPIPS** | ↓ | 학습된 특징 기반 **지각적** 거리 |

- 키: `ssim_mean/ssim_best`, `psnr_mean/psnr_best`, `lpips_mean/lpips_best`, `n`, `eval_split`, `prompt_mode`.
- **매우 중요한 해석 주의**: **생성 ≠ 복원**. 하나의 리포트는 여러 real 이미지에 대응하므로
  **SSIM/PSNR은 낮게 나오는 게 정상**이고 **거친 proxy**일 뿐이다. 절대값으로 품질을 판단하지 말 것.
  지각 지표 **LPIPS**와 분포 지표 **FID/FDS**가 더 신뢰성 있다.

### 4.3 특징공간 시각화 — T-SNE (정성)

- 산출물: `tsne.png`(파랑=real, 빨강=generated), `tsne_embedding.npz`.
- **읽는 법**:
  - 두 색이 **겹치면 mode coverage 양호**, 빨강이 **한쪽에 뭉치거나 분리되면 collapse/도메인 이탈**.
  - `summary.tsne_perplexity` 기록됨. 축·전역거리는 **의미 없음**(정성 도구), seed 고정됨.

### 4.4 텍스트-이미지 정합 — CLIPScore (MedCLIP, 기본 백엔드)

- 개념: `CLIPScore = w·max(cos(f_img, f_txt), 0)` (w=2.5). MedCLIP의 이미지·텍스트 인코더가
  **공유 공간**에 정렬 → 코사인이 "이미지가 프롬프트 의미를 담았는가"를 측정. **(↑)**.
- `clipscore.json` / summary 키:
  - `clip_gen_clipscore_mean/std`, `clip_gen_cos_mean/std`, `clip_gen_n` — **생성물 정합**.
  - (real 기준선 활성 시) `clip_real_clipscore_mean/...` — real (image,report) 정합 **상한(ceiling)**.
  - **`clip_gap` = real − gen (↓)** — 인코더마다 코사인 스케일이 달라 **절대값 대신 gap으로 모델 간 비교**.
  - `clip_backend`(=medclip).
- **읽는 법**: `clip_gap`이 작을수록 생성물이 real 수준의 텍스트 정합에 근접. ε가 낮아질수록
  gap이 커지는 경향을 기대(=프라이버시 강화 시 정합 저하).

### 4.5 (계획) Downstream classification — TSTR / label-agreement

- 아직 미구현(다음 단계). TSTR=생성셋으로 분류기 학습 후 real test AUROC, 또는 사전학습 CXR
  분류기의 조건 라벨 회수율. ε별 AUROC 곡선으로 privacy–utility tradeoff 정량화 예정.

## 5. 실행 방법 (재현/추가 생성이 필요할 때)

**A) 생성 + 평가 원샷** (ckpt만 있고 이미지가 아직 없을 때):
```bash
python Eval_metric/generate_and_eval.py \
  --dp_ckpt ./checkpoints/ldm/ldm_epoch0100.pt \
  --lora_ckpt ./finetune_dp/.../ldm_lora_eps1.pt \
  --vae_ckpt ./checkpoints/vae/vae_ep0070.pt \
  --prompt_source split --n_prompts 361 --n_samples 1 \
  --device_id 0 --gen_output_dir ./EVAL/gen_out/eps1 --out_dir ./eval/eps1 \
  --root_path /storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0 \
  --eval_split test --max_real 361 \
  --eval_model xrv --pca_dim 64 \
  --metrics fds tsne fid clip --clip_backend medclip
```

**B) 디스크의 기존 생성물만 평가**:
```bash
python Eval_metric/run_feature_eval.py \
  --gen_dir ./EVAL/gen_out/eps1/samples --out_dir ./eval/eps1 \
  --root_path /storage/.../mimic-cxr/2.1.0 --eval_split test --max_real 361 \
  --device cuda:0 --eval_model xrv --pca_dim 64 \
  --metrics fds tsne fid clip --clip_backend medclip \
  --ckpt ./finetune_dp/.../ldm_lora_eps1.pt
```

**C) paired SSIM/PSNR/LPIPS(+FID)** — 같은 `--output_dir`에 먼저 실행하면 summary에 누적:
```bash
python LDM_dp_eval.py --dp_ckpt ... --lora_ckpt ... --vae_ckpt ... \
  --root_path /storage/.../mimic-cxr/2.1.0 --eval_split test --max_eval 361 \
  --metrics ssim psnr lpips --compute_fid true --eval_model xrv \
  --output_dir ./eval/eps1
```

**CLIPScore 단독**:
```bash
python Eval_metric/clipscore.py --gen_dir ./EVAL/gen_out/eps1/samples \
  --backend medclip --device cuda:0 \
  --root_path /storage/.../mimic-cxr/2.1.0 --eval_split test --max_real 361 \
  --output ./eval/eps1/clipscore.json
```

의존성: `pip install torch torchvision numpy scipy scikit-learn pillow matplotlib torchxrayvision medclip`.

## 6. 결과 분석 지침 (분석 채팅용 핵심)

1. **비교 단위**: ε(1/3/5/10)별 `./eval/eps<N>/eval_summary.json`을 한 표로 모아 비교.
   각 행의 정체는 `ckpt_info.json`(epsilon_spent, lora_rank/alpha, args)로 확인.
2. **기대 서사(privacy–utility tradeoff)**: ε가 **낮을수록**(프라이버시 강함, 노이즈↑)
   → **FID/FDS ↑, LPIPS ↑, CLIP gap ↑, SSIM/PSNR ↓, T-SNE 클러스터 붕괴** 경향.
   이 단조 경향이 보이면 프레임워크가 정상 작동한다는 신호.
3. **신뢰도 우선순위**: 분포·지각 지표(**FID, FDS, LPIPS, CLIP gap**) > 픽셀 proxy(SSIM/PSNR).
   SSIM/PSNR이 낮다고 "품질 나쁨"으로 결론짓지 말 것(생성≠복원).
4. **공정 비교 조건**: 모든 ε에서 **동일한 real셋·동일 n·동일 backbone·동일 pca_dim**이어야 함.
   `n_real`, `n_gen`, `fid_backbone`, `pca_dim`, `eval_split`을 각 json에서 대조 확인.
5. **FDS 방향 해석**: `fds_real_given_gen` vs `fds_gen_given_real`의 대소로 **collapse vs hallucination**
   진단. 절대값은 표본 수에 민감하니 **모델 간 상대 비교**로 사용.
6. **CLIP은 gap으로**: `clip_gap`(real−gen)으로 비교. 절대 `clip_gen_clipscore_mean`만으로 판단 금지.
7. **T-SNE는 근거 보강용**: 수치(FID/FDS)의 결론을 시각적으로 확인하는 용도.

## 7. 알아둘 제약/함정

- **real test 상한**: 미다운로드 DICOM이 있어 test split 실제 사용량이 제한될 수 있음(예: 361장).
  이는 FID/FDS 공분산 추정 품질의 상한을 정한다.
- **FDS와 표본 수**: `n ≤ feature_dim`이면 공분산이 특이 → 값 폭발. 코드가 **Ledoit-Wolf(`--fds_cov lw`, 기본)**
  로 안정화하고 경고를 출력한다. 절대값 안정화를 원하면 **생성 샘플 증량** 또는 **`--pca_dim 64`**
  (PCA는 real에 fit → subspace 고정으로 모델 간 비교 유지). 단 **gen도 `pca_dim`보다 커야** 함.
- **프롬프트 다양성**: 소수 프롬프트로 대량 생성하면 분포가 인위적으로 좁아짐 →
  분포 지표엔 **`--prompt_source split`**(test 리포트 사용)이 바람직.
- **PyTorch ≥2.6**: 우리 ckpt는 비-텐서 객체 포함 → `torch.load(..., weights_only=False)` 사용(코드 반영됨).
- **CLIP 백엔드 부재 시**: 라이브러리/가중치 없으면 드라이버가 `[clip] skipped ...`로 안전하게 건너뜀
  (다른 지표는 정상 진행).
- **feature 캐시**: `_features/`는 재실행 가속용. real셋/설정을 바꾸면 캐시가 자동 무효화(키 불일치)된다.

## 8. 코드 위치 (참조)

- 지표 구현: `Eval_metric/`
  - `features.py`(feature 추출+캐시), `fds.py`(FDS), `tsne_viz.py`(T-SNE),
    `clipscore.py`(CLIPScore: medclip/biovil-t/cxr-clip), `fid.py`(FID),
    `psnr.py`/`ssim.py`(paired 픽셀 지표), `../Loss/lpips.py`(LPIPS).
  - 드라이버: `run_feature_eval.py`(기존 이미지 평가), `generate_and_eval.py`(생성+평가 원샷).
- paired 생성-평가: `LDM_dp_eval.py`. 생성: `LDM_dp_inference.py`.
- 계획 문서: `docs/EVAL_PLAN.md`, 프레임워크: `docs/DP_LDM_FRAMEWORK_PLAN.md`.
