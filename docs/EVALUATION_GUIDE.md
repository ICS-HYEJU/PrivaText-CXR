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
   └─ DP 미세조정         (LDM_dp_finetune.py, DP-SGD [+LoRA])
                              → LoRA 어댑터 ldm_lora_eps{1,3,5,10}.pt  또는
                                전체 DP 체크포인트 ldm_dp_final.pt (ckpt 안에 args/epsilon_spent 저장)
   └─ 추론/생성           (LDM_dp_inference.py 또는 generate_and_eval.py)
                              → ./<gen>/samples/*.png + descriptions.csv + grid_all.png
                                (--text_mode로 프롬프트에 쓸 리포트 섹션 선택)
   └─ 평가                (아래 §4 스크립트)                  → ./eval/eps<N>/*.json, tsne.png
```

- 생성물 폴더 규약: `./generated_eps<N>/` (또는 임의 `--gen_output_dir`) 안에
  `samples/`(개별 PNG), `grid_all.png`(n×n 미리보기), `descriptions.csv`(index,description,file).
- **`descriptions.csv`** 는 각 생성 이미지가 어떤 프롬프트로 만들어졌는지의 매핑 →
  CLIPScore가 이미지–프롬프트 페어링에 사용.

## 3. 결과 디렉토리 구조 (분석 시 읽을 파일)

```
./eval/eps<N>/
    eval_summary.json      # 모든 지표가 병합되는 메인 파일 (여기부터 보면 됨)
    fds.json               # FDS 단독 결과 (양방향 KL)
    clipscore.json         # CLIPScore(gen/real/gap/retrieval/negative-control)
    label_agreement.json   # label-agreement per-pathology AUROC + support (all/nih)
    label_auroc_roc.png    # label ROC 곡선 (gen vs real, 분류기×신뢰 병변)
    tsne.png               # real vs generated 특징공간 산점도
    tsne_embedding.npz     # t-SNE 2D 좌표 (재플롯용: emb, n_real, perplexity)
    ckpt_info.json         # 이 결과를 만든 ckpt의 학습 args, epsilon_spent, lora_rank/alpha
    _features/             # feature 캐시(real_*.npz, gen_*.npz) — 재실행 가속용, 분석엔 불필요
    _clip_real_tmp/        # CLIP real 기준선용 임시 PNG (분석엔 불필요)
    eval_pairs.csv, pairs/ # (LDM_dp_eval.py 사용 시) paired ssim/psnr/lpips 상세
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

### 4.2 이미지 품질 — 픽셀/지각 기반 (paired)

> 두 경로에서 생성됨: (1) `LDM_dp_eval.py`, 또는 (2) 통합 드라이버
> `run_feature_eval.py`/`generate_and_eval.py`에 `--metrics ... ssim psnr lpips` 추가 시.
> 통합 드라이버의 paired 지표는 **생성 프롬프트가 eval split에서 왔을 때만**(gen 인덱스↔real 인덱스
> 매핑이 성립) 계산된다: `generate_and_eval`은 `--prompt_source split`이면 자동 활성,
> `run_feature_eval`은 `--paired_from_split` 플래그 필요. 조건 미충족 시 `[paired] skipped ...`.
> 키에 `n_paired`, `*_std`가 함께 기록됨.

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

- 개념: MedCLIP의 이미지·텍스트 인코더가 **공유 공간**에 정렬 → 코사인이 "이미지가 프롬프트
  의미를 담았는가"를 측정. **(↑)**.
- **주 지표 = `cos_mean`(raw 코사인)**. `clipscore_mean = w·max(cos,0)`(w=2.5)은 OpenAI CLIP
  기준 스케일이라 MedCLIP에선 **참고값**으로만 본다. 둘 다 json에 기록됨.
- **텍스트 섹션 = `--text_mode`** ∈ `FINDINGS` / `FINDINGS/IMPRESSION`(기본) / `FULL`.
  **생성 프롬프트(inference)와 CLIP 텍스트가 동일 모드**를 씀(`text_utils.extract_report_sections`).
  `generate_and_eval`는 `--text_mode`가 생성·CLIP 둘 다 지배(clip은 미지정 시 자동 일치).
  - ⚠️ **fallback**: 요청한 섹션이 없는 리포트(예: FINDINGS 없이 IMPRESSION만, 또는 둘 다 없음)는
    **빈 프롬프트 방지를 위해 원문 전체로 대체**된다. 그래서 `--text_mode FINDINGS`인데도 일부
    프롬프트가 `IMPRESSION: ...`/`FINAL REPORT ...`로 보일 수 있다(= FINDINGS 섹션이 없는 리포트).
- **negative control(항상 산출)**: `cos_shuffled_mean`(이미지 vs 무작위 다른 리포트=노이즈 바닥),
  **`cos_signal = cos_mean − cos_shuffled_mean`**. signal이 0에 가까우면 그 인코더는 이 데이터에서
  변별력 없음(작은 gap 무의미). signal이 뚜렷이 양수여야 CLIP 해석이 유효.
- **중복 프롬프트**: `--clip_dedup` 주면 retrieval을 **고유 프롬프트 부분집합**에서 계산(반복 많을 때 공정).
- `clipscore.json` / summary 키:
  - `clip_gen_cos_mean/std`(주), `clip_gen_clipscore_mean/std`(참고), `clip_gen_n`.
  - **negative control**: `clip_gen_cos_shuffled_mean`, `clip_gen_cos_signal`(=cos−shuffled).
  - retrieval(i2t·t2i): `clip_gen_R@{1,5,10}_*`(Recall@k, ↑), `clip_gen_P@{1,5,10}_*`
    (Precision@k, ↑), `clip_gen_mAP_*`(mean Average Precision, ↑),
    `clip_gen_median_rank_*`(↓), `clip_gen_mean_rank_*`, `clip_gen_duplicate_prompts`.
    정답 판정은 텍스트 동일성(중복 프롬프트=다중 정답 반영).
  - (real 기준선 활성 시) `clip_real_*` — real (image,report) 정합/retrieval **상한(ceiling)**.
  - **`clip_gap_cos` = real_cos − gen_cos (↓)** ⭐ 주 비교값 (스케일 제거). `clip_gap_clipscore`도 병기.
  - `clip_backend`(=medclip), `clip_text_mode`.
- **읽는 법**: `clip_gap_cos`가 작을수록 생성물이 real 수준 정합에 근접. retrieval R@k가 높을수록
  "생성 이미지가 자기 프롬프트를 잘 회수" = 정합 강함. ε↓일수록 gap↑·R@k↓ 경향 기대.
- **주의**: retrieval은 **프롬프트가 다양·고유할 때** 의미가 큼(`--prompt_source split --n_samples 1`
  권장). 중복 프롬프트가 많으면 `duplicate_prompts`로 확인. 또한 **FINDINGS/IMPRESSION 프롬프트로
  생성한 이미지**여야 정합 해석이 맞으므로, 정책 변경 시 **모든 ε 재생성** 권장.

### 4.4b CLIP 유효성 진단 — `clip_diagnose` (opt-in)

real image-report cosine이 shuffled와 거의 같으면(signal≈0) CLIP 지표를 그대로 쓸 수 없다.
`--metrics clip_diagnose`는 **real 이미지에서 인코더가 실제로 작동하는지** 검증(기존 지표/기본값
불변, `clip_diagnose.json` + `clipdiag_*` 키만 추가):
- **sanity**: `logit_scale`(학습된 CLIP ~50–100, ~14/1이면 미로드 의심), 임베딩 반복성(cos≈1).
- **matched vs shuffled 통계**: `signal_mean` + **95% bootstrap CI**(`signal_ci_low/high`,
  `signal_ci_excludes_zero`) + **permutation p** + effect size. tiny signal이 통계적으로 유의한지.
- **zero-shot label AUROC**(⭐결정적): 짧은 병변 프롬프트 vs CheXpert GT.
  **≫0.5면 인코더 정상 → 긴 report 텍스트가 문제**; **≈0.5면 인코더/전처리가 깨짐**.
- **label-level retrieval**: 같은 CheXpert 라벨 공유=정답(관대). exact-text는 0이라도 여기서
  유의미하면 CLIP을 **보조 semantic 지표**로 사용 가능.
- 실행: `python Eval_metric/clip_diagnose.py --backend medclip --root_path ... --eval_split test
  --max_real 361 --text_mode FINDINGS --output ./.../clip_diagnose.json` (또는 드라이버 metric).

### 4.5 Downstream classification — label-agreement (구현됨), TSTR (계획)

- **label-agreement**(`--metrics ... label`): 사전학습 **XRV DenseNet-121**을 생성/real 이미지에
  적용해 **CheXpert GT 라벨(mimic-cxr-2.0.0-chexpert.csv[.gz])** 대비 AUROC 산출.
- 두 가중치를 함께: `densenet121-res224-nih`(NIH=사전학습 도메인, out-of-domain·엄격) +
  `densenet121-res224-all`(MIMIC 포함·in-domain·상한 높음, **FID 백본과 공유이므로 덜 독립적**).
- `label_agreement.json` / summary 키 (weight별 short=`all`/`nih`):
  - `label_<short>_auroc_gen_macro`, `label_<short>_auroc_real_macro`,
    **`label_<short>_auroc_gap_macro`(=real−gen, ↓)** ⭐, `label_<short>_auroc_ratio_macro`(=gen/real, ↑).
  - per-pathology AUROC(gen/real)와 support는 `label_agreement.json`에만.
- **읽는 법**: **분류기 절대 AUROC는 weight set 간 직접 비교 금지**. 각 분류기의 **gen을 자기 real과**
  비교(gap/ratio)해야 공정. gap↓·ratio→1일수록 생성물이 real 수준의 병변 판별력 보유.
- **ROC 그래프**: `label`을 돌리면 `out_dir`에 **`label_auroc_roc.png`** 저장(양쪽 eval 경로 모두).
  분류기(all/nih) × 신뢰 병변별로 **gen(빨강 실선) vs real(파랑 점선) ROC 곡선**, 범례에 AUROC 표기.
- **제약**: `--paired_from_split` 필요(gen index→split study join). 불확실 라벨(-1) **drop**.
  채점 라벨 = 각 분류기 유효라벨(op_threshs) ∩ CheXpert GT (nih≈7, all≈11).
- **macro 신뢰도**: per-pathology AUROC는 양성이 1~2개면 `1.0`/`0.5` 같은 노이즈가 됩니다.
  그래서 **macro는 양성·음성 각각 `--label_min_pos`(기본 10) 이상인 병변만** 평균합니다
  (`macro_pathologies`에 포함 목록 기록). per-pathology와 `support`(n_pos/n_neg)는 전부 남으니
  **신뢰 판단은 support로**. n=361처럼 표본이 작으면 대개 Cardiomegaly/Effusion/Edema 정도만 신뢰 가능.
- **TSTR**(Train on Synthetic, Test on Real)는 계획 단계(생성셋으로 분류기 학습 후 real AUROC).

## 5. 실행 방법 (재현/추가 생성이 필요할 때)

> **⚠️ 반드시 `/opt/medclip_env/bin/python`으로 실행** (§9 환경 참고). MedCLIP은 transformers
> 4.24.0 + `wget`가 있는 이 venv에서만 동작하며, 이 venv가 torch·torchxrayvision·medclip을
> 공유하므로 **모든 지표(fds/fid/tsne/label/clip/ssim/psnr/lpips)가 한 env에서 다 돈다.**

**A) 생성 + 평가 원샷** — 한 명령으로 전체 summary 생성:
```bash
/opt/medclip_env/bin/python Eval_metric/generate_and_eval.py \
  --dp_ckpt   ./finetune_dp/lr2e-3/ldm_dp_final.pt \
  --vae_ckpt  ./checkpoints/vae/vae_ep0070.pt \
  --device_id 0 \
  --prompt_source split --n_prompts 361 --n_samples 1 --text_mode FINDINGS \
  --gen_output_dir ./EVAL/gen_out/lr2e-3/eps10/findings \
  --out_dir        ./EVAL/metric/lr2e-3/eps10/findings \
  --root_path /storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0 \
  --eval_split test --max_real 361 --eval_model xrv --pca_dim 64 \
  --metrics fds tsne fid clip ssim psnr lpips label
```
- `--lora_ckpt`는 LoRA 어댑터를 쓸 때만. 전체 DP 체크포인트(`ldm_dp_final.pt`)면 `--dp_ckpt`만.
- `--prompt_source split`이면 paired 지표(ssim/psnr/lpips/label)가 자동 활성(`paired_from_split`).
- `--text_mode` ∈ FINDINGS / FINDINGS/IMPRESSION / FULL — **생성 프롬프트와 CLIP 텍스트에 동시 적용**.

**B) 디스크의 기존 생성물만 평가** (paired/label엔 `--paired_from_split` 필수):
```bash
/opt/medclip_env/bin/python Eval_metric/run_feature_eval.py \
  --gen_dir ./EVAL/gen_out/lr2e-3/eps10/findings/samples \
  --out_dir ./EVAL/metric/lr2e-3/eps10/findings \
  --root_path /storage/.../mimic-cxr/2.1.0 --eval_split test --max_real 361 \
  --paired_from_split --device cuda:0 --eval_model xrv --pca_dim 64 \
  --clip_text_mode FINDINGS \
  --metrics fds tsne fid clip ssim psnr lpips label \
  --ckpt ./finetune_dp/lr2e-3/ldm_dp_final.pt
```

**CLIPScore 단독** (negative-control·중복제외 포함):
```bash
/opt/medclip_env/bin/python Eval_metric/clipscore.py \
  --gen_dir ./EVAL/gen_out/lr2e-3/eps10/findings/samples \
  --backend medclip --text_mode FINDINGS --dedup --device cuda:0 \
  --root_path /storage/.../mimic-cxr/2.1.0 --eval_split test --max_real 361 \
  --output ./EVAL/metric/lr2e-3/eps10/findings/clipscore.json
```

**label-agreement 단독**:
```bash
/opt/medclip_env/bin/python Eval_metric/downstream_cls.py \
  --gen_dir ./EVAL/gen_out/lr2e-3/eps10/findings/samples \
  --root_path /storage/.../mimic-cxr/2.1.0 --eval_split test --max_real 361 \
  --device cuda:0 --label_min_pos 10 \
  --output ./EVAL/metric/lr2e-3/eps10/findings/label_agreement.json
```

### 주요 옵션 요약
| 옵션 | 기본 | 의미 |
|---|---|---|
| `--eval_model` | xrv | feature backbone (xrv 1024-d / inception 2048-d) |
| `--pca_dim` | none | FDS: real에 PCA fit 후 축소(작은 셋 안정화). gen도 pca_dim보다 커야 |
| `--text_mode` | FINDINGS/IMPRESSION | 리포트 섹션 (생성+CLIP 공통) |
| `--clip_dedup` | off | CLIP retrieval을 고유 프롬프트 부분집합에서 |
| `--label_min_pos` | 10 | label macro에 넣을 최소 양성·음성 수 |
| `--xrv_weights` | all + nih | label 분류기 가중치들 |
| `--paired_from_split` | off | run_feature_eval에서 paired/label 활성(gen index→split) |

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
6. **CLIP은 gap으로**: **`clip_gap_cos`**(real_cos−gen_cos)로 비교. 절대 `clip_gen_cos_mean`만으로 판단 금지.
7. **CLIP 신뢰도 먼저 확인**: `clip_real_cos_signal`(=real cos − shuffled)이 ~0이면 그 인코더는 이
   데이터에서 변별력 없음 → CLIP 계열(cos/gap/R@k/P@k/mAP) 전부 참고만. 뚜렷이 양수라야 해석 유효.
8. **label은 gap/ratio + support**: `label_*_auroc_gap_macro`/`_ratio_macro`로 분류기별 gen↔real 비교.
   weight set 간 절대 AUROC 비교 금지. per-pathology는 `support`(n_pos/n_neg≥10)만 신뢰.
9. **T-SNE는 근거 보강용**: 수치(FID/FDS)의 결론을 시각적으로 확인하는 용도.

## 7. 알아둘 제약/함정

- **real test 상한**: 미다운로드 DICOM이 있어 test split 실제 사용량이 제한될 수 있음(예: 361장).
  이는 FID/FDS 공분산 추정 품질의 상한을 정한다.
- **FDS와 표본 수**: `n ≤ feature_dim`이면 공분산이 특이 → 값 폭발. 코드가 **Ledoit-Wolf(`--fds_cov lw`, 기본)**
  로 안정화하고 경고를 출력한다. 절대값 안정화를 원하면 **생성 샘플 증량** 또는 **`--pca_dim 64`**
  (PCA는 real에 fit → subspace 고정으로 모델 간 비교 유지). 단 **gen도 `pca_dim`보다 커야** 함.
- **프롬프트 다양성**: 소수 프롬프트로 대량 생성하면 분포가 인위적으로 좁아짐 →
  분포 지표엔 **`--prompt_source split`**(test 리포트 사용)이 바람직.
- **PyTorch ≥2.6**: 우리 ckpt는 비-텐서 객체 포함 → `torch.load(..., weights_only=False)` 사용(코드 반영됨).
- **MedCLIP/transformers 버전**: MedCLIP은 **transformers 4.24.0**에서 동작(신버전 5.x는 `CLIPFeatureExtractor`
  제거로 실패). 그래서 **`/opt/medclip_env/bin/python`(4.24.0 + wget)** 로 실행해야 clip이 산다. clip이
  `[clip] skipped (ImportError ...)`면 대개 base conda(5.8.1)로 실행한 것. §9 참고.
- **CLIP 백엔드 부재 시**: 라이브러리/가중치 없으면 드라이버가 `[clip] skipped ...`로 안전하게 건너뜀
  (다른 지표는 정상 진행).
- **feature 캐시**: `_features/`는 재실행 가속용. real셋/설정을 바꾸면 캐시가 자동 무효화(키 불일치)된다.
- **text_mode fallback**: `--text_mode FINDINGS`라도 FINDINGS 섹션이 없는 리포트는 원문으로 대체됨
  (§4.4). FINDINGS만 깨끗하게 쓰려면 후속으로 `--section_fallback` 옵션(현재 미구현)을 추가해야 한다.

## 8. 코드 위치 (참조)

- 지표 구현: `Eval_metric/`
  - `features.py`(feature 추출+캐시), `fds.py`(FDS), `tsne_viz.py`(T-SNE),
    `clipscore.py`(CLIPScore: medclip/biovil-t/cxr-clip + retrieval + negative-control),
    `downstream_cls.py`(label-agreement AUROC + ROC png), `fid.py`(FID),
    `psnr.py`/`ssim.py`(paired 픽셀 지표), `text_utils.py`(리포트 섹션 추출), `../Loss/lpips.py`(LPIPS).
  - 드라이버: `run_feature_eval.py`(기존 이미지 평가), `generate_and_eval.py`(생성+평가 원샷).
    두 진입점 모두 `run_feature_eval.run(cfg)`를 공유 → 지표 로직 단일화.
- paired 생성-평가(별도): `LDM_dp_eval.py`. 생성: `LDM_dp_inference.py`.
- 계획 문서: `docs/EVAL_PLAN.md`, 프레임워크: `docs/DP_LDM_FRAMEWORK_PLAN.md`.

## 9. 실행 환경 (Docker / Python) — 매우 중요

컨테이너 `CXR_dp_medclip` 안에 **Python 환경이 둘**이고 `transformers` 버전이 다르다.

| 환경 | Python | transformers | 용도 |
|---|---|---|---|
| **A (base conda)** | `/opt/conda/bin/python` | **5.8.1** | 학습/DP-SGD (Opacus). **MedCLIP 불가**(CLIPFeatureExtractor 제거) |
| **B (medclip venv)** | **`/opt/medclip_env/bin/python`** | **4.24.0** | **평가 실행용**. MedCLIP + wget 있음 |

- **평가는 전부 B로 실행**: `/opt/medclip_env/bin/python Eval_metric/...`. B는 `--system-site-packages`라
  torch·torchxrayvision·medclip을 A와 공유 → 모든 지표가 한 env에서 동작.
- 프로젝트 경로(컨테이너): **`/workspace/PrivaText-CXR`**. 데이터: `/storage/hjchoi/...`.
- MedCLIP 패키지는 `/opt/conda/.../medclip`에 있고 B가 공유. **medclip 소스를 임의 수정하지 말 것**
  (예전에 tf5용으로 패치했다가 B(tf4.24)가 깨졌음 → 원본 유지가 맞음).
- 실제 실행 Python 확인: `python -c "import sys; print(sys.executable)"`.
