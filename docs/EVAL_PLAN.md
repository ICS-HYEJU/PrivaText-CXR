# 생성 output 평가 프레임워크 — 개념 & 적용 Plan (B안: 신규 지표 우선)

대상: DP-LDM(VAE + UNet diffusion, BioBERT 텍스트 조건) LoRA 미세조정 모델.
비교 축: epsilon(1/3/5/10) · LoRA alpha · MTDDPM 등 모델 변형 간 score 비교.
평가 축(요청): ① Image quality(FID/PSNR/LPIPS/**FDS**) ② **T-SNE** 특징공간 ③ text-image alignment(**Downstream classification**, **CLIPscore**).

기존 구현: FID(InceptionV3), PSNR, SSIM, LPIPS, paired eval(`LDM_dp_eval.py`).
신규 필요: **FDS(FeatureKL) / T-SNE / CLIPscore / Downstream classification**.

---

## A. Image Quality — FDS (Feature KL Divergence)

### 개념 / 메커니즘
- FID 복습: real/gen 이미지를 사전학습 CNN에 통과시켜 feature를 얻고, 두 집합을 각각 다변량 가우시안 N(μ,Σ)로 근사. FID = 두 가우시안 사이 Fréchet(2-Wasserstein) 거리
  `||μr−μg||² + Tr(Σr+Σg−2(ΣrΣg)^½)`.
- **FDS/FeatureKL**: 같은 가우시안 근사에서 거리 대신 **KL divergence** 사용.
  두 다변량 가우시안의 KL은 닫힌형:
  `D_KL(N0||N1) = ½[ tr(Σ1⁻¹Σ0) + (μ1−μ0)ᵀΣ1⁻¹(μ1−μ0) − k + ln(detΣ1/detΣ0) ]`.
  값이 **낮을수록**(↓) 생성 feature 분포가 real 분포에 가까움.
- FID와 차이: FID는 대칭 거리(metric), KL은 **비대칭**. 방향 선택이 의미를 가짐
  - `D_KL(gen || real)`: "생성 분포가 real이 커버하는 영역 밖으로 얼마나 새는가"(mode invention / hallucination에 민감).
  - `D_KL(real || gen)`: "real의 어떤 mode를 생성이 놓치는가"(mode collapse에 민감).

### 내 연구 적용
- 기존 `fid.py`의 feature 파이프라인 재사용(중복 forward 없이 μ,Σ 계산 후 KL).
- **backbone은 도메인 특화 권장**: ImageNet InceptionV3 대신 TorchXRayVision DenseNet-121 feature가 CXR에서 임상적으로 유의미(다른 브랜치에 XRV 옵션 존재 → 통합).
- eps/alpha 모델마다 FDS를 뽑아 FID와 나란히 표에 → "MTDDPM + alpha 모델간 비교"의 한 열.
- 양방향(gen||real, real||gen) 모두 보고 → DP 노이즈가 hallucination을 키우는지 / mode를 죽이는지 구분.
- 수치 안정화: 공분산 shrinkage(Σ + εI), Cholesky 기반 log-det/solve.

### 구현
- `Eval_metric/fds.py`: `compute_fds(feats_real, feats_gen, direction, reg)`; feature 추출은 fid.py에서 import.

---

## B. T-SNE 특징공간 시각화

### 개념 / 메커니즘
- 고차원 feature(예: DenseNet 1024-d / Inception 2048-d)를 2D로 비선형 임베딩.
  고차원에서 점들의 이웃 관계를 가우시안 확률로, 저차원에서 t-분포 확률로 두고 **두 분포의 KL을 최소화** → 국소 이웃 구조 보존.
- 성질/주의: **정성적** 도구. 축은 의미 없음, 전역 거리·밀도는 왜곡됨, `perplexity`에 민감, seed 고정 필요.

### 내 연구 적용
- real vs generated feature를 한 공간에 임베딩하고 색으로 구분 → 생성 분포가 real을 얼마나 덮는지(**mode coverage vs collapse**)를 시각 진단.
- 색을 **finding label**(pneumonia/effusion/normal 등)로 바꿔서 → 생성이미지가 클래스별 real 클러스터에 안착하는지 = **텍스트 조건이 특징공간에 반영되는지** 확인.
- eps/alpha 모델별 패널 병렬 배치 → privacy budget이 줄수록 클러스터가 뭉개지는지 관찰(문서의 privacy–utility 서사와 연결).
- (옵션) UMAP 병행: 전역 구조 보존이 나아 mode coverage 판단에 보완적.

### 구현
- `Eval_metric/tsne_viz.py`: feature 캐시(.npy) → sklearn `TSNE` → matplotlib scatter(hue=source/label), 모델별 subplot png 저장.

---

## C. Text-Image Alignment

### C-1. CLIPscore (도메인 인코더: BioViL-T / MedCLIP / CXR-CLIP)

#### 개념 / 메커니즘
- CLIP류: 이미지 인코더 `f_img`, 텍스트 인코더 `f_txt`가 contrastive 학습으로 **공유 임베딩 공간**에 정렬.
  `CLIPScore = w · max(cos(f_img(x), f_txt(t)), 0)` (원 논문 w=2.5). ↑ 높을수록 의미 정합.
- 일반 CLIP은 자연영상 학습이라 CXR에서 부정확 → **도메인 특화 인코더** 사용:
  - **BioViL-T**: 흉부X선+리포트(시계열 포함) 학습 image-text 모델(Microsoft).
  - **MedCLIP**: 이미지-리포트 semantic matching(unpaired 학습 지원).
  - **CXR-CLIP**: CXR 특화 CLIP.
- 우리 학습엔 CLIP/contrastive loss 없음(문서 명시) → 순수 **평가 전용** 외부 인코더. 조건에 쓰는 BioBERT와 다른 모델이므로 circularity 없음.

#### 내 연구 적용
- 각 (생성이미지, 조건프롬프트) 쌍의 CLIPScore 평균/분포.
- **핵심: real-normalized 비교** — `cos(gen_img, prompt)` vs `cos(real_img, prompt)`(paired). real 대비 gap이 모델간 공정 비교 기준(절대값은 인코더마다 스케일 다름).
- **retrieval 검증**: 생성이미지가 자기 프롬프트를 후보군에서 top-k로 되찾는지(R@k), mismatched 프롬프트를 negative control로.
- eps/alpha별 → alignment가 privacy budget에 따라 얼마나 저하되는지 정량화.

#### 구현 / 리스크
- `Eval_metric/clipscore.py`: `encode_image/encode_text` 어댑터 뒤에 3개 백엔드. 가중치 다운로드 필요(네트워크 정책 확인) — 없으면 graceful skip.

### C-2. Downstream Task (Classification utility)

#### 개념 / 메커니즘
- "생성 데이터가 실제로 쓸모 있나"를 재는 utility 검증. 두 프로토콜:
  - **TSTR (Train on Synthetic, Test on Real)**: 생성이미지+라벨로 분류기 학습 → real test AUROC. real 학습(TRTR) 대비 gap이 작을수록 합성데이터 utility 높음. **DP 생성모델 평가의 표준**.
  - **Label agreement**: 사전학습 CXR 분류기(TorchXRayVision)를 real·gen에 적용 → 조건 라벨과 gen 예측 일치(정확도/AUROC). 학습 불필요.
- 핵심: 프롬프트의 "pneumonia"로 생성한 이미지를 독립 분류기가 pneumonia로 인식하는가 = 텍스트 의미가 픽셀까지 전달됐는가.

#### 내 연구 적용
- 1차(가벼움): **XRV 사전학습 다분류기**로 label-agreement(조건 라벨 회수율) — 즉시 가능.
- 2차(정식): **TSTR** — 생성셋으로 DenseNet 학습, MIMIC real test AUROC(vs TRTR).
- eps/alpha별 AUROC 곡선 → **privacy–utility tradeoff** 정량화(문서 핵심 서사와 직결).

#### 구현
- `Eval_metric/downstream_cls.py`: (a) xrv label-agreement 모드, (b) TSTR 학습·평가 모드. 라벨은 `Data/cls_emb.py`/CheXpert csv에서.

---

## 통합 러너 & 산출물
- `Eval_metric/run_eval.py` + `eval_config.yaml`: 모델 리스트(eps/alpha/MTDDPM ckpt) × 지표 → 결과 테이블(csv) + T-SNE png + summary.md.
- **feature 캐시(.npy)**: FID/FDS/T-SNE가 같은 feature 재사용(중복 forward 방지).

## 제안 구현 순서
1. 공통 인프라: 생성셋/실제셋 로더 + feature 추출 캐시(fid.py 재사용, XRV backbone 통합).
2. **FDS**(feature 재사용 → 가장 빠름).
3. **T-SNE/UMAP**(같은 feature 재사용).
4. **CLIPscore**(BioViL-T 우선, 가중치 확보 확인).
5. **Downstream**: xrv label-agreement → TSTR.
6. 통합 러너 + 모델간 비교 테이블.

## 확인 필요 사항
- 생성 output(이미지+프롬프트 쌍) 저장 위치/포맷.
- 네트워크 정책: BioViL-T/MedCLIP/CXR-CLIP/torchxrayvision 가중치 다운로드 가능 여부.
- 실제 real held-out 라벨 소스(CheXpert csv vs cls_emb).
