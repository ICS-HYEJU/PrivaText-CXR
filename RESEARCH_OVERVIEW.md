# PrivaText-CXR 연구 종합 정리

최종 갱신: 2026-10-27 (Asia/Seoul)

## 1. 한 문장으로 설명하면

이 연구는 흉부 X-ray와 판독문으로 학습한 latent diffusion model을
differential privacy(DP) 아래에서 미세조정하여, 환자 프라이버시를 보호하면서도
판독문에 맞는 임상 소견을 가진 합성 흉부 X-ray를 생성할 수 있는지 검증하는 연구다.

단순히 영상이 그럴듯한지를 보는 데서 끝나지 않는다. 생성 영상이 다음 세 조건을
동시에 만족하는지를 구분해서 평가한다.

1. 실제 흉부 X-ray 분포와 시각적·통계적으로 유사한가?
2. 입력한 병변 조건이 생성 영상에 실제로 반영되는가?
3. 합성 영상을 실제 임상 라벨 분류기의 학습 데이터로 사용할 수 있는가?

현재까지 가장 중요한 발견은 **분포 유사도, 조건 충실도, downstream utility가 서로
다른 축이며 한 축의 개선이 다른 축의 개선을 보장하지 않는다**는 것이다.

---

## 2. 연구 동기

MIMIC-CXR 같은 의료영상 데이터는 임상적으로 가치가 높지만 환자 개인정보 때문에
공유와 재사용이 어렵다. 생성 모델을 이용하면 실제 환자 영상을 직접 공개하지 않고도
연구용 데이터를 만들 가능성이 있다. 그러나 생성 영상이 실제 데이터와 비슷하다는
것만으로는 충분하지 않다.

- 모델이 학습 환자를 기억하거나 복제하면 개인정보 보호에 실패한다.
- 판독문에 “Pleural Effusion”이라고 써도 영상에 흉수가 나타나지 않으면 조건부 생성에
  실패한 것이다.
- 생성 영상에 원본 report의 라벨을 그대로 붙였지만 실제 영상 내용과 라벨이 다르면,
  합성 데이터가 downstream classifier를 오히려 악화시킨다.
- FID가 좋아도 병변 분류나 retrieval 성능은 나쁠 수 있다.

따라서 이 연구는 privacy, fidelity, clinical conditionality, downstream utility를
분리해서 측정한다.

---

## 3. 핵심 연구 질문

### RQ1. DP 학습으로 유용한 흉부 X-ray를 생성할 수 있는가?

privacy budget `epsilon`을 낮추면 보호는 강해지지만 gradient noise가 커진다. 이때
영상 품질과 임상 정보가 얼마나 유지되는지 평가한다.

### RQ2. 판독문 조건이 생성 결과를 실제로 제어하는가?

같은 noise에서 병변 존재/부재 prompt만 바꾸거나, 정답 report와 shuffled report를
바꾸었을 때 생성 결과 및 denoising loss가 달라지는지 측정한다.

### RQ3. 합성 영상은 실제 분류기 학습에 도움이 되는가?

real-only, synthetic-only, real+synthetic 학습을 비교하고, 합성 영상의 label
consistency filtering과 혼합 전략이 utility를 개선하는지 평가한다.

### RQ4. 성능 병목은 어디에 있는가?

VAE 압축, text projection, LoRA capacity, DP clipping/noise, sampling algorithm,
조건 loss, 데이터 split을 각각 분리 실험한다.

---

## 4. 전체 파이프라인

```text
MIMIC-CXR JPG + report + CheXpert labels
                  |
                  v
        report 전처리: LABEL + IMPRESSION
                  |
          BioBERT -> 768→512 projection
                  |             
CXR -> frozen VAE encoder -> latent z (1×16×16)
                  |             |
                  +------ conditional UNet ------+
                         DP-SGD + LoRA             |
                                |                  |
                        predicted noise            |
                                |                  |
                     DDIM/DDPM sampling -----------+
                                |
                        frozen VAE decoder
                                |
                        synthetic CXR 256×256
                                |
          +---------------------+----------------------+
          |                     |                      |
    fidelity 평가       clinical condition 평가   downstream 분류 학습
```

### 4.1 기반 모델

- 이미지 크기: grayscale `256×256`
- VAE: NIH ChestX-ray 기반 pretrained VAE
- latent: `1×16×16`
- diffusion: 1,000-step epsilon-prediction latent diffusion
- UNet context dimension: `512`
- 기반 LDM: NIH class-label-conditioned pretrained checkpoint

VAE와 UNet backbone은 기본적으로 고정한다. MIMIC-CXR의 report 조건으로 바꾸기 위해
BioBERT encoder와 `768→512` projection을 사용한다.

### 4.2 미세조정 대상

초기 주요 checkpoint들은 cross/self-attention Linear layer에 rank-4 LoRA를 주입해
학습했다. 이후 진단 결과 새 BioBERT projection을 고정한 것이 큰 병목임을 확인하여,
최근 classification 실험에서는 projection과 LoRA를 함께 학습한다.

- LoRA rank/alpha 기본값: `4/4`
- projection parameter: 약 `393,728`
- projection+LoRA 총 trainable parameter: 약 `738,816`
- 전체 모델 대비 약 `0.22%`

### 4.3 기본 DP 설정

최근 통제 실험의 공통 설정은 다음과 같다.

- target epsilon: `10`
- delta: `1e-5`
- clipping norm: `0.001`
- logical/physical batch: `256/4`
- learning rate: `0.002`
- epochs: `3`
- optimizer/privacy engine: Opacus DP-SGD
- 실제 3-epoch privacy expenditure: 대체로 epsilon `8.2672`

`sigma=0` clipping-only arm은 학습 가능성의 상한을 확인하기 위한 비공개 대조군이며,
privacy guarantee가 없으므로 epsilon은 `infinity`다.

주의할 점은 각 실험의 epsilon을 합쳐 프로젝트 전체 privacy budget이라고 부를 수
없다는 것이다. 같은 private dataset에 대한 반복 탐색을 실제 공개 메커니즘으로
운영한다면 composition을 별도로 계산해야 한다.

---

## 5. 데이터

### 5.1 사용 데이터

- 이미지: MIMIC-CXR-JPG
- 판독문: MIMIC-CXR report
- 병변 라벨: MIMIC-CXR CheXpert CSV의 14개 label
- 기반 사전학습: NIH ChestX-ray 계열 VAE/LDM checkpoint

CheXpert label 상태는 positive, negative, uncertain, missing으로 구분한다. 대부분의
분류 loss와 AUROC에서는 uncertain/missing을 제외한다.

### 5.2 기존 custom JPG split

주요 JPG 재학습 및 E001~E028은 다음 custom split을 사용했다.

- train: p10/p11의 공식 train
- validation: p10/p11의 공식 validate
- p10/p11 공식 test: 제외
- test: p12 전체를 prefix-based custom holdout으로 사용
- patient overlap: train/validation/test 모두 0

사용 가능한 이미지 수:

| 단계 | 일반 JPG manifest | LABEL+IMPRESSION 사용 가능 |
|---|---:|---:|
| train | 53,857 | 49,907 |
| validation | 802 | 748 |
| p12 test | 37,190 | 35,107 |

train에서는 single-positive No Finding을 가장 큰 abnormal single-positive class의
2배로 제한했다. validation과 test는 원 분포를 유지했다.

이 split은 **공식 MIMIC-CXR test split이 아니다**. prefix와 No-Finding 재표집 정책이
성능에 미치는 영향이 섞여 있다는 한계가 있다.

### 5.3 진행 중인 공식 split 실험 E029

E029는 MIMIC 공식 split CSV의 `train/validate/test` 값을 그대로 보존한다.
prefix 재분할과 No-Finding downsampling을 하지 않는다.

현재 감사된 로컬 JPG/report 수:

| 공식 split | 이미지 | 환자 |
|---|---:|---:|
| train | 108,869 | 19,027 |
| validate | 971 | 146 |
| test | 1,170 | 75 |

- split 간 patient overlap: 0
- LABEL+IMPRESSION train: 102,859장
- 공식 split CSV SHA256:
  `1fb1d5964ca4b64af0e64d36cfc26fb12de415bffacc34530cd7d5ea312c9da8`

2026-10-27 현재 E029 VAE cache 생성과 E004 공식 validation/test baseline 생성이
진행 중이다. 결과가 나오기 전까지 custom split과 official split의 우열을 주장하지
않는다.

### 5.4 입력 text

주요 실험은 `LABEL+IMPRESSION`을 사용한다.

```text
LABEL: Cardiomegaly, Pleural Effusion
IMPRESSION: ...
```

positive CheXpert label을 앞에 명시하고 가능한 경우 report의 IMPRESSION을 뒤에 붙인다.
positive label이 없는 행은 이 모드의 학습에서 제외된다.

---

## 6. 생성과 checkpoint

대표 JPG checkpoint는 privacy budget과 epoch 수의 2×2 조합이다.

| checkpoint | 실제 epsilon | epochs | 비고 |
|---|---:|---:|---|
| eps1/ep10 | 0.8803 | 10 | 강한 privacy |
| eps10/ep10 | 8.5207 | 10 | 완화된 privacy |
| eps1/ep30 | 0.9168 | 30 | 강한 privacy, 장기 학습 |
| eps10/ep30 | 8.8107 | 30 | 완화된 privacy, 장기 학습 |

모든 모델은 동일한 10,240-image p12 test subsample에서 평가했다.

LoRA checkpoint를 생성에 사용할 때는 `--dp_ckpt`뿐 아니라 반드시
`--lora_ckpt`도 지정해야 한다. 같은 파일을 두 인자에 전달하더라도 필요하다.
그렇지 않으면 `strict=False` 로딩이 LoRA delta를 조용히 버리고 base model로 생성할
수 있다.

---

## 7. 평가 체계

이 연구에서는 하나의 점수로 모든 성능을 요약하지 않는다.

### 7.1 분포 및 시각 품질

- FID: 낮을수록 좋음
- Ledoit-Wolf FDS: 낮을수록 좋음
- MT-DDPM FDS: t-SNE 2-D embedding 기반 별도 지표, 낮을수록 좋음
- SSIM/PSNR: 높을수록 좋음
- LPIPS: 낮을수록 좋음

Ledoit-Wolf FDS와 MT-DDPM FDS는 같은 이름의 변형이지만 서로 다른 질문을 측정하며,
실제로 checkpoint 순위를 반대로 매겼다. 따라서 보고할 때 반드시 어느 FDS인지 쓴다.

### 7.2 임상 병변 표현

- XRV DenseNet-121 multi-label AUROC
- MedCLIP zero-shot label AUROC
- real 대비 generated AUROC ratio
- 병변별 support와 AUROC/AUPRC

`clip_label_clipzs_*`는 accuracy가 아니라 zero-shot AUROC다. retrieval R@k/mAP 및
real-image encoder validity 지표와 혼동하지 않는다.

### 7.3 이미지·텍스트 retrieval

- RoentGen-style image-to-image 및 image-to-text retrieval
- 실제 공개 checkpoint 제약 때문에 현재 구현은 두 방향 모두 MedCLIP backend를 사용
- 저장 필드의 `Prec@k`는 conventional precision이 아니라 query-level Hit@k 의미

### 7.4 조건 충실도

- 같은 noise에서 positive/negative prompt를 바꾼 paired score delta
- prompt-intent AUROC
- 정답 report와 patient-shuffled report의 timestep별 denoising MSE 차이
- patient-cluster bootstrap confidence interval

### 7.5 Downstream utility

- DenseNet-121을 `weights=None`에서 scratch로 학습
- masked multi-label BCE
- real-only, synthetic-only, real+synthetic 비교
- 고정 real test에서 병변별 및 macro AUROC

생성 checkpoint의 frozen-XRV 점수와 합성 데이터로 새 classifier를 학습한 downstream
utility는 서로 다른 평가다.

---

## 8. 대표 생성 성능 결과

### 8.1 privacy budget과 epoch 수의 4-way 비교

| metric | eps1/ep10 | eps10/ep10 | eps1/ep30 | eps10/ep30 |
|---|---:|---:|---:|---:|
| FID ↓ | 3.7695 | 3.3589 | 3.3984 | **2.8365** |
| Ledoit-Wolf FDS symmetric ↓ | 501.31 | 477.04 | 481.99 | **455.46** |
| SSIM ↑ | 0.2230 | 0.2269 | 0.2274 | **0.2314** |
| LPIPS ↓ | 0.5361 | 0.5345 | 0.5343 | **0.5324** |
| XRV label AUROC ratio ↑ | 0.6768 | 0.6902 | 0.6903 | **0.6935** |
| CLIP zero-shot AUROC ratio ↑ | **0.7971** | 0.7891 | 0.7933 | 0.7894 |
| Roent i2i Prec@5 ↑ | **0.390** | 0.360 | 0.325 | 0.335 |
| Roent i2t Prec@5 ↑ | **0.530** | 0.465 | 0.500 | 0.515 |

핵심 해석:

- FID, Ledoit-Wolf FDS, pixel metric, XRV ratio는 대체로 eps10과 30 epochs를 선호한다.
- CLIP zero-shot 및 Roent retrieval은 eps1을 선호하거나 거의 비슷하다.
- 더 오래 학습하면 분포 품질은 좋아지지만 semantic retrieval이 같이 좋아지지는 않는다.
- 따라서 “가장 좋은 모델”은 연구 목적에 따라 달라진다.

### 8.2 절대 병변 분류 성능

ep30 생성 이미지의 절대 macro AUROC는 다음과 같다.

| evaluator | real | eps1 synthetic | eps10 synthetic |
|---|---:|---:|---:|
| Frozen XRV | 0.7320 | 0.5053 | 0.5077 |
| MedCLIP | 0.6413 | 0.5087 | 0.5062 |

비율 지표가 약 0.79여도 synthetic의 절대 AUROC는 chance 부근일 수 있다. 이 결과는
FID가 양호하다는 사실과 임상 라벨이 올바르게 표현된다는 사실을 분리해야 함을 보여준다.

---

## 9. Downstream 합성 데이터 utility

### 9.1 필터링 전 결과

5,000장 real test, reliable 12-label macro AUROC:

| classifier train arm | macro AUROC |
|---|---:|
| Real 1,100 | **0.6461** |
| eps1 Real1,100 + Synth1,100 | 0.6369 |
| eps10 Real1,100 + Synth1,100 | 0.6275 |
| eps1 Synth5,500 only | 0.5211 |
| eps10 Synth5,500 only | 0.5006 |
| eps1 Real1,100 + Synth5,500 | 0.6014 |
| eps10 Real1,100 + Synth5,500 | 0.5903 |

결론은 합성 이미지 수를 늘리는 것만으로는 utility가 개선되지 않으며, label-image
mismatch와 synthetic domain 비중이 classifier를 악화시킬 수 있다는 것이다.

### 9.2 consistency filtering

XRV와 MedCLIP이 원본 CheXpert label과 얼마나 일치하는지를 이용해 synthetic pool의
상위 subset만 선택했다.

- 단일 seed에서는 dual-consistency top25가 `0.6678`로 상승했다.
- 3-seed 검증에서는 real-only mean `0.6382`, filtered augmentation mean `0.6513`으로
  평균 `+0.0131`이었지만 seed 하나에서 효과가 반전됐다.
- threshold 세분화 결과 top5 mean `0.6489`, top10 `0.6485`, top15 `0.6470`,
  top20 `0.6406`, top25 `0.6452`였다.

현재 판단:

- filtering은 유망한 신호지만 재현된 확정적 개선은 아니다.
- 같은 5,000-image test를 반복 관찰했으므로 해당 test는 사실상 development set이다.
- 후속 최종 검증은 top5와 top15만 유지하고 새로운 patient-disjoint holdout을 써야 한다.
- `R1100+S275`에는 동일 총량 real-only `R1375`, `R1100+S825`에는 `R1925` 대조군이
  필요하다.
- 최소 5 classifier seeds, 동일 optimizer exposure, 병변별 support 및 calibration을
  함께 보고해야 한다.

---

## 10. 병목 진단 실험 E001~E029

### 10.1 진단 및 기본 conditioning: E001~E008

| ID | 질문 | 핵심 결과 |
|---|---|---|
| E001 | VAE가 병변 정보를 보존하는가? | XRV `0.7765→0.7396`, MedCLIP `0.5706→0.5037`; 특히 MedCLIP 정보 손실 |
| E002 | prompt가 동일 noise 생성 결과를 바꾸는가? | intent AUROC 약 `0.50`; 기존 생성기는 조건 반응이 거의 없음 |
| E003 | padding mask/frozen BERT eval 수정 효과 | 변화 미미, CI가 0 포함 |
| E004 | BioBERT projection을 학습해야 하는가? | XRV `+0.0317`, CI 양수; 가장 명확한 구조적 개선 |
| E005 | clipping과 Gaussian noise의 영향 | clipping-only `0.6374` > no-clip `0.5934` > DP `0.5613`; DP noise가 주요 병목 |
| E006 | 정답 report가 shuffled report보다 유리한가? | denoising 이득 최대 `0.08%`; condition dependence가 매우 약함 |
| E007 | structured label token만 추가하면 좋아지는가? | 유의한 개선 없음 |
| E008 | clipping norm/LR 2×2 조정 효과 | 기본 `C=.001, lr=.002`가 가장 높음 |

### 10.2 Downstream utility 개선: E009~E015

| ID | 질문 | 핵심 결과 |
|---|---|---|
| E009 | XRV-consistent synthetic만 고르면 도움이 되는가? | top10/top25 단일 seed 상승, 50% 이상 포함 시 악화 |
| E010 | XRV+MedCLIP dual filtering이 나은가? | top25 단일 seed `0.6678`, 유망하지만 미확정 |
| E011 | 3 classifier seeds에서 재현되는가? | 평균 `+0.0131`, 한 seed에서 반전 |
| E012 | balanced sampling 또는 synthetic pretrain이 나은가? | pretrain→real fine-tune 단일 seed만 상승 |
| E013 | DDIM50 개선이 DDPM1000에서도 유지되는가? | `0.5613/0.5166 → 0.5182/0.4911`; 유지되지 않음 |
| E014 | pretrain→fine-tune seed 재현성 | seed43/44에서 기준보다 낮아 기각 |
| E015 | filtering threshold 세분화 | top5/top15 후보만 남지만 확정 효과 없음 |

### 10.3 구조·DP·보조손실 탐색: E016~E026

| ID | 질문 | 핵심 결과 |
|---|---|---|
| E016 | projection-only 효과 | XRV `0.5533`; projection 자체는 유효 |
| E017 | LoRA-only 효과 | corrected와 동일; 단독 기여 없음 |
| E018 | LoRA rank8 | rank4보다 낮음 |
| E019/E022 | projection LR `.004/.001` | 기본 `.002`보다 낮음 |
| E020 | condition dropout | XRV/MedCLIP 모두 개선 없음 |
| E021 | structured label + projection/LoRA | 개선 없음 |
| E023 | clipping norm `.002` | 사실상 개선 없음 |
| E024 | frozen-XRV auxiliary BCE | XRV `+0.0004`, MedCLIP 하락; 효과 없음 |
| E025 | auxiliary weight 10배 | auxiliary가 diffusion loss를 지배하여 사전 중단 |
| E026 | low-timestep stratification | XRV 미세 상승, MedCLIP `-0.0139`; teacher 편향으로 종료 |

작은 LR/rank/dropout/clipping sweep과 frozen-XRV BCE 계열은 더 반복하지 않는다.

### 10.4 조건 대조 학습: E027~E028

#### E027: 고정 No-Finding counterfactual

같은 latent/noise에서 정답 report가 고정 No-Finding 문장보다 낮은 denoising MSE를
갖도록 margin-ranking loss를 추가했다.

- 49,907장, 3 epochs, epsilon `8.2672`
- XRV: `0.5613 → 0.6112`, delta `+0.0499`, CI `[0.0226, 0.0730]`
- MedCLIP: `0.5166 → 0.5257`, delta `+0.0091`, CI가 0 포함
- prompt intent mean:
  - XRV `0.5325 → 0.6538`
  - MedCLIP `0.5261 → 0.5728`
- correct-vs-shuffle CI 하한이 양수인 timestep은 4개 중 1개

지금까지 가장 강한 조건 반응 개선이지만 사전 엄격 gate는 실패했다. 고정 No-Finding
negative가 개별 병변 grounding 대신 abnormal-vs-normal shortcut을 만들었을 수 있다.

#### E028: 병변 하나만 뒤집은 hard negative

shortcut을 제거하기 위해 다른 label은 유지하고 positive pathology 하나만 negative로
반전했다. 먼저 Gaussian noise가 없는 clipping-only upper bound에서 검증했다.

| arm | XRV | MedCLIP |
|---|---:|---:|
| structured control | 0.6526 | 0.5134 |
| label-flip rank loss | 0.6459 | 0.4956 |

두 encoder 모두 악화했다. 즉 실패 원인은 DP noise가 아니며 현재 병변별 label-flip
margin loss 자체가 유효하지 않았다. epsilon-10 DP 후속 arm은 실행하지 않는다.

### 10.5 E029: 공식 split 재검증

현재 진행 중이다. 기존 custom split의 prefix 정책, No-Finding downsampling, 데이터
규모가 결론에 미친 영향을 공식 split 경로에서 분리한다. E029 결과 전에는 현재
classification 결론을 공식 MIMIC split의 일반화 결과로 표현하지 않는다.

---

## 11. 현재까지의 과학적 결론

### 결론 1. 좋은 FID는 좋은 임상 조건 생성을 보장하지 않는다

eps10/ep30은 FID와 distribution metric에서 가장 좋지만, synthetic 병변 AUROC는
chance에 가깝고 downstream synthetic-only 분류도 낮다.

### 결론 2. DP noise가 utility를 낮추지만 유일한 원인은 아니다

clipping-only가 DP보다 훨씬 높아 DP noise 영향은 분명하다. 그러나 E028은 noise가
없는 상태에서도 실패했으므로 loss/conditioning 설계도 독립적인 병목이다.

### 결론 3. text projection 적응이 필수적이다

무작위로 새로 연결된 BioBERT projection을 동결한 초기 경로는 심각한 병목이었다.
projection 학습은 지금까지 가장 재현 가능한 구조적 개선 요인이다.

### 결론 4. 조건 사용과 병변별 grounding은 다르다

E027은 prompt response와 XRV를 크게 높였지만 E028의 병변별 hard-negative는 실패했다.
모델이 조건에 반응하는 것과 원하는 개별 병변을 정확히 그리는 것은 별개의 문제다.

### 결론 5. 합성 데이터는 선택 없이 섞으면 분류기를 악화시킨다

합성 비중이 높을수록 성능이 낮아졌다. label-consistent filtering은 가능성을 보였지만
seed 안정성과 독립 test 검증이 부족하다.

### 결론 6. privacy-utility 관계가 모든 지표에서 단조롭지 않다

distribution metric은 eps10을 선호하지만 CLIP zero-shot, Roent retrieval 및 일부
downstream 비교에서는 eps1이 더 좋았다. 이는 하나의 지표만으로 privacy budget을
선택하면 안 된다는 근거다.

---

## 12. 현재 연구의 한계

1. 기존 핵심 결과는 공식 split이 아닌 p10/p11-train, p12-test custom split이다.
2. validation 802장은 일부 병변의 positive/negative support가 매우 작다.
3. 여러 아이디어를 같은 validation/test에서 반복 확인해 선택 편향 가능성이 있다.
4. scratch downstream classifier의 seed variance가 합성 효과와 비슷한 크기다.
5. `1×16×16` latent는 작은 병변과 MedCLIP 의미 정보를 크게 잃을 수 있다.
6. square resize와 view(AP/PA/lateral) 혼합이 병변 표현에 미치는 영향이 충분히
   분리되지 않았다.
7. report 길이 제한과 LABEL+IMPRESSION 필터가 학습 population을 바꾼다.
8. secure RNG는 실험 속도를 위해 꺼져 있다. 실제 privacy release 전에는 secure mode로
   최종 재학습해야 한다.
9. 반복 private 실험의 전체 privacy composition은 별도 release 시나리오에서 계산해야
   한다.
10. 현재 worktree에는 장기 실험 중 생성된 수정·미추적 산출물이 많아 checkpoint와
    source version의 연결을 지속적으로 기록해야 한다.

---

## 13. 다음 연구 우선순위

### 1순위: E029 공식 split 완료

- E004 구조의 official validate/test baseline
- official-train projection+LoRA DP model
- XRV/MedCLIP 병변별·macro AUROC와 patient bootstrap
- 데이터 양 효과와 split 정책 효과를 후속 matched-size 대조로 분리

### 2순위: 최종 평가 protocol 동결

- 개발에 사용하지 않은 patient-disjoint holdout 확보
- primary endpoint와 최소 효과 크기 사전 지정
- sampler, reliable-label 기준, bootstrap 단위 고정

### 3순위: downstream top5/top15 독립 검증

- `R1100`, `R1100+S275`, `R1100+S825`, `R1375`, `R1925`
- 최소 5 seeds
- optimizer update와 real exposure 통제
- 새 holdout을 한 번만 사용

### 4순위: conditioning 상한과 VAE 구조 개선

E029 후에도 clinical conditionality가 낮다면 다음 순서로 진행한다.

1. 더 큰 latent(`4×32×32` 등) 또는 CXR semantic-preserving VAE의 reconstruction
   XRV/MedCLIP 상한 측정
2. public/non-private paired CXR-text 또는 structured label로 condition projection을
   사전 정렬한 뒤 DP fine-tuning
3. condition objective의 clipping-only 성공을 먼저 확인
4. 성공한 목적함수만 DP로 이동

E028에서 실패한 현재 absolute/relative margin label-flip loss나 frozen-XRV BCE의 단순
weight 조정은 반복하지 않는다.

### 5순위: sampler를 별도 연구 변수로 취급

DDPM1000이 DDIM50보다 낮았으므로, 성공 후보에 대해서만 DDIM50/100/200과 DDPM1000을
비교한다. sampler 선택 후 새 holdout에서 최종 성능을 측정한다.

---

## 14. 이 연구에서 주장할 수 있는 것과 아직 주장할 수 없는 것

### 현재 근거로 주장할 수 있는 것

- DP-LDM이 MIMIC-CXR 분포와 유사한 영상을 생성할 수 있다.
- privacy budget과 epoch 수는 distribution fidelity에 영향을 준다.
- distribution fidelity와 clinical/downstream utility는 일치하지 않는다.
- 초기 모델의 report conditioning은 매우 약했고 projection 학습이 이를 일부 개선했다.
- 고정 No-Finding counterfactual은 강한 조건 반응 신호를 만들었지만 병변별 grounding
  개선으로 확정할 수 없다.
- 무필터 synthetic data는 현재 설정에서 real-only classifier를 개선하지 못했다.

### 아직 주장하면 안 되는 것

- 생성 영상이 실제 임상 학습 데이터를 안정적으로 대체하거나 증강한다.
- eps1 또는 eps10이 모든 utility 측면에서 일관되게 우수하다.
- E027이 개별 병변을 정확히 제어한다.
- custom p12 결과가 공식 MIMIC test 성능을 대표한다.
- filtering augmentation이 독립 holdout과 충분한 seeds에서 유의하게 개선됐다.
- 여러 실험 checkpoint를 각각 공개해도 전체 privacy budget이 개별 epsilon과 같다.

---

## 15. 재현 환경과 중요한 실행 규칙

- Docker container: `CXR_dp_medclip`
- host project: `/home/hjchoi/PycharmProjects/PrivaText-CXR`
- container project: `/workspace/PrivaText-CXR`
- 학습/생성 Python: `/opt/conda/bin/python`
- MedCLIP 평가 Python: `/opt/medclip_env/bin/python`
- 평가 PYTHONPATH: `/workspace/PrivaText-CXR/src`
- CLIP 계열 평가 device: `cuda:0`

핵심 함정:

1. LoRA 생성에는 `--dp_ckpt`와 `--lora_ckpt`를 모두 지정한다.
2. 같은 `--out_dir`에 평가 프로세스 두 개를 동시에 실행하지 않는다.
3. CLIP 계열은 `cuda:0`을 사용한다.
4. canonical 평가 driver는 루트의 `Eval_metric/run_feature_eval.py`다.
5. 학습 시 VAE/base checkpoint 경로를 명시적으로 전달한다.
6. 생성 이미지의 md5가 checkpoint 간 동일하면 LoRA 미로딩을 먼저 의심한다.
7. 결과를 비교할 때 DICOM 361장, custom JPG 10,240장, official split 결과를 섞지
   않는다.

세션 시작 시 반드시 다음 순서를 따른다.

1. `pwd`
2. `claude-progress.md`, `feature_list.json`, 이 문서 읽기
3. `git log --oneline -5`, `git worktree list`, `git branch -a`
4. 실행 중인 GPU/process와 다른 worktree 소유 기능 확인
5. `./init.sh`

---

## 16. 주요 파일 안내

| 목적 | 파일/경로 |
|---|---|
| 기능 및 검증 단일 진실 원천 | `feature_list.json` |
| 세션 진행 기록 | `claude-progress.md` |
| 빠른 환경 스모크 | `init.sh` |
| JPG 데이터 파이프라인 | `docs/MIMIC_JPG_PIPELINE.md` |
| 4-way privacy×epoch 결과 | `eval_analysis/jpg_4way_report.md` |
| 초기 synthetic utility | `eval_analysis/synth_utility_report.md` |
| 분류 utility 원인 진단 | `classification-utility-diagnosis.md` |
| E001~E029 개별 보고서 | `classification_experiments/NNN_*.md` |
| 최근 분석 hand-off | `classification_experiments/HANDOFF_EXPERIMENT_ANALYSIS_2026-10-26.md` |
| 평가 결과 JSON | `EVAL/metric/**/eval_summary.json` |
| 분류 실험 evidence | `classification_experiments/evidence/` |
| custom manifest | `data_manifests/mimic_p10_p12/` |
| official split manifest | `data_manifests/mimic_official_split_029/` |

---

## 17. 연구를 처음 듣는 사람에게 설명하는 요약

이 프로젝트는 환자 개인정보를 보호하기 위해 DP-SGD로 미세조정한 text-conditioned
latent diffusion model로 흉부 X-ray를 생성한다. 모델은 실제 영상과 비슷한 분포의
이미지를 만들고, privacy budget을 완화하거나 더 오래 학습하면 FID 같은 분포 지표는
좋아진다. 하지만 생성 영상이 report의 병변을 정확히 표현하는 능력은 훨씬 약하다.
그 결과 필터링하지 않은 합성 영상을 분류기 학습에 추가하면 real-only보다 성능이
낮아진다.

실험을 통해 무작위 BioBERT projection을 동결한 것이 큰 병목이며 projection 학습이
효과적이라는 것을 확인했다. 또한 DP Gaussian noise가 utility를 낮추지만, noise가
없는 조건에서도 잘못 설계된 병변 대조 loss가 실패하므로 DP만이 원인은 아니다.
고정 No-Finding 문장과 비교한 E027은 지금까지 가장 큰 조건 반응 향상을 만들었지만,
이것이 개별 병변 제어인지 단순 abnormal-vs-normal 구분인지는 확정되지 않았다.

현재 연구는 기존 custom data split의 영향을 제거하기 위해 공식 MIMIC split에서 전체
파이프라인을 재검증하고 있다. 최종 목표는 단순히 FID가 낮은 모델이 아니라, 명시한
병변 조건을 재현하고 독립 real holdout에서 실제 분류 utility를 개선하면서도 명확한
DP guarantee를 갖는 생성 모델을 찾는 것이다.

