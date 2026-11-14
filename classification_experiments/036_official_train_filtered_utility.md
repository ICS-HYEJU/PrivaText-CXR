# 036_official_train_filtered_utility

## 1. 실험 주제

공식 MIMIC-CXR train에서 생성하고 XRV+MedCLIP dual consistency 상위 25%로
필터링한 합성 흉부 X-ray가 scratch DenseNet-121의 공식 test downstream
classification AUROC와 accuracy를 개선하는지 평가한다.

## 2. 실험 선정 근거

E034 `filtered_top025`는 공식 validate와 test prompt를 합쳐 만들었으므로 이를
classifier 학습에 사용하면 test 정보가 누출된다. E036은 공식 train에서만 22,000개
study를 추출하고 공식 test 환자와 중복이 0임을 확인한 뒤 별도의 합성·필터링 풀을
만들었다.

## 3. 실험 방식

- 공식 train에서 seed 42로 study당 한 영상, 총 22,000건을 추출했다.
- E033 RadBERT+E027 epsilon 1/10 checkpoint로 각각 DDIM50 이미지 22,000장을 생성했다.
- XRV와 MedCLIP label consistency 평균을 dual consistency로 사용해 eps별 상위
  25%=5,500장을 선택했다.
- pretrained weight가 없는 DenseNet-121을 30 epochs, batch 32, masked BCE로
  arm마다 새로 학습했다.
- 공통 real 1,100장과 eps별 filtered synthetic 1,100/5,500장을 조합했다.
- 공식 test 3,028장은 최종 평가에만 사용했다. 신뢰 가능한 11개 병변의 macro
  AUROC와 threshold 0.5 accuracy를 계산했다.

## 4. 실험 결과

| epsilon | arm | train n | macro AUROC | macro ACC | micro ACC | exact-match ACC |
|---:|---|---:|---:|---:|---:|---:|
| - | R1.1k | 1,100 | 0.6536 | 0.7749 | 0.7422 | 0.4819 |
| 1 | R1.1k+S1.1k | 2,200 | 0.6584 | 0.7462 | 0.7745 | 0.5798 |
| 1 | S1.1k | 1,100 | 0.6292 | 0.6124 | 0.5915 | 0.2560 |
| 1 | S5.5k | 5,500 | 0.6568 | 0.6741 | 0.6962 | 0.4649 |
| 1 | R1.1k+S5.5k | 6,600 | 0.6520 | 0.7388 | 0.7705 | 0.5569 |
| 10 | R1.1k+S1.1k | 2,200 | **0.6663** | 0.6919 | 0.7171 | 0.4693 |
| 10 | S1.1k | 1,100 | 0.6169 | 0.6409 | 0.6084 | 0.3033 |
| 10 | S5.5k | 5,500 | 0.6419 | 0.6841 | 0.7106 | 0.4864 |
| 10 | R1.1k+S5.5k | 6,600 | 0.6579 | 0.7596 | **0.7922** | **0.5941** |

전체 수치와 병변별 결과는
`RESULT_20261110_OFFICIAL_TRAIN_FILTERED_UTILITY/classification_results.json` 및 각
arm의 `eval_summary.json`에 있다.

## 5. 실험 결과 분석

가장 높은 macro AUROC는 epsilon 10의 R1.1k+S1.1k(0.6663)로 real-only 대비
+0.0127이었다. epsilon 1의 R1.1k+S1.1k도 +0.0048 높았다. 반면 synthetic-only
S1.1k는 두 epsilon 모두 real-only보다 낮았다. 합성량을 5.5k로 늘리면 epsilon 1
synthetic-only AUROC는 real-only를 소폭 넘었지만 epsilon 10은 넘지 못했다.

Accuracy와 AUROC 순위는 일치하지 않는다. 고정 threshold 0.5 accuracy는 class
prevalence와 calibration에 민감하며, real-only가 가장 높은 macro ACC를 보인 반면
epsilon 10 R1.1k+S5.5k가 가장 높은 micro/exact-match ACC를 보였다. 따라서 accuracy를
AUROC 대체 지표로 해석하지 않는다. 또한 단일 classifier seed 결과이므로
R1.1k+S1.1k의 AUROC 상승을 확정적 일반화 개선으로 주장하려면 seed 반복과 환자 단위
confidence interval이 필요하다.
