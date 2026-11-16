# 037_top10_filtered_utility

## 1. 실험 주제

E036 공식-train synthetic 22,000장의 dual-consistency 상위 10%=2,200장을 사용한
scratch DenseNet-121 downstream classification utility 평가.

## 2. 실험 선정 근거

Top25 5,500장 결과와 비교해 더 엄격한 filtering이 synthetic label noise를 줄이는지
확인한다. 새로운 생성이나 test 기반 선택 없이 E036의 train-only 점수를 재사용한다.

## 3. 실험 방식

eps1/eps10별 기존 dual-consistency 점수를 내림차순 정렬해 2,200장을 선택했다.
각 epsilon에서 S2.2k와 동일 real 1.1k를 더한 R1.1k+S2.2k를 구성했다. classifier는
weights=None DenseNet-121, 30 epochs, seed42, masked BCE이며 공식 test 3,028장에서
11개 신뢰 가능 병변의 macro AUROC와 threshold 0.5 accuracy를 계산했다.

## 4. 실험 결과

| epsilon | arm | train n | macro AUROC (vs R1.1k) | macro ACC | micro ACC | exact-match ACC |
|---:|---|---:|---:|---:|---:|---:|
| - | R1.1k 기준 | 1,100 | 0.6536 | 0.7749 | 0.7422 | 0.4819 |
| 1 | S2.2k | 2,200 | 0.6388 (-0.0147) | 0.6934 | 0.7321 | 0.5095 |
| 1 | R1.1k+S2.2k | 3,300 | 0.6544 (+0.0008) | 0.7339 | 0.7589 | 0.5487 |
| 10 | S2.2k | 2,200 | 0.6326 (-0.0210) | 0.6140 | 0.6607 | 0.4271 |
| 10 | R1.1k+S2.2k | 3,300 | 0.6410 (-0.0125) | 0.6260 | 0.6393 | 0.4025 |

## 5. 실험 결과 분석

Top10 S2.2k 단독 학습은 두 epsilon 모두 real-only보다 낮았다. eps1의 real+synthetic
조합은 real-only와 거의 같은 AUROC(+0.0008)였고 eps10 조합은 -0.0125 낮았다.
따라서 현재 단일 seed에서는 top10의 더 엄격한 filtering과 2,200장 증강이 E036
top25의 eps10 R1.1k+S1.1k 개선(+0.0127)을 재현하지 못했다. Accuracy는 prevalence와
고정 threshold calibration에 민감하므로 AUROC의 대체 지표로 해석하지 않는다.

전체 병변별 결과: `RESULT_20261111_OFFICIAL_TRAIN_TOP10_UTILITY/classification_results.json`.
