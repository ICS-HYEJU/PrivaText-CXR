# 002_prompt_response

## 1. 실험 주제
동일 DDPM 난수에서 병변 조건의 존재·부재 반응

## 2. 실험 선정 근거
생성기가 조건을 무시하는지, morphology 변화가 목표 병변 score 변화로 연결되는지 확인한다.

## 3. 실험 방식
두 JPG ep30 checkpoint, 4병변×존재/부재×32 seeds=256장씩. 존재는 LABEL: 병변+positive impression, 부재는 LABEL: No Finding+negative impression. 각 pair의 전체 DDPM 난수 seed를 동일하게 재설정, full 1000 steps. checkpoint/LoRA 동시 로딩. XRV-all 및 bare-label MedCLIP. paired score Δ/seed bootstrap 2000회와 intent-label AUROC. 이 AUROC는 실제 환자 라벨 평가와 별개이며 부재 arm의 No Finding은 다른 소견도 줄일 수 있다.

## 4. 실험 결과
| checkpoint | 병변 | XRV AUROC | XRV paired Δscore | MedCLIP AUROC |
|---|---|---:|---:|---:|
| eps1_ep30 | Cardiomegaly | 0.5059 | +0.000708 | 0.5068 |
| eps1_ep30 | Pleural Effusion | 0.5039 | +0.000830 | 0.4980 |
| eps1_ep30 | Edema | 0.5088 | +0.001320 | 0.5166 |
| eps1_ep30 | Atelectasis | 0.4990 | +0.000651 | 0.5020 |
| eps10_ep30 | Cardiomegaly | 0.5137 | +0.002657 | 0.5039 |
| eps10_ep30 | Pleural Effusion | 0.5020 | -0.001398 | 0.5029 |
| eps10_ep30 | Edema | 0.5068 | +0.000109 | 0.5117 |
| eps10_ep30 | Atelectasis | 0.5049 | -0.000537 | 0.5029 |

raw predictions와 CI: artifacts/002_response/results.json 및 checkpoint별 scores.csv.

## 5. 실험 결과 분석
paired Δ가 0 부근이거나 부호가 불안정하면 조건 반영이 약한 증거다. 양의 Δ라도 작은 값/낮은 AUROC이면 임상적으로 충분한 분리라고 결론내리지 않는다. 프롬프트 intent는 영상에 실제 병변이 있다는 검증이 아니며 teacher score 자체도 ground truth가 아니다. 최종 utility와 다중 병변 데이터 평가를 따로 유지한다.


관측 결과는 XRV 0.4990~0.5137, MedCLIP 0.4980~0.5166으로 두 checkpoint의 모든 병변에서 거의 chance 수준이다. paired XRV 평균 변화도 −0.001398~+0.002657로 작고 일부 음수다. 단순 프롬프트와 동일 난수에서도 조건별 분리가 약해, 다음 실험에서는 조건 전달 및 projection 적응을 우선한다.
