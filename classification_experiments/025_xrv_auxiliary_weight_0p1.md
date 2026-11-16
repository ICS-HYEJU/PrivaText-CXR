# 025_xrv_auxiliary_weight_0p1

## 1. 실험 주제
E024의 frozen XRV 병변 보존 보조손실 가중치를 `0.01`에서 `0.1`로 높여, DP clipping 전 label-aware gradient 비중을 강화한다.

## 2. 실험 선정 근거
E024는 구현과 gradient 전달은 정상이고 XRV AUROC가 `+0.00040` 변했지만 신뢰구간이 0을 포함했으며 MedCLIP AUROC는 `-0.00086` 변했다. 학습 로그에서 weighted auxiliary loss가 주로 `0.001–0.01` 수준이고 대부분의 표본 gradient가 clipping norm `0.001`을 넘었다. 따라서 보조 신호가 DP clipping 이후 충분히 남지 않았을 가능성을 검증하기 위해, privacy 조건을 바꾸지 않고 가중치만 10배 높인다.

## 3. 실험 방식
- 비교군: E004 Projection+LoRA rank4와 E024 XRV auxiliary `lambda=0.01`.
- 실험군: 동일한 frozen XRV auxiliary 구조에서 `lambda_aux=0.1`.
- 고정 조건: target epsilon `10`, delta `1e-5`, clipping norm `0.001`, seed `42`, projection/LoRA learning rate `0.002`, LoRA rank/alpha `4/4`, 3 epochs, logical/physical batch `256/4`, auxiliary 적용 timestep `t<=250`, 동일 train/validation split.
- 실행 전 실제 모델 스모크에서 loss와 gradient finite 여부, projection·LoRA auxiliary gradient, frozen VAE/XRV, Opacus physical batch 4, checkpoint round-trip, 실제 가중치 기록을 확인한다. weighted auxiliary 항이 diffusion loss보다 큰 경우 전체 학습을 중단한다.
- 스모크 통과 후 전체 DP 학습을 수행하고 실제 epsilon이 10 이내인지 확인한다.
- E004/E024와 동일한 validation 802장, DDIM 50-step, eta 0, image seed로 생성한다. XRV 및 MedCLIP macro/per-pathology AUROC와 E004 대비 patient-paired bootstrap CI를 계산한다.
- 동일 noise의 positive/negative prompt pair로 4개 병변의 prompt-label response를 평가한다.
- XRV와 MedCLIP이 모두 E004보다 개선될 때만 full DDPM 후보로 올린다. XRV만 개선되고 MedCLIP이 하락하면 XRV teacher 과적합으로 판정하며, 둘 다 개선되지 않으면 다음 실험은 low-noise timestep stratified sampling으로 진행한다.

## 4. 실험 결과
실제 모델 및 Opacus 스모크까지 수행했으며, 사전 지정한 보조손실 지배 중단 기준에 걸려 전체 학습은 실행하지 않았다.

- auxiliary-only preflight loss: `1.5834`
- projection auxiliary gradient norm: `0.4793`
- LoRA auxiliary gradient norm: `0.1179`
- VAE/XRV frozen 및 checkpoint round-trip: 통과
- Opacus physical batch 4 DP step: 통과
- 실제 epsilon: `8.7488` at delta `1e-5` (4-step smoke 설정)
- 보조손실 활성 DP step: diffusion loss `0.3543`, weighted XRV loss `0.5715`, total loss `0.9258`
- weighted XRV / diffusion loss 비율: `1.61`

스모크 evidence는 `classification_experiments/evidence/025_smoke.json`, 전용 스모크 checkpoint는 `classification_experiments/artifacts/xrv_aux_lambda0p1_smoke/ldm_dp_final.pt`에 저장했다. E024 checkpoint와 evidence는 덮어쓰지 않았다.

## 5. 실험 결과 분석
`lambda_aux=0.1`은 활성 표본이 포함된 DP step에서 diffusion loss보다 더 큰 목적함수 항을 만들었다. 이 상태로 전체 학습하면 생성 품질과 MedCLIP 일반화보다 frozen XRV teacher의 결정을 우선하도록 최적화할 위험이 크다. E024에서 이미 MedCLIP이 소폭 하락했으므로 teacher-specific overfitting 위험도 방향상 일치한다.

따라서 E025는 성능 평가 실패가 아니라 사전 안전 기준에 따른 학습 후보 기각이다. 가중치 자체를 10배 높이는 방식은 사용하지 않는다. 다음 E026에서는 `lambda_aux=0.01`과 privacy/clipping 조건을 복원하고, uniform diffusion timestep과 별도로 낮은 timestep의 auxiliary 경로를 구성해 표본별 보조 gradient 크기를 폭증시키지 않으면서 label 신호가 DP step마다 전달되도록 검증한다.
