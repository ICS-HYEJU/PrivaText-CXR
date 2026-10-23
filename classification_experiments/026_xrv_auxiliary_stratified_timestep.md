# 026_xrv_auxiliary_stratified_timestep

## 1. 실험 주제
XRV 보조손실 가중치는 E024의 `0.01`로 유지하고, 각 physical batch의 timestep을 low/high 구간에서 층화 추출해 보조손실 활성 표본을 안정적으로 확보한다.

## 2. 실험 선정 근거
E024의 uniform timestep에서는 `t<=250`인 약 25.1% 표본에만 보조손실이 적용되며 physical batch 4에서 활성 표본이 하나도 없을 확률이 약 31.5%다. E025처럼 가중치를 0.1로 올리면 활성 step의 weighted auxiliary loss가 diffusion loss의 1.61배가 되어 중단 기준에 걸렸다. 따라서 표본별 보조손실 크기는 유지하면서 활성 빈도의 분산을 줄이는 방법을 검증한다.

## 3. 실험 방식
- 각 physical batch의 절반은 `t=0..250`, 절반은 `t=251..999`에서 균등 추출하고 위치를 무작위로 섞는다.
- diffusion objective는 원래의 1000-step uniform objective와 기대값이 같도록 importance weight를 적용한다. batch 4 기준 low 표본 weight는 `0.251/0.5=0.502`, high 표본 weight는 `0.749/0.5=1.498`이다.
- XRV auxiliary loss는 low 표본에만 적용하며 가중치는 `0.01`이다. 이에 따라 physical batch마다 정확히 50%가 활성화되지만 표본별 auxiliary 배율은 E024와 같다.
- target epsilon `10`, delta `1e-5`, clipping norm `0.001`, seed `42`, learning rate `0.002`, rank/alpha `4/4`, 3 epochs, batch `256/4`, 데이터와 split은 E004/E024와 동일하게 고정한다.
- 스모크에서 timestep 층화, finite loss/gradient, auxiliary 비지배, frozen VAE/XRV, Opacus 및 checkpoint를 검증한 뒤 전체 학습한다.
- 학습 후 validation 802장 DDIM50 XRV/MedCLIP AUROC, E004 paired patient bootstrap CI, 동일-noise prompt response를 평가한다. 두 encoder가 모두 E004보다 개선될 때만 full DDPM으로 진행한다.

## 4. 실험 결과
진행 중. 구현 및 실제 모델 스모크 검증을 수행한다.

## 5. 실험 결과 분석
결과 대기 중이다. 이 실험은 auxiliary 신호의 평균 크기와 활성 빈도를 함께 바꾸므로, 결과가 개선되면 후속 실험에서 25% 층화 비율과 별도 auxiliary timestep 경로를 비교해 원인을 분리한다.
