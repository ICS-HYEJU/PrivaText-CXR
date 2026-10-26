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
구현, 실제 모델/Opacus 스모크, 전체 DP 학습, validation DDIM50 생성·평가 및 동일-noise prompt-response 평가를 완료했다.

### 스모크 및 전체 학습

- projection auxiliary gradient norm `0.4793`, LoRA auxiliary gradient norm `0.1179`로 gradient 전달을 확인했다.
- VAE/XRV frozen, uncertain/missing mask, checkpoint round-trip, physical batch 4 Opacus 검증을 모두 통과했다.
- 스모크의 weighted auxiliary loss는 `0.0036–0.0357`로 weighted diffusion loss `0.1386–0.3935`보다 작았다.
- 49,907장, 3 epochs, 582 logical steps를 `1시간 35분 37초`에 완료했다.

| epoch | train loss | epsilon |
|---:|---:|---:|
| 1 | 0.2501 | 6.2851 |
| 2 | 0.2459 | 7.4000 |
| 3 | 0.2450 | 8.2672 |

최종 epsilon은 delta `1e-5`에서 `8.2672`로 target 10 이내였다. 체크포인트는 `classification_experiments/artifacts/ablation/xrv_aux_stratified/ldm_dp_final.pt`에 저장했다.

### DDIM50 validation 평가

E004와 동일한 validation 802장과 생성 seed를 사용했으며, 두 체크포인트의 첫 이미지 md5가 달라 E026 LoRA가 실제 생성에 반영됐음을 확인했다.

| arm | XRV AUROC | MedCLIP AUROC |
|---|---:|---:|
| E004 projection+LoRA | 0.5613 | 0.5166 |
| E024 uniform auxiliary | 0.5617 | 0.5157 |
| E026 stratified auxiliary | 0.5628 | 0.5027 |

- XRV E026−E004: `+0.00152`, patient-paired bootstrap 95% CI `[-0.01554, 0.01655]`
- MedCLIP E026−E004: `-0.01387`, patient-paired bootstrap 95% CI `[-0.03736, 0.00439]`

### 동일-noise prompt-label response

4개 병변×존재/부재×32 seed의 256장을 평가했다.

| encoder | 4병변 intent AUROC 평균 | paired score delta 평균 |
|---|---:|---:|
| XRV | 0.5393 | +0.01542 |
| MedCLIP | 0.5151 | +0.000731 |

병변별 intent AUROC는 XRV `0.5176–0.5635`, MedCLIP `0.4971–0.5410`이었다. 통합 evidence는 `classification_experiments/evidence/026_results.json`, 원시 평가는 `xrv_aux_stratified_validation.json`, paired 비교는 `026_xrv_auxiliary_stratified_timestep_comparison.json`, prompt 평가는 `026_prompt_response.json`에 저장했다. 두 encoder 동시 개선 gate를 통과하지 못해 full DDPM은 실행하지 않았다.

## 5. 실험 결과 분석
층화 timestep은 E024보다 XRV AUROC를 `+0.00112` 더 높였지만 E004 대비 변화량은 `+0.00152`에 불과하고 신뢰구간이 넓게 0을 포함했다. 반면 MedCLIP은 E004보다 `0.01387`, E024보다 `0.01301` 낮았다. 따라서 label 보존이 일반적으로 개선됐다는 근거는 없으며, XRV teacher 방향으로 최적화가 편향된 신호가 더 강하다.

Prompt response에서도 XRV 평균은 E024의 `0.5327`에서 `0.5393`으로 소폭 올랐지만 MedCLIP은 `0.5195`에서 `0.5151`로 낮아졌고, 두 값 모두 chance 부근이다. 보조손실 활성 빈도를 높이는 것만으로 conditioning 반응 실패를 해결하지 못했다.

E025와 E026을 함께 보면 auxiliary 가중치를 직접 높이면 단일 teacher loss가 diffusion loss를 지배하고, 가중치를 유지한 채 low-t 노출을 늘리면 XRV 지표만 미세하게 움직이면서 MedCLIP 일반화가 악화된다. 현재 frozen-XRV BCE auxiliary 계열은 최종 후보에서 제외한다. 다음 학습 방법은 teacher-specific output BCE 대신 report condition이 생성 결과에 반영되도록 직접 최적화하는 encoder-independent 조건 대조 학습 또는 condition–image alignment loss를 우선 검토한다.
