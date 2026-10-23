# 024_xrv_auxiliary_loss

## 1. 실험 주제
Projection+LoRA rank4 diffusion 학습에 frozen XRV 병변 분류 보조 loss를 추가해 predicted x0의 label 보존을 직접 최적화한다.

## 2. 실험 선정 근거
E016~E023에서 projection은 유효했지만 LoRA rank, projection learning rate, conditioning dropout, structured token, clipping norm 변경은 E004를 넘지 못했다. 기존 diffusion loss는 병변 보존을 직접 평가하지 않으므로 image-label supervision을 추가한다.

## 3. 실험 방식
### 3.1 전체 학습 구조

기존 report conditioning과 diffusion 학습 경로를 유지하면서, 동일한 UNet forward 결과로부터 복원한 predicted x0에 XRV 보조 분류 loss를 추가한다.

```text
Report -> BioBERT -> Projection -> UNet + LoRA -> predicted epsilon
                                             |-> diffusion loss
                                             |-> predicted x0 latent
                                                 -> frozen VAE decoder
                                                 -> predicted CXR
                                                 -> frozen XRV-DenseNet121
                                                 -> masked CheXpert BCE loss
```

전체 목적함수는 다음과 같다.

```text
L_total = L_diffusion + lambda_aux * L_xrv
```

### 3.2 고정 기준과 비교 조건

- 비교 기준: E004 Projection+LoRA rank4 결과(XRV AUROC 0.5613, MedCLIP AUROC 0.5166)
- privacy target: epsilon 10, delta 1e-5로 고정
- DP clipping norm: 0.001
- seed: 기존 실험에서 가장 좋은 결과를 사용한 42
- projection learning rate: 0.002
- LoRA learning rate: 0.002
- LoRA rank/alpha: 4/4
- epoch: 3
- logical batch size: 256
- physical batch size: 4
- 생성 및 평가 조건: E004와 동일

위 조건은 변경하지 않고 XRV 보조 loss의 효과만 비교한다.

### 3.3 데이터와 라벨 처리

데이터셋은 각 표본에 대해 latent, report, CheXpert label, label mask를 반환하도록 확장한다.

- positive label은 1, negative label은 0으로 사용한다.
- uncertain 및 missing label은 loss 계산에서 제외한다.
- CSV 열 순서에 의존하지 않고 pathology 이름을 기준으로 XRV 출력과 라벨을 명시적으로 매핑한다.
- 표본별 loss는 유효한 라벨에 대해서만 평균한다. 유효 라벨이 없는 표본은 보조 loss에서 제외한다.
- 기존 train/test split은 변경하지 않는다.

### 3.4 predicted x0 복원

UNet을 두 번 호출하지 않고 기존 denoising forward의 predicted epsilon을 재사용한다. timestep t의 noisy latent x_t에서 다음 식으로 predicted x0를 복원한다.

```text
x0_pred = (x_t - sqrt(1 - alpha_bar_t) * epsilon_pred) / sqrt(alpha_bar_t)
```

고잡음 timestep에서는 x0 복원이 불안정하고 분류 loss가 diffusion 학습을 방해할 수 있으므로, 보조 loss는 `t <= 250`인 표본에만 적용한다. diffusion loss는 기존과 같이 모든 timestep에서 계산한다. 이 기준은 validation 결과를 보고 조정하지 않고 실행 전에 고정한다.

### 3.5 VAE decode와 XRV 입력 전처리

- VAE 파라미터는 모두 freeze한다.
- predicted latent에서 projection 및 LoRA까지 gradient가 전달되어야 하므로 `differentiable_decode_first_stage` 경로를 사용한다.
- XRV 모델은 `densenet121-res224-all`을 사용하고 모든 파라미터를 freeze한 뒤 eval mode로 유지한다.
- XRV 파라미터는 optimizer와 checkpoint 저장 대상에서 제외한다.
- decoded image를 XRV 입력 규격에 맞게 224x224로 resize하고 값 범위를 clamp한 후 XRV가 요구하는 intensity scale을 적용한다.
- 전처리 과정은 tensor 연산으로만 구성해 gradient graph를 끊지 않는다.

### 3.6 보조 loss

XRV logits와 매핑된 CheXpert label 사이에 masked `BCEWithLogitsLoss`를 계산한다. 초기 실험의 auxiliary weight는 `lambda_aux=0.01`로 사전 고정한다. 이 실험에서는 validation 성능에 따라 weight를 탐색하지 않는다. 보조 loss가 지나치게 크거나 작지 않은지는 학습 로그의 diffusion loss, raw XRV loss, weighted XRV loss, 적용 표본 비율로 확인한다.

### 3.7 DP 학습 적용

기존 Opacus 학습 구조와 privacy accountant를 유지한다. XRV와 VAE는 frozen module이지만 XRV loss에서 발생한 gradient는 decoder 입력을 거쳐 projection과 LoRA까지 전달된다. optimizer에는 기존과 동일하게 projection과 LoRA의 trainable parameter만 포함한다. 학습 종료 시 실제 epsilon이 목표 10 이내인지 확인한다.

### 3.8 실행 전 스모크 검증

본 학습 전에 다음 검증을 수행한다.

1. VAE와 XRV 파라미터가 학습 전후 동일한지 확인한다.
2. XRV 보조 loss만 backward했을 때 projection과 LoRA gradient가 0이 아니며 finite인지 확인한다.
3. uncertain/missing label이 loss에서 실제로 제외되는지 확인한다.
4. `t > 250` 표본에서는 total loss가 기존 diffusion loss와 동일한지 확인한다.
5. Opacus 환경에서 per-sample gradient가 정상 생성되는지 확인한다.
6. checkpoint 저장 후 다시 로드했을 때 projection과 LoRA 출력이 동일한지 확인한다.
7. physical batch size 4에서 GPU 메모리 부족 없이 한 step이 완료되는지 확인한다.

### 3.9 학습 후 평가 순서

학습 checkpoint로 먼저 DDIM 50-step 이미지를 생성하고 다음 평가를 순서대로 수행한다.

1. XRV 병변별 및 macro AUROC
2. MedCLIP zero-shot 병변별 및 macro AUROC
3. prompt-label response와 conditioning 보존 여부
4. E004 대비 paired 비교
5. DDIM 결과가 기준을 넘는 경우 full DDPM 생성 및 재평가

## 4. 실험 결과
구현, 실제 모델 스모크, 전체 DP 학습, validation DDIM50 생성 및 평가를 완료했다.

### 구현 및 스모크

- 라벨 mapping/mask: XRV 11개 병변을 이름으로 CheXpert 열과 연결했으며 uncertain/missing label 제외 검증을 통과했다.
- auxiliary-only gradient: loss `1.5834`, projection gradient norm `0.4793`, LoRA gradient norm `0.1179`로 두 학습 대상 모두 gradient가 전달됐다.
- Opacus: physical batch 4에서 per-sample gradient가 finite였고 실제 DP 학습 step을 통과했다.
- frozen/저장 검증: VAE와 XRV 파라미터 해시가 학습 전후 동일했고, checkpoint의 projection/LoRA tensor가 메모리 상태와 정확히 일치했으며 XRV parameter는 checkpoint에 포함되지 않았다.
- smoke evidence: `classification_experiments/evidence/024_smoke.json`

### 전체 학습

49,907장, 3 epochs, 총 582 logical step을 완료했다. 소요 시간은 1시간 33분 2초이고 최종 privacy expenditure는 epsilon `8.2672` at delta `1e-5`로 목표 epsilon 10 이내였다.

| epoch | train loss | epsilon |
|---:|---:|---:|
| 1 | 0.2475 | 6.2851 |
| 2 | 0.2420 | 7.4000 |
| 3 | 0.2416 | 8.2672 |

최종 checkpoint는 `classification_experiments/artifacts/ablation/xrv_aux/ldm_dp_final.pt`에 저장했다.

### DDIM50 validation 평가

E004와 동일한 validation 802장, DDIM 50-step, eta 0, image seed 조건으로 비교했다.

| arm | XRV AUROC | MedCLIP AUROC |
|---|---:|---:|
| E004 projection+LoRA | 0.5613 | 0.5166 |
| E024 XRV auxiliary | 0.5617 | 0.5157 |

- XRV: E024−E004 Δ=`+0.00040`, paired patient bootstrap 95% CI=`[-0.00547, 0.00702]`
- MedCLIP: E024−E004 Δ=`-0.00086`, paired patient bootstrap 95% CI=`[-0.01376, 0.01085]`

### 동일 난수 prompt-label response

E004와 E024 각각 4병변×존재/부재×32 seed, 총 256장을 DDIM50으로 생성했다. 각 존재/부재 pair는 동일한 초기 noise seed를 사용했다.

| arm | XRV intent AUROC 평균 | MedCLIP intent AUROC 평균 | XRV paired score Δ 평균 | MedCLIP paired score Δ 평균 |
|---|---:|---:|---:|---:|
| E004 | 0.5325 | 0.5261 | +0.01182 | +0.000584 |
| E024 | 0.5327 | 0.5195 | +0.01299 | +0.000603 |

E024 병변별 intent AUROC는 XRV `0.5254–0.5566`, MedCLIP `0.5078–0.5312`였다. 상세 결과는 `classification_experiments/evidence/024_prompt_response.json`에 저장했다.

통합 evidence는 `classification_experiments/evidence/024_results.json`, DDIM50 원시 평가는 `classification_experiments/evidence/xrv_aux_validation.json`, E004 paired 비교는 `classification_experiments/evidence/024_xrv_auxiliary_loss_comparison.json`에 저장했다. 사전 지정한 두 encoder 동시 개선 조건을 충족하지 않아 full DDPM은 실행하지 않았다.

## 5. 실험 결과 분석
E024의 XRV AUROC는 E004보다 `0.0004` 높았지만 신뢰구간이 0을 포함했고, MedCLIP은 `0.0009` 낮으면서 역시 신뢰구간이 0을 포함했다. 따라서 label-aware XRV auxiliary loss가 validation DDIM50 분류 성능을 개선했다는 근거는 확인되지 않았다.

prompt response도 E004와 실질적으로 같았다. XRV intent AUROC 평균은 `+0.0002` 높았지만 MedCLIP intent AUROC 평균은 `-0.0066` 낮았고, 두 모델 모두 약 `0.52–0.53`으로 chance에 가까웠다. 보조 loss가 기존 모델의 약한 조건 반응을 의미 있게 강화하지 못했다.

학습 자체는 정상 작동했다. projection과 LoRA에 보조 gradient가 도달했고 loss와 privacy accountant가 안정적이었으므로 실패 원인은 구현 단절보다 효과 부족에 가깝다. 특히 대부분의 per-sample gradient가 clipping norm `0.001`을 초과했고, auxiliary term은 timestep 제한과 weight `0.01` 때문에 배치 평균 total loss에 주로 `0.001–0.01` 수준으로 기여했다. 이 관측은 보조 신호가 DP clipping 이후 충분히 남지 않았을 가능성을 제시하지만, 이번 단일 weight 실험만으로 원인을 확정하지 않는다.

사전 결정 규칙상 XRV와 MedCLIP이 함께 개선되지 않았으므로 E024를 최종 후보로 선정하지 않고 full DDPM을 실행하지 않는다. 다음 학습 방법 실험은 privacy target과 E004 기준을 유지하면서 auxiliary gradient가 clipping 전에 diffusion gradient와 균형을 이루도록 설계해야 하며, validation 결과를 이용한 사후 weight 선택은 별도 탐색 실험으로 명시해야 한다.
