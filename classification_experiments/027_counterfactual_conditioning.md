# 027_counterfactual_conditioning

## 1. 실험 주제
정답 `LABEL+IMPRESSION` 조건의 denoising loss가 고정 `No Finding` 반대 조건보다 낮아지도록 condition-contrastive margin loss를 추가한다.

## 2. 실험 선정 근거
E006에서 정답 report와 다른 report의 denoising MSE 차이가 거의 0이었고, E024~E026의 frozen-XRV BCE는 XRV 방향의 미세한 변화와 MedCLIP 하락을 보여 teacher-specific 보조손실을 종료했다. E027은 외부 분류기 없이 UNet이 report 조건을 실제 denoising에 사용하도록 직접 최적화한다.

## 3. 실험 방식
- 각 표본에서 동일한 latent, timestep, Gaussian noise로 정답 조건과 반대 조건을 각각 한 번 forward한다.
- 정답 조건은 해당 표본의 실제 `LABEL+IMPRESSION`, 반대 조건은 모든 표본에 공통인 공개 문장 `LABEL: No Finding IMPRESSION: No acute cardiopulmonary abnormality.`이다.
- 다른 환자의 report를 negative로 사용하지 않으므로 한 표본의 loss가 다른 사람의 데이터에 의존하지 않으며 DP 표본 단위를 유지한다.
- `L_rank=max(0, margin + MSE_correct - MSE_counterfactual)`, `L_total=L_diffusion+lambda_rank*L_rank`를 사용한다. 사전 고정값은 margin `0.01`, lambda `0.1`이다.
- target epsilon `10`, delta `1e-5`, clipping norm `0.001`, seed `42`, projection/LoRA learning rate `0.002`, rank/alpha `4/4`, 3 epochs, logical/physical batch `256/4`, 데이터 및 split은 E004와 동일하다.
- 전체 학습 전 실제 모델 스모크에서 두 UNet forward의 Opacus 처리, 표본별 gradient, projection/LoRA rank-loss gradient, frozen VAE, checkpoint round-trip, rank loss 비지배 여부를 확인한다.
- 학습 후 validation 802장 DDIM50 XRV/MedCLIP AUROC, E004 patient-paired bootstrap, E006 방식의 correct-vs-shuffled denoising gap, 동일-noise prompt response를 평가한다. XRV와 MedCLIP이 모두 E004보다 개선될 때만 full DDPM을 수행한다.

## 4. 실험 결과
진행 중. E027 전용 학습 구현과 실제 모델/Opacus 스모크를 수행한다.

## 5. 실험 결과 분석
결과 대기 중이다. 정답 조건 gap이 증가해도 AUROC가 개선되지 않으면 조건 사용 자체와 임상적으로 올바른 조건 반영을 구분하며, AUROC와 prompt response를 우선 판단 기준으로 사용한다.
