# 028_pathology_label_flip_clipping_control

## 1. 실험 주제
병변별 structured-label hard negative와 timestep-normalized margin loss가 조건 학습에 실제로 학습 가능한지 Gaussian noise가 없는 clipping-only 대조 실험으로 검증한다.

## 2. 실험 선정 근거
E006은 정답 report의 denoising 이득이 최대 0.08%에 불과했고 E027의 단일 고정 `No Finding` negative는 abnormal-vs-normal shortcut과 절대 margin의 timestep scale 불일치 위험이 있다. E005에서 동일 clipping 조건의 sigma=0 모델이 DP 모델보다 높은 XRV utility를 보였으므로, 새 목적함수를 곧바로 epsilon-10 DP로 평가하지 않고 noise 없는 learnability upper bound를 먼저 확인한다. E024~E026에서 실패한 frozen-XRV teacher loss는 사용하지 않는다.

## 3. 실험 방식
- 하나의 실험 안에서 동일 초기화·데이터 순서·Poisson sampler·latent/timestep RNG를 사용하는 두 clipping-only arm을 비교한다.
  - `structured_clip_control`: structured CheXpert token + 기존 diffusion loss.
  - `label_flip_clip`: 위 구조에 pathology-specific rank loss만 추가.
- 두 arm 모두 Gaussian noise `sigma=0`, clipping norm `0.001`, epsilon `inf`인 비공개 learnability 대조군이다. seed `42`, projection/LoRA learning rate `0.002`, rank/alpha `4/4`, 3 epochs, logical/physical batch `256/4`를 고정한다.
- 각 record에서 `No Finding`과 `Support Devices`를 제외한 양성 병변 중 하나를 stable hash로 결정해 상태 `positive -> negative`로 반전한다. 다른 13개 상태와 원 report text는 유지한다. 양성 병변이 없는 No-Finding-only record는 rank loss에서 제외한다.
- correct와 label-flip negative는 동일 latent, timestep, Gaussian noise를 사용한다. `relative_gap=(MSE_flip-MSE_correct)/detach(MSE_correct)`로 정규화하고 `L_rank=max(0, rho-relative_gap)`을 사용한다. 사전 고정 `rho=0.01`, `lambda_rank=0.1`이며 validation을 보고 조정하지 않는다.
- rank loss는 각 record 자체의 영상·report·label에서만 계산하고 다른 환자 데이터를 사용하지 않는다.
- 실제 모델 스모크에서 structured payload의 정확히 한 병변만 반전되는지, No-Finding-only 제외, 두 UNet forward, projection/LoRA/label-token gradient, finite loss, frozen VAE, checkpoint round-trip을 확인한다.
- 고정 smoke cohort에서 diffusion gradient와 rank gradient의 표본별 norm·cosine을 각각 계산한다. `C=0.001`로 표본별 total gradient를 모의 clipping한 뒤 unclipped aggregate와 clipped aggregate의 cosine 및 rank 기여 방향을 evidence에 기록한다.
- 두 arm을 validation 802장 DDIM50으로 생성해 XRV/MedCLIP macro·병변별 AUROC와 patient-paired CI를 비교한다. 병변별 correct-vs-flipped 및 E006-compatible correct-vs-shuffled denoising gap, 동일-noise prompt response도 평가한다.
- epsilon-10 DP 후속 실험 승인 기준은 `label_flip_clip - structured_clip_control`에서 다음을 모두 충족하는 경우로 사전 고정한다: (1) XRV와 MedCLIP macro AUROC가 각각 최소 `+0.01`이고 paired CI 하한이 0보다 큼, (2) 4개 timestep 중 최소 3개에서 correct-vs-flipped relative gap이 `0.5%` 이상이고 patient CI 하한이 0보다 큼, (3) XRV와 MedCLIP prompt intent AUROC 평균이 각각 최소 `+0.02`, (4) gradient audit에서 rank gradient가 finite이고 clipping 후 aggregate가 diffusion-only 방향과 음의 cosine을 보이지 않음. 하나라도 실패하면 DP arm은 실행하지 않는다.

## 4. 실험 결과
실행 대기 중이다. 구현과 정적 검증을 완료했으며 현재 E027 학습이 GPU 1에서 진행 중이다. 핸드오프의 순서 조건에 따라 E027 학습·자동 평가·엄격 판정이 끝나기 전에는 E028 GPU 스모크나 학습을 시작하지 않는다.

- `label_flip_training.py`: 한 양성 병변만 stable-hash로 선택해 positive→negative 반전, 상대 MSE margin loss, matched structured control, clipping-only 실행을 구현했다.
- 정적 검증: exactly-one flip과 No-Finding-only 제외 규칙, Python compile, launcher shell syntax를 통과했다.
- gradient audit: 표본별 diffusion/rank norm과 cosine, projection/LoRA/label-token rank gradient, `C=0.001` 모의 clipping 전후 aggregate cosine을 기록하도록 구현했다.
- `queue_028_after_027.sh`: E027의 전체 자동 평가 완료 marker를 기다린 뒤 E027 adjudication을 저장하고, E028 candidate/control 스모크가 모두 통과할 때만 두 arm 전체 학습을 시작하도록 기동했다.

## 5. 실험 결과 분석
결과 대기 중이다. clipping-only에서도 승인 기준을 충족하지 못하면 label-flip 목적함수 계열을 폐기한다. clipping-only는 성공하지만 후속 DP가 실패하면 조건 loss를 추가 탐색하지 않고 DP gradient SNR 개선을 별도 연구 질문으로 전환한다.
