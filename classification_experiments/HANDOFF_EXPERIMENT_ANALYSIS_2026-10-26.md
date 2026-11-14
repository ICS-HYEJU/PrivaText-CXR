# Classification Experiments Analysis Handoff

작성일: 2026-10-26 (Asia/Seoul)

## 1. 이 문서의 목적

이 문서는 `classification_experiments/`에서 수행한 E001~E027의 목적과 결과를
지도교수 관점에서 요약하고, 다음 세션이 이미 실패한 탐색을 반복하지 않으면서 가장
근거가 강한 후속 실험을 선택하도록 하기 위한 hand-off이다.

다음 세션은 반드시 저장소 루트의 `AGENTS.md` 시작 워크플로우를 먼저 수행한다.
학습·생성은 `/opt/conda/bin/python`, MedCLIP 평가는
`/opt/medclip_env/bin/python`을 사용하고, 평가 시
`PYTHONPATH=/workspace/PrivaText-CXR/src`를 설정한다.

## 2. 현재 실행 상태

- E027 `counterfactual conditioning` 전체 DP 학습이 GPU1에서 실행 중이다.
- 2026-10-26 점검 시 epoch 1, logical step 약 125까지 진행했다.
- 실행 파일: `classification_experiments/counterfactual_training.py`
- 학습 로그: `classification_experiments/artifacts/counterfactual_train.log`
- 체크포인트 출력 예정 위치:
  `classification_experiments/artifacts/ablation/counterfactual/`
- 자동 후속 평가 스크립트: `classification_experiments/continue_027.sh`
- smoke evidence: `classification_experiments/evidence/027_smoke.json`
- smoke는 projection/LoRA rank-loss gradient, 두 번의 UNet forward under Opacus,
  frozen VAE, checkpoint round-trip을 모두 통과했다.
- 학습 로그에서 per-sample gradient clipping fraction은 대체로 0.5~1.0이며,
  `condition_gap`은 아직 0 부근 또는 음수이다. E027을 성공으로 미리 간주하지 않는다.

E027이 실행 중인 동안 같은 GPU 또는 같은 output 경로를 사용하는 학습·생성·평가를
추가로 시작하지 않는다. 프로세스를 임의로 중단하거나 checkpoint를 덮어쓰지 않는다.

## 3. 지금까지 확인된 핵심 결론

### 3.1 VAE는 병목이지만 전체 실패를 설명하지는 못한다

- E001 real -> deterministic VAE reconstruction:
  - XRV AUROC: `0.7765 -> 0.7396`
  - MedCLIP AUROC: `0.5706 -> 0.5037`
- VAE가 특히 MedCLIP 의미 정보를 크게 손실한다.
- 반면 XRV 정보는 상당 부분 남으므로 생성 XRV AUROC가 chance 부근인 현상을 VAE
  하나만으로 설명할 수 없다.

### 3.2 가장 큰 생성기 병목은 약한 condition dependence이다

- E002 prompt-response는 두 ep30 checkpoint 모두 XRV/MedCLIP intent AUROC가
  대체로 `0.50~0.52`였다.
- E006 correct-vs-shuffled report denoising은 기존 ep30 모델에서조차 상대 이득이
  최대 `0.08%`였고, 새 3-epoch 모델에서는 CI가 모두 0을 포함했다.
- 따라서 report가 전혀 사용되지 않는다고 단정할 수는 없지만, 임상 병변을 제어할
  만큼 강하게 사용된다는 근거는 없다.

### 3.3 Projection 학습만 명확한 구조적 개선 신호를 보였다

- E004 Projection+LoRA DDIM50:
  - XRV `0.5296 -> 0.5613`, delta `+0.0317`, paired CI가 0을 포함하지 않음
  - MedCLIP `0.5047 -> 0.5166`, CI는 0 포함
- E016 projection-only XRV는 `0.5533`이었다.
- E017 LoRA-only는 corrected와 동일한 `0.5296/0.5047`이었다.
- 결론: projection 적응이 주요 유효 요인이고 LoRA는 projection과 함께 있을 때만
  제한적으로 기여할 가능성이 있다.

### 3.4 단순 hyperparameter/구조 조정은 E004를 넘지 못했다

다음 계열은 우선 종료한다.

- LoRA rank 4 -> 8: `0.5517/0.5148`
- projection LR `.004`: `0.5603/0.5100`
- projection LR `.001`: `0.5578/0.5121`
- condition dropout 0.1: `0.5558/0.4871`
- structured labels + projection/LoRA: `0.5604/0.5072`
- clipping norm `.002`: `0.5572/0.5167`

작은 learning-rate, rank, dropout, clipping-norm sweep을 추가로 반복하지 않는다.

### 3.5 DP Gaussian noise가 중요한 병목이다

- E005 XRV:
  - clipping-only, sigma=0: `0.6374`
  - no clipping, sigma=0: `0.5934`
  - DP: `0.5613`
- clipping 자체가 실패 원인이라는 근거는 없고, 같은 clipping에 Gaussian noise를
  추가했을 때 utility가 크게 하락했다.
- 새로운 loss를 제안할 때는 곧바로 DP 전체 학습으로 가지 말고 동일 구조의
  clipping-only learnability control을 먼저 수행한다.

### 3.6 Frozen-XRV auxiliary BCE 계열은 종료한다

- E024: E004 대비 XRV `+0.00040`, MedCLIP `-0.00086`; 두 CI 모두 0 포함.
- E025 lambda 0.1: weighted auxiliary가 diffusion loss의 1.61배가 되어 사전 gate로
  전체 학습을 중단했다.
- E026 stratified timestep: XRV `+0.00152`, MedCLIP `-0.01387`; XRV teacher 방향의
  편향은 커지고 일반화 개선은 없었다.
- frozen-XRV BCE weight/timestep을 더 탐색하지 않는다.

### 3.7 DDPM1000을 자동으로 최종 정답 sampler로 취급하지 않는다

- E004 checkpoint가 DDIM50에서는 `XRV=0.5613`, `MedCLIP=0.5166`이었으나
  DDPM1000에서는 `0.5182/0.4911`로 하락했다.
- DDIM50은 단순 screening proxy가 아니라 실제로 더 나은 sampler 후보일 수 있다.
- 최종 후보는 DDIM 50/100/200과 DDPM1000을 독립 ablation으로 비교하고 sampler를
  고정한 뒤 새 holdout에서 평가한다.

## 4. 합성 데이터 downstream classification 해석

- unfiltered synthetic은 도움되지 않았다. 기존 real-only `0.6461`보다 모든
  synthetic-containing arm이 낮았고 synthetic-only는 chance에 가까웠다.
- E009/E010에서 teacher-consistency filtering의 top25가 단일 seed에서는
  `0.6580/0.6678`로 상승했다.
- E011의 3-seed dual-top25 결과:
  - real-only mean `0.6382`, SD `0.0099`
  - filtered augmentation mean `0.6513`, SD `0.0078`
  - paired delta: `+0.0188`, `+0.0251`, `-0.0046`
  - mean delta `+0.0131`이지만 한 seed에서 방향이 반전됐고 n=3이라 확정 근거가 아님.
- E015에서는 top5 mean `0.6489`, top10 `0.6485`, top15 `0.6470`, top20
  `0.6406`, top25 `0.6452`였다. threshold별 차이는 seed 변동과 비슷하거나 작다.
- 최초 real-only seed42 `0.6461`과 재실행 seed42 `0.6305`의 차이만 약 1.56%p로,
  주장하려는 synthetic 효과와 비슷하다.
- 같은 5,000-image test를 반복 관찰했으므로 이 test는 사실상 development set이
  되었다. 더 이상 최종 일반화 주장에 사용하지 않는다.

Downstream 후속 후보는 top25가 아니라 다음 두 개만 유지한다.

1. dual-consistency top5: 관측 평균이 가장 높음
2. dual-consistency top15: 관측 seed 분산이 가장 작음

다음 검증에는 아래 arm이 모두 필요하다.

- `R1100`
- `R1100 + S275(top5)`
- `R1100 + S825(top15)`
- 동일 총량 real-only `R1375`
- 동일 총량 real-only `R1925`

동일 총량 real-only가 없으면 synthetic의 효과와 단순한 데이터/optimizer-step 증가를
구분할 수 없다. 최소 5개 classifier seed와 아직 결과를 보지 않은 patient-disjoint
holdout을 사용한다. 병변별 AUROC, macro AUROC, calibration, label support를 모두
보고한다.

## 5. E027의 해석상 주의점

E027은 방향성은 타당하지만 현재 형태를 최종 설계로 승격하지 않는다.

1. 모든 표본의 negative가 하나의 고정 No-Finding 문장이라 모델이 개별 병변 대신
   `abnormal vs No Finding` shortcut만 배울 수 있다.
2. No-Finding 표본에서는 correct와 negative가 의미상 거의 동일하다.
3. 고정 margin `0.01`은 timestep별 MSE scale을 반영하지 않는다. E006에서 MSE는
   timestep에 따라 약 `0.91`에서 `0.0036`까지 변했다. 따라서 같은 margin이 어떤
   timestep에서는 미미하고 다른 timestep에서는 원 diffusion loss보다 상대적으로
   매우 커진다.
4. 높은 clipping fraction 때문에 rank loss의 크기를 단순히 올려도 유효 신호가
   보존된다는 보장이 없다. diffusion gradient와 rank gradient의 per-sample norm 및
   cosine을 별도로 감사해야 한다.

E027 완료 후 판정 순서는 다음과 같다.

1. E004 대비 XRV/MedCLIP paired comparison
2. correct-vs-shuffled denoising gap
3. 동일-noise prompt intent AUROC 및 paired score delta
4. 두 encoder가 모두 개선되고 효과가 CI/사전 최소효과 기준을 통과할 때만 sampler
   ablation 또는 더 큰 최종 평가로 이동

점추정치가 E004보다 아주 조금 높다는 이유만으로 성공 처리하지 않는다.

## 6. 선정된 다음 주력 실험

E027 이후 가장 우선할 생성기 실험은 다음이다.

**Pathology-specific label-flip counterfactual conditioning with a clipping-only
learnability control**

권장 설계:

- 각 record의 structured CheXpert 상태에서 한 병변만 반전한 deterministic negative를
  만든다.
- 나머지 병변 상태는 correct와 동일하게 유지해 hard negative로 만든다.
- 다른 환자의 report를 사용하지 않아 record-level DP 경계를 유지한다.
- No-Finding-only 표본은 rank loss에서 제외하거나 별도의 명시적 규칙을 사전 고정한다.
- margin은 고정 절대값 대신 detached correct MSE 또는 timestep별 baseline으로
  정규화한다.
- 동일 `z`, timestep, Gaussian noise에서 correct/negative를 비교한다.
- 먼저 clipping-only sigma=0 arm으로 목적함수의 learnability upper bound를 확인한다.
- clipping-only가 성공한 경우에만 동일 설계의 epsilon-10 DP arm을 실행한다.
- gradient audit에 diffusion/rank 각각의 per-sample norm, cosine, total clipping 이후의
  방향 변화를 포함한다.

판정 논리:

- clipping-only도 실패: counterfactual/loss 설계를 폐기한다.
- clipping-only 성공, DP 실패: 추가 auxiliary loss가 아니라 DP gradient SNR 개선이
  다음 연구 질문이다.
- denoising gap만 개선되고 prompt/AUROC가 실패: condition 사용은 늘었지만 임상적으로
  올바른 condition grounding에는 실패한 것이다.
- XRV만 개선하고 MedCLIP이 하락: teacher/shortcut 편향으로 판정한다.

## 7. 평가 프로토콜 개선

- 현재 validation 802장은 일부 병변에서 positive 또는 negative support가 매우 작아
  macro AUROC가 소수 reliable label에 의존한다.
- 모델/목적함수/sampler 선택은 기존 validation에서 하되, 최종 후보는 아직 사용하지
  않은 큰 patient-disjoint holdout에서 한 번만 평가한다.
- 새 holdout을 보기 전에 primary endpoint, reliable-label 기준, 최소 효과 크기,
  bootstrap 단위, sampler를 고정한다.
- XRV와 MedCLIP을 합쳐 하나의 점수로 만들지 않는다. 두 encoder가 같은 방향인지
  별도로 보고한다.
- 생성 conditionality와 downstream classifier utility는 별개의 endpoint로 유지한다.

## 8. 저장소 기록상 정리할 항목

새 실험을 시작하기 전에 다음 기록 불일치를 수정한다.

- `020_condition_dropout.md`는 결과 대기라고 쓰여 있지만 evidence에는
  `0.5558/0.4871` 결과가 존재한다.
- `022_dp_epsilon5.md`와 `022_projection_lr001.md`가 같은 번호를 사용한다.
- epsilon5 실험은 완료 evidence를 찾지 못했으므로 완료 여부를 명시해야 한다.
- `015_filter_threshold_seeds.md`의 표에 literal `\n`이 있고 결과 분석이 미완성이다.
- 현재 worktree가 매우 dirty하고 E027이 실행 중이므로 관련 없는 파일을 정리하거나
  대규모 커밋에 포함하지 않는다.

## 9. 다음 세션에 그대로 전달할 프롬프트

아래 블록을 새 세션의 사용자 프롬프트로 그대로 사용할 수 있다.

```text
# Goal
classification_experiments의 현재 실험을 이어받아 E027을 엄격히 판정하고,
근거가 있을 때만 다음 개선 실험을 설계·실행한다.

# Role
지도교수 겸 재현성 감사자 역할을 수행한다. 점추정치 상승보다 실험 통제,
paired uncertainty, independent holdout, DP 경계와 다음 세션의 재현 가능성을 우선한다.

# Required context
1. 저장소 루트 AGENTS.md의 시작 워크플로우를 먼저 수행한다.
2. claude-progress.md와 feature_list.json을 읽는다.
3. classification_experiments/HANDOFF_EXPERIMENT_ANALYSIS_2026-10-26.md를 전문으로 읽는다.
4. E027 프로세스, counterfactual_train.log, checkpoint, continue_027.sh 및
   evidence/027* 상태를 확인한다.
5. 기존 dirty worktree와 실행 중인 프로세스를 보존하고 관련 없는 변경을 수정하거나
   커밋하지 않는다.

# Immediate task
- E027이 아직 실행 중이면 중복 학습을 시작하지 말고 완료/오류 여부를 감시한다.
- 완료되면 자동 평가가 실제로 모두 수행됐는지 확인하고 다음을 하나의 결과표로 만든다:
  E004 대비 XRV/MedCLIP macro 및 병변별 AUROC, patient-paired CI,
  correct-vs-shuffled denoising gap, prompt intent AUROC와 paired score delta,
  최종 epsilon, clipping/gradient audit.
- LoRA checkpoint가 생성에 실제 반영됐는지 md5 또는 checkpoint loading log로 확인한다.
- 단순히 두 macro point estimate가 높다는 이유로 성공 처리하지 않는다.

# Decision rules
- 두 encoder, denoising gap, prompt response가 함께 개선될 때만 E027을 후보로 유지한다.
- XRV만 오르고 MedCLIP이 하락하면 teacher/shortcut 편향으로 판정한다.
- denoising gap만 오르고 AUROC/prompt response가 오르지 않으면 condition use와
  clinical grounding이 분리된 실패로 판정한다.
- DDPM1000을 자동으로 최종 정답으로 취급하지 않는다. 통과한 후보에 대해서만
  DDIM50/100/200과 DDPM1000 sampler ablation을 사전 지정해 비교한다.

# Selected next experiment, only after E027 adjudication
병변별 label-flip counterfactual conditioning을 설계한다.
- 한 record의 다른 label 상태는 유지하고 하나의 병변만 반전한 deterministic hard
  negative를 사용한다.
- 다른 환자 report를 사용하지 않는다.
- No-Finding-only record의 rank-loss 처리 규칙을 사전에 고정한다.
- absolute margin 0.01을 그대로 재사용하지 말고 timestep/MSE scale에 정규화한다.
- diffusion/rank per-sample gradient norm과 cosine을 감사한다.
- 먼저 clipping-only sigma=0 learnability control을 실행한다.
- clipping-only가 명확히 성공한 경우에만 epsilon10 DP 전체 학습을 승인한다.

# Downstream utility track
같은 5,000 test set으로 threshold를 더 탐색하지 않는다. top5와 top15만 유지하고,
R1100, R1100+S275, R1100+S825, R1375, R1925를 최소 5 seeds와 새로운
patient-disjoint holdout에서 비교하는 계획을 작성한다. 동일 총량 real-only 대조군 없이
synthetic utility 개선을 주장하지 않는다.

# Deliverables
- E027 최종 판정 보고서와 evidence 경로
- 채택/기각 결정 및 수치 근거
- 실행했다면 새 실험의 사전 명세, verification command, evidence
- claude-progress.md와 feature_list.json의 정확한 상태 업데이트
- 미해결 위험 및 다음 세션용 handoff
- 안전할 때 관련 파일만 설명적인 메시지로 커밋
```

## 10. 최종 우선순위

1. E027 완료 및 엄격한 판정
2. 병변별 label-flip counterfactual의 clipping-only learnability control
3. 성공 시 epsilon-10 DP 재검증
4. 성공 후보에 한해 sampler ablation
5. downstream top5/top15를 matched-real 및 새 holdout에서 재검증
6. 그래도 MedCLIP이 chance이면 더 높은 해상도/다채널 latent 또는 CXR 의미보존
   목적함수를 갖는 VAE 개선 검토

