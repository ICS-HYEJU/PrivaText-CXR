---
name: train-eval-loop
description: PrivaText-CXR DP finetune의 학습·평가·하이퍼파라미터 조정 루프를 1 iteration 수행한다. 학습 실행, 메트릭 평가, 원인 진단, 다음 파라미터 결정, 기록까지를 하나의 절차로 묶는다. 사용자가 "루프 돌려", "다음 실험", "학습 시작", "런 상태 확인"을 요청할 때 사용.
---

# Train-Eval Loop (1 iteration)

## 사전 조건 — 하나라도 미충족이면 중단하고 보고

```
[ ] harness/feature_list.json 의 critical_path (F-01,F-02,F-04,F-05,F-08) 가 done 인가
[ ] --vae_ckpt / --pretrained_ckpt 실제 파일이 존재하는가        (Q-02)
[ ] harness/experiment_spec.md 의 TODO_CONFIRM 임계값이 확정되었는가  (F-09)
[ ] tier 가 결정되었는가 (T2면 사람 승인을 받았는가)
```

임계값(F-09)이 미확정이면 **런을 실행하되 자동 pass/fail 판정은 하지 않고** 결과만 보고합니다.

---

## Step 1 — 이력 확인

`harness/RUN_LOG.md`를 읽습니다.

- 직전 런의 결론과 "다음 조치"가 무엇이었는지 확인
- 이번에 실행하려는 설정의 `hash8`이 이미 존재하면 **중복 실행하지 않고** 기존 결과를 보고

## Step 2 — 하이퍼파라미터 결정

- 첫 런이면 default = 베이스라인
- 아니면 직전 런의 `analysis.md` → "다음 조치"에 적힌 파라미터 1개만 변경
- 변경 값은 `harness/hparam_space.json`의 `range`와 `step_rule` 안에 있어야 함
- **한 번에 하나만.** 예외는 `coupled_with`에 명시된 쌍뿐
- `tunable: false` 파라미터는 건드리지 않음

## Step 3 — 실행

```bash
harness/launch_run.sh --tier T1 <변경된 인자들>
```

- 반드시 **detached**. 포그라운드로 기다리지 마십시오
- `runs/<run_id>/config.json`이 생성되었는지 확인 후 run_id를 사용자에게 보고
- T0는 15분 내 종료 예상이므로 완료까지 기다려도 됨

## Step 4 — 감시 (T1/T2)

`harness/watch_run.py`가 트립와이어를 검사합니다. 에이전트는 폴링 결과만 확인합니다.

- **에이전트가 sleep으로 대기하지 마십시오.** T2는 5일입니다
- 주기적 확인이 필요하면 `/loop` 또는 스케줄된 체크인을 사용
- 트립와이어 발동 시 → Step 5로 (조기 진단)

검사 항목은 `harness/decision_rules.md` §1 (TW-01 ~ TW-06).

## Step 5 — 평가

`runs/<run_id>/metrics.jsonl`을 읽습니다. **stdout.log를 정규식으로 긁지 마십시오.**

```
epsilon_spent ≤ target_epsilon ?   ← 하드 제약. 위반 시 품질과 무관하게 실패
val_loss      ≤ THRESH_VAL_LOSS ?
FID           ≤ THRESH_FID ?       ← F-07 구현 후
```

모두 통과 → **루프 종료.** 사용자에게 최종 설정과 결과를 보고합니다.

## Step 6 — 진단

`harness/decision_rules.md` §2를 **D-01부터 순서대로** 검사하고, **첫 번째로 매칭되는 규칙 하나만**
채택합니다.

- T0/T1 첫 런에서는 **D-05(설정 오류)를 먼저 확인**하십시오. 초기 실패의 대부분은
  하이퍼파라미터가 아니라 체크포인트 경로 문제입니다
- D-03/D-04는 `grad_norm_pre_clip`/`clipped_frac`이 필요합니다(F-03). 없으면 D-99로 처리
- **어느 규칙에도 매칭되지 않으면(D-99) 파라미터를 추측으로 바꾸지 말고 사람에게 보고**하십시오

## Step 7 — 기록

`runs/<run_id>/analysis.md`를 `decision_rules.md` §4 템플릿대로 작성합니다.

**진단에는 `metrics.jsonl`의 실제 수치를 인용해야 합니다.** 수치 없는 서술적 진단
("학습이 불안정해 보임")은 무효입니다.

`harness/RUN_LOG.md`에 1행 append. MIMIC 사용 런은 별도 표시(프라이버시 회계용).

## Step 8 — 다음 iteration

Step 2로 돌아갑니다. 단:

- T1 → T2 승격은 `decision_rules.md` §3의 5개 조건을 모두 만족해야 하며 **사람 승인 필수**
- 같은 규칙이 3회 연속 발동하고 개선이 없으면 루프를 멈추고 보고 — 규칙이 이 상황에
  맞지 않는다는 뜻입니다

---

## 절대 금지

`harness/decision_rules.md` §5와 동일합니다. 요약:

1. `target_epsilon` / `target_delta` 변경 — 기준을 맞추기 위한 목표 조작
2. 기준 미달 런을 "통과"로 기록
3. 한 번에 2개 이상 파라미터 변경 (coupled 예외 제외)
4. `TODO_CONFIRM` 임계값에 대한 자동 pass/fail 판정
5. 사람 승인 없는 T2 시작
6. stall(TW-05) 자동 재시작
7. D-99에서 추측에 의한 파라미터 변경
