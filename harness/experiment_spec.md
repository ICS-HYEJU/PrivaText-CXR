# Experiment Spec — PrivaText-CXR DP Finetune 자동화

> **버전** v0.1 (draft) · **대상** `LDM_dp_finetune.py` · **작성 목적** 에이전트가 학습·평가 루프를
> 재현 가능하게 수행하기 위한 단일 진실 공급원(single source of truth).
> `TODO_CONFIRM` 표시는 사람이 확정해야 하는 값입니다. 확정 전에는 루프를 **T2(full run)까지 진행하지 않습니다.**

---

## 1. 목표

MIMIC-CXR 리포트 조건부 latent diffusion 모델을 **differential privacy 예산 내에서** fine-tune 하되,
생성 품질(FID) 및 학습 안정성(val_loss) 기준을 만족하는 하이퍼파라미터 조합을 자동으로 탐색한다.

## 2. 최우선 제약 — 1회 학습 = 5일 3시간

실측: **100 epoch / LoRA 미적용 / 약 5d 3h 8m** → **약 1.24 h/epoch**.

이 제약이 루프 설계 전체를 규정합니다.

> **"학습 완료 → 평가 → 조정 → 재학습" 형태의 닫힌 루프는 이 프로젝트에서 성립하지 않습니다.**
> 5일짜리 반복을 10회만 돌려도 50일입니다. 대신 아래 3-tier 구조를 사용합니다.

### Tier 구조

| Tier | 이름 | 설정 | 소요(목표) | 목적 | 루프 반복 |
|------|------|------|-----------|------|----------|
| **T0** | smoke | `--epochs 1`, subset ~200 samples, `--val_batches 5` | **≤ 15분** | 하네스 배관 검증. 코드/경로/인자 오류 조기 검출 | 코드 변경 시마다 |
| **T1** | proxy | `--epochs 3`, subset 10~20%, `--save_every 1` | **2~5시간** | 하이퍼파라미터 후보 **순위 결정**. 루프 반복의 90%가 여기서 발생 | 다수 (자동) |
| **T2** | full | `--epochs 100`, 전체 데이터 | **~5일** | T1 승격 후보의 최종 확인 | 소수 (승인 필요) |

**T2는 사람의 명시적 승인 없이 시작하지 않습니다.** (`decision_rules.md` §5 참조)

### T2의 루프는 "재학습"이 아니라 "비행 중 감시(in-flight monitoring)"

T2가 도는 5일 동안 에이전트는 대기하지 않습니다. `--save_every 5` 시점마다 기록되는
`metrics.jsonl`을 폴링하여 **트립와이어 위반 시 런을 조기 중단**시킵니다.
5일을 다 태우고 실패를 발견하는 것을 막는 것이 T2 자동화의 실질적 가치입니다.

프로세스는 반드시 **detached(nohup/tmux)** 로 띄웁니다. 에이전트 세션이 5일간 포그라운드
프로세스를 붙들고 있을 수 없습니다.

---

## 3. ⚠ DP 고유 문제 — 하이퍼파라미터 튜닝 자체가 프라이버시를 소모합니다

이것은 이 프로젝트의 자동 루프에서 **가장 중요한 연구 정합성 이슈**입니다.

동일한 private 데이터셋(MIMIC)에 대해 하이퍼파라미터를 반복 탐색하면, 최종 선택된 설정 자체가
데이터에 대한 정보를 누설합니다. 각 런이 `ε=10`을 보고하더라도 **20번 튜닝한 뒤 고른 설정의
실제 프라이버시 보장은 ε=10이 아닙니다.** 논문에 `ε=10`으로 보고하면 부정확한 주장이 됩니다.

**따라서 본 하네스의 기본 정책:**

1. **T1(탐색)은 MIMIC이 아닌 대체 데이터에서 수행한다.** 이 리포는 이미 NIH(`Data/nih.py`)와
   MIMIC(`Data/mimic_cxr.py`)을 모두 갖고 있습니다. NIH를 튜닝용 프록시로 사용합니다.
2. **T2(확정)만 MIMIC에서 수행하며, 횟수를 사전에 정하고 기록한다.** (`RUN_LOG.md`)
3. MIMIC에서 수행한 모든 런은 탐색용이었더라도 `RUN_LOG.md`에 기록한다. 논문에 튜닝 횟수를
   명시할 수 있어야 한다.

> `TODO_CONFIRM`: T1 프록시로 NIH를 쓰는 것이 타당한지 확인 필요.
> NIH는 label-conditional, MIMIC은 report(text)-conditional 이라 컨텍스트 인코더 경로가 다릅니다.
> 대안: MIMIC train split을 쪼개 "튜닝 전용 hold-out shard"를 만들고, 최종 런은 나머지로 수행.

---

## 4. 평가 메트릭

### 4.1 1차 지표 (현재 코드에서 즉시 취득 가능)

| 메트릭 | 출처 | 방향 | 기준값 | 상태 |
|--------|------|------|--------|------|
| `val_loss` | `LDM_dp_finetune.py:383 evaluate()` — 매 epoch | 낮을수록 좋음 | `TODO_CONFIRM` | ✅ 구현됨 |
| `epsilon_spent` | `privacy_engine.get_epsilon()` — 매 epoch | **제약(constraint)** | `≤ target_epsilon (10.0)` | ✅ 구현됨 |

`epsilon_spent`는 **최적화 대상이 아니라 하드 제약**입니다. 초과 시 해당 런은 품질과 무관하게 실패 처리합니다.

### 4.2 2차 지표 (구현 필요)

| 메트릭 | 출처 | 방향 | 기준값 | 상태 |
|--------|------|------|--------|------|
| `FID` | `Eval_metric/fid.py` | 낮을수록 좋음 | `TODO_CONFIRM` (제안: ≤ 50) | ❌ **파일 621줄 전부 주석 — 동작 안 함** |
| `SSIM` | `Eval_metric/ssim.py` | 높을수록 좋음 | `TODO_CONFIRM` (제안: ≥ 0.85) | ⚠ VAE recon 전용. LDM 생성물용 경로 없음 |
| `PSNR` | `Eval_metric/psnr.py` | 높을수록 좋음 | `TODO_CONFIRM` (제안: ≥ 28 dB) | ⚠ 동일 |

> **주의:** SSIM/PSNR은 *재구성(reconstruction)* 지표입니다. Diffusion **생성물**에는 참조 이미지가
> 없으므로 그대로 적용할 수 없습니다. LDM 단계의 주 지표는 **FID**이며, SSIM/PSNR은 VAE 단계
> 품질 게이트로만 사용합니다. → `feature_list.json` F-07 참조.

### 4.3 정지 조건 (루프 6번)

모두 만족 시 루프 종료:

```
epsilon_spent ≤ target_epsilon
AND val_loss  ≤ THRESH_VAL_LOSS      (TODO_CONFIRM)
AND FID       ≤ THRESH_FID           (TODO_CONFIRM)
```

`TODO_CONFIRM` 값이 비어 있으면 에이전트는 **자동 승격/정지 판단을 하지 않고 사람에게 보고**합니다.

---

## 5. 런 디렉토리 규약

현재 `--save_dir ./finetune_dp` 고정이라 서로 다른 하이퍼파라미터 런이 체크포인트를 덮어씁니다.
루프의 전제 조건으로 아래 구조를 도입합니다 (`feature_list.json` F-01).

```
runs/
  <run_id>/                    # run_id = {tier}_{YYYYMMDD-HHMMSS}_{hash8}
    config.json                # vars(args) 전체 — 루프 2번 "사용된 하이퍼파라미터 저장"
    metrics.jsonl              # epoch당 1줄 append — 루프 3번 입력
    stdout.log                 # nohup 리다이렉트
    analysis.md                # 루프 4번 "원인 분석 및 기록"
    status.json                # {state: running|passed|failed|aborted, reason: ...}
    ckpt/
      ep0005.pt ...
```

`hash8` = 튜닝 대상 하이퍼파라미터만 정렬 직렬화한 sha256 앞 8자리.
동일 설정 재실행을 에이전트가 감지할 수 있게 합니다.

### metrics.jsonl 스키마 (1 line = 1 epoch)

```json
{"run_id":"T1_20260803-101500_a1b2c3d4","epoch":3,"global_step":1240,
 "train_loss":0.1832,"val_loss":0.1975,"epsilon_spent":2.41,
 "lr":8.7e-05,"grad_norm_pre_clip":1.83,"clipped_frac":0.62,
 "wall_sec":4471,"timestamp":"2026-08-03T10:15:00Z"}
```

`grad_norm_pre_clip`, `clipped_frac`은 현재 기록되지 않지만 **DP 진단의 핵심 신호**입니다
(`decision_rules.md` D-03/D-04). → `feature_list.json` F-03.

---

## 6. 범위 밖 (이번 하네스에서 다루지 않음)

- VAE 사전학습(`Autoencoder_train.py`) 자동화 — DP finetune 확정 후 동일 구조 재사용
- 멀티 GPU / 분산 학습
- 하이퍼파라미터 베이지안 최적화 — 우선 규칙 기반(`decision_rules.md`)으로 시작
