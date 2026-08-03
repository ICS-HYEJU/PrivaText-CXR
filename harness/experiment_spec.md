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

## 3. 데이터 분할 — search / train / test

**확정(2026-08-03):** MIMIC train split을 **환자 ID(`patient_id`) 기준으로** `search` / `train` 두
그룹으로 분리합니다. 하이퍼파라미터 탐색(T1)은 `search`에서만 수행하고, 최종 확정 런(T2)은 `train`에서
수행합니다. 데이터셋 추가 다운로드가 진행 중이므로 이를 반영한 전처리가 선행되어야 합니다
(`feature_list.json` F-11).

```
MIMIC train
  ├── search  (T1 하이퍼파라미터 탐색 전용)   ← 환자 단위로 분리
  └── train   (T2 최종 확정 런)
MIMIC test    (최종 평가, 어느 런에서도 학습에 쓰지 않음)
```

**환자 단위 분리가 필수인 이유**: 한 환자는 여러 study·여러 view를 갖습니다. 이미지 단위로 나누면
같은 환자의 다른 촬영본이 search와 train에 걸쳐 들어가 탐색 결과가 낙관적으로 편향됩니다.
`Data/mimic_cxr.py:172`가 이미 샘플마다 `patient_id`를 들고 있으므로 구현은 가능합니다.
(`Data/mimic_cxr_old.py:316` 주석에 과거 환자 단위 분할 로직이 남아 있습니다 — 참고용)

### 3.1 하이퍼파라미터 튜닝의 프라이버시 소모

동일 데이터에서 하이퍼파라미터를 반복 탐색하면 최종 선택된 설정 자체가 데이터 정보를 누설합니다.
search/train 분리로 **train에 대한** 누설은 차단되지만, `search` 그룹의 환자들에 대해서는 반복
탐색만큼 프라이버시가 소모됩니다.

**정책:**
1. `search` 그룹에서 수행한 모든 런을 `RUN_LOG.md`에 기록한다 (탐색 횟수 자체가 보고 대상).
2. `search` 그룹의 환자는 최종 모델 학습에 **재사용하지 않는다.**
3. 논문에는 "하이퍼파라미터는 학습에 사용되지 않은 별도 환자 그룹에서 탐색했다"고 명시한다.

### 3.2 ⚠ Sample-level DP vs Patient-level DP — 확인 필요

**Opacus가 제공하는 것은 sample-level DP입니다.** 즉 "이미지 1장이 학습에 포함되었는지"를 숨깁니다.
데이터셋을 환자 단위로 다루고 계시므로, 의도하신 프라이버시 단위가 **환자**일 가능성이 높습니다.

한 환자가 `k`장의 이미지를 갖는다면, 보고되는 `ε=10`은 **환자 단위로는 성립하지 않습니다.**
group privacy로 환산하면 대략 `(k·ε, k·e^{(k-1)ε}·δ)` 수준으로 급격히 약해집니다.
MIMIC은 환자당 이미지가 여러 장이므로 이 차이는 무시할 수 없습니다.

**선택지:**
- **(a) sample-level로 명시** — 현재 구현 그대로. 논문에 "image-level DP"라고 정확히 기술.
- **(b) patient-level 보장** — 환자당 이미지를 1장으로 제한하거나, Poisson 샘플링을 환자 단위로
  바꾸고 환자별 gradient를 합산 후 클리핑. 구현 난이도가 상당함.

> `TODO_CONFIRM`: (a)/(b) 중 선택. **이 결정은 루프를 돌리기 전에 내려야 합니다.**
> 나중에 (b)로 바꾸면 σ 계산과 sample_rate가 전부 달라져 그동안의 T1 탐색 결과가 무효가 됩니다.

---

## 3.5 자원 — GPU 2개

- **기본 구성**: T1 후보 2개를 병렬 실행하여 탐색 처리량을 2배로.
- **T2 진행 중**: GPU 1개는 T2 전용, 나머지 1개로 T1 탐색 계속.
- 병렬 실행 시에도 **`decision_rules.md`의 "한 번에 한 파라미터" 원칙은 유지**합니다.
  베이스라인 대비 서로 다른 파라미터를 각각 1개씩 바꾼 2개 런을 동시에 돌리는 것은 허용되지만,
  **한 런 안에서 2개를 바꾸는 것은 금지**입니다 (원인 귀속 불가).
- 두 GPU에 동일 설정을 seed만 바꿔 돌리면 **런간 분산**을 측정할 수 있습니다.
  T1 개선폭이 이 분산보다 작으면 승격하지 않습니다 (`decision_rules.md` §3-4).

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
