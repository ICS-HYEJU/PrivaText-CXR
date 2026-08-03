# PrivaText-CXR

Report-conditional latent diffusion for chest X-ray generation, fine-tuned under differential privacy.

## 파이프라인

```
1) VAE 사전학습     Autoencoder_train.py    → checkpoints/vae/*.pt
2) LDM 학습(non-DP) LDM_train.py            → NIH, class-conditional
3) DP fine-tune     LDM_dp_finetune.py      → MIMIC, report-conditional, Opacus  ← 자동화 대상
```

세 스크립트 모두 **단일 argparse**로 모든 하이퍼파라미터를 노출합니다. 하드코딩된 설정 파일은 없습니다.

## 하네스 문서 (자동화 작업 시 반드시 먼저 읽을 것)

| 파일 | 역할 |
|------|------|
| `harness/experiment_spec.md` | 목표, tier 구조, 메트릭 정의, 임계값, 런 디렉토리 규약 |
| `harness/decision_rules.md` | **메트릭 미달 → 원인 진단 → 조정 파라미터 결정 테이블.** 진단은 여기서만 한다 |
| `harness/hparam_space.json` | 조정 가능 파라미터 화이트리스트와 안전 범위 |
| `harness/feature_list.json` | 하네스 구축에 필요한 구현 작업 목록 |
| `harness/RUN_LOG.md` | 누적 실험 이력 (아직 미생성 — F-08) |

## 핵심 제약

**1회 전체 학습 = 약 5일 3시간** (100 epoch, LoRA 미적용, 약 1.24 h/epoch).

- 전체 학습을 반복하는 닫힌 루프는 성립하지 않습니다. **T0/T1/T2 tier 구조**를 사용합니다
  (`experiment_spec.md` §2).
- T2(full run)는 **사람의 명시적 승인 없이 시작하지 않습니다.**
- 5일 런은 반드시 **detached**로 띄웁니다. 포그라운드로 대기하지 마십시오.

**DP 예산은 최적화 대상이 아니라 하드 제약입니다.**

- `--target_epsilon` / `--target_delta`를 기준 통과 목적으로 변경하는 것은 **금지**입니다.
- private 데이터(MIMIC)에서의 반복 튜닝은 그 자체로 프라이버시를 소모합니다
  (`experiment_spec.md` §3). MIMIC 사용 런은 전부 `RUN_LOG.md`에 기록합니다.

## 코드 작업 시 주의

- **체크포인트 경로**: `--vae_ckpt` / `--pretrained_ckpt`가 미지정이면 경고만 내고 **random weight로
  학습이 진행됩니다** (`LDM_dp_finetune.py:510`, `:542`). 학습 실행 전 항상 확인하십시오.
- **`Eval_metric/fid.py`는 전체가 주석 처리되어 동작하지 않습니다.** 현재 사용 가능한 LDM 품질 지표는
  `val_loss` 하나뿐입니다.
- **SSIM/PSNR은 재구성 지표**입니다. diffusion 생성물에 그대로 적용하지 마십시오. VAE 단계 전용입니다.
- **`physical_batch`는 품질 파라미터가 아닙니다.** OOM 회피용 자원 파라미터이며 gradient accumulation으로
  `logical_batch`가 보존되므로 결과 비교에 영향을 주지 않습니다.
- 다수 소스 파일이 **cp949로 저장되어 주석이 깨져 있습니다**(`¡æ` 등). 인코딩 정리는 별도 커밋으로
  분리하십시오 (diff가 커집니다).
- `.gitignore`가 없습니다. 체크포인트(`*.pt`)를 커밋하지 마십시오.

## 실행 예시

```bash
# DP fine-tune
python LDM_dp_finetune.py \
  --root_path /storage/hjchoi/mimic/split \
  --vae_ckpt        <required> \
  --pretrained_ckpt <required> \
  --target_epsilon 10.0 --max_grad_norm 1.0 \
  --logical_batch 256 --physical_batch 8 \
  --epochs 100 --lr 1e-4
```
