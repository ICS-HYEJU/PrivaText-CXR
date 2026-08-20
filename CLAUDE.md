# CLAUDE.md

이 저장소는 MIMIC-CXR 기반 differentially-private latent diffusion 흉부 X-ray 생성
파이프라인(학습 → 생성 → 평가 → 리포트)이며, 여러 세션에 걸쳐 이어지는 장기 작업을
전제로 설계되었습니다. 목표는 코드량을 늘리는 것이 아니라, 다음 세션이 추측 없이
이어받을 수 있는 상태로 저장소를 남기는 것입니다.

## 실행 환경 (필수 전제 — 이것부터 확인하지 않으면 아무 것도 진행 불가)

- 모든 학습/생성/평가는 Docker 컨테이너 **`CXR_dp_medclip`** 안에서 실행됩니다.
  `docker inspect -f '{{.State.Running}}' CXR_dp_medclip`
- 컨테이너 안에 Python 환경이 **두 개**이며 절대 섞어쓰면 안 됩니다:
  - `/opt/conda/bin/python` — 학습/생성용 (Opacus 포함, MedCLIP 불가)
  - `/opt/medclip_env/bin/python` — 평가용 (MedCLIP, transformers 4.24.0)
- 프로젝트 경로: 호스트 `/home/hjchoi/PycharmProjects/PrivaText-CXR`
  == 컨테이너 `/workspace/PrivaText-CXR` (bind mount, 완전 동일)
- 평가 스크립트(`Eval_metric/run_feature_eval.py` 등) 실행 시 반드시
  `PYTHONPATH=/workspace/PrivaText-CXR/src`를 설정합니다 — top-level `Eval_metric/`와
  `src/Eval_metric/`가 (둘 다 `__init__.py`가 없는) namespace package로 병합되어 동작하는
  구조라, 이 PYTHONPATH가 없으면 `clip_label` 등 top-level 전용 지표가 임포트되지 않습니다.
- JPG custom split은 `docs/MIMIC_JPG_PIPELINE.md`를 먼저 읽습니다. report/metadata와
  JPG가 서로 다른 루트에 있으며, p10/p11 official train+validate를 학습/검증에 사용하고
  p10/p11 official test를 제외하며 p12 전체를 patient-disjoint custom test로 사용합니다.
  JPG 경로는 `--manifest_csv`를 넘긴 경우에만 활성화되어 기존 DICOM 실행과 공존합니다.

## 시작 워크플로우

1. `pwd` 확인
2. `claude-progress.md` 읽기 — 최근 검증 상태와 알려진 함정 확인
3. `feature_list.json` 읽기 — 최우선 미완료 항목 선택
4. `git log --oneline -5`, `git worktree list`, `git branch -a` 확인 — 다른
   worktree/브랜치에서 진행 중인 작업과 충돌하지 않는지 확인
5. `./init.sh` 실행 (컨테이너/경로/compileall + 1장짜리 생성 스모크)
6. 기준선이 이미 깨져 있으면, 새 기능 작업 전에 그것부터 고칩니다

## 작업 규칙

- **한 번에 기능(feature) 하나.** 단, 이 저장소는 여러 git worktree가 동시에 존재할 수
  있으므로(`.claude/worktrees/...`), 이 규칙은 **worktree(브랜치) 단위**로 적용합니다.
  `claude-progress.md`에 "이 worktree가 소유한 기능"을 명시해 충돌을 방지하세요.
- 코드가 추가됐다고 기능을 완료(`passing`)로 표시하지 않습니다. `feature_list.json`의
  `verification` 커맨드가 실제로 통과하고, `evidence`(산출물 경로)가 기록되어야 합니다.
- **LoRA 체크포인트 로딩 함정**: 체크포인트가 LoRA 파인튜닝 결과물(`args.use_lora=True`)이면
  생성 시 반드시 `--lora_ckpt`를 함께 지정합니다 (`--dp_ckpt`와 같은 파일이라도). 이것 없이
  `--dp_ckpt`만 넘기면 `load_state_dict(strict=False)`가 LoRA 델타 텐서를 조용히 버리고
  pretrained base로 생성해버립니다. **서로 다른 체크포인트에서 생성한 이미지의 md5가
  동일하면 바로 이 버그를 의심하세요.**
- `run_feature_eval.py`를 같은 `--out_dir`에 대해 두 프로세스로 동시 실행하지 않습니다
  (`eval_summary.json`이 read-merge-write 방식이라 경쟁 조건 발생).
- CLIP 계열 지표(`clip`, `clip_label`, `clip_diagnose`)는 `--device cuda:0`으로 고정합니다
  (MedCLIP이 내부적으로 cuda:0을 하드코딩하는 것으로 보이며, 다른 device를 주면
  "Expected all tensors to be on the same device" 에러로 조용히 skip됩니다).
- 구현 중에 검증 규칙을 조용히 바꾸지 않습니다. 채팅 요약보다 저장소에 남는 산출물을
  우선합니다.

## 필수 산출물

- `feature_list.json` — 파이프라인 능력(pipeline-*) + 체크포인트별 평가 커버리지(eval-*)의
  단일 진실 원천(single source of truth)
- `claude-progress.md` — 세션 로그 + 현재 검증 상태 + 알려진 함정
- `init.sh` — 표준 빠른 스모크 검증 (풀 파이프라인 검증은 `feature_list.json`의 개별
  `verification` 커맨드로 별도 실행 — 생성+평가는 체크포인트당 1시간 이상 걸리므로 매
  세션 자동 실행 대상이 아닙니다)
- `session-handoff.md` — 다음 세션이 바로 이어받을 수 있는 간결한 핸드오프 (선택, 큰 세션
  마무리 시 작성)

## 완료 정의 (Definition of Done)

다음이 모두 충족되어야 기능이 완료됩니다.
- 목표 동작이 구현됨
- 필요한 검증(생성 커맨드 + 평가 커맨드)이 실제로 실행됨
- 증거가 `feature_list.json` 또는 `claude-progress.md`에 기록됨
  (예: `EVAL/metric/<name>/eval_summary.json` 경로, 주요 수치)
- 저장소가 `./init.sh`로 재시작 가능한 상태를 유지함

## 세션 종료 전

1. `claude-progress.md` 업데이트
2. `feature_list.json` 업데이트
3. 미해결 위험/차단 항목 기록
4. 안전한 상태에서 설명적인 메시지로 커밋
5. 다음 세션이 즉시 `./init.sh`를 실행할 수 있을 만큼 저장소를 정리
