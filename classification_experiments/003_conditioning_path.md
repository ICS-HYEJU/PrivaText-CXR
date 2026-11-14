# 003_conditioning_path

## 1. 실험 주제
Padding mask 전달 및 frozen BioBERT eval 고정의 생성 병변 분리력 영향.

## 2. 실험 선정 근거
002에서 두 기존 checkpoint가 명시적 병변 조건에도 AUROC 약 0.50을 보였다. 실제 UNet의 padding 출력 차이를 mask가 제거하는 검증도 통과했다.

## 3. 실험 방식
A baseline 대 B corrected. 공통 NIH pretrained 시작, MIMIC train 전체 49,907장, 3 epochs, seed 42, target epsilon 10/delta 1e-5, rank/alpha 4/4, logical/physical batch 256/4, lr .002, warmup 100. A는 기존 BERT train-mode 및 mask 없음, B는 frozen BERT eval + UNet mask 전달. projection은 양쪽 동결. float32 VAE mean/logvar 캐시 사용, posterior sampling/BERT는 online. DP noise 및 latent/timestep RNG는 BERT dropout 소비와 분리해 고정한다. 독립 validation 생성 평가를 사용하며 과거 30-epoch 수치와 직접 효과 비교하지 않는다. 실행 명령은 run_ablation.sh baseline 0 / corrected 1.

평가 프로토콜(결과 확인 전 고정): validation 802장 전체, index별 동일 seed, 0~999 timestep 전체 범위를 균등 선택한 50-step DDIM eta=0. 단순 50-step DDPM truncation이 아니다. XRV/MedCLIP explicit-label AUROC와 patient bootstrap. 이 screening은 과거 full DDPM과 직접 수치 비교하지 않으며 최종 후보는 full DDPM으로 확인한다.

## 4. 실험 결과
진행 중. 시작 전 trained UNet padding invariance 및 synthetic-input Opacus update 검증 통과. 학습과 생성 평가 완료 후 실제 수치를 기록한다.

## 5. 실험 결과 분석
미완료. 3-epoch 단일 seed의 full-cohort screening이며 장기 학습/seed 일반화 결론은 내리지 않는다. 성능 개선 여부 미확정.
