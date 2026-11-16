# 001_vae_reconstruction

## 1. 실험 주제
VAE 재구성이 병변 분류 신호를 보존하는가?

## 2. 실험 선정 근거
현재 latent 1×16×16의 병변 정보 손실과 diffusion 조건 학습 실패를 분리한다.

## 3. 실험 방식
validation 전체 802장. 원본/결정론적 posterior.mode/seed=20260908 posterior.sample 비교. 동일 256 PNG→224 XRV 전처리. XRV-all, MedCLIP bare-label cosine의 multilabel AUROC/AUPRC. uncertain/missing 제외, 양성·음성 각각 10개 이상인 동일 label 집합의 macro. 기존 single-abnormality CLIP metric과 구별. 환자 단위 paired bootstrap 500회. 실행: conda diagnostics.py reconstruct; medclip_env diagnostics.py eval-recon. 모든 실행은 CXR_dp_medclip 내부.

## 4. 실험 결과
| 입력 | XRV AUROC | MedCLIP AUROC |
|---|---:|---:|
| real | 0.7765 | 0.5706 |
| mode | 0.7396 | 0.5037 |
| sample | 0.7418 | 0.5050 |

상세 수치·support·paired CI: `artifacts/001_vae/results.json`. 원본/재구성 및 raw score CSV 보존.

## 5. 실험 결과 분석
XRV mode−real ΔAUROC=-0.0369. 이 차이는 재구성 전처리와 VAE를 통과할 때의 병변 분리력 변화를 나타내며 생성 모델의 성능 상한을 엄밀히 증명하지는 않는다. 관측된 재구성 손실만으로 기존 합성 AUROC 약 0.50을 설명할 수 있는지 조건 반응 실험과 함께 판단한다. validation 결과이며 최종 p12 test와 혼합하지 않는다. 다음 별도 실험은 동일 난수의 병변 조건 반응이다.


XRV mode−real paired 95% CI=[−0.0562, −0.0105], MedCLIP=[−0.1061, −0.0301]. XRV는 재구성에서도 0.7396을 유지하지만 MedCLIP은 0.5037로 저하한다. 따라서 VAE 손실은 존재하며 encoder에 따라 민감도가 다르다. 동일 label-set validation 결과로 해석하고 두 metric을 하나로 합치지 않는다. 외부 VAE와 pretrained/DP checkpoint 내장 VAE의 모든 state tensor 동일성도 확인했다.
