# PrivaText-CXR : DP-LDM Framework 구현 지시서

> 이 문서는 MIMIC-CXR(private) / NIH(public) 기반 Differentially-Private Latent
> Diffusion Model 프레임워크의 설계 결정, 근거, 구현 로드맵을 정리한 것이다.
> 다른 세션(채팅)에서도 이 문서를 근거로 작업을 이어갈 수 있도록 작성되었다.

---

## 0. 전체 개요

```
NIH (public, 질병명 label)  ──pretrain(비-DP)──►  pretrained LDM (weight 보유)
                                                        │
MIMIC (private, report)     ──DP-SGD finetune──────────┘──►  DP model
                                                        │
                                                        └──►  decoder 연결 → 이미지 생성
```

- **최종 목표**: budget(ε)별 LoRA weight를 로드해 description-conditioned CXR 이미지를
  생성하는 프레임워크.
- **conditioning encoder**: `Diffusion/context_encoder.py`의 `BioBERTContextEncoder`
  (단일 공유 인코더, 두 모드 지원).

---

## 1. 핵심 설계 결정 (확정)

### 1.1 Conditioning encoder 통일
- NIH(public): `encoder(labels, mode='label')`  → CLS token → `[B,1,512]`
- MIMIC(private): `encoder(reports, mode='description')` → mean-pool → `[B,1,512]`
- 두 출력 shape가 동일(`[B,1,512]`)하므로 **UNet 구조 변경 불필요**. pretrain 지식이
  finetune으로 전이됨.

> **버그 주의**: `BioBERTContextEncoder.forward(texts, mode='label')`의 기본값이
> `'label'`이다. MIMIC report를 `mode` 없이 호출하면 `max_length=32`로 잘리고 CLS만
> 사용되어 report 정보가 대량 손실된다. **MIMIC은 반드시 `mode='description'`** 으로
> 호출해야 한다. (`Data/mimic_cxr.py`의 `collate_fn`, `LDM_dp.py`의 `get_input_dp`)

### 1.2 DP 입도(granularity): 이미지 단위(sample-level)
- **이미지 단위 DP** = 이미지 1장의 존재/부재를 보호. Opacus 기본, DP-LDM 표준.
- **환자 단위 DP** = 환자 전체를 보호. **더 강한 보호**지만 노이즈↑ → utility↓, 구현 복잡.
- **결정**: 이미지 단위 사용. 이유는 "표준이고 utility가 좋음" (강도가 더 높아서가 **아님**).
- 논문에서 프라이버시 강도를 서술할 때 이 구분을 혼동하지 말 것.

### 1.3 학습 loss
- `LDM.py:p_losses` — diffusion noise-prediction MSE (`loss_simple`) + VLB 항만 사용.
- **별도 text-image alignment loss 없음** (CLIP/contrastive 아님).
- 근거: conditioning `c`가 loss의 **입력 조건**으로 들어가므로, noise 예측 loss 최소화가
  곧 cross-attention의 조건 정렬을 유도(**implicit conditioning**). DP-LDM 레퍼런스와 동일.

---

## 2. 데이터 분할 (privacy budget 격리)

### 2.1 원칙: parallel composition
- 같은 데이터로 여러 번 DP 학습하면 ε가 **합산**된다(재학습, 하이퍼파라미터 재탐색 포함,
  weight를 초기화해도 동일 — 회계는 "데이터를 몇 번 봤는가" 기준).
- **disjoint 데이터**로 나누면, 한 이미지는 한 쪽에만 속하므로 parallel composition에 의해
  privacy loss가 **합산되지 않고 각자 max로 제한**된다.

### 2.2 분할 구성
```
D_search : 하이퍼파라미터 탐색 전용 (여러 번 실험, ε는 여기서만 소진)
D_train  : 최종 DP 학습 전용 (1회, 최종 ε 보고 대상)
D_test   : 평가 (gradient 없음, ε 소비 없음)
```
- 셋은 **환자(pXXXXXXXX) 단위로 disjoint**. 환자 disjoint면 이미지도 자동 disjoint이므로
  이미지 단위 DP에서 안전.

### 2.3 D_search = p10 폴더의 일부만 사용 (확정 가능)
- MIMIC의 `p10~p19`는 환자 ID 앞자리 버킷. 한 환자는 정확히 한 prefix에만 속함.
- p10 내부 **환자 폴더 목록을 잘라** 일부를 D_search로, 나머지를 D_train으로 배정.
```
p10/ ├── (앞쪽 K명)  → D_search
     └── (나머지)    → D_train  ┐
p11 ~ p18/                      ├─ D_train
p19/                            ┘ → D_test  (예시, 비율 조정 가능)
```
- 구현: `Data/make_dp_splits.py`가 환자→split 배정 manifest(JSON) 생성.
  `MIMICCXRDataset`가 `patient_whitelist`로 필터.

---

## 3. 구현 로드맵

| 단계 | 내용 | LoRA | ε 소비 | 상태 |
|------|------|:---:|:---:|------|
| S0 | pretrain (NIH) | — | X | **완료 (weight 보유)** |
| S1 | encoder 통일 + `mode='description'` 수정 | — | X | TODO |
| S2 | 데이터 분할 유틸 (`make_dp_splits.py` + whitelist) | — | X | TODO |
| S3 | **end-to-end inference 파이프라인 (기존 DP ckpt로 검증)** | ✗ | X | **우선** |
| S4 | LoRA 주입 구현 (`Model/lora.py`) | — | X | TODO |
| S5 | D_search에서 DP-LoRA 하이퍼파라미터 탐색 | ✓ | D_search | TODO |
| S6 | D_train에서 최종 DP-LoRA 1회 | ✓ | D_train | TODO |
| S7 | budget별 LoRA 로드 inference framework | ✓ | X | TODO |

> **진행 순서(사용자 결정)**: S3(end-to-end 검증)를 먼저 완성해 결과물을 확인한 뒤,
> LoRA framework(S4~S7)로 재사용/확장한다.

---

## 4. S3 : End-to-End Inference 파이프라인 (기존 full-attention DP ckpt)

목적: decoder까지 연결된 생성 루프가 실제 동작하는지 검증. **LoRA 아님.**

```
description(str)
  └─ BioBERTContextEncoder(mode='description') → context [B,1,512]
       └─ LatentDiffusionDP.sample(c)  (p_sample_loop, DDPM T steps)
            └─ latent z [B,1,16,16]
                 └─ decode_first_stage(z) → VAE.decode(z/scale_factor)
                      └─ image [B,1,256,256]
                           └─ 이미지 + description 저장 (grid PNG + CSV)
```

- 진입점: `LDM_dp_inference.py` (신규)
- 입력: `--dp_ckpt`, `--descriptions`(txt 파일/직접입력), `--n_samples`, `--output_dir`
- 저장:
  - `grid_all.png` — description을 캡션으로 단 이미지 grid
  - `samples/{idx}_sample{k}.png` — 개별 이미지
  - `descriptions.csv` — idx, description, 파일명 매핑

---

## 5. S4~S7 : LoRA budget-swap framework

### 5.1 LoRA 개념
- cross-attention linear(`to_q/to_k/to_v/to_out`)에 저랭크 어댑터
  `ΔW = (α/r)·B·A`만 주입, **base는 frozen, LoRA만 DP-SGD 학습**.
- 장점: 학습 파라미터↓ → 클리핑/노이즈가 적은 차원에 집중 → DP privacy-utility 개선.
- budget별로 작은 LoRA 파일 저장 → inference 시 base + LoRA(budget) 로드/merge.

### 5.2 기존 DP 체크포인트와의 관계 (중요)
- 이미 수행한 #1 DP finetune은 **full-attention 방식**(LoRA 아님).
- **full-attention ckpt는 LoRA 파일로 사후 변환 불가** (파라미터화가 다름).
- 따라서 LoRA framework는 **pretrained에서 새 DP-LoRA 학습**이 필요(새 ε 소비 →
  §2의 D_search/D_train 분할 적용).
- 기존 ckpt는 S3(end-to-end 검증) 용도로만 재사용.

### 5.3 구현 지점
```
Model/lora.py           : LoRALinear(W0 frozen, A/B 학습, scale α/r)
attention_module_dp.py  : CrossAttention.to_q/k/v/out → LoRALinear 래핑 patch
LDM_dp.py               : configure_dp_params → LoRA 파라미터만 반환
LDM_dp_finetune.py      : make_private가 LoRA 파라미터에만 hook
                          ε 마일스톤(1,3,5,10 등)마다 LoRA-only state_dict 저장
LDM_dp_inference.py     : --budget → 해당 LoRA 파일 로드 → base와 merge → sample
```
- Opacus 호환: LoRA는 표준 `nn.Linear` 조합 → GradSampleModule per-sample grad 정상.

---

## 6. 코드베이스 참조 (실제 경로/사실)

| 대상 | 실제 위치 | 비고 |
|------|-----------|------|
| MIMIC 로더 | `Data/mimic_cxr.py` `MIMICCXRDataset` | pre-split dir, `p10~p19`, sample=(image, report) |
| NIH 로더 | `Data/dataset.py` `NIH` | description = 질병명 label (`"Pneumonia\|Effusion"`) |
| NIH 라벨 임베더 | `Data/class_label.py` `ClassLabelEmbedder` | vocab 18토큰 |
| BioBERT 인코더 | `Diffusion/context_encoder.py` `BioBERTContextEncoder` | `mode='label'`/`'description'`, `.proj`(768→512) |
| DP LDM | `Diffusion/LDM_dp.py` `LatentDiffusionDP` | `configure_dp_params`, `get_input_dp` |
| DP finetune | `Diffusion/LDM_dp_finetune.py` | Opacus make_private, VirtualBatch |
| LDM/sampling | `Diffusion/LDM.py` | `sample()`, `decode_first_stage()` 완비 |
| VAE decode | `Model/autoencoder.py` `AutoencoderKL.decode` + `Model/decoder.py` | |
| LoRA | **없음** | 신규 구현 필요 |

> 과거 문서에 등장한 `Modules/BioBERT_embedder.py`, `Data/nih.py`,
> `Data/split_dataset.py`, `Model/VAE/` 는 **현재 브랜치에 존재하지 않음**.
> 실제 경로는 위 표 기준.

---

## 7. 미결 결정 사항

1. D_search / D_train / D_test 비율 및 p10 내 D_search 환자 수(K).
2. ε 마일스톤 값 집합 (예: {1, 3, 5, 10}) — budget별 LoRA 저장 지점.
3. LoRA rank `r`, scaling `α` 기본값.
4. NIH 라벨 pseudo-report 변환 적용 여부 (선택, conditioning 격차 축소용).

---

_생성 브랜치: `claude/build-encoder-module-3Qc7S`_
