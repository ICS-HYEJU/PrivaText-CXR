# PrivaText-CXR : DP-LDM Framework 구현 지시서

> MIMIC-CXR(private) / NIH(public) 기반 Differentially-Private Latent Diffusion
> Model 프레임워크의 설계 결정, 근거, 구현 로드맵. 다른 세션에서도 이 문서를 근거로
> 작업을 이어갈 수 있도록 작성됨.
>
> 대상 브랜치: `claude/mimic-cxr-dataset-6oMiO` (Option 2 데이터 로딩 + model-parallel 계열)

---

## 0. 전체 개요

```
NIH (public, 질병명 label)  ──pretrain(비-DP)──►  pretrained LDM (weight 보유)
                                                        │
MIMIC (private, report)     ──DP-SGD finetune──────────┘──►  DP model
                                                        │
                                                        └──►  decoder 연결 → 이미지 생성
```

- **최종 목표**: budget(ε)별 LoRA weight를 로드해 description-conditioned CXR 생성.
- **conditioning encoder**: `Modules/BioBERT_embedder.py`의 `BioBERTEmbedder`.

---

## 1. 핵심 설계 결정 (확정)

### 1.1 Conditioning encoder
- 이 브랜치에서 실제 사용하는 인코더는 `Modules/BioBERT_embedder.py`의 **`BioBERTEmbedder`**.
  - `forward(texts)` → `proj(last_hidden_state)` = `[B, seq_len, 512]` (per-token full sequence).
  - **mode 인자 없음.** 긴 report도 full-sequence로 처리하므로 별도 처리 불필요.
- NIH(public)은 짧은 질병명 label, MIMIC(private)은 긴 report지만, 동일 `BioBERTEmbedder`가
  둘 다 tokenize→proj로 처리 → `[B, seq_len, 512]`. UNet cross-attention 구조 동일.

> 참고: `Model/Diffusion/context_encoder.py`의 `BioBERTContextEncoder`(별도 클래스)는
> `mode='label'/'description'`을 갖고 기본값이 `'label'`이라 report에 쓰면 32토큰으로
> 잘리는 문제가 있음. **이 브랜치의 학습/추론 경로는 `BioBERTEmbedder`를 쓰므로 해당
> 문제 없음.** 혼동 주의.

### 1.2 DP 입도(granularity): 이미지 단위(sample-level)
- **이미지 단위 DP** = 이미지 1장 보호. Opacus 기본, DP-LDM 표준.
- **환자 단위 DP** = 환자 전체 보호. **더 강한 보호**지만 노이즈↑ → utility↓, 구현 복잡.
- **결정**: 이미지 단위 사용. 이유는 "표준·utility 우수"이지 "보호가 더 강해서"가 **아님**.
  논문에서 프라이버시 강도 서술 시 이 구분을 혼동하지 말 것.

### 1.3 학습 loss
- `Model/Diffusion/LDM.py:p_losses` — diffusion noise-prediction MSE(`loss_simple`) + VLB만 사용.
- **별도 text-image alignment loss 없음** (CLIP/contrastive 아님).
- 근거: conditioning `c`가 loss의 **입력 조건**이므로, noise 예측 loss 최소화가 곧
  cross-attention의 조건 정렬을 유도(**implicit conditioning**). DP-LDM 레퍼런스와 동일.

---

## 2. 데이터 분할 (privacy budget 격리)

### 2.1 원칙: parallel composition
- 같은 데이터로 여러 번 DP 학습하면 ε가 **합산**(재학습·하이퍼파라미터 재탐색 포함,
  weight를 초기화해도 동일 — 회계는 "데이터를 몇 번 봤는가" 기준).
- **disjoint 데이터**로 나누면 한 이미지는 한 쪽에만 속하므로 parallel composition에 의해
  privacy loss가 **합산되지 않고 각자 max로 제한**.

### 2.2 분할 구성
```
D_search : 하이퍼파라미터 탐색 전용 (여러 번 실험, ε는 여기서만 소진)
D_train  : 최종 DP 학습 전용 (1회, 최종 ε 보고 대상)
D_test   : 평가 — 공식 split CSV의 'test'/'validate' 사용 (train과 이미 disjoint)
```
- D_search / D_train은 **train split 환자를 patient(pXXXXXXXX) 단위로 disjoint 분할**.
  환자 disjoint → 이미지 disjoint 자동 보장 → 이미지 단위 DP에서 안전.

### 2.3 D_search = p10 폴더의 일부만 사용 (확정)
- MIMIC subject_id 앞자리로 prefix(p10~p19)가 갈리며, 한 환자는 한 prefix에만 속함.
- `Data/make_dp_splits.py`가 **split CSV**(`mimic-cxr-2.0.0-split.csv`)를 읽어 train 환자를
  분할. `--search_prefixes p10 --search_count K` → p10의 앞 K명만 D_search, 나머지 p10 + 그
  외 prefix → D_train.
- 산출 manifest key = `patient_id`(예: `"p10000032"`) → `MIMICCXRDataset(patient_whitelist=...)`
  로 그대로 필터.

---

## 3. 구현 로드맵

| 단계 | 내용 | LoRA | ε 소비 | 상태 |
|------|------|:---:|:---:|------|
| S0 | pretrain (NIH) | — | X | **완료 (weight 보유)** |
| S2 | 데이터 분할 유틸 (`make_dp_splits.py` + `patient_whitelist`) | — | X | **완료** |
| S3 | **end-to-end inference (기존 DP ckpt로 검증)** | ✗ | X | **완료 (실행 검증 대기)** |
| S4 | LoRA 주입 구현 (`Model/lora.py`) | — | X | TODO |
| S5 | D_search에서 DP-LoRA 하이퍼파라미터 탐색 | ✓ | D_search | TODO |
| S6 | D_train에서 최종 DP-LoRA 1회 | ✓ | D_train | TODO |
| S7 | budget별 LoRA 로드 inference framework | ✓ | X | TODO |

> **진행 순서(사용자 결정)**: S3(end-to-end 검증)를 먼저 완성해 결과물 확인 후 LoRA(S4~S7)로 확장.

---

## 4. S3 : End-to-End Inference (구현 완료)

`LDM_dp_inference.py` (프로젝트 루트). 기존 full-attention DP 체크포인트 재사용(LoRA 아님).

```
description(str)
  └─ BioBERTEmbedder → context [B, seq_len, 512]
       └─ LatentDiffusionDP.sample(c)  (p_sample_loop, DDPM T steps)
            └─ latent z → decode_first_stage(z) → VAE.decode(z/scale_factor)
                 └─ image [B,1,256,256]
                      └─ grid_all.png + samples/*.png + descriptions.csv
```

실행:
```bash
python LDM_dp_inference.py \
    --dp_ckpt ./finetune_dp/ldm_dp_final.pt \
    --vae_ckpt ./checkpoints/vae/vae_ep0020.pt \
    --biobert_path /storage/hjchoi \
    --descriptions ./prompts.txt --n_samples 2 --output_dir ./generated
```
- 모델 args(`--vae_*`, `--unet_*`)는 **학습 때 값과 동일**해야 함(기본값은 `LDM_dp_finetune.py`와 일치).

---

## 5. S4~S7 : LoRA budget-swap framework

### 5.1 개념
- cross-attention linear(`to_q/to_k/to_v/to_out`)에 저랭크 `ΔW=(α/r)·B·A`만 주입,
  **base frozen, LoRA만 DP-SGD 학습**.
- 장점: 학습 파라미터↓ → 클리핑/노이즈가 적은 차원에 집중 → DP privacy-utility 개선.
- budget마다 작은 LoRA 파일 저장 → inference 시 base + LoRA(budget) 로드/merge.

### 5.2 기존 DP 체크포인트와의 관계 (중요)
- 이미 수행한 #1 DP finetune은 **full-attention 방식**(LoRA 아님).
- **full-attention ckpt는 LoRA 파일로 사후 변환 불가**(파라미터화가 다름).
- LoRA framework는 **pretrained에서 새 DP-LoRA 학습** 필요(새 ε 소비 → §2 분할 적용).
- 기존 ckpt는 S3(end-to-end 검증) 용도로만 재사용.

### 5.3 구현 지점 (이 브랜치 경로 기준)
```
Model/lora.py                      : LoRALinear(W0 frozen, A/B 학습, scale α/r)
Model/attention_module_dp.py       : CrossAttention.to_q/k/v/out → LoRALinear 래핑
Model/Diffusion/LDM_dp.py          : configure_dp_params → LoRA 파라미터만 반환
LDM_dp_finetune.py                 : make_private가 LoRA 파라미터에만 hook
                                     ε 마일스톤(1,3,5,10 등)마다 LoRA-only state_dict 저장
LDM_dp_inference.py                : --budget → 해당 LoRA 로드 → base와 merge → sample
```
- Opacus 호환: LoRA는 표준 `nn.Linear` 조합 → GradSampleModule per-sample grad 정상.

---

## 6. 코드베이스 참조 (이 브랜치 실제 경로)

| 대상 | 위치 | 비고 |
|------|------|------|
| MIMIC 로더 | `Data/mimic_cxr.py` `MIMICCXRDataset` | 원본 dir + split CSV 직접 읽기, `patient_id="p10000032"`, `patient_whitelist` 지원 |
| DP split 생성 | `Data/make_dp_splits.py` | CSV 기반, p10 일부→search 지원 |
| BioBERT 인코더 | `Modules/BioBERT_embedder.py` `BioBERTEmbedder` | `forward(texts)`→`[B,seq,512]`, mode 없음, `.proj`(768→512) |
| DP LDM | `Model/Diffusion/LDM_dp.py` `LatentDiffusionDP` | `configure_dp_params`(type().__name__ 사용), `get_input_dp`, `device` 인자 |
| DP finetune | `LDM_dp_finetune.py` (루트) | Opacus + BatchMemoryManager(Poisson), model-parallel |
| LDM/sampling | `Model/Diffusion/LDM.py` | `sample()`, `decode_first_stage()` 완비 |
| VAE | `Model/VAE/Autoencoder.py` (+ Encoder/Decoder) | `VAE(cfg)`, `decode(z)` |
| UNet | `Model/Diffusion/UNetmodel.py` | `UNetModel` (파일명 소문자 m) |
| **end-to-end inference** | `LDM_dp_inference.py` (루트) | **신규 (S3)** |
| LoRA | **없음** | 신규 구현 필요 (S4~) |

---

## 7. 미결 결정 사항

1. D_search 환자 수 K (p10 중 몇 명), base_split 비율.
2. ε 마일스톤 값 집합 (예: {1, 3, 5, 10}) — budget별 LoRA 저장 지점.
3. LoRA rank `r`, scaling `α` 기본값.
4. NIH 라벨 pseudo-report 변환 적용 여부 (선택, conditioning 격차 축소용).

---

_대상 브랜치: `claude/mimic-cxr-dataset-6oMiO`_
