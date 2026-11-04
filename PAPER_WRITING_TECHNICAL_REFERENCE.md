# PrivaText-CXR 논문 작성용 기술 명세

최종 갱신: 2026-11-03  
대상 논문 구성: Introduction → Preliminary → Method → Implementation and Performance Evaluation  
근거: 저장소 구현, 저장된 VAE/LDM 체크포인트 메타데이터, 공식 split manifest 및 검증 문서

## 문서의 목적과 범위

이 문서는 PrivaText-CXR 논문을 작성할 때 필요한 **연구 논리, 수식, 네트워크 구조,
데이터 경계, 사전학습 및 DP 미세조정 절차, 구현 설정, 평가 방법**을 한 파일에서
확인할 수 있도록 정리한 기술 명세다. 개별 실험 번호의 진행 일지나 체크포인트별
성능표를 재현하는 문서가 아니다.

2026-11-03 현재 p10--p15 공식 split을 이용한 RadBERT/E027, 30-epoch,
epsilon 1/10 학습이 진행 중이므로 다음 내용은 의도적으로 포함하지 않는다.

- 학습 중인 모델의 최종 privacy expenditure
- 학습 중인 모델의 최종 loss와 수렴 판단
- 생성 영상에 대한 FID, FDS, SSIM, PSNR, LPIPS
- XRV 또는 MedCLIP 성능
- downstream classification utility
- 모델 간 우열 또는 통계적 유의성에 관한 결론
- 최종 Results 절의 수치와 해석

반면, 현재 결과 없이도 확정할 수 있는 학습 프로토콜과 평가 정의는 포함한다.
데이터 수는 학습 성능이 아니라 저장된 manifest를 감사하여 얻은 코호트 기술값이다.

이 문서의 서술 상태는 다음 세 가지로 구분한다.

- **구현 확인**: 현재 코드 또는 체크포인트에서 직접 확인한 사실
- **프로토콜 확정**: 최종 학습·평가에 사용하도록 저장소 규칙으로 정한 사항
- **결과 후 입력**: 학습·생성·평가 완료 후에만 채울 수 있는 사항

---

# I. Introduction

## 1. 연구 배경

흉부 방사선영상은 폐렴, 기흉, 흉수, 심비대 등 다양한 흉부 이상을 평가하는 데
널리 사용된다. 대규모 흉부 X-ray와 판독문 데이터는 생성 모델과 임상 의사결정
보조 모델을 개발하는 데 유용하지만, 의료영상은 환자 수준의 민감정보이며 자유 형식
판독문과 결합될 때 데이터 공유와 재사용에 추가적인 제약이 생긴다. MIMIC-CXR은
비식별화된 흉부 방사선영상과 자유 형식 판독문을 제공하지만, credentialing과 데이터
사용 계약을 요구하는 controlled-access 자원이다. 따라서 이 논문에서 MIMIC-CXR을
"private dataset"이라고 부를 때는 원자료가 비공개라는 뜻이 아니라, **DP 보호가
필요한 미세조정 도메인**이라는 의미로 한정해야 한다.

합성 의료영상은 실제 환자 영상을 직접 배포하지 않고 연구용 데이터를 제공할 가능성이
있다. 그러나 생성 영상이 시각적으로 그럴듯하다는 사실만으로 안전하거나 유용하다고
말할 수는 없다. 생성 모델이 학습 표본에 과도하게 의존할 수 있고, 판독문에 명시된
병변이 생성 영상에 반영되지 않을 수 있으며, 높은 분포 유사도가 downstream classifier
학습의 향상을 보장하지도 않는다. 그러므로 privacy, distributional fidelity, clinical
consistency, semantic alignment, downstream utility를 서로 다른 평가 축으로 다뤄야 한다.

## 2. 문제 정의

공개 NIH ChestX-ray14 데이터에서 영상 표현과 class-conditioned latent diffusion prior를
학습한 뒤, MIMIC-CXR의 영상--판독문 쌍을 이용하여 text-conditioned generator로
미세조정한다. MIMIC-CXR 단계에는 DP-SGD를 적용하고, 전체 모델을 갱신하는 대신
text projection과 UNet cross-attention의 LoRA adapter만 학습한다. 목표는 다음 조건을
동시에 만족하는 생성 모델을 구축하는 것이다.

1. 실제 흉부 X-ray의 전반적 분포와 구조를 재현한다.
2. 입력 판독문의 임상 의미가 생성 영상에 반영된다.
3. 학습 과정이 명시적인 \((\epsilon,\delta)\)-DP 보장을 제공한다.
4. 생성 영상이 임상 라벨 평가와 downstream 학습에서 측정 가능한 정보를 보존한다.

## 3. 연구 질문

논문에서는 다음 연구 질문을 중심으로 결과를 구성한다.

- **RQ1 -- Privacy-preserving generation:** 공개 데이터 사전학습 후 제한된 파라미터만
  DP 미세조정하여 흉부 X-ray를 생성할 수 있는가?
- **RQ2 -- Privacy--utility trade-off:** 더 작은 privacy budget이 분포 품질, 임상
  일관성, 의미 정합성에 각각 어떤 영향을 주는가?
- **RQ3 -- Conditionality:** 동일한 latent와 noise에서 판독문 조건만 바꾸었을 때
  생성 또는 denoising 반응이 체계적으로 달라지는가?
- **RQ4 -- Clinical utility:** 합성 영상이 실제 영상에서 평가되는 병변 정보를
  보존하며, 새로운 분류기의 학습 자료로 이용될 수 있는가?
- **RQ5 -- Metric disagreement:** 분포적 품질, 조건 충실도, 병변 분류 성능 및
  downstream utility가 같은 모델 순위를 만드는가?

## 4. 연구 기여로 정리할 항목

현재 구현을 기준으로 논문의 기여는 다음과 같이 기술할 수 있다. 성능 우월성은 평가가
완료되기 전에는 기여 문장에 포함하지 않는다.

1. Swin self-attention을 포함한 VAE와 text-conditioned latent diffusion을 결합하여
   256×256 단일 채널 흉부 X-ray를 1×16×16 latent에서 생성하는 파이프라인을 제시한다.
2. 공개 NIH 데이터에서 VAE와 class-conditioned LDM을 사전학습하고, credentialed
   MIMIC-CXR에서만 DP 미세조정하는 단계적 학습 방식을 사용한다.
3. frozen RadBERT, 새로 초기화한 768→512 projection, rank-4 cross-attention LoRA를
   결합하여 DP 학습 파라미터를 제한한다.
4. 동일 latent, timestep 및 noise에서 정답 판독문과 고정 No-Finding 조건을 비교하는
   per-record condition-contrastive objective를 사용한다.
5. MIMIC-CXR 공식 patient-disjoint split을 보존하고, 분포·임상·의미·downstream
   utility를 분리한 다축 평가 프로토콜을 제시한다.

## 5. Introduction에서 피해야 할 표현

- DP를 적용했다는 사실만으로 생성 영상이 재식별 불가능하다고 단정하지 않는다.
- 낮은 FID를 임상적으로 정확한 생성의 증거로 표현하지 않는다.
- 판독문 조건에 반응한다는 사실을 병변 위치까지 grounding되었다고 확대 해석하지 않는다.
- frozen evaluator의 AUROC를 방사선 전문의 진단 정확도로 표현하지 않는다.
- 한 번의 training seed 또는 validation 점추정치를 일반적 우월성으로 표현하지 않는다.

---

# II. Preliminary

## 1. Variational Autoencoder

입력 흉부 X-ray를 \(x\in\mathbb{R}^{1\times256\times256}\)라 하자. VAE encoder는
대각 Gaussian posterior의 평균과 log-variance를 출력한다.

\[
q_\phi(z\mid x)=
\mathcal{N}\!\left(\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))\right).
\]

미분 가능한 sampling을 위해 reparameterization trick을 사용한다.

\[
z=\mu_\phi(x)+
\exp\!\left(\tfrac{1}{2}\log\sigma_\phi^2(x)\right)\odot\eta,
\qquad \eta\sim\mathcal{N}(0,I).
\]

Decoder \(p_\psi(x\mid z)\)는 latent sample을 영상 \(\hat{x}\)로 복원한다. 본
구현의 latent는 \(z\in\mathbb{R}^{1\times16\times16}\)이며, 입력 대비 각 공간 축이
16배 감소하고 전체 scalar 수는 256배 감소한다.

본 연구의 실제 VAE checkpoint는 단순 ELBO만 사용하지 않고 다음 목적함수를 사용한다.

\[
\mathcal{L}_{\mathrm{VAE}} =
\lambda_{\mathrm{rec}}\mathcal{L}_{1}
+\lambda_{\mathrm{ssim}}\mathcal{L}_{\mathrm{SSIM}}
+\lambda_{\mathrm{perc}}\mathcal{L}_{\mathrm{LPIPS}}
+\lambda_{\mathrm{KL}}\mathcal{L}_{\mathrm{KL}}
+\lambda_{\mathrm{MMD}}\mathcal{L}_{\mathrm{MMD}}.
\]

각 항은 다음과 같다.

\[
\mathcal{L}_{1}=\|x-\hat{x}\|_1,
\qquad
\mathcal{L}_{\mathrm{SSIM}}=1-\operatorname{SSIM}(x,\hat{x}),
\]

\[
\mathcal{L}_{\mathrm{KL}}=
D_{\mathrm{KL}}(q_\phi(z\mid x)\,\|\,\mathcal{N}(0,I)).
\]

MMD 항은 posterior latent와 표준 Gaussian prior 사이의 multi-kernel Gaussian MMD다.
구현은 spatial latent를 평균 pooling하고, posterior latent를 표준화한 뒤 base bandwidth의
\([0.1,0.5,1,2,5,10]\)배 kernel을 평균한다. LPIPS 계산 시 단일 채널 영상을 세 채널로
반복하여 VGG 기반 perceptual network에 입력한다.

## 2. Latent Diffusion Model

VAE가 생성한 clean latent를 \(z_0\)라 하자. Forward diffusion은 \(T=1000\)개의
step에 걸쳐 Gaussian noise를 추가한다.

\[
q(z_t\mid z_{t-1})=
\mathcal{N}(\sqrt{1-\beta_t}\,z_{t-1},\beta_tI),
\]

\[
z_t=\sqrt{\bar{\alpha}_t}z_0+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I),
\]

여기서 \(\alpha_t=1-\beta_t\),
\(\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s\)다. UNet은 noisy latent \(z_t\),
timestep \(t\), condition \(c\)를 입력으로 받아 noise를 예측한다.

\[
\hat{\epsilon}=\epsilon_\theta(z_t,t,c).
\]

기본 학습 목적함수는 epsilon-prediction MSE다.

\[
\mathcal{L}_{\mathrm{diff}}=
\mathbb{E}_{z_0,t,\epsilon,c}
\left[\|\epsilon-epsilon_\theta(z_t,t,c)\|_2^2\right].
\]

Text condition은 SpatialTransformer의 cross-attention으로 전달된다. Latent feature가
query, text sequence가 key와 value를 구성한다.

\[
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}+M\right)V,
\]

여기서 \(M\)은 padding token이 attention에 기여하지 않도록 하는 context mask다.

## 3. Low-Rank Adaptation

LoRA는 pretrained linear weight \(W_0\)를 고정하고 저차원 update만 학습한다.

\[
y=W_0x+\Delta Wx,
\qquad
\Delta W=\frac{\alpha}{r}BA,
\]

\[
A\in\mathbb{R}^{r\times d_{\mathrm{in}}},
\qquad
B\in\mathbb{R}^{d_{\mathrm{out}}\times r}.
\]

본 구현은 \(A\)를 Kaiming uniform으로 초기화하고 \(B\)를 0으로 초기화한다. 따라서
학습 시작 시 \(\Delta W=0\)이고 네트워크 출력은 pretrained base와 같다. 추론 시에는

\[
W_{\mathrm{merged}}=W_0+\frac{\alpha}{r}BA
\]

로 병합하여 일반 linear layer로 실행한다. 본 연구는 \(r=4\), \(\alpha=4\)를 사용하므로
scaling factor \(\alpha/r=1\)이다.

## 4. Differential Privacy

무작위 메커니즘 \(\mathcal{M}\)은 하나의 보호 record가 다른 인접 데이터셋
\(D,D'\)에 대해 모든 measurable output set \(S\)에 다음을 만족하면
\((\epsilon,\delta)\)-differentially private다.

\[
\Pr[\mathcal{M}(D)\in S]
\le e^\epsilon\Pr[\mathcal{M}(D')\in S]+\delta.
\]

DP-SGD는 각 record의 gradient \(g_i\)를 norm \(C\)로 제한한다.

\[
\bar{g}_i=g_i\min\left(1,\frac{C}{\|g_i\|_2}\right).
\]

Clipped gradient 합에 Gaussian noise를 추가해 update한다.

\[
\tilde{g}=\frac{1}{|B|}
\left(\sum_{i\in B}\bar{g}_i+
\mathcal{N}(0,\sigma^2C^2I)\right),
\]

여기서 \(C\)는 maximum gradient norm이고 \(\sigma\)는 noise multiplier다. 본 구현은
Poisson sampling과 Opacus RDP accountant를 사용한다. 목표 \(\epsilon\), \(\delta\),
sampling rate, epoch 수를 이용해 \(\sigma\)를 계산하고, 실제 소비 \(\epsilon\)은
checkpoint 저장 시 accountant에서 다시 읽는다.

Physical batch는 메모리에 한 번에 올리는 크기이고 logical batch는 한 번의 DP update를
구성하는 목표 batch다. `BatchMemoryManager`가 하나의 Poisson-sampled logical batch를
여러 physical batch로 나누되, clipping과 noise addition은 logical update 의미를 보존한다.

---

# III. Method

## 1. Network Architecture

### 1.1 VAE with Swin Attention Blocks

#### 1.1.1 Checkpoint로 확인한 전체 구조

논문 구조도와 표는 `checkpoints/vae/vae_ep0070.pt`의 저장 인자를 기준으로 작성해야 한다.
현재 argparse default가 checkpoint 구성과 다를 수 있으므로 default만 보고 구조를
추정하지 않는다.

| 항목 | 값 |
|---|---:|
| 입력/출력 | 1×256×256 grayscale |
| base channel | 128 |
| channel multipliers | [1, 2, 4, 4, 4] |
| level별 channel | [128, 256, 512, 512, 512] |
| level별 ResBlock | 2 |
| attention resolution | 32×32, 16×16 |
| latent posterior parameter | 2×16×16 |
| sampled latent | 1×16×16 |
| dropout | 0 |
| downsampling | stride-2 3×3 convolution |
| upsampling | nearest-neighbor ×2 + 3×3 convolution |

Encoder의 tensor 흐름은 다음과 같다.

```text
1×256×256 input
 -> 3×3 Conv, 1→128
 -> L0: ResBlock×2                         128×256×256
 -> Downsample
 -> L1: ResBlock(128→256) + ResBlock       256×128×128
 -> Downsample
 -> L2: ResBlock(256→512) + ResBlock       512×64×64
 -> Downsample
 -> L3: [ResBlock -> Swin]×2               512×32×32
 -> Downsample
 -> L4: [ResBlock -> Swin]×2               512×16×16
 -> Middle: ResBlock -> Swin -> ResBlock   512×16×16
 -> GroupNorm -> SiLU -> 3×3 Conv          2×16×16
 -> split into mu and log-variance         each 1×16×16
```

Decoder는 다음과 같이 대칭적으로 영상을 복원한다.

```text
z: 1×16×16
 -> 1×1 post-quantization Conv
 -> 3×3 Conv, 1→512
 -> Middle: ResBlock -> Swin -> ResBlock
 -> L4: [ResBlock -> Swin]×2               512×16×16
 -> Upsample
 -> L3: [ResBlock -> Swin]×2               512×32×32
 -> Upsample
 -> L2: ResBlock×2                         512×64×64
 -> Upsample
 -> L1: ResBlock(512→256) + ResBlock       256×128×128
 -> Upsample
 -> L0: ResBlock(256→128) + ResBlock       128×256×256
 -> GroupNorm -> SiLU -> 3×3 Conv          1×256×256
```

Encoder와 decoder 사이에는 U-Net식 장거리 skip connection이 없다. Skip connection은
ResBlock과 Swin block 내부에만 존재한다.

#### 1.1.2 Residual block

Main branch는 `GroupNorm(32) → SiLU → 3×3 Conv → GroupNorm(32) → SiLU →
Dropout(0) → zero-initialized 3×3 Conv` 순서다. 입력과 출력 channel이 같으면 identity
skip을, 다르면 1×1 convolution skip을 사용한다. VAE ResBlock에는 timestep embedding이
없다.

#### 1.1.3 Swin block

Swin block은 `LayerNorm → window multi-head self-attention → residual addition →
LayerNorm → MLP → residual addition`으로 구성된다.

- window: 8×8
- heads: 8
- MLP ratio: 4
- QKV bias: 사용
- relative position bias: 사용
- 각 level의 첫 attention shift: (0, 0)
- 각 level의 두 번째 attention shift: (4, 4)
- middle attention shift: (0, 0)

Attention은 32×32 및 16×16 feature map과 16×16 bottleneck에서만 사용한다.

#### 1.1.4 VAE 학습 objective와 checkpoint

Epoch-70 checkpoint의 확인값은 다음과 같다.

| 설정 | 값 |
|---|---:|
| batch size | 8 |
| learning rate | 1×10⁻⁴ |
| weight decay | 1×10⁻⁴ |
| \(\lambda_{rec}\) | 1.0 |
| \(\lambda_{ssim}\) | 1.0 |
| \(\lambda_{perc}\) | 1.0 |
| \(\lambda_{KL}\) | 1×10⁻⁴ |
| \(\lambda_{MMD}\) | 1×10⁻³ |
| MMD base sigma | 1.0 |
| multi-kernel MMD | 사용 |

입력은 \([-1,1]\)로 정규화하며 decoder 마지막에는 sigmoid나 tanh가 없다. MIMIC-CXR
미세조정에서는 VAE를 eval mode로 고정하고 gradient를 계산하지 않는다. 현재 학습은
VAE posterior의 mean과 log-variance를 float32 cache로 저장하되, 매 epoch posterior
sample은 새 Gaussian noise를 사용하여 다시 뽑는다. 즉, posterior sample 자체를 고정
cache하지 않는다.

### 1.2 Text Encoder

#### 1.2.1 입력 text 생성

MIMIC-CXR 학습 condition은 `LABEL+IMPRESSION` 모드다. CheXpert에서 값이 1인 positive
label을 `LABEL:` 뒤에 나열하고, 사용 가능한 경우 판독문의 `IMPRESSION` section을
뒤에 붙인다. 학습 manifest에서 positive CheXpert label이 없는 영상은 이 모드의
eligible set에서 제외한다. Text와 image의 정렬은 manifest row 순서를 단일 기준으로
유지한다.

Tokenizer 설정은 다음과 같다.

- local RadBERT tokenizer 사용
- batch 단위 padding
- truncation 사용
- maximum length: 512 tokens
- attention mask를 UNet cross-attention까지 전달

#### 1.2.2 RadBERT와 projection

최종 프로토콜은 `StanfordAIMI/RadBERT`의 로컬 snapshot을 사용한다.

| 항목 | 값 |
|---|---|
| local path | `/storage/hjchoi/models/StanfordAIMI_RadBERT` |
| revision | `c8f97547ff6dd96debf4e50022d9679219e6987b` |
| `pytorch_model.bin` SHA256 | `8141cb71657b0d527cf6d9a1509e680d3dcc45544fba3bdf64876a4ce539632e` |
| architecture | BERT |
| vocabulary size | 28,996 |
| hidden size | 768 |
| output projection | Linear 768→512 |
| maximum length | 512 |

RadBERT backbone은 frozen eval mode다. 따라서 backbone parameter는 gradient를 받지 않고
dropout도 비활성화된다. Projection은 매 신규 미세조정에서 새로 초기화하며 LoRA와 함께
DP-SGD로 학습한다. Projection은 bias를 포함하므로 parameter 수는
\(768\times512+512=393{,}728\)개다.

Text sequence \(H\in\mathbb{R}^{B\times L\times768}\)는

\[
C=HW_p^\top+b_p,
\qquad C\in\mathbb{R}^{B\times L\times512}
\]

로 projection되어 UNet context가 된다. 학습과 생성은 반드시 동일 RadBERT snapshot을
사용해야 한다. BioBERT 기반 checkpoint를 RadBERT로 생성하거나 그 반대로 실행하는 것은
허용하지 않는다.

### 1.3 Text-Conditioned Latent Diffusion Model with LoRA

#### 1.3.1 UNet

NIH pretrained LDM checkpoint `checkpoints/ldm/ldm_epoch0100.pt`의 구조는 다음과 같다.

| 항목 | 값 |
|---|---:|
| latent input/output | 1×16×16 |
| base channels | 128 |
| channel multipliers | [1, 2, 4] |
| ResBlock per level | 2 |
| attention resolutions | [1, 2, 4] in latent hierarchy |
| transformer depth | 1 |
| context dimension | 512 |
| attention head channel | 8 |
| diffusion timesteps | 1,000 |
| beta schedule | linear |
| parameterization | epsilon prediction |
| scale factor | 1.0 |

Cross-attention LoRA는 16개의 SpatialTransformer block 안의 32개 CrossAttention module에
적용된다. 각 CrossAttention의 `to_q`, `to_k`, `to_v`, `to_out` linear를 감싸므로 총
128개 linear layer에 adapter가 삽입된다. Base UNet과 VAE는 고정되고, LoRA A/B와 text
projection만 학습된다. 현재 기본 구성의 trainable parameter는 약 738,816개다.

#### 1.3.2 Condition-contrastive objective

각 record \(i\)에 대해 정답 condition \(c_i^+\)는 해당 영상의 `LABEL+IMPRESSION`,
negative condition \(c^-\)는 다음 공개 고정 문장이다.

```text
LABEL: No Finding IMPRESSION: No acute cardiopulmonary abnormality.
```

두 condition forward는 동일한 clean latent \(z_{0,i}\), timestep \(t_i\), Gaussian
noise \(\epsilon_i\), noisy latent \(z_{t,i}\)를 공유한다. 따라서 두 denoising error의
차이는 stochastic input 차이가 아니라 condition 차이로 해석할 수 있다.

\[
e_i^+=\frac{1}{CHW}
\left\|\epsilon_i-epsilon_\theta(z_{t,i},t_i,c_i^+)\right\|_2^2,
\]

\[
e_i^-=\frac{1}{CHW}
\left\|\epsilon_i-epsilon_\theta(z_{t,i},t_i,c^-)\right\|_2^2.
\]

Diffusion 항은 구현의 timestep-dependent log-variance와 original ELBO weight를 포함한다.
핵심 ranking 항은

\[
\mathcal{L}_{\mathrm{rank},i}
=\max(0,m+e_i^+-e_i^-)
\]

이며, 전체 loss는

\[
\mathcal{L}_{\mathrm{total}}
=\frac{1}{B}\sum_i
\left(\mathcal{L}_{\mathrm{diff},i}
+\lambda_{rank}\mathcal{L}_{\mathrm{rank},i}\right)
\]

이다. 기본값은 \(m=0.01\), \(\lambda_{rank}=0.1\)이다.

Negative sentence는 다른 환자에게서 가져오지 않으므로 한 record의 loss가 다른 private
record에 직접 의존하지 않는다. 다만 고정 No-Finding condition은 abnormal/normal 구분을
강화할 수 있을 뿐, 특정 병변의 위치나 세부 형태가 올바르게 grounding되었음을 보장하지
않는다. 이 한계는 Methods와 Discussion 양쪽에 명시한다.

## 2. Training

### 2.1 Dataset

#### 2.1.1 Public Dataset: NIH Chest X-ray Dataset

NIH ChestX-ray14는 공개 흉부 X-ray 사전학습 데이터다. 저장소 loader는 미리 구성된
`train_val/train_val.csv`와 `test/test.csv`를 읽고, `Finding Labels`의 pipe-separated
multi-label string을 condition으로 사용한다. 논문에 전체 NIH 공개 데이터 규모를
기술할 때는 원 논문의 112,120 images/30,805 patients와 **실제 로컬 split에서 loader가
읽은 표본 수**를 구분해 보고해야 한다. 후자는 최종 논문 작성 시 manifest 또는 CSV로
다시 감사한다.

NIH 영상 전처리는 다음과 같다.

- PIL grayscale 변환
- 256×256 resize
- training에서 contrast factor 1.5 적용
- training에서 random horizontal flip, probability 0.5
- tensor 변환 후 mean 0.5, standard deviation 0.5로 정규화하여 \([-1,1]\) 범위 사용
- validation/test에서는 resize와 normalization만 사용

NIH multi-label condition은 14개 disease token과 `PAD`, `BOS`, `EOS`, `SEP`를 포함한
18-token vocabulary로 변환한다. Condition sequence는

```text
BOS label_1 SEP label_2 SEP ... label_n EOS PAD ...
```

형태다. Token embedding dimension은 256, UNet context projection은 512다. `No Finding`
처럼 vocabulary에 없는 label만 있는 표본은 `[BOS, EOS]`가 된다.

NIH는 두 단계에서 사용한다.

1. Swin-VAE 사전학습
2. Frozen VAE latent에서 class-label-conditioned LDM 사전학습

두 단계는 공개 데이터 학습이므로 DP-SGD를 적용하지 않는다.

#### 2.1.2 Private Dataset: MIMIC-CXR Database

MIMIC-CXR은 DICOM 영상과 자유 형식 판독문을 제공하고, MIMIC-CXR-JPG는 변환된 JPG,
CheXpert label 및 권장 split을 제공한다. 본 연구는 MIMIC-CXR-JPG v2.1.0 영상,
MIMIC-CXR report, CheXpert label, 공식 split CSV를 연결한다.

2026-10-28 이후 신규 학습의 데이터 경계는
`mimic-cxr-2.0.0-split.csv`의 `train`, `validate`, `test`를 그대로 따른다.

- `train`: DP 미세조정
- `validate`: 모델 및 protocol 선택
- `test`: protocol 동결 후 한 번의 최종 평가

Patient prefix로 split을 다시 정하지 않고 No-Finding downsampling을 적용하지 않는다.
Manifest 생성 시 `(dicom_id, subject_id, study_id)`를 official split CSV와 one-to-one으로
검증하고, split 간 `subject_id` 교집합이 0인지 확인한다.

현재 진행 중인 p10--p15 학습 manifest의 감사값은 다음과 같다. 이는 결과가 아니라
가용 코호트 기술값이다.

| 구분 | 전체 paired JPG/report | LABEL+IMPRESSION eligible |
|---|---:|---:|
| train | 219,619 | 207,481 |
| validate | 1,845 | 1,715 |
| test | 3,028 | 2,881 |
| 합계 | 224,492 | 212,077 |

Eligible은 `chexpert_found=True`이고 positive label count가 1 이상인 행이다. Split 간
patient overlap은 0이다. Manifest와 official split의 provenance는 다음과 같다.

- manifest: `data_manifests/mimic_official_split_p10_p15/ldm_official_split_manifest.csv.gz`
- manifest SHA256: `78bf5dd350b61fa5941c73d5665b01cbad3a01421e331f2b9549ae760badd9d5`
- official split CSV SHA256:
  `1fb1d5964ca4b64af0e64d36cfc26fb12de415bffacc34530cd7d5ea312c9da8`
- policy: p10--p15 paired JPGs, official split only, prefix reassignment/downsampling 없음

하나의 study에 여러 image가 있으면 각 image가 학습 행이 되지만 patient와 study 식별자를
보존한다. 통계적 불확실성은 image가 아니라 patient cluster를 기준으로 계산한다.

### 2.2 Pre-training

#### 2.2.1 Stage 1: VAE pre-training

NIH training image를 이용해 Swin-VAE를 non-private 방식으로 학습한다. 실제 사용
checkpoint는 epoch 70의 `checkpoints/vae/vae_ep0070.pt`다. VAE 학습 loss와 설정은
Section III.1.1.4의 확인값을 사용한다. 최종 논문에는 checkpoint 선택 기준과 NIH
train/validation 표본 수를 추가 감사한 후 기록한다.

#### 2.2.2 Stage 2: class-conditioned LDM pre-training

VAE를 고정하고 NIH image를 latent로 변환한 뒤, NIH disease-label token sequence를
cross-attention condition으로 사용하여 UNet을 학습한다. 실제 base checkpoint는
`checkpoints/ldm/ldm_epoch0100.pt`다.

| 설정 | checkpoint 확인값 |
|---|---:|
| epochs | 100 |
| global steps | 1,081,600 |
| batch size | 8 |
| learning rate | 1×10⁻⁴ |
| diffusion steps | 1,000 |
| beta schedule | linear |
| latent scale factor | 1.0 |
| EMA | 사용 |
| class embedding | 256 |
| context dimension | 512 |

#### 2.2.3 Pre-training에서 fine-tuning으로의 전환

NIH 단계는 disease-label token condition을 사용하지만 MIMIC 단계는 RadBERT report
sequence를 사용한다. 두 단계의 context dimension은 모두 512이므로 pretrained UNet의
cross-attention 구조를 유지할 수 있다. MIMIC 단계에서 VAE와 base UNet을 재사용하고,
RadBERT용 768→512 projection과 LoRA adapter는 새로 초기화한다.

### 2.3 Fine-tuning

#### 2.3.1 현재 확정된 학습 구성

현재 p10--p15 공식 split 학습 protocol은 다음과 같다.

| 항목 | 설정 |
|---|---|
| image size | 256×256 grayscale |
| text | LABEL+IMPRESSION |
| maximum tokens | 512 |
| VAE | `vae_ep0070.pt`, frozen |
| base LDM | `ldm_epoch0100.pt`, base frozen |
| text encoder | RadBERT, frozen eval |
| text projection | 768→512, trainable |
| LoRA | rank 4, alpha 4, dropout 0 |
| LoRA target | all 16 SpatialTransformer blocks, q/k/v/out |
| objective | epsilon prediction + condition ranking |
| rank margin/weight | 0.01 / 0.1 |
| optimizer | AdamW |
| learning rate | 0.002 |
| scheduler | 100-step warmup 후 linear decay |
| epochs | 30 |
| logical/physical batch | 256 / 4 |
| clipping | flat per-sample global norm |
| clipping norm | 0.001 |
| target delta | 1×10⁻⁵ |
| target epsilon arms | 1, 10 |
| seed | 42 |
| sampling | Poisson sampling |
| validation during training | 비활성; 별도 공식 validation generation/evaluation |
| checkpoint interval | 10 epochs |

`target_epsilon`은 noise multiplier를 정하기 위한 목표값이다. 논문에는 목표값만 보고하지
않고, 최종 checkpoint의 accountant가 기록한 `epsilon_spent`를 함께 보고해야 한다.
해당 값은 현재 학습 완료 전이므로 이 문서에 기입하지 않는다.

#### 2.3.2 학습 가능한 파라미터

- RadBERT backbone: frozen
- VAE encoder/decoder: frozen
- pretrained base UNet: frozen
- RadBERT 768→512 projection: trainable
- cross-attention LoRA A/B: trainable

Flat clipping은 위 trainable parameter의 per-sample gradient를 하나의 vector로 간주하여
global norm 0.001로 clipping한다. Projection과 LoRA를 서로 다른 clipping group으로
나누는 방식은 ablation이며 현재 기본 protocol이 아니다.

#### 2.3.3 DP 학습 알고리즘

논문 Algorithm에는 다음 순서를 사용한다.

```text
Input:
  private training records D={(x_i, r_i)}, frozen VAE E/D,
  frozen RadBERT T, pretrained UNet theta_0,
  trainable projection P and LoRA parameters A,
  target (epsilon, delta), clipping norm C

1. Build a Poisson-sampled logical-batch loader.
2. Compute noise multiplier sigma from target epsilon, delta,
   sampling rate, and number of epochs.
3. For each logical batch:
   a. Split it into physical batches for memory safety.
   b. Obtain VAE posterior parameters for x_i from the verified cache.
   c. Sample z_i from the posterior with an independent seeded RNG.
   d. Sample timestep t_i and Gaussian noise epsilon_i.
   e. Form z_t from z_i, t_i, and epsilon_i.
   f. Encode the correct report r_i and the fixed No-Finding sentence
      with frozen RadBERT and trainable projection P.
   g. Predict noise twice using the same z_t, t_i, and epsilon_i.
   h. Compute diffusion loss and margin-ranking loss.
   i. Compute per-record gradients for P and A.
   j. Clip each gradient to global norm C.
   k. Add Gaussian noise with standard deviation sigma*C.
   l. Apply the AdamW update at the logical-batch boundary.
   m. Update the RDP accountant.
4. Save full-state and adapter checkpoints with epsilon_spent and provenance.
```

Data order, posterior/timestep RNG, DP noise RNG은 서로 분리된 seed stream을 사용한다.
이는 text encoder dropout이나 다른 연산의 RNG 소비가 Poisson sampling 또는 DP noise를
바꾸지 않도록 하기 위함이다.

#### 2.3.4 Checkpoint와 생성 시 일관성

Full checkpoint에는 base model, LoRA, projection, optimizer, epoch, global step,
`epsilon_spent`, 전체 training args가 저장된다. LoRA-only checkpoint에는 adapter와
rank/alpha, privacy metadata가 저장된다.

LoRA fine-tuned checkpoint를 생성에 사용할 때는 `--dp_ckpt`만 지정하면 안 된다.
`strict=False` base loading이 LoRA tensor를 조용히 버릴 수 있으므로 `--lora_ckpt`를
함께 지정하고, adapter를 주입·복원한 뒤 merge해야 한다. 생성 전 다음을 확인한다.

1. checkpoint의 text encoder provenance가 RadBERT snapshot과 일치한다.
2. 예상 LoRA key가 모두 로드되고 unexpected key가 없다.
3. projection weight가 checkpoint에서 복원된다.
4. 서로 다른 checkpoint의 동일-seed sample hash가 동일하지 않다.

---

# IV. Implementation and Performance Evaluation

## 1. Implementation

### 1.1 실행 환경

모든 학습, 생성, 평가는 Docker container `CXR_dp_medclip`에서 실행한다. Host의
프로젝트 경로 `/home/hjchoi/PycharmProjects/PrivaText-CXR`는 container의
`/workspace/PrivaText-CXR`에 bind mount된다.

학습/생성과 평가는 dependency 충돌 때문에 별도 Python environment를 사용한다.

| 용도 | Python | 주요 확인 버전 |
|---|---|---|
| 학습/생성 | `/opt/conda/bin/python` | Python 3.11.12, PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128, Opacus 1.6.0, transformers 5.8.1 |
| 평가 | `/opt/medclip_env/bin/python` | Python 3.11.12, PyTorch 2.7.0+cu128, transformers 4.24.0 |

공통 확인 버전은 NumPy 2.2.5, scikit-learn 1.8.0이며 학습 환경의 pandas는 3.0.3이다.
최종 논문에는 실제 사용 GPU model, CUDA driver, OS/container image digest를 추가한다.

### 1.2 생성 설정

Validation screening과 최종 평가는 동일한 checkpoint provenance, prompt order, seed를
사용해야 한다. DDIM50은 빠른 validation screening에 사용할 수 있으나, 최종 protocol이
DDPM 또는 다른 step 수를 사용하면 sampler를 별도 연구 변수로 명시한다. Sampler가 다른
결과를 직접 비교하여 모델 자체의 차이라고 해석하면 안 된다.

생성 결과는 각 PNG와 함께 `descriptions.csv`에 image file, prompt, prompt index를
저장한다. 이 index가 official validation/test manifest row와 일치할 때만 paired metric과
CheXpert label evaluation을 수행한다.

### 1.3 재현성 및 provenance

최종 산출물은 다음 정보를 보존해야 한다.

- Git commit
- official split CSV hash
- exact manifest hash
- VAE 및 pretrained LDM checkpoint path/hash
- RadBERT revision 및 weight hash
- target epsilon/delta와 actual epsilon spent
- clipping norm과 noise multiplier
- LoRA rank/alpha 및 주입 block
- seed, sampler, sampling steps
- prompt manifest와 생성 index mapping
- evaluator 이름과 weight identifier

## 2. Performance Evaluation

이 절은 평가 **방법**만 정의한다. 현재 학습 중 모델의 수치나 순위는 포함하지 않는다.

### 2.1 공통 평가 원칙

1. 공식 `validate`는 모델 선택과 protocol screening에 사용한다.
2. 공식 `test`는 방법과 분석 계획을 동결한 뒤 최종 평가에만 사용한다.
3. 비교 모델은 동일 real reference, prompt order, seed, sampler, sampling step을 사용한다.
4. 동일 환자의 여러 영상이 독립 표본인 것처럼 CI를 좁히지 않도록 patient-cluster
   bootstrap을 사용한다.
5. Distribution, clinical, semantic, downstream metric을 하나의 composite score로
   임의 결합하지 않는다.
6. Metric별 유효 표본 수, 제외 기준, pathology support를 함께 보고한다.

### 2.2 Distributional Fidelity

#### 2.2.1 Feature extraction

Real과 generated image에서 동일한 frozen feature extractor로 feature를 한 번 추출하고
cache한다. 기본 임상 feature extractor는 TorchXRayVision이며, 일반 Inception feature를
사용한 경우 이를 명시한다. 서로 다른 backbone에서 얻은 FID/FDS 값은 직접 비교하지 않는다.

#### 2.2.2 Fréchet distance

Real feature의 평균·공분산을 \((\mu_r,\Sigma_r)\), generated feature를
\((\mu_g,\Sigma_g)\)라 하면

\[
\operatorname{FID}=
\|\mu_r-\mu_g\|_2^2+
\operatorname{Tr}\left(
\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}
\right).
\]

낮을수록 두 feature distribution이 가깝다. FID는 feature marginal distribution을
평가하며 prompt-level 임상 일치를 직접 측정하지 않는다.

#### 2.2.3 Ledoit--Wolf FDS

고차원 feature의 covariance를 Ledoit--Wolf shrinkage로 추정한다. Real과 generated
Gaussian 사이의 양방향 KL을 계산한다.

\[
\operatorname{FDS}_{sym}=
\tfrac12\left[
D_{KL}(\mathcal{N}_r\|\mathcal{N}_g)+
D_{KL}(\mathcal{N}_g\|\mathcal{N}_r)
\right].
\]

`fds_real_given_gen`, `fds_gen_given_real`, `fds_symmetric`을 구분해 보고한다.

#### 2.2.4 MT-DDPM FDS

MT-DDPM FDS는 real/generated feature를 joint t-SNE 2차원 공간으로 투영한 뒤 Gaussian
KL을 계산하는 별도 지표다. Full-covariance Ledoit--Wolf FDS와 정보 차원 및 geometry가
다르므로 두 값을 같은 FDS로 혼용하지 않는다. 논문 표에는 각각
`FDS (full-feature, Ledoit--Wolf)`와 `MT-DDPM FDS (joint t-SNE 2-D)`로 표기한다.

#### 2.2.5 Paired image metric

Prompt index가 real image index와 정확히 일치할 때만 다음을 계산한다.

- SSIM: 높을수록 구조적 유사성이 큼
- PSNR: 높을수록 pixel error가 작음
- LPIPS: 낮을수록 deep perceptual distance가 작음

Conditional generation은 real image의 deterministic reconstruction이 아니므로 paired
pixel metric은 보조 지표다. 높은 pixel similarity가 더 나은 다양성 또는 임상적
정확성을 뜻하지 않는다.

### 2.3 Clinical Consistency

#### 2.3.1 Frozen clinical evaluator

TorchXRayVision DenseNet-121의 `densenet121-res224-all`과
`densenet121-res224-nih` weight를 평가 전용으로 사용한다. 생성 모델 학습에는 evaluator
gradient를 사용하지 않으므로 circular training objective와 분리된다.

#### 2.3.2 Ground truth와 label 처리

Official split의 `(subject_id, study_id)`를 MIMIC-CXR CheXpert CSV와 연결한다.
Pathology value가 0 또는 1인 경우만 AUROC에 사용하고 uncertain/missing은 해당 pathology
계산에서 mask한다. 각 pathology가 최소 10 positive와 10 negative를 모두 가질 때 macro
평균에 포함한다.

Generated image와 paired real image에 대해 같은 classifier와 ground truth를 사용한다.
각 weight family별로 다음을 보고한다.

- pathology별 generated AUROC
- pathology별 real AUROC
- macro AUROC와 포함 pathology 목록
- support
- real-minus-generated gap
- generated/real ratio
- patient-cluster bootstrap CI

이 지표는 생성 영상에 원래 study label을 예측할 수 있는 feature가 남아 있는지 측정한다.
생성 모델 자체의 임상 진단 정확도나 실제 임상 안전성을 직접 입증하지 않는다.

### 2.4 Semantic Alignment

#### 2.4.1 Report-level MedCLIP alignment

MedCLIP image encoder와 text encoder로 normalized embedding을 만들고 cosine similarity를
계산한다. Report-level score는 전반적 image--text alignment를 측정하지만, 보고서의
길이와 반복 표현에 영향을 받고 특정 병변의 grounding을 직접 보장하지 않는다.

#### 2.4.2 Label-level zero-shot AUROC

명확한 pathology-level 평가를 위해 CheXpert abnormality가 정확히 하나인 study만
선택한다. `No Finding`과 `Support Devices`는 abnormality 후보에서 제외한다. 각 label
prompt는 기본적으로 label 이름 자체인 `"{label}"`이며, image--prompt cosine을 해당
label의 score로 사용한다. Label별 one-vs-rest AUROC와 macro AUROC를 real/generated에
대해 각각 계산한다.

이 값은 **zero-shot label AUROC**이며 accuracy가 아니다. Macro에는 최소 positive
support 10을 만족하는 label만 포함한다.

#### 2.4.3 Retrieval

두 종류를 구분한다.

1. Label-prompt retrieval: candidate가 pathology label prompt이며 정답 label 하나가
   relevant하다. R@1은 이 제한된 후보 집합에서 top-1 label retrieval을 의미한다.
2. Impression retrieval: candidate가 study의 IMPRESSION embedding이며 동일 single
   abnormality label을 가진 text를 relevant로 정의한다. R@K, P@K, mAP를 계산한다.

RoentGen-style image-to-image 및 image-to-text retrieval도 별도로 계산할 수 있다. 현재
저장소 구현은 원 논문의 공개되지 않거나 접근 불가능한 weight 대신 두 방향 모두 MedCLIP
backend를 사용하므로, 결과를 "RoentGen checkpoint 성능"으로 표현하지 않는다.

#### 2.4.4 Conditioning fidelity

다음 paired control을 이용해 모델이 condition을 사용하는지 평가한다.

- correct report vs patient-shuffled report의 denoising MSE
- correct `LABEL+IMPRESSION` vs fixed `No Finding` condition gap
- 동일 initial noise에서 positive/negative prompt를 바꾼 생성 score delta
- pathology별 prompt-intent AUROC
- XRV와 MedCLIP의 독립 evaluator 결과

Condition gap은

\[
\Delta e=e^- - e^+
\]

로 정의할 수 있으며 양수이면 정답 condition의 denoising error가 더 작다는 뜻이다.
하지만 이것만으로 lesion localization이나 fine-grained pathology grounding을 주장하지
않는다.

### 2.5 Downstream Classification Utility

Downstream utility는 frozen evaluator로 생성 영상을 채점하는 평가와 다르다. 새로운
DenseNet-121을 `weights=None`에서 학습하고 고정된 real test set에서 평가한다.

비교 arm의 기본 형태는 다음과 같다.

- real-only
- synthetic-only
- real + synthetic
- 필요 시 privacy budget별 synthetic arm

Classifier는 grayscale image를 세 채널로 반복하고 14-way CheXpert head를 사용한다.
Loss는 uncertain/missing entry를 제외한 masked multi-label BCE이며, class imbalance를
위해 training positive prevalence 기반 `pos_weight`를 사용한다.

\[
\mathcal{L}_{cls}=
\frac{\sum_{i,k}m_{ik}
\operatorname{BCEWithLogits}(y_{ik},\hat{y}_{ik};w_k^+)}
{\sum_{i,k}m_{ik}},
\]

여기서 \(m_{ik}\)는 label이 관측된 경우 1이다. 모든 arm은 동일 real test manifest,
augmentation, optimizer policy 및 평가 code를 사용한다. 다음을 보고한다.

- pathology별 AUROC
- 최소 positive support를 만족한 pathology의 macro AUROC
- training seed별 값, 평균, 표준편차
- real-only 대비 차이와 불확실성

단일 seed의 작은 상승은 안정적인 utility 개선으로 결론내리지 않는다.

### 2.6 통계 분석 계획

- 비교 단위는 가능한 한 같은 환자에 대한 paired difference로 구성한다.
- 95% confidence interval은 patient-cluster bootstrap으로 계산한다.
- Macro AUROC는 사전 정의한 support 기준을 만족한 pathology의 산술평균으로 계산한다.
- CI가 0을 포함하는 모델 차이는 통계적으로 확인된 개선이라고 표현하지 않는다.
- 여러 pathology를 탐색한 결과는 primary endpoint와 exploratory endpoint를 구분한다.
- Model-selection validation과 final test 결과를 혼합하지 않는다.
- Training seed가 하나인 경우 구조 선택을 위한 screening으로 표현하고 일반화 주장을
  제한한다.

---

# V. Results 작성 전까지 보류할 항목

현재 학습 완료 전에는 Results 본문을 작성하지 않는다. 다음 값이 모두 생성되고 검증된
후, 실험 번호가 아니라 연구 질문별로 결과를 구성한다.

## 1. Privacy and optimization

- target epsilon별 actual epsilon spent
- noise multiplier, sample rate, logical steps
- epoch별 total/diffusion/rank loss
- clipped fraction과 gradient norm audit
- 학습 완료 및 checkpoint integrity

## 2. Distributional fidelity

- FID
- Ledoit--Wolf FDS의 양방향 및 symmetric 값
- MT-DDPM FDS
- SSIM, PSNR, LPIPS
- t-SNE 및 정성 영상

## 3. Clinical consistency

- XRV all/NIH pathology별 및 macro AUROC
- real/generated gap과 ratio
- patient-paired CI

## 4. Semantic alignment and conditioning

- MedCLIP report alignment
- label-level zero-shot AUROC
- retrieval R@K 및 mAP
- correct-vs-shuffled denoising
- paired prompt response
- XRV/MedCLIP 간 일치 여부

## 5. Downstream utility

- real-only, synthetic-only, mixed classifier 결과
- epsilon별 차이
- multi-seed 평균과 분산

결과가 확보된 뒤에도 다음 문장 구조를 유지한다.

- "FID가 감소했다"와 "임상적 유용성이 개선됐다"를 별도 주장으로 쓴다.
- 서로 다른 metric이 반대 방향이면 한 metric을 선택적으로 제거하지 않고 평가 공간의
  차이로 보고한다.
- Condition response를 lesion grounding으로 확대하지 않는다.
- Validation에서 선택한 모델 하나만 동결된 official test로 넘긴다.

---

# VI. 논문용 표와 그림 설계

## Figure 1. 전체 파이프라인

다음 흐름을 한 그림에 나타낸다.

```text
Public NIH CXR
  -> Swin-VAE pre-training
  -> class-conditioned LDM pre-training

Credentialed MIMIC-CXR image + LABEL/IMPRESSION
  -> frozen VAE posterior
  -> latent noising
  -> frozen RadBERT + trainable 768→512 projection
  -> LoRA-adapted cross-attention UNet
  -> DP-SGD
  -> DDIM/DDPM sampling
  -> frozen VAE decoder
  -> synthetic CXR
```

Frozen block, DP-trainable block, public-data training, private-domain training을 색으로 구분한다.

## Figure 2. Swin-VAE

- 다섯 encoder/decoder level과 tensor shape
- 32×32 및 16×16 Swin 위치
- mean/log-variance와 reparameterization
- 1×16×16 latent
- encoder--decoder long skip이 없음을 명확히 표시
- ResBlock 및 Swin inset

## Figure 3. DP condition-contrastive fine-tuning

하나의 record에서 같은 \(z_t,t,\epsilon\)을 공유하는 두 branch를 그린다.

- correct LABEL+IMPRESSION
- fixed public No Finding
- 두 predicted noise
- positive/negative MSE
- margin-ranking loss
- diffusion + weighted rank loss
- per-sample clipping과 Gaussian noise

## Table 1. Dataset cohort

- NIH 전체 공개 규모와 실제 local pretraining split 규모를 분리
- MIMIC p10--p15 official train/validate/test
- paired 및 LABEL+IMPRESSION eligible 수
- patient 수와 split overlap
- exclusion rule

## Table 2. Architecture

- VAE level, resolution, channel, block, attention
- UNet latent resolution, channel multiplier, attention
- RadBERT hidden/context dimension
- LoRA target/rank/alpha

## Table 3. Training configuration

- public pretraining과 DP fine-tuning을 별도 column으로 구성
- optimizer, LR, batch, epochs, diffusion steps
- DP target epsilon/delta, clipping norm, actual epsilon
- rank margin/weight

## Table 4. Evaluation definitions

- 평가 축
- metric
- feature/evaluator
- 방향
- 표본 포함 기준
- 해석 범위와 주의점

Results table은 현재 문서에서 생성하지 않는다.

---

# VII. 주장 범위와 한계

## 현재 방법론만으로 말할 수 있는 것

- 명시된 trainable parameter에 Opacus DP-SGD를 적용하는 구현을 구축했다.
- 공개 NIH 사전학습과 MIMIC 공식 split DP 미세조정을 분리했다.
- frozen RadBERT와 trainable projection/LoRA를 사용한다.
- 정답 condition과 공개 fixed negative를 동일 noise에서 비교하는 loss를 사용한다.
- 임상·의미·분포·downstream 평가를 분리한 protocol을 정의했다.

## 결과 없이 말하면 안 되는 것

- 모델이 기존 방법보다 우수하다.
- epsilon 1 또는 epsilon 10 중 하나가 모든 metric에서 우수하다.
- 생성 영상이 실제 임상 학습 데이터를 대체할 수 있다.
- 병변이 정확한 위치에 grounding된다.
- synthetic data 공개가 모든 privacy risk를 제거한다.
- 현재 학습 중 모델이 목표 epsilon을 정확히 달성했다.

## 구조적 한계

- Local MIMIC subset은 전체 MIMIC-CXR을 대표하지 않을 수 있다.
- 단일 기관 데이터이므로 외부 기관 generalization이 검증되지 않는다.
- Positive-label-required `LABEL+IMPRESSION` eligibility가 cohort를 바꾼다.
- Fixed No-Finding negative는 abnormal/normal shortcut을 유도할 수 있다.
- VAE reconstruction error가 diffusion 성능의 상한을 제한한다.
- XRV와 MedCLIP도 각자 pretraining domain과 label ontology의 영향을 받는다.
- DP는 학습 알고리즘의 안정성 보장이지 임상적 정확성 보장이 아니다.
- 같은 private dataset에서 여러 모델을 공개한다면 release-level privacy composition을
  별도로 고려해야 한다.

---

# VIII. 코드 및 증거 추적표

| 논문 항목 | 근거 파일 |
|---|---|
| VAE encoder/decoder | `src/Model/VAE/Encoder.py`, `src/Model/VAE/Decoder.py` |
| VAE checkpoint 구조 | `checkpoints/vae/vae_ep0070.pt`, `vae-handoff.md` |
| VAE loss | `src/Loss/vae_loss.py` |
| NIH loader | `src/Data/nih.py` |
| NIH label condition | `src/Data/cls_emb.py` |
| LDM pre-training | `src/LDM_train.py` |
| Latent diffusion | `src/Model/Diffusion/LDM.py` |
| DP fine-tuning | `src/LDM_dp_finetune.py` |
| DP model parameter selection | `src/Model/Diffusion/LDM_dp.py` |
| LoRA | `src/Model/lora.py` |
| Text encoder/projection | `src/Modules/BioBERT_embedder.py` |
| Condition-contrastive loss | `classification_experiments/counterfactual_training.py` |
| RadBERT protocol | `classification_experiments/031_radbert_training.py` |
| p10--p15 training entry | `classification_experiments/033_p10_p15_radbert_training.py` |
| p10--p15 launcher | `classification_experiments/run_033_p10_p15_radbert.sh` |
| official manifest construction | `classification_experiments/033_build_p10_p15_manifest.py` |
| manifest audit | `data_manifests/mimic_official_split_p10_p15/split_summary.json` |
| evaluation driver | `Eval_metric/run_feature_eval.py` |
| XRV clinical metric | `src/Eval_metric/downstream_cls.py` |
| MedCLIP label metric | `Eval_metric/clip_labelret.py` |
| FID/FDS | `src/Eval_metric/fid.py`, `src/Eval_metric/fds.py` |
| downstream classifier | `Eval_metric/train_downstream_cls.py` |

---

# IX. 참고문헌 초안

아래 목록은 논문 본문에서 사용하는 핵심 방법과 데이터의 1차 출처다. 최종 제출 형식에
맞춰 BibTeX를 정리한다.

1. Kingma, D. P., & Welling, M. *Auto-Encoding Variational Bayes*.
   arXiv:1312.6114, 2013. <https://arxiv.org/abs/1312.6114>
2. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B.
   *High-Resolution Image Synthesis With Latent Diffusion Models*.
   CVPR, 2022. <https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html>
3. Hu, E. J., et al. *LoRA: Low-Rank Adaptation of Large Language Models*.
   ICLR, 2022. <https://arxiv.org/abs/2106.09685>
4. Abadi, M., et al. *Deep Learning with Differential Privacy*.
   ACM CCS, 2016. <https://arxiv.org/abs/1607.00133>
5. Wang, X., et al. *ChestX-ray8: Hospital-scale Chest X-ray Database and
   Benchmarks on Weakly-Supervised Classification and Localization of Common
   Thorax Diseases*. CVPR, 2017. <https://arxiv.org/abs/1705.02315>
6. Johnson, A. E. W., et al. *MIMIC-CXR, a de-identified publicly available
   database of chest radiographs with free-text reports*. Scientific Data 6,
   317, 2019. <https://doi.org/10.1038/s41597-019-0322-0>
7. Johnson, A. E. W., et al. *MIMIC-CXR-JPG - chest radiographs with structured
   labels*, v2.1.0. PhysioNet. <https://physionet.org/content/mimic-cxr-jpg/2.1.0/>
8. Yan, A., et al. *RadBERT: Adapting Transformer-based Language Models to
   Radiology*. Radiology: Artificial Intelligence, 2022.
   <https://pubs.rsna.org/doi/10.1148/ryai.210258>
9. Wang, Z., Wu, Z., Agarwal, D., & Sun, J. *MedCLIP: Contrastive Learning from
   Unpaired Medical Images and Text*. EMNLP, 2022.
   <https://aclanthology.org/2022.emnlp-main.256/>
10. Cohen, J. P., et al. *TorchXRayVision: A library of chest X-ray datasets and
    models*. MIDL, 2022. <https://proceedings.mlr.press/v172/cohen22a.html>
11. Heusel, M., et al. *GANs Trained by a Two Time-Scale Update Rule Converge to
    a Local Nash Equilibrium*. NeurIPS, 2017.
    <https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html>
12. Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O.
    *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*.
    CVPR, 2018. <https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html>

---

# X. 학습 완료 후 이 문서에 추가할 최소 정보

학습이 끝나면 이 문서를 실험 일지로 바꾸지 않고 아래 확정값만 추가한다.

1. 각 최종 checkpoint의 epoch, global step, noise multiplier, actual epsilon spent
2. 실제 사용한 GPU와 wall-clock time
3. 모델 선택에 사용한 validation primary endpoint
4. 동결된 test protocol과 test를 처음 연 시점
5. 사전 정의한 metric의 최종 표 및 patient-bootstrap CI
6. 반복 seed가 있다면 mean±standard deviation
7. 대표 성공/실패 영상의 사전 정의된 선택 규칙
8. 결과에 근거한 주장과 주장하지 못하는 내용의 최종 경계

개별 실험 번호, 프로세스 ID, queue script, 중간 실패 로그, dashboard URL은 논문 작성용
기술 명세에 추가하지 않는다.
