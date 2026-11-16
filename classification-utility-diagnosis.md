# 합성 CXR classification utility 진단 (2026-09-07)

이 분석의 소유 worktree는 `claude/fid-multi-model-eval-l7hpk4`다. 소스·체크포인트·기존 산출물 확인과 작은 CPU 진단만 수행했다. 개선 구현/재학습/전체 재평가는 미실행이다.

현재 근거는 생성 조건과 병변 영상 특징의 연결이 약하고 그 합성 영상에 원본 라벨을 그대로 부여하는 것이 utility 저하의 주요 원인 후보임을 가리킨다. DP noise만으로 원인을 확정할 수 없다.

## 관측 결과와 평가 구분

JPG p12 custom holdout 결과이며 예전 DICOM 361장 결과와 다르다. 현 JPG LDM 학습은 LABEL+IMPRESSION 필터 후 49,907장이다.

| macro AUROC | real | eps1 ep30 synthetic | eps10 ep30 synthetic |
|---|---:|---:|---:|
| Frozen XRV-all | 0.73199 | 0.50531 | 0.50767 |
| MedCLIP label zero-shot | 0.64126 | 0.50872 | 0.50622 |

출처: `EVAL/metric/JPG_LABEL_IMPRESSION/{eps1_ep30,eps10_ep30}/eval_summary.json`. 고정 10,240-image 평가, CLIP은 그중 single-positive abnormality 부분집합.

| utility classifier training | eps1 | eps10 |
|---|---:|---:|
| R1100 (공유 대조군) | 0.64609 | 0.64609 |
| R1100+S1100 | 0.63690 | 0.62746 |
| S5500 | 0.52108 | 0.50061 |
| R1100+S5500 | 0.60140 | 0.59030 |

출처: `EVAL/metric/synth_utility/*/eval_summary.json`, `eval_analysis/synth_utility_comparison.csv`. 고정 real test 5,000장, reliable 12-label macro, 각 arm 단일 seed.

- `src/Eval_metric/downstream_cls.py`: 동결한 XRV pretrained all/nih scorer. uncertain/missing 제외, 양성/음성 각각 10개 이상인 pathology를 macro에 포함. `[-1,1]*1024` 범위 변환은 맞다.
- `Eval_metric/clip_labelret.py`: 정확히 한 positive abnormality가 있는 study, `cos(image,"{label}")` one-vs-rest AUROC. No Finding/Support Devices 제외. 다른 병변의 unknown/uncertain을 확정 음성으로 검증하지 않으므로 일반 multilabel AUROC와 다르다.
- `Eval_metric/train_downstream_cls.py`: XRV pretrained 재학습이 아니라 **torchvision DenseNet121(weights=None) scratch 학습**. grayscale 3채널 복제, 224 resize, ±10° 회전, masked BCE, pos_weight=n_neg/n_pos(상한 20), Adam lr=1e-4, 30 epochs. validation/early stopping 없이 마지막 epoch 평가.

AUROC ratio 0.79는 0.506/0.641이며 높은 분류 성능이 아니다. FID가 좋아도 라벨이 틀린 샘플은 좋은 증강 데이터가 아닐 수 있다. 기존 기록의 good CLIP-zeroshot 표현은 절대 AUROC로 교정해야 한다.

## 실제 생성 모델 경로

NIH VAE → NIH class-label-conditioned LDM → MIMIC BioBERT-conditioned DP LoRA → prompt별 DDPM sampling.

실제 checkpoint CPU 검사:

- `checkpoints/vae/vae_ep0070.pt`: NIH, 256×256, ch_mult=[1,2,4,4,4], z_channels=1. latent는 1×16×16. loss_ver=vae, lambda_perc=1 (L1/SSIM/LPIPS/KL/MMD 계열; perceptual loss가 없다는 주장은 틀림).
- `checkpoints/ldm/ldm_epoch0100.pt`: NIH, emb_target=cls, context_dim=512, 1000-step epsilon prediction. embedder.* 키가 0개.
- `finetune_dp/JPG_LABEL_IMPRESSION/{eps1_ep30,eps10_ep30}/ldm_dp_final.pt`: rank=4/alpha=4, finetune_biobert=False, C=0.001, logical/physical batch=256/4, lr=0.002, warmup=500, EMA=False. epsilon_spent=0.91683194/8.81070823.
- trainable LoRA 345,088개(전체 약 0.103%). VAE/UNet backbone/BioBERT/projection은 동결. 16 SpatialTransformer 내 32 attention의 128 linear에 주입. 타입 기반 탐색이라 attn2 cross-attention뿐 아니라 attn1 self-attention도 포함한다.
- Opacus Poisson+BatchMemoryManager, scheduler는 logical update에서 진행. physical batch=4를 noise 적용 batch=4로 혼동하면 안 된다.
- frozen VAE posterior latent + random timestep noise MSE. 병변 classification/조건 일치 loss와 CFG/null-conditioning training은 현재 경로에 없다.

소스: `src/LDM_train.py:267,337,394`, `src/Data/cls_emb.py`, `src/Modules/BioBERT_embedder.py`, `src/Model/Diffusion/LDM_dp.py:107,190,258`, `src/Model/Diffusion/LDM.py:248`, `src/Model/lora.py:116`, `src/LDM_dp_finetune.py:918,966,1033`, `src/LDM_dp_inference.py:315,404`.

## 원인 우선순위와 개선

### 1. 조건 공간 전환과 무작위 projection 동결 (사실 확인, 주요 성능 원인 가설)

현재 NIH pretraining 소스는 model 밖의 ClassLabelEmbedder 출력을 detach하고 optimizer/checkpoint에 넣지 않는다. random label code도 일관되면 conditional 학습은 가능하지만 저장하지 않으면 재현/재사용이 어렵다.

DP에서는 pretrained BioBERT에 **새 Linear(768,512)**를 붙인다. pretrained LDM missing=201은 새 BioBERT/projection 전체 201키와 일치한다. projection은 로드되지 않은 무작위 초기화 상태로 동결한다. eps1/eps10 ep30 projection weight는 비트 단위로 같다. 즉 NIH label-code와 새 BioBERT 공간의 연결을 작은 rank-4 LoRA가 DP 아래에서 익혀야 한다.

우선 projection frozen/trainable ablation. projection 393,728개는 LoRA 345,088개보다 많으므로 DP에서 무조건 이득이라 가정하지 않는다. 장기적으로 공개/사용 가능한 사전학습 데이터에서 일관된 text 또는 structured-label conditioning을 정렬하고 encoder/projection도 저장한다. NIH에 없는 report를 가정하지 말고 공통 label template부터 연결한다.

### 2. Padding mask와 frozen BioBERT mode (실행 진단 확인)

BioBERT attention_mask는 BERT 내부에만 쓰고 UNet까지 전달하지 않는다. padded query output/projection은 0이 아니어서 UNet이 이를 조건처럼 읽는다. 학습은 길이가 다른 report 4개씩, 생성은 1개씩 처리하므로 padding 분포가 다르다.

CPU에서 현 BioBERTEmbedder와 작은 DP CrossAttention으로 재현: 짧은 문장은 단독 9 tokens, batch에서 32 tokens. 유효 token 최대 차이 1.61e-6, unmasked attention 출력 차이 0.04276, mask 적용 후 8.94e-8. 이는 새로 초기화한 작은 attention 진단이며 실제 checkpoint AUROC 영향 크기 측정은 아니다.

ldm.train()은 frozen BERT도 train mode로 바꾼다. requires_grad=False는 dropout을 끄지 않는다. 실제 p=0.1, 같은 입력 반복 context 최대 차이 0.96959 확인. dropout 자체가 반드시 해로운 것은 아니지만 고정 encoder 의도와 실행은 다르다.

개선: mask를 UNet/cross-attention까지 전달하고 frozen BERT만 eval로 고정. projection 학습 여부와 BERT mode를 분리한다. 수정은 기존 inference만 바꿔도 분포가 달라지므로 새 경로 재학습/평가를 별도로 기록한다.

### 3. 작은 VAE latent (구조 확인, 병변 보존 손실은 미측정)

256×256에서 1×16×16, 즉 256개 실수 latent로 압축한다(엄밀한 bit-rate 계산 아님). 미세 병변 손실 및 NIH→MIMIC domain shift 가능성이 있다.

먼저 같은 validation에서 real 원본 → VAE reconstruction(mode/sample) → conditional synthesis의 XRV/MedCLIP AUROC를 비교한다. reconstruction이 무너지면 VAE를 먼저, 유지되면 conditioning을 먼저 고친다. 필요 시 downsampling factor 8/latent channels 4 등을 비교하되 VAE/UNet 재학습과 checkpoint 비호환을 감안한다. 병변 보존을 VAE 선택 기준에 넣는다.

### 4. Noise MSE와 DP 제약 (분리 진단 필요)

Noise MSE는 전체 형태/질감이 좋아져도 내려갈 수 있고 작은 병변을 직접 감독하지 않는다. C=0.001은 점검 대상이나 per-sample norm/clipping fraction을 기록하지 않아 과도한 clipping을 확정할 수 없다.

동일 초기화/데이터의 non-DP, clipping-only, DP 비교 및 norm/clipping fraction/update norm/noise 대비 신호 진단을 제안한다. **현재 main은 use_dp 값과 별개로 make_private를 호출하므로 --use_dp False만으로 non-DP 대조군이 되지 않는다.** 여러 boolean parser도 문자열 False가 truthy가 될 수 있다.

C/lr은 같이 튜닝한다. sigma/sampling/steps가 같으면 C 변경은 clipping 기준과 absolute noise를 함께 바꾼다. C 증가가 privacy 자동 악화 또는 성능 자동 개선을 뜻하지 않는다. 공개/승인된 탐색 집합에서 진단하고 private 반복 탐색은 budget을 별도 관리한다. [Opacus FAQ](https://opacus.ai/docs/faq), [optimizer 구현](https://github.com/meta-pytorch/opacus/blob/main/opacus/optimizers/optimizer.py).

CFG는 conditioning dropout/null branch 학습 후 validation에서 scale을 비교한다. 기존 checkpoint에 scale만 추가하는 것으로 충분하지 않다. 다양성 감소도 확인한다. [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).

병변 auxiliary loss를 쓰면 별도 teacher와 독립 evaluator/downstream test를 유지하고 DP per-sample gradient 구조를 보존한다.

### 5. 합성 영상에 원본 라벨 부여, 합성 비중 83%

utility loader는 source study의 CheXpert 라벨을 합성 영상에 그대로 붙인다. 실제 생성 병변을 검증하지 않는다. 두 frozen scorer가 chance 부근이면 label-image noise를 의심해야 한다. synthetic-only real-test만으로 domain shift와 label mismatch를 완전히 분리할 수는 없다.

R1100+S5500은 83.3% 합성이므로 loss/BatchNorm 통계에 synthetic domain이 크게 기여한다. eps10 RS5500에서 cardiomegaly 0.7406→0.6111, edema 0.7599→0.6511, effusion 0.7368→0.6029. 일부 병변은 개선되므로 전 병변 악화는 아니다.

real batch를 유지하고 L=L_real+lambda_syn*L_syn으로 합성 기여를 분리한다. 낮은 lambda(예: 0.1/0.25/0.5), 저비중 RS, synthetic pretrain→real-only finetune을 비교한다. real-only/real augmentation/반복 대조군의 optimizer steps와 real exposure를 맞춘다. quality filtering은 독립 teacher/label별 기준을 쓰고 유지율·다양성·분포 변화를 기록한다.

### 6. Scratch classifier와 라벨 감독의 한계

1,100장 scratch, 단일 seed, 마지막 epoch 평가라 작은 차이는 불안정하다. 동일 30 epochs여도 R/RS1100/S5500/RS5500의 optimizer updates는 약 1,020/2,040/5,130/6,180이다. validation, 3+ seeds, patient-cluster bootstrap CI, pathology AUPRC를 추가하고 test는 튜닝에 쓰지 않는다. 실제 pretrained XRV adaptation은 별도 실험한다.

No Finding은 pool 5,500장 중 positive=950/explicit negative=0/missing=4,550이다. missing 제외+pos_weight=n_neg/n_pos이므로 가중치가 0이 되어 이 label의 직접 BCE 기여가 0이다. AUROC도 산출되지 않는다. Pleural Other도 support 기준 부족으로 macro 제외, 14-output이지만 12-label macro다.

Cardiomegaly pos/neg/missing=1,422/373/3,556, Support Devices=1,790/88/3,614로 명시 음성 감독이 희소하다. missing/uncertain을 조용히 0으로 바꾸지 말고 의미를 검토해 label별 정책을 사전 고정한 별도 실험으로 다룬다.

## 확인한 매핑 및 남은 한계

- pool 5,500 images, 5,271 studies, 3,261 patients. real test와 patient 교집합 0.
- eps1/eps10 각각 index unique=5,500, newline/strip 정규화 후 prompt 5,500/5,500 일치.
- eval 10,240 prompt LABEL prefix는 대응 manifest의 CheXpert positive와 10,240/10,240 일치.
- dataset max_length=512는 문자, BioBERT는 token. pool 382/5,500(6.95%)가 512자 경계. 손실된 원문 병변 정보는 미측정. LABEL은 앞에 있어 절단이 모든 문제의 설명은 아니다.
- square resize로 aspect ratio가 변한다. XRV 공식 예제의 center-crop+resize와 geometry 차이를 validation에서 비교한다. 범위 변환은 맞다. [TorchXRayVision](https://github.com/mlmed/torchxrayvision).
- manifest loader 자체는 AP/PA/lateral을 제한/conditioning하지 않는다. 실제 view 분포와 study report 공유 영향을 점검해야 한다. view 혼합이 주원인이라는 결론은 아직 아니다.
- 기존 ep30 생성의 adapter key/md5 검증은 기록에서 확인. 이번 init.sh도 missing=0/unexpected=0. 전체 기존 PNG를 이번에 다시 hash하지는 않았다.
- checkpoint 생성 당시 source와 현재 dirty working tree의 완전한 일치는 보장하지 않는다. 설정은 실제 checkpoint args에서 확인했다.

## 다음 실험과 완료 기준

1. 고정 validation의 원본/VAE reconstruction, 동일 noise의 병변 positive-negative prompt 교체, 정답/셔플 context의 timestep별 denoising loss로 VAE와 conditioning 병목 분리.
2. BERT eval/mask 수정, projection frozen/trainable 비교.
3. non-DP/clipping-only/DP 및 C/lr 비교로 DP 추가 손실 분리.
4. reconstruction 병목이면 큰 latent, 아니면 정렬된 conditioning, 그 후 CFG.
5. 저비중/가중 RS 및 synthetic→real finetune, scratch/pretrained 각각 3+ seeds와 real augmentation 대조군 비교.

조건 반응 확인 전 수천 장 재생성을 반복하지 않는다. 성공 기준은 FID 감소만이 아니라 독립 real holdout에서 real-only 대비 utility 개선이다. 병변별 support/AUROC/AUPRC, 다양성, epsilon을 함께 기록한다. 성능 개선은 아직 미검증이다.

## 이번 검증

./init.sh 통과: container running, compile, eval namespace imports, 1-image/50-step known-good LoRA smoke. 실제 checkpoint 4개 CPU 검사, 기존 결과 확인, 전체 utility prompt mapping/patient overlap/eval LABEL prefix 검사, 작은 CPU padding/dropout 재현 완료. 학습 모듈은 container /opt/conda/bin/python, 평가 데이터 검사는 /opt/medclip_env/bin/python을 사용했다. 전체 재생성/재평가 및 reconstruction AUROC는 실행하지 않았다.
