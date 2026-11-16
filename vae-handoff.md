# VAE Architecture Figure Handoff

## Purpose

This document contains only the context needed to continue drawing and documenting the
VAE architecture used in PrivaText-CXR. It is intended to be supplied to a new Codex
Web/Cloud session so that the next session does not have to rediscover the implementation.

## Repository and Runtime

- Host repository: `/home/hjchoi/PycharmProjects/PrivaText-CXR`
- Container repository: `/workspace/PrivaText-CXR`
- Required Docker container: `CXR_dp_medclip`
- Python for VAE/model inspection: `/opt/conda/bin/python`
- Active branch during inspection: `claude/fid-multi-model-eval-l7hpk4`
- Standard startup smoke: `./init.sh`
- `./init.sh` passed on 2026-11-02, including compilation, eval imports, and one-image
  generation.

## Scope of the Completed Inspection

The architecture below was derived from the implementation and cross-checked against the
checkpoint actually used by the latent diffusion pipeline:

- Checkpoint: `checkpoints/vae/vae_ep0070.pt`
- Checkpoint epoch: 70
- Encoder implementation: `src/Model/VAE/Encoder.py`
- Decoder implementation: `src/Model/VAE/Decoder.py`
- VAE wrapper: `src/Model/VAE/Autoencoder.py`
- Residual/downsample/upsample blocks: `src/Model/UNetBlock_module.py`
- Swin attention: `src/Model/swin_attention.py`
- VAE loss: `src/Loss/vae_loss.py`

Do not infer the paper figure only from current argparse defaults. The checkpoint contains
the authoritative architecture and loss arguments; some current training defaults differ
from those saved in the epoch-70 checkpoint.

## Verified Checkpoint Configuration

```text
Input/output channels:     1 / 1
Input resolution:          256 x 256
Base channels:             128
Channel multipliers:       [1, 2, 4, 4, 4]
Channels by level:         [128, 256, 512, 512, 512]
Residual blocks per level: 2
Attention resolutions:     [32, 16]
Latent channels:           1
double_z:                  True
Dropout:                   0.0
Convolution dimensions:    2-D
Downsampling:              strided 3 x 3 convolution
Upsampling:                nearest-neighbor x2 followed by 3 x 3 convolution
```

The encoder produces two latent channels at `16 x 16`, which are split into posterior
mean and log variance. Reparameterization then produces a one-channel `16 x 16` latent.

## Main Architecture

```text
Input chest X-ray: 1 x 256 x 256
    |
    v
3x3 Conv: 1 -> 128
    |
    v
Encoder L0: ResBlock x2                         -> 128 x 256 x 256
    |
    v  Downsample (3x3 Conv, stride 2)
Encoder L1: ResBlock(128->256) + ResBlock       -> 256 x 128 x 128
    |
    v  Downsample
Encoder L2: ResBlock(256->512) + ResBlock       -> 512 x 64 x 64
    |
    v  Downsample
Encoder L3: [ResBlock -> Swin] x2               -> 512 x 32 x 32
    |
    v  Downsample
Encoder L4: [ResBlock -> Swin] x2               -> 512 x 16 x 16
    |
    v
Middle: ResBlock -> Swin -> ResBlock            -> 512 x 16 x 16
    |
    v
GroupNorm -> SiLU -> 3x3 Conv (512->2)          -> 2 x 16 x 16
    |
    +--> mean (mu):       1 x 16 x 16
    +--> log variance:    1 x 16 x 16
    |
    v
z = mu + exp(0.5 * logvar) * epsilon,
epsilon ~ N(0, I)                               -> 1 x 16 x 16
    |
    v
1x1 post-quantization Conv (1->1)
    |
    v
3x3 Conv: 1 -> 512                             -> 512 x 16 x 16
    |
    v
Middle: ResBlock -> Swin -> ResBlock
    |
    v
Decoder L4: [ResBlock -> Swin] x2               -> 512 x 16 x 16
    |
    v  Upsample
Decoder L3: [ResBlock -> Swin] x2               -> 512 x 32 x 32
    |
    v  Upsample
Decoder L2: ResBlock x2                         -> 512 x 64 x 64
    |
    v  Upsample
Decoder L1: ResBlock(512->256) + ResBlock       -> 256 x 128 x 128
    |
    v  Upsample
Decoder L0: ResBlock(256->128) + ResBlock       -> 128 x 256 x 256
    |
    v
GroupNorm -> SiLU -> 3x3 Conv (128->1)
    |
    v
Reconstructed chest X-ray: 1 x 256 x 256
```

There are four spatial downsampling and four upsampling operations:

```text
256 -> 128 -> 64 -> 32 -> 16
16  -> 32  -> 64 -> 128 -> 256
```

There are no U-Net-style long skip connections between the encoder and decoder. Residual
connections exist only inside the ResBlocks and Swin Transformer blocks.

## ResBlock Inset

Use the following structure for a magnified ResBlock inset:

```text
Main branch:
Input -> GroupNorm(32) -> SiLU -> 3x3 Conv
      -> GroupNorm(32) -> SiLU -> Dropout(p=0)
      -> zero-initialized 3x3 Conv -> Add -> Output

Skip branch:
Input -------------------------------------> Add
      or 1x1 Conv when input/output channels differ
```

The VAE ResBlocks have no timestep embedding.

## Swin Transformer Inset

```text
Input
  -> LayerNorm
  -> 8x8 window multi-head self-attention
  -> residual addition
  -> LayerNorm
  -> MLP (C -> 4C -> C, GELU)
  -> residual addition
  -> Output
```

Verified Swin settings:

- Window size: `8 x 8`
- Attention heads: 8
- MLP ratio: 4
- QKV bias: enabled
- Relative position bias: enabled
- First attention block at a level: shift `(0, 0)`
- Second attention block at a level: shift `(4, 4)`
- Middle attention block: shift `(0, 0)`

Attention is used only at spatial resolutions `32 x 32` and `16 x 16`, plus the middle
bottleneck at `16 x 16`.

## Latent Posterior

Use the following notation in the figure or caption:

```text
q_phi(z|x) = N(mu_phi(x), diag(sigma_phi(x)^2))
z = mu + exp(0.5 * logvar) * epsilon,  epsilon ~ N(0, I)
```

The latent tensor has shape `1 x 16 x 16`. Relative to the `1 x 256 x 256` input, the
spatial resolution is reduced by a factor of 16 along each axis and the scalar element
count is reduced by a factor of 256.

## Verified Training Objective for `vae_ep0070.pt`

The saved checkpoint arguments specify:

```text
loss_ver       = vae
lambda_rec     = 1.0
lambda_ssim    = 1.0
lambda_perc    = 1.0
lambda_kl      = 1e-4
lambda_mmd     = 1e-3
```

The corresponding figure/caption notation is:

```text
L_VAE = L_L1 + L_SSIM + L_LPIPS + 1e-4 L_KL + 1e-3 L_MMD
L_SSIM = 1 - SSIM(x, x_hat)
```

The final decoder projection has no sigmoid or tanh activation. It ends with
`GroupNorm -> SiLU -> 3x3 Conv`, and reconstruction losses are evaluated against images
normalized to `[-1, 1]`.

## Recommended Figure Layout

Draw the main model horizontally:

1. Grayscale CXR input on the left.
2. Five blue encoder stages with decreasing spatial size.
3. Purple Swin blocks only at the `32 x 32` and `16 x 16` stages.
4. Green posterior node showing `mu`, `log variance`, and stochastic sampling.
5. One narrow latent block labelled `z: 1 x 16 x 16`.
6. A symmetric decoder on the right.
7. Reconstructed grayscale CXR at the far right.
8. ResBlock and Swin block details as two small insets below the main architecture.

Suggested colors:

- Blue: convolutional ResBlocks
- Orange: downsampling/upsampling
- Purple: Swin Transformer blocks
- Green: posterior and latent variables
- Gray: input and reconstructed radiographs
- Dashed arrow: stochastic reparameterization path

Do not draw a long encoder-to-decoder skip arrow.

## Draft Figure Caption

> Architecture of the variational autoencoder used for latent chest X-ray
> representation. A 256 x 256 single-channel chest radiograph is encoded through five
> residual stages into the parameters of a spatial Gaussian posterior with a resolution
> of 16 x 16 and one latent channel. Swin Transformer blocks with 8 x 8 windows and eight
> attention heads are applied at the 32 x 32 and 16 x 16 resolutions and in the
> bottleneck. A latent sample is obtained using the reparameterization trick and
> reconstructed by a symmetric decoder with residual blocks and four upsampling
> operations. No long-range skip connections are used between the encoder and decoder.

## Verification Command

To re-check the saved checkpoint settings inside the required container:

```bash
docker exec CXR_dp_medclip sh -lc '
  cd /workspace/PrivaText-CXR &&
  /opt/conda/bin/python -B -c "
import torch
c = torch.load(\"checkpoints/vae/vae_ep0070.pt\", map_location=\"cpu\")
keys = [
    \"in_channels\", \"out_channels\", \"ch\", \"ch_mult\",
    \"num_res_blocks\", \"attn_resolutions\", \"resolution\",
    \"z_channels\", \"double_z\", \"dropout\", \"resamp_with_conv\",
    \"loss_ver\", \"lambda_rec\", \"lambda_ssim\", \"lambda_perc\",
    \"lambda_kl\", \"lambda_mmd\"
]
print(\"epoch\", c.get(\"epoch\"))
print({k: c.get(\"args\", {}).get(k) for k in keys})
"'
```

## Prompt for a New Codex Web/Cloud Session

```text
Open the PrivaText-CXR repository and read AGENTS.md, then read vae-handoff.md in full.
Use vae-handoff.md as the source of truth for the VAE architecture figure. Do not modify
the VAE implementation. Help me produce a publication-quality architecture figure while
preserving every tensor shape, attention location, and the absence of encoder-decoder skip
connections documented there. If implementation details need verification, inspect the
listed source files and checkpoints/vae/vae_ep0070.pt inside CXR_dp_medclip.
```

## Remaining Work

- Choose the final publication format: vector SVG/PDF, draw.io, PowerPoint, Mermaid, or
  TikZ.
- Draw the architecture based on this handoff.
- Check readability at the target paper column width.
- Decide whether the loss equation belongs in the main architecture figure or only in the
  caption/method section.
- Export a vector version and a high-resolution raster preview.

