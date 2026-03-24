"""
VAELoss
=======
L_total = λ_rec  * ||x - x̂||₁
        + λ_ssim * (1 - SSIM(x, x̂))          # via torchmetrics functional
        + λ_kl   * D_KL(q(z|x) || N(0,I))
        + λ_mmd  * MMD(q(z), p(z))             # Gaussian kernel (InfoVAE)
        + λ_perc * LPIPS(x, x̂)               # learned perceptual (VGG)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# torchmetrics functional API — stateless, no accumulation side-effects
try:
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
except ImportError:  # older torchmetrics
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn


class VAELoss(nn.Module):
    """
    Args
    ----
    lambda_rec   : weight for L1 reconstruction loss
    lambda_ssim  : weight for SSIM loss
    lambda_kl    : weight for KL divergence
    lambda_mmd   : weight for MMD regularisation
    lambda_perc  : weight for LPIPS perceptual loss; 0.0 = disabled
    mmd_sigma    : bandwidth σ_k of the Gaussian MMD kernel
    data_range   : value range of images (2.0 for [-1, 1] normalised input)
    """
    def __init__(self, args, device):
        super().__init__()
        self.device     = device
        self.lambda_rec  = args.lambda_rec
        self.lambda_ssim = args.lambda_ssim
        self.lambda_kl   = args.lambda_kl
        self.lambda_mmd  = args.lambda_mmd
        self.lambda_perc = args.lambda_perc
        self.mmd_sigma   = args.mmd_sigma
        self.data_range  = args.data_range

        # Load LPIPS only when needed — avoids unnecessary checkpoint download
        if self.lambda_perc > 0.0:
            from Loss.lpips import LPIPS
            self.lpips = LPIPS().eval()
        else:
            self.lpips = None

    # ── 1. Reconstruction (L1) ────────────────────────────────────────────────
    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """L_rec = mean( |x - x̂| )"""
        return F.l1_loss(x_hat, x, reduction='mean')

    # ── 2. SSIM Loss ──────────────────────────────────────────────────────────
    def ssim_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        L_ssim = 1 - SSIM(x, x̂)
        data_range must match image normalisation; default 2.0 for [-1, 1].
        """
        ssim_val = ssim_fn(x_hat, x, data_range=self.data_range)
        return 1.0 - ssim_val

    # ── 3. LPIPS Perceptual Loss ───────────────────────────────────────────────
    def perceptual_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        L_perc = LPIPS(x, x̂)

        LPIPS.forward() returns [B, 1, 1, 1]; reduced to scalar via .mean().
        Grayscale inputs (C=1) are expanded to 3 channels before VGG.
        Returns tensor(0.0) — no gradient — when lambda_perc == 0.0.
        """
        if self.lpips is None:
            return torch.tensor(0.0, device=x.device)

        # Grayscale → pseudo-RGB for VGG ScalingLayer (expects 3 channels)
        if x.shape[1] == 1:
            x     = x.repeat(1, 3, 1, 1)
            x_hat = x_hat.repeat(1, 3, 1, 1)

        # [B, 1, 1, 1] → scalar
        return self.lpips(x.contiguous(), x_hat.contiguous()).mean()

    # ── 4. KL Divergence ──────────────────────────────────────────────────────
    def kl_loss(self, posterior) -> torch.Tensor:
        """DiagonalGaussianDistribution.kl() returns shape [B]; we average."""
        return posterior.kl().mean()

    # ── 5. MMD Loss (InfoVAE) ─────────────────────────────────────────────────
    def mmd_loss(self, z_q: torch.Tensor, z_p: torch.Tensor = None) -> torch.Tensor:
        """
        MMD(q(z), p(z)) = E[k(z,z')] + E[k(z̃,z̃')] - 2·E[k(z,z̃)]
        k(z, z') = exp( -||z - z'||² / (2σ²) )
        z_p defaults to N(0, I) samples of the same shape as z_q.
        """
        if z_p is None:
            z_p = torch.randn_like(z_q)

        z_q = z_q.reshape(z_q.size(0), -1)   # [B, D] = [B, C*H*W]
        z_p = z_p.reshape(z_p.size(0), -1)   # [B, D]

        k_qq = self._gaussian_kernel(z_q, z_q)   # q(z) self-similarity
        k_pp = self._gaussian_kernel(z_p, z_p)   # p(z) self-similarity
        k_qp = self._gaussian_kernel(z_q, z_p)   # cross-similarity

        return k_qq.mean() + k_pp.mean() - 2.0 * k_qp.mean()

    def _gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """k(x, y) = exp(-||x-y||² / 2σ²),  x:[N,D] y:[M,D] → [N,M]"""
        x_sq  = (x ** 2).sum(1, keepdim=True)          # [N, 1]
        y_sq  = (y ** 2).sum(1, keepdim=True).t()      # [1, M]
        dist2 = x_sq + y_sq - 2.0 * (x @ y.t())       # [N, M]
        return torch.exp(-dist2 / (2.0 * self.mmd_sigma ** 2))

    # ── 6. Shared computation ─────────────────────────────────────────────────
    def _compute(self, x, x_hat, posterior, z_q):
        l_rec  = self.reconstruction_loss(x, x_hat)
        l_ssim = self.ssim_loss(x, x_hat)
        l_perc = self.perceptual_loss(x, x_hat)   # 0.0 when lpips is None
        l_kl   = self.kl_loss(posterior)
        l_mmd  = self.mmd_loss(z_q)

        total = (
            self.lambda_rec  * l_rec
            + self.lambda_ssim * l_ssim
            + self.lambda_perc * l_perc
            + self.lambda_kl   * l_kl
            + self.lambda_mmd  * l_mmd
        )
        return total, l_rec, l_ssim, l_perc, l_kl, l_mmd

    # ── 7. Forward ────────────────────────────────────────────────────────────
    def forward(self, x, x_hat, posterior, z_q):
        """
        Returns
        -------
        total     : scalar loss tensor (differentiable)
        loss_dict : dict of float values for logging
        """
        total, l_rec, l_ssim, l_perc, l_kl, l_mmd = self._compute(x, x_hat, posterior, z_q)
        return total, {
            "loss_total": total.item(),
            "loss_rec":   l_rec.item(),
            "loss_ssim":  l_ssim.item(),
            "loss_perc":  l_perc.item(),
            "loss_kl":    l_kl.item(),
            "loss_mmd":   l_mmd.item(),
        }

    # ── 8. Debug forward ──────────────────────────────────────────────────────
    def debug_forward(self, x, x_hat, posterior, z_q):
        """
        Same as forward but prints a detailed breakdown of every loss term
        including tensor shapes, value ranges, and weighted contributions.
        """
        total, l_rec, l_ssim, l_perc, l_kl, l_mmd = self._compute(x, x_hat, posterior, z_q)

        W = 65
        print("\n" + "=" * W)
        print("  [VAELoss Debug]")
        print("-" * W)
        print(f"  {'x (input)':<18}: shape={tuple(x.shape)}"
              f"  range=[{x.min():.3f}, {x.max():.3f}]")
        print(f"  {'x_hat (recon)':<18}: shape={tuple(x_hat.shape)}"
              f"  range=[{x_hat.min():.3f}, {x_hat.max():.3f}]")
        print(f"  {'z_q (latent)':<18}: shape={tuple(z_q.shape)}"
              f"  mean={z_q.mean():.4f}  std={z_q.std():.4f}")
        print(f"  {'posterior mean':<18}: shape={tuple(posterior.mean.shape)}"
              f"  mean.mean={posterior.mean.mean():.4f}")
        print(f"  {'posterior std':<18}: shape={tuple(posterior.std.shape)}"
              f"  std.mean={posterior.std.mean():.4f}")
        print("-" * W)
        print(f"  {'Term':<12}  {'weight':>8}  {'raw value':>12}  {'weighted':>12}")
        print(f"  {'-' * 12}  {'-' * 8}  {'-' * 12}  {'-' * 12}")
        print(f"  {'L_rec':<12}  {self.lambda_rec:>8.4f}  {l_rec.item():>12.6f}"
              f"  {(self.lambda_rec * l_rec).item():>12.6f}")
        print(f"  {'L_ssim':<12}  {self.lambda_ssim:>8.4f}  {l_ssim.item():>12.6f}"
              f"  {(self.lambda_ssim * l_ssim).item():>12.6f}")
        print(f"  {'L_perc(LPIPS)':<12}  {self.lambda_perc:>8.4f}  {l_perc.item():>12.6f}"
              f"  {(self.lambda_perc * l_perc).item():>12.6f}")
        print(f"  {'L_kl':<12}  {self.lambda_kl:>8.4f}  {l_kl.item():>12.6f}"
              f"  {(self.lambda_kl * l_kl).item():>12.6f}")
        print(f"  {'L_mmd':<12}  {self.lambda_mmd:>8.4f}  {l_mmd.item():>12.6f}"
              f"  {(self.lambda_mmd * l_mmd).item():>12.6f}")
        print("-" * W)
        print(f"  {'L_total':<12}                            {total.item():>12.6f}")
        print("=" * W + "\n")

        return total, {
            "loss_total": total.item(),
            "loss_rec":   l_rec.item(),
            "loss_ssim":  l_ssim.item(),
            "loss_perc":  l_perc.item(),
            "loss_kl":    l_kl.item(),
            "loss_mmd":   l_mmd.item(),
        }
