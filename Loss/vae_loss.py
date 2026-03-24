"""
InfoVAELoss — MMD-based InfoVAE for Medical Chest X-Ray VAE
============================================================

InfoVAE paper (Zhao et al., 2017) proposes a generalised ELBO:

    L_InfoVAE = E[log p(x|z)]
              - (1 - α)  · KL( q(z|x) || p(z) )
              - (α + λ - 1) · D( q(z) || p(z) )          [eq. 7]

Converting to a minimisation objective and substituting:
  · -E[log p(x|z)]  → L_rec + L_ssim + L_perc   (L1 + SSIM + LPIPS)
  · D( q(z) || p(z) ) → MMD                      (tractable divergence estimate)

gives the final training loss:

    L_total = λ_rec  · L_rec
            + λ_ssim · L_ssim
            + λ_perc · L_perc
            + (1 - α)          · KL( q(z|x) || p(z) )
            + (α + λ_info - 1) · MMD( q(z), p(z) )

where
    L_rec   = mean( |x - x̂| )                              (L1)
    L_ssim  = 1 - SSIM(x, x̂)                              (structural)
    L_perc  = LPIPS(x, x̂)                                  (learned perceptual)
    KL      = 0.5 · Σ( μ² + σ² - log σ² - 1 )            (closed-form)
    MMD     = E[k(z,z')] + E[k(z̃,z̃')] - 2·E[k(z,z̃)]    (Gaussian kernel)

Parameter intuition
-------------------
α (alpha)       : controls how much weight is shifted from KL to MMD.
                  α=0  → standard β-VAE-like (pure KL regularisation)
                  α=1  → pure MMD-VAE (KL weight = 0)
λ_info          : overall strength of the InfoVAE divergence term.
                  λ_info=1, α=0  → standard VAE (MMD weight = 0)
                  λ_info>1       → extra pressure on q(z) ≈ p(z)
λ_rec, λ_ssim   : independent reconstruction weights (not part of InfoVAE,
                  added for medical-imaging perceptual quality).
λ_perc          : LPIPS perceptual loss weight. Set to 0.0 to disable
                  (avoids loading LPIPS model entirely).

Requires
--------
    pip install torchmetrics torchvision taming-transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# torchmetrics functional API — stateless, no per-epoch accumulation side-effects
try:
    from torchmetrics.functional.image import structural_similarity_index_measure as _ssim_fn
except ImportError:                                         # older torchmetrics (<0.11)
    from torchmetrics.functional import structural_similarity_index_measure as _ssim_fn


class InfoVAELoss(nn.Module):
    """
    InfoVAE loss with explicit α / λ_info coefficient structure.

    Args
    ----
    lambda_rec   : weight for L1 reconstruction loss  (λ_rec)
    lambda_ssim  : weight for SSIM perceptual loss    (λ_ssim)
    lambda_perc  : weight for LPIPS perceptual loss   (λ_perc); 0.0 = disabled
    alpha        : InfoVAE α — shifts regularisation weight from KL → MMD
    lambda_info  : InfoVAE λ — overall divergence regularisation strength
    mmd_sigma    : Gaussian kernel bandwidth σ for MMD
    data_range   : pixel value range passed to SSIM
                   (2.0 for images normalised to [-1, 1])

    Effective regularisation weights (derived, not free parameters)
    ---------------------------------------------------------------
    KL  weight = (1 - alpha)
    MMD weight = (alpha + lambda_info - 1)
    """

    def __init__(
        self,
        lambda_rec:  float = 1.0,
        lambda_ssim: float = 1.0,
        lambda_perc: float = 0.0,
        alpha:       float = 0.0,
        lambda_info: float = 1.0,
        mmd_sigma:   float = 1.0,
        data_range:  float = 2.0,
    ):
        super().__init__()
        self.lambda_rec  = lambda_rec
        self.lambda_ssim = lambda_ssim
        self.lambda_perc = lambda_perc
        self.alpha       = alpha
        self.lambda_info = lambda_info
        self.mmd_sigma   = mmd_sigma
        self.data_range  = data_range

        # Load LPIPS only when needed — avoids unnecessary checkpoint download
        if lambda_perc > 0.0:
            from Loss.lpips import LPIPS
            self.lpips = LPIPS().eval()
        else:
            self.lpips = None

    # ── 1. Reconstruction (L1) ────────────────────────────────────────────────
    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """L_rec = mean( |x - x̂| )  — pixel-wise L1, averaged over the batch."""
        return F.l1_loss(x_hat, x, reduction="mean")

    # ── 2. SSIM loss ──────────────────────────────────────────────────────────
    def ssim_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        L_ssim = 1 - SSIM(x, x̂)

        SSIM ∈ [0, 1], so this loss ∈ [0, 1].
        Uses torchmetrics stateless functional API — safe to call per batch.
        data_range must match image normalisation; default 2.0 for [-1, 1].
        """
        ssim_val = _ssim_fn(x_hat, x, data_range=self.data_range)
        return 1.0 - ssim_val

    # ── 3. LPIPS perceptual loss ──────────────────────────────────────────────
    def perceptual_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        L_perc = LPIPS(x, x̂)

        LPIPS.forward() returns [B, 1, 1, 1]; we reduce to a scalar via .mean().
        Grayscale inputs (C=1) are expanded to 3 channels before passing to VGG.
        Returns 0.0 (no gradient) when lambda_perc == 0.0 (LPIPS not loaded).
        """
        if self.lpips is None:
            return torch.tensor(0.0, device=x.device)

        # Grayscale → pseudo-RGB for VGG
        if x.shape[1] == 1:
            x     = x.repeat(1, 3, 1, 1)
            x_hat = x_hat.repeat(1, 3, 1, 1)

        # LPIPS returns [B, 1, 1, 1] — reduce to scalar
        return self.lpips(x.contiguous(), x_hat.contiguous()).mean()

    # ── 4. KL divergence ──────────────────────────────────────────────────────
    def kl_loss(self, posterior) -> torch.Tensor:
        """
        KL( q(z|x) || N(0, I) ) — closed-form solution for diagonal Gaussian:
            = 0.5 · Σ_i ( μ_i² + σ_i² - log(σ_i²) - 1 )

        posterior.kl() is expected to return per-sample KL of shape [B];
        we average over the batch.
        """
        return posterior.kl().mean()

    # ── 5. MMD — aggregate posterior vs prior ─────────────────────────────────
    def mmd_loss(self, z_q: torch.Tensor, z_p: torch.Tensor = None) -> torch.Tensor:
        """
        Unbiased MMD² estimate using a Gaussian kernel:
            MMD²(q, p) = E_{z,z'~q}[k(z,z')]
                       + E_{z,z'~p}[k(z,z')]
                       - 2 · E_{z~q, z'~p}[k(z,z')]

        z_q  : latent samples from the encoder  ( aggregate posterior q(z) )
        z_p  : prior samples from N(0, I); if None, sampled automatically.
        """
        if z_p is None:
            z_p = torch.randn_like(z_q)

        z_q = z_q.reshape(z_q.size(0), -1)   # [B, D]
        z_p = z_p.reshape(z_p.size(0), -1)   # [B, D]

        k_qq = self._gaussian_kernel(z_q, z_q)
        k_pp = self._gaussian_kernel(z_p, z_p)
        k_qp = self._gaussian_kernel(z_q, z_p)

        return k_qq.mean() + k_pp.mean() - 2.0 * k_qp.mean()

    def _gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """k(x, y) = exp( -‖x - y‖² / (2σ²) ),  x:[N,D] y:[M,D] → [N,M]"""
        x_sq  = (x ** 2).sum(1, keepdim=True)
        y_sq  = (y ** 2).sum(1, keepdim=True).t()
        dist2 = x_sq + y_sq - 2.0 * (x @ y.t())
        return torch.exp(-dist2 / (2.0 * self.mmd_sigma ** 2))

    # ── 6. InfoVAE coefficient structure ──────────────────────────────────────
    def infovae_weights(self) -> tuple:
        """
        Derive effective KL and MMD weights from the InfoVAE hyperparameters.

        KL  weight = (1 - α)
        MMD weight = (α + λ_info - 1)
        """
        kl_weight  = 1.0 - self.alpha
        mmd_weight = self.alpha + self.lambda_info - 1.0
        return kl_weight, mmd_weight

    # ── 7. Shared computation (reused by forward and debug_forward) ───────────
    def _compute(self, x, x_hat, posterior, z_q):
        l_rec  = self.reconstruction_loss(x, x_hat)
        l_ssim = self.ssim_loss(x, x_hat)
        l_perc = self.perceptual_loss(x, x_hat)   # 0.0 scalar when lpips is None
        l_kl   = self.kl_loss(posterior)
        l_mmd  = self.mmd_loss(z_q)

        kl_weight, mmd_weight = self.infovae_weights()

        total = (
            self.lambda_rec  * l_rec
            + self.lambda_ssim * l_ssim
            + self.lambda_perc * l_perc
            + kl_weight        * l_kl
            + mmd_weight       * l_mmd
        )
        return total, l_rec, l_ssim, l_perc, l_kl, l_mmd, kl_weight, mmd_weight

    # ── 8. Forward ────────────────────────────────────────────────────────────
    def forward(self, x, x_hat, posterior, z_q):
        """
        Parameters
        ----------
        x         : original input image  [B, C, H, W]
        x_hat     : reconstructed image   [B, C, H, W]
        posterior : DiagonalGaussianDistribution (must implement .kl() → [B])
        z_q       : reparameterised latent sample  [B, D, ...]

        Returns
        -------
        total     : scalar loss tensor (differentiable; use for .backward())
        loss_dict : dict of float scalars for logging / TensorBoard
        """
        total, l_rec, l_ssim, l_perc, l_kl, l_mmd, kl_w, mmd_w = self._compute(
            x, x_hat, posterior, z_q
        )
        return total, {
            "loss_total"        : total.item(),
            "loss_rec"          : l_rec.item(),
            "loss_ssim"         : l_ssim.item(),
            "loss_perc"         : l_perc.item(),
            "loss_kl"           : l_kl.item(),
            "loss_mmd"          : l_mmd.item(),
            "weight_rec"        : self.lambda_rec,
            "weight_ssim"       : self.lambda_ssim,
            "weight_perc"       : self.lambda_perc,
            "alpha"             : self.alpha,
            "lambda_info"       : self.lambda_info,
            "weight_kl_eff"     : kl_w,
            "weight_mmd_eff"    : mmd_w,
        }

    # ── 9. Debug forward ──────────────────────────────────────────────────────
    def debug_forward(self, x, x_hat, posterior, z_q):
        """
        Identical to forward() but also prints a detailed breakdown:
          - tensor shapes and value ranges
          - InfoVAE α / λ values and derived effective weights
          - per-term raw values, weights, and weighted contributions
          - total loss
        """
        total, l_rec, l_ssim, l_perc, l_kl, l_mmd, kl_w, mmd_w = self._compute(
            x, x_hat, posterior, z_q
        )

        W = 78
        print("\n" + "=" * W)
        print("  [InfoVAELoss Debug]")
        print("-" * W)
        print(f"  {'x (input)':<22}: shape={tuple(x.shape)}"
              f"  range=[{x.min():.4f}, {x.max():.4f}]")
        print(f"  {'x_hat (recon)':<22}: shape={tuple(x_hat.shape)}"
              f"  range=[{x_hat.min():.4f}, {x_hat.max():.4f}]")
        print(f"  {'z_q (latent)':<22}: shape={tuple(z_q.shape)}"
              f"  mean={z_q.mean():.4f}  std={z_q.std():.4f}")
        print(f"  {'posterior.mean':<22}: shape={tuple(posterior.mean.shape)}"
              f"  mean={posterior.mean.mean():.4f}")
        print(f"  {'posterior.std':<22}: shape={tuple(posterior.std.shape)}"
              f"  mean={posterior.std.mean():.4f}")
        print("-" * W)
        print(f"  α (alpha)       = {self.alpha:.6f}")
        print(f"  λ (lambda_info) = {self.lambda_info:.6f}")
        print(f"  KL  weight      = (1 - α)              = {kl_w:.6f}")
        print(f"  MMD weight      = (α + λ_info - 1)     = {mmd_w:.6f}")
        print("-" * W)
        print(f"  {'Term':<14} {'Raw':>14} {'Weight':>14} {'Weighted':>14}")
        print(f"  {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
        print(f"  {'L_rec':<14} {l_rec.item():>14.6f}"
              f" {self.lambda_rec:>14.6f} {(self.lambda_rec * l_rec).item():>14.6f}")
        print(f"  {'L_ssim':<14} {l_ssim.item():>14.6f}"
              f" {self.lambda_ssim:>14.6f} {(self.lambda_ssim * l_ssim).item():>14.6f}")
        print(f"  {'L_perc(LPIPS)':<14} {l_perc.item():>14.6f}"
              f" {self.lambda_perc:>14.6f} {(self.lambda_perc * l_perc).item():>14.6f}")
        print(f"  {'L_kl':<14} {l_kl.item():>14.6f}"
              f" {kl_w:>14.6f} {(kl_w * l_kl).item():>14.6f}")
        print(f"  {'L_mmd':<14} {l_mmd.item():>14.6f}"
              f" {mmd_w:>14.6f} {(mmd_w * l_mmd).item():>14.6f}")
        print("-" * W)
        print(f"  {'L_total':<14} {'':>14} {'':>14} {total.item():>14.6f}")
        print("=" * W + "\n")

        return total, {
            "loss_total"        : total.item(),
            "loss_rec"          : l_rec.item(),
            "loss_ssim"         : l_ssim.item(),
            "loss_perc"         : l_perc.item(),
            "loss_kl"           : l_kl.item(),
            "loss_mmd"          : l_mmd.item(),
            "weight_rec"        : self.lambda_rec,
            "weight_ssim"       : self.lambda_ssim,
            "weight_perc"       : self.lambda_perc,
            "alpha"             : self.alpha,
            "lambda_info"       : self.lambda_info,
            "weight_kl_eff"     : kl_w,
            "weight_mmd_eff"    : mmd_w,
        }


# ── Example / smoke-test ──────────────────────────────────────────────────────
if __name__ == "__main__":

    class DummyPosterior:
        """Minimal stand-in for DiagonalGaussianDistribution."""

        def __init__(self, mean, logvar):
            self.mean   = mean
            self.logvar = logvar
            self.std    = torch.exp(0.5 * logvar)

        def kl(self) -> torch.Tensor:
            kl = -0.5 * (1 + self.logvar - self.mean.pow(2) - self.logvar.exp())
            return kl.view(kl.size(0), -1).sum(dim=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, H, W  = 4, 1, 128, 128
    latent_dim  = 64

    x     = torch.rand(B, C, H, W, device=device) * 2 - 1
    x_hat = torch.rand(B, C, H, W, device=device) * 2 - 1

    mu     = torch.randn(B, latent_dim, device=device)
    logvar = torch.randn(B, latent_dim, device=device)
    posterior = DummyPosterior(mu, logvar)

    eps = torch.randn_like(mu)
    z_q = mu + eps * posterior.std

    criterion = InfoVAELoss(
        lambda_rec  = 1.0,
        lambda_ssim = 0.5,
        lambda_perc = 0.1,    # LPIPS perceptual loss weight
        alpha       = 0.5,
        lambda_info = 1.2,
        mmd_sigma   = 1.0,
        data_range  = 2.0,
    ).to(device)

    total, loss_dict = criterion.debug_forward(x, x_hat, posterior, z_q)
