import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    """
    Total VAE loss:
        L_total = λ_rec * L_rec  +  λ_ssim * L_ssim  +  λ_kl * L_kl  +  λ_mmd * L_mmd

    Components
    ----------
    L_rec  : L1 reconstruction loss         ||x - x_hat||_1
    L_ssim : SSIM-based perceptual loss     1 - SSIM(x, x_hat)
    L_kl   : KL divergence from N(0, I)    0.5 * Σ(μ² + σ² - log σ² - 1)
    L_mmd  : MMD between q(z) and p(z)     using Gaussian kernel
    """

    def __init__(
        self,
        lambda_rec:  float = 1.0,
        lambda_ssim: float = 1.0,
        lambda_kl:   float = 1e-4,
        lambda_mmd:  float = 1e-3,
        ssim_window_size: int   = 11,
        ssim_sigma:       float = 1.5,
        mmd_sigma:        float = 1.0,
    ):
        super().__init__()
        self.lambda_rec  = lambda_rec
        self.lambda_ssim = lambda_ssim
        self.lambda_kl   = lambda_kl
        self.lambda_mmd  = lambda_mmd
        self.ssim_window_size = ssim_window_size
        self.ssim_sigma       = ssim_sigma
        self.mmd_sigma        = mmd_sigma

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Reconstruction Loss  (L1)
    # ──────────────────────────────────────────────────────────────────────────
    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """L_rec = ||x - x_hat||_1  (mean over all elements)"""
        return F.l1_loss(x_hat, x, reduction="mean")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. SSIM Loss
    # ──────────────────────────────────────────────────────────────────────────
    def ssim_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """L_ssim = 1 - SSIM(x, x_hat)"""
        return 1.0 - self._ssim(x, x_hat)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        SSIM(x, y) = (2μ_x μ_y + C1)(2σ_xy + C2)
                     ─────────────────────────────────
                     (μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)

        Uses a 2-D Gaussian kernel for local statistics.
        Returns the mean SSIM over the batch.
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        k  = self.ssim_window_size
        C  = x.shape[1]                      # number of channels

        kernel = self._gaussian_kernel_2d(k, self.ssim_sigma, C).to(x.device)
        pad    = k // 2

        mu_x  = F.conv2d(x,   kernel, padding=pad, groups=C)
        mu_y  = F.conv2d(y,   kernel, padding=pad, groups=C)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sig_x2 = F.conv2d(x * x, kernel, padding=pad, groups=C) - mu_x2
        sig_y2 = F.conv2d(y * y, kernel, padding=pad, groups=C) - mu_y2
        sig_xy = F.conv2d(x * y, kernel, padding=pad, groups=C) - mu_xy

        ssim_map = ((2.0 * mu_xy + C1) * (2.0 * sig_xy + C2)) / \
                   ((mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2))

        return ssim_map.mean()

    def _gaussian_kernel_2d(self, kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
        """Create a (channels, 1, k, k) separable Gaussian kernel."""
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        g = g / g.sum()
        kernel_2d = g.outer(g)                                   # (k, k)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)          # (1, 1, k, k)
        return kernel_2d.expand(channels, 1, kernel_size, kernel_size)  # (C, 1, k, k)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. KL Divergence Loss
    # ──────────────────────────────────────────────────────────────────────────
    def kl_loss(self, posterior) -> torch.Tensor:
        """
        L_kl = D_KL( q_phi(z|x) || p(z) )
             = 0.5 * Σ_i ( μ_i² + σ_i² - log(σ_i²) - 1 )

        DiagonalGaussianDistribution.kl() already computes this sum
        over spatial + channel dims and returns shape [B].
        """
        return posterior.kl().mean()

    # ──────────────────────────────────────────────────────────────────────────
    # 4. MMD Loss  (InfoVAE regularization)
    # ──────────────────────────────────────────────────────────────────────────
    def mmd_loss(self, z_q: torch.Tensor, z_p: torch.Tensor = None) -> torch.Tensor:
        """
        MMD²( q(z), p(z) ) =
            E_{z,z'~q}[ k(z,z') ] + E_{z,z'~p}[ k(z,z') ] - 2·E_{z~q,z'~p}[ k(z,z') ]

        Gaussian kernel:  k(z, z') = exp( -||z - z'||² / (2 σ_k²) )

        Args:
            z_q: latent samples from encoder  [B, C, H, W]
            z_p: prior samples N(0,I)         [B, C, H, W]  (sampled here if None)
        """
        if z_p is None:
            z_p = torch.randn_like(z_q)

        # Flatten to [B, D]
        z_q = z_q.reshape(z_q.size(0), -1)
        z_p = z_p.reshape(z_p.size(0), -1)

        k_qq = self._gaussian_kernel_mmd(z_q, z_q)   # [B, B]
        k_pp = self._gaussian_kernel_mmd(z_p, z_p)   # [B, B]
        k_qp = self._gaussian_kernel_mmd(z_q, z_p)   # [B, B]

        return k_qq.mean() + k_pp.mean() - 2.0 * k_qp.mean()

    def _gaussian_kernel_mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        k(x, y) = exp( -||x - y||² / (2 σ²) )

        Args:
            x: [N, D]
            y: [M, D]
        Returns:
            kernel matrix [N, M]
        """
        x_sq = (x ** 2).sum(dim=1, keepdim=True)    # [N, 1]
        y_sq = (y ** 2).sum(dim=1, keepdim=True).t() # [1, M]
        xy   = x @ y.t()                              # [N, M]
        dist2 = x_sq + y_sq - 2.0 * xy               # [N, M]
        return torch.exp(-dist2 / (2.0 * self.mmd_sigma ** 2))

    # ──────────────────────────────────────────────────────────────────────────
    # forward
    # ──────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x:         torch.Tensor,
        x_hat:     torch.Tensor,
        posterior,
        z_q:       torch.Tensor,
    ):
        """
        Args:
            x        : original image              [B, C, H, W]
            x_hat    : reconstructed image         [B, C, H, W]
            posterior: DiagonalGaussianDistribution
            z_q      : sampled latent z            [B, z_channels, h, w]

        Returns:
            total_loss (scalar Tensor),
            loss_dict  (dict of float values for logging)
        """
        l_rec  = self.reconstruction_loss(x, x_hat)
        l_ssim = self.ssim_loss(x, x_hat)
        l_kl   = self.kl_loss(posterior)
        l_mmd  = self.mmd_loss(z_q)

        total = (self.lambda_rec  * l_rec  +
                 self.lambda_ssim * l_ssim +
                 self.lambda_kl   * l_kl   +
                 self.lambda_mmd  * l_mmd)

        loss_dict = {
            "loss_total": total.item(),
            "loss_rec"  : l_rec.item(),
            "loss_ssim" : l_ssim.item(),
            "loss_kl"   : l_kl.item(),
            "loss_mmd"  : l_mmd.item(),
        }
        return total, loss_dict
