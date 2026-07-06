"""
Diffusion/privacy/privacy_analysis.py  ?  DP-SGD Privacy Accounting
=====================================================================

compute_noise_multiplier:
    Given target (¥å, ¥ä, sample_rate, epochs), return the Gaussian noise
    multiplier ¥ò such that the training run satisfies (¥å, ¥ä)-DP.

    Uses Opacus's built-in RDP accountant (same as DP-LDM approach).

Usage:
    from Diffusion.privacy import compute_noise_multiplier

    sigma = compute_noise_multiplier(
        target_epsilon = 10.0,
        target_delta   = 1e-5,
        sample_rate    = batch_size / dataset_size,
        epochs         = 30,
    )
    print(f"noise_multiplier = {sigma:.4f}")
"""

import math


def compute_noise_multiplier(
    target_epsilon: float,
    target_delta  : float,
    sample_rate   : float,
    epochs        : int,
    accountant    : str = 'rdp',
    epsilon_tol   : float = 0.01,
) -> float:
    """
    Compute the Gaussian noise multiplier ¥ò for (¥å, ¥ä)-DP training.

    The total number of optimizer steps is estimated as:
        steps = ceil(epochs / sample_rate)
              = epochs * (dataset_size / batch_size)

    Args:
        target_epsilon : target ¥å privacy budget (smaller = more private)
        target_delta   : target ¥ä (typically 1/dataset_size or 1e-5)
        sample_rate    : q = batch_size / dataset_size  (Poisson subsampling rate)
        epochs         : total training epochs
        accountant     : Opacus accountant type ('rdp' recommended)
        epsilon_tol    : convergence tolerance for binary search

    Returns:
        float : noise multiplier ¥ò (Gaussian std relative to clipping bound)

    Raises:
        ImportError : if opacus is not installed
        ValueError  : if no valid ¥ò found in search range [0.01, 1000]
    """
    try:
        from opacus.accountants.utils import get_noise_multiplier
    except ImportError:
        raise ImportError(
            "opacus is required for privacy accounting. "
            "Install with: pip install opacus"
        )

    steps = math.ceil(epochs / sample_rate)

    sigma = get_noise_multiplier(
        target_epsilon = target_epsilon,
        target_delta   = target_delta,
        sample_rate    = sample_rate,
        steps          = steps,
        accountant     = accountant,
        epsilon_tolerance = epsilon_tol,
    )

    print(f'[privacy_analysis] eps={target_epsilon}  delta={target_delta}  '
          f'q={sample_rate:.6f}  epochs={epochs}  steps={steps}  '
          f'sigma={sigma:.6f}')
    return sigma


def get_epsilon_spent(
    privacy_engine,
    target_delta: float,
) -> float:
    """
    Query the current ¥å spent from a running PrivacyEngine.

    Args:
        privacy_engine : opacus.PrivacyEngine (after make_private)
        target_delta   : ¥ä value to compute ¥å at
    Returns:
        float : ¥å spent so far
    """
    return privacy_engine.get_epsilon(target_delta)


def print_privacy_summary(
    noise_multiplier: float,
    max_grad_norm   : float,
    sample_rate     : float,
    epochs          : int,
    target_epsilon  : float,
    target_delta    : float,
):
    """Print a human-readable summary of DP training configuration."""
    steps = math.ceil(epochs / sample_rate)
    print(
        f"\n{'='*60}\n"
        f"  DP-SGD Configuration\n"
        f"{'='*60}\n"
        f"  noise_multiplier  : {noise_multiplier:.4f}\n"
        f"  max_grad_norm     : {max_grad_norm}\n"
        f"  sample_rate (q)   : {sample_rate:.6f}\n"
        f"  epochs            : {epochs}\n"
        f"  total steps       : {steps}\n"
        f"  target eps        : {target_epsilon}\n"
        f"  target delta      : {target_delta}\n"
        f"{'='*60}\n"
    )