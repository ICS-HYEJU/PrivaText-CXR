"""
Diffusion/privacy/privacy_analysis.py  –  DP-SGD Privacy Accounting
=====================================================================

compute_noise_multiplier:
    Given target (ε, δ, sample_rate, epochs), return the Gaussian noise
    multiplier σ such that the training run satisfies (ε, δ)-DP.

    sample_rate = logical_batch_size / dataset_size
                = 1 / steps_per_epoch   (Poisson subsampling rate q)

    This must match the batch_size passed to make_private() divided by N.
    DP-LDM uses accountant='prv' (Privacy Random Variables) which gives
    tighter epsilon bounds than 'rdp', especially for larger epsilon values.

Usage:
    from Diffusion.privacy import compute_noise_multiplier

    sigma = compute_noise_multiplier(
        target_epsilon = 10.0,
        target_delta   = 1e-5,
        sample_rate    = logical_batch / dataset_size,
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
    accountant    : str = 'prv',
    epsilon_tol   : float = 1e-3,
) -> float:
    """
    Compute the Gaussian noise multiplier σ for (ε, δ)-DP training.

    steps = ceil(epochs / sample_rate)
          = epochs * ceil(dataset_size / logical_batch_size)

    This equals the total number of optimizer.step() calls (one per logical
    batch), which is what Opacus's privacy accountant tracks.

    Args:
        target_epsilon : target ε privacy budget
        target_delta   : target δ (recommend 1/dataset_size or 1e-5)
        sample_rate    : q = logical_batch / dataset_size
                         Must equal batch_size/N used in make_private()
        epochs         : total training epochs
        accountant     : 'prv' (DP-LDM default, tighter bounds) or 'rdp'
        epsilon_tol    : binary search convergence tolerance

    Returns:
        float : noise multiplier σ

    Raises:
        ImportError : if opacus is not installed
        ValueError  : if no valid σ found in search range [0.01, 1000]
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
        target_epsilon    = target_epsilon,
        target_delta      = target_delta,
        sample_rate       = sample_rate,
        steps             = steps,
        accountant        = accountant,
        epsilon_tolerance = epsilon_tol,
    )

    print(f'[privacy_analysis] ε={target_epsilon}  δ={target_delta}  '
          f'q={sample_rate:.6f}  epochs={epochs}  steps={steps}  '
          f'accountant={accountant}  → σ={sigma:.6f}')
    return sigma


def get_epsilon_spent(
    privacy_engine,
    target_delta: float,
) -> float:
    """
    Query the current ε spent from a running PrivacyEngine.

    Args:
        privacy_engine : opacus.PrivacyEngine (after make_private)
        target_delta   : δ value to compute ε at
    Returns:
        float : ε spent so far
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
        f"  target ε          : {target_epsilon}\n"
        f"  target δ          : {target_delta}\n"
        f"{'='*60}\n"
    )
