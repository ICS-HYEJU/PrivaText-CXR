"""
fid_from_png.py  –  FID from saved debug grid PNG files
=========================================================

각 PNG 파일은 train_vae.py의 _save_recon_images()가 생성한 matplotlib 그림입니다:
  - 상단 subplot: Original  (make_grid, nrow=n, padding=2)
  - 하단 subplot: Reconstruction

원본 이미지 파일명을 몰라도 PNG에서 직접 FID를 계산합니다.
torchmetrics / torch-fidelity 의존성 없이 torchvision + scipy만으로 동작합니다.

Usage:
    python fid_from_png.py \
        --img_dir ./checkpoints/vae/debug_imgs_2026-03-25_07-03-00 \
        --n_images 4

    # 여러 debug_imgs 디렉토리를 한 번에:
    python fid_from_png.py \
        --img_dir ./checkpoints/vae \
        --recursive \
        --n_images 4

    # 분할 결과를 시각 확인 (첫 PNG만 저장):
    python fid_from_png.py \
        --img_dir ./checkpoints/vae/debug_imgs_2026-03-25_07-03-00 \
        --debug_split

Dependencies:
    pip install scipy          (numpy / torch / torchvision은 이미 설치돼 있어야 함)
"""

import argparse
import glob
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID from debug grid PNG files")

    parser.add_argument("--img_dir",   required=True, type=str,
                        help="Directory containing recon_ep*.png  (or parent dir if --recursive)")
    parser.add_argument("--n_images",  default=4,   type=int,
                        help="Number of images per grid row (same as n in _save_recon_images, default 4)")
    parser.add_argument("--resize",    default=299, type=int,
                        help="Resize extracted sub-images to this size before Inception (default 299)")
    parser.add_argument("--recursive", action="store_true",
                        help="Search all subdirectories of --img_dir for recon_ep*.png files")
    parser.add_argument("--debug_split", action="store_true",
                        help="Save the first PNG's split result to debug_split.png for visual verification")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# PNG parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_subplot_split_row(img_gray: np.ndarray) -> int:
    """
    Find the y-coordinate of the horizontal gap between the two matplotlib subplots.

    Strategy: scan the central 40% of the figure for the row with the
    highest average brightness (= white background between subplots).

    Args:
        img_gray: (H, W) uint8 grayscale array

    Returns:
        Integer row index where the figure splits into top / bottom halves.
    """
    H = img_gray.shape[0]
    search_start = int(H * 0.30)
    search_end   = int(H * 0.70)

    row_brightness = img_gray[search_start:search_end].mean(axis=1)

    # Smooth to avoid noise from a single bright pixel inside the CXR image
    kernel_size = max(3, int(H * 0.02))
    smoothed = np.convolve(row_brightness, np.ones(kernel_size) / kernel_size, mode='same')

    split_row = search_start + int(np.argmax(smoothed))
    return split_row


def crop_title_and_margins(half: np.ndarray,
                           title_frac: float = 0.16,
                           lr_margin_frac: float = 0.01) -> np.ndarray:
    """
    Remove the matplotlib subplot title (top) and figure margins (left/right).

    _save_recon_images() layout:
        figsize=(n*3, 6), tight_layout → each subplot ~300 px tall
        Title text ('Original' / 'Reconstruction') occupies ≈ 16 % of the subplot height.

    Args:
        half:           (H, W, 3) or (H, W) numpy array for one subplot half
        title_frac:     fraction of half-height to skip at the top (title area)
        lr_margin_frac: fraction of full width to skip on each side (axes margin)

    Returns:
        Cropped content array.
    """
    H, W = half.shape[:2]
    y0 = int(H * title_frac)
    x0 = int(W * lr_margin_frac)
    x1 = W - x0
    return half[y0:, x0:x1]


def split_grid_row(content: np.ndarray, n: int) -> list:
    """
    Split a horizontal image grid into n individual sub-images.

    make_grid(nrow=n, padding=2) adds 2 px padding between images.
    After matplotlib rendering the padding is proportionally small,
    so equal-width splitting is sufficient for FID.

    Args:
        content: (H, W, C) or (H, W) array containing the grid
        n:       number of images in the grid

    Returns:
        List of n sub-image arrays.
    """
    W = content.shape[1]
    strip_w = W // n
    strips = []
    for i in range(n):
        x0 = i * strip_w
        x1 = x0 + strip_w if i < n - 1 else W
        strips.append(content[:, x0:x1])
    return strips


def extract_pairs_from_png(png_path: str, n_images: int = 4):
    """
    Extract original / reconstruction image pairs from one debug grid PNG.

    Args:
        png_path: path to recon_ep*.png
        n_images: number of images per grid row

    Returns:
        orig_strips:  list of n_images (H, W, 3) uint8 arrays  (originals)
        recon_strips: list of n_images (H, W, 3) uint8 arrays  (reconstructions)
    """
    img = np.array(Image.open(png_path).convert('RGB'))   # (H, W, 3)
    gray = img.mean(axis=2)                                # (H, W) for brightness scan

    # ── 1. Split into top (original) and bottom (reconstruction) ──────────
    split_row = find_subplot_split_row(gray)
    top_half  = img[:split_row]
    bot_half  = img[split_row:]

    # ── 2. Remove title text and figure margins ────────────────────────────
    top_content = crop_title_and_margins(top_half)
    bot_content = crop_title_and_margins(bot_half)

    # ── 3. Split each content area into n_images sub-images ───────────────
    orig_strips  = split_grid_row(top_content, n_images)
    recon_strips = split_grid_row(bot_content, n_images)

    return orig_strips, recon_strips


# ──────────────────────────────────────────────────────────────────────────────
# Inception v3 feature extractor
# ──────────────────────────────────────────────────────────────────────────────

def build_inception(device: torch.device) -> torch.nn.Module:
    """
    Load Inception v3 pretrained, strip the final FC layer → 2048-dim pool features.

    torchvision >= 0.13: uses Inception_V3_Weights.DEFAULT
    torchvision <  0.13: falls back to pretrained=True
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = models.inception_v3(
                weights=models.Inception_V3_Weights.DEFAULT,
                transform_input=False,
            )
        except AttributeError:
            model = models.inception_v3(pretrained=True, transform_input=False)

    model.fc          = torch.nn.Identity()   # 2048-dim pool → output
    model.aux_logits  = False                 # disable aux during eval
    model.eval()
    return model.to(device)


@torch.no_grad()
def extract_features(imgs: list, inception: torch.nn.Module,
                     device: torch.device, batch_size: int = 32) -> np.ndarray:
    """
    Run Inception v3 on a list of (H, W, 3) uint8 arrays and return 2048-dim features.

    Args:
        imgs:       list of (H, W, 3) uint8 numpy arrays (any size; will be resized to 299)
        inception:  feature extractor from build_inception()
        device:     torch device
        batch_size: images processed per forward pass

    Returns:
        (N, 2048) float32 numpy array
    """
    feats = []
    for start in range(0, len(imgs), batch_size):
        batch_np = imgs[start:start + batch_size]
        tensors  = []
        for arr in batch_np:
            pil = Image.fromarray(arr.astype(np.uint8), mode='RGB').resize((299, 299), Image.BILINEAR)
            t   = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0  # [0, 1]
            tensors.append(t)
        batch = torch.stack(tensors).to(device)                     # (B, 3, 299, 299)
        out   = inception(batch)                                     # (B, 2048)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)                             # (N, 2048)


# ──────────────────────────────────────────────────────────────────────────────
# Fréchet distance
# ──────────────────────────────────────────────────────────────────────────────

def frechet_distance(feats_real: np.ndarray, feats_fake: np.ndarray) -> float:
    """
    Compute FID between two sets of Inception features.

        FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2·sqrt(Σ_r·Σ_f))

    Args:
        feats_real: (N, 2048) features for real images
        feats_fake: (N, 2048) features for reconstructed images

    Returns:
        FID score (float)
    """
    from scipy.linalg import sqrtm

    mu_r, mu_f = feats_real.mean(0), feats_fake.mean(0)
    sig_r = np.cov(feats_real, rowvar=False)
    sig_f = np.cov(feats_fake, rowvar=False)

    diff    = mu_r - mu_f
    covmean = sqrtm(sig_r @ sig_f)

    # Numerical stability: sqrtm can return tiny imaginary parts
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print("  [WARN] sqrtm produced significant imaginary component; taking real part.")
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sig_r + sig_f - 2.0 * covmean))
    return fid


# ──────────────────────────────────────────────────────────────────────────────
# Tensor conversion (kept for debug_split visualisation)
# ──────────────────────────────────────────────────────────────────────────────

def array_to_uint8_rgb_tensor(img_array: np.ndarray, resize: int = 299) -> torch.Tensor:
    pil = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
    if resize != pil.width or resize != pil.height:
        pil = pil.resize((resize, resize), Image.BILINEAR)
    t = torch.from_numpy(np.array(pil)).permute(2, 0, 1).unsqueeze(0)
    return t.to(torch.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Debug helper
# ──────────────────────────────────────────────────────────────────────────────

def save_debug_split(png_path: str, n_images: int, out_path: str = "debug_split.png"):
    """
    Save a visualisation of the splitting result for one PNG file.
    Shows: original sub-images (row 1) and reconstruction sub-images (row 2).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        orig_strips, recon_strips = extract_pairs_from_png(png_path, n_images)

        fig, axes = plt.subplots(2, n_images, figsize=(n_images * 3, 6))
        for j, (o, r) in enumerate(zip(orig_strips, recon_strips)):
            axes[0, j].imshow(o, cmap='gray')
            axes[0, j].set_title(f"Orig {j+1}")
            axes[0, j].axis("off")
            axes[1, j].imshow(r, cmap='gray')
            axes[1, j].set_title(f"Recon {j+1}")
            axes[1, j].axis("off")

        plt.suptitle(f"Split verification: {os.path.basename(png_path)}", y=1.01)
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"  [Debug] Split visualisation saved → {out_path}")

    except Exception as e:
        print(f"  [Debug] Could not save split visualisation: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Collect PNG files ─────────────────────────────────────────────────────
    if args.recursive:
        pattern = os.path.join(args.img_dir, "**", "recon_ep*.png")
        png_files = sorted(glob.glob(pattern, recursive=True))
    else:
        pattern = os.path.join(args.img_dir, "recon_ep*.png")
        png_files = sorted(glob.glob(pattern))

    if not png_files:
        raise FileNotFoundError(
            f"No 'recon_ep*.png' files found.\n"
            f"  Searched: {pattern}\n"
            f"  Tip: use --recursive to search subdirectories."
        )

    expected_samples = len(png_files) * args.n_images
    print(f"\n[FID from PNG]  Found {len(png_files)} PNG files")
    print(f"               n_images={args.n_images}  resize={args.resize}")
    print(f"               Expected image pairs: {expected_samples}")
    if expected_samples < 2048:
        print(f"  [WARN] {expected_samples} samples is below the recommended minimum of 2048.")
        print(f"         FID values may have high variance.")

    # ── Optional: save split visualisation ────────────────────────────────────
    if args.debug_split:
        save_debug_split(png_files[0], args.n_images, out_path="debug_split.png")

    # ── Extract all sub-images from PNGs ──────────────────────────────────────
    real_imgs = []
    fake_imgs = []
    n_failed  = 0

    for i, png_path in enumerate(png_files):
        try:
            orig_strips, recon_strips = extract_pairs_from_png(png_path, args.n_images)
            real_imgs.extend(orig_strips)
            fake_imgs.extend(recon_strips)
        except Exception as e:
            print(f"  [WARN] Skipped {os.path.basename(png_path)}: {e}")
            n_failed += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(png_files):
            print(f"  [{i+1}/{len(png_files)}] {len(real_imgs)} pairs collected", end="\r")

    print()

    if len(real_imgs) == 0:
        print("[ERROR] No images could be extracted. Check --img_dir and --n_images.")
        return None

    # ── Build Inception v3 feature extractor ──────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    print(f"  Loading Inception v3 …")
    inception = build_inception(device)

    # ── Extract Inception features ────────────────────────────────────────────
    print(f"  Extracting features from {len(real_imgs)} real images …")
    feats_real = extract_features(real_imgs, inception, device)

    print(f"  Extracting features from {len(fake_imgs)} reconstructed images …")
    feats_fake = extract_features(fake_imgs, inception, device)

    # ── Compute FID ───────────────────────────────────────────────────────────
    print(f"  Computing Fréchet distance …")
    fid_score  = frechet_distance(feats_real, feats_fake)

    print(f"\n{'='*52}")
    print(f"  FID Score   : {fid_score:.4f}")
    print(f"  Image pairs : {len(real_imgs)}  ({n_failed} files failed)")
    print(f"  Source dir  : {args.img_dir}")
    print(f"{'='*52}\n")

    return fid_score


if __name__ == "__main__":
    main()
