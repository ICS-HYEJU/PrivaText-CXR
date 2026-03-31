"""
fid_from_png.py  –  FID from saved debug grid PNG files
=========================================================

각 PNG 파일은 train_vae.py의 _save_recon_images()가 생성한 matplotlib 그림입니다:
  - 상단 subplot: Original  (make_grid, nrow=n, padding=2)
  - 하단 subplot: Reconstruction

원본 이미지 파일명을 몰라도 PNG에서 직접 FID를 계산합니다.

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
    pip install torchmetrics[image]
"""

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image


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
# Tensor conversion
# ──────────────────────────────────────────────────────────────────────────────

def array_to_uint8_rgb_tensor(img_array: np.ndarray, resize: int = 299) -> torch.Tensor:
    """
    Convert (H, W, C) uint8 numpy array → (1, 3, resize, resize) uint8 tensor.

    The sub-images extracted from the PNG are already in [0, 255] uint8.
    We convert to RGB and resize for the Inception feature extractor.

    Args:
        img_array: (H, W, 3) uint8 array
        resize:    target size (299 for Inception v3)

    Returns:
        (1, 3, resize, resize) torch.uint8 tensor
    """
    pil = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
    if resize != pil.width or resize != pil.height:
        pil = pil.resize((resize, resize), Image.BILINEAR)
    t = torch.from_numpy(np.array(pil)).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
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

    # ── torchmetrics FID ──────────────────────────────────────────────────────
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        raise ImportError(
            "torchmetrics[image] is required.\n"
            "  pip install torchmetrics[image]"
        )

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
        print(f"         FID values may have high variance. Use more epochs or larger --n_images.")

    # ── Optional: save split visualisation for the first PNG ──────────────────
    if args.debug_split:
        save_debug_split(png_files[0], args.n_images, out_path="debug_split.png")

    # ── FID computation ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    print(f"  device: {device}")

    n_extracted = 0
    n_failed    = 0

    for i, png_path in enumerate(png_files):
        try:
            orig_strips, recon_strips = extract_pairs_from_png(png_path, args.n_images)

            for orig, recon in zip(orig_strips, recon_strips):
                real_t = array_to_uint8_rgb_tensor(orig,  args.resize).to(device)
                fake_t = array_to_uint8_rgb_tensor(recon, args.resize).to(device)

                fid_metric.update(real_t, real=True)
                fid_metric.update(fake_t, real=False)
                n_extracted += 1

        except Exception as e:
            print(f"  [WARN] Skipped {os.path.basename(png_path)}: {e}")
            n_failed += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(png_files):
            print(f"  [{i+1}/{len(png_files)}] {n_extracted} pairs extracted", end="\r")

    print()

    # ── Report ────────────────────────────────────────────────────────────────
    if n_extracted == 0:
        print("[ERROR] No images could be extracted. Check --img_dir and --n_images.")
        return None

    fid_score = fid_metric.compute().item()

    print(f"\n{'='*52}")
    print(f"  FID Score   : {fid_score:.4f}")
    print(f"  Image pairs : {n_extracted}  ({n_failed} files failed)")
    print(f"  Source dir  : {args.img_dir}")
    print(f"{'='*52}\n")

    return fid_score


if __name__ == "__main__":
    main()
