"""
Eval_metric/features.py  —  Shared feature extraction + cache (Eval step 1)
===========================================================================
The common backbone for FID / FDS / T-SNE.  All three consume the SAME feature
vectors, so we extract them ONCE per (image_dir, eval_model) and cache to .npy.

Design
------
- Reads generated / real images directly from a FOLDER of PNG/JPG files
  (e.g. ./generated_eps5/samples/  and a folder of held-out real test PNGs).
  This keeps evaluation decoupled from the generation pipeline: you evaluate
  whatever was saved to disk.
- Images are loaded as 1-channel, resized, normalised to [-1, 1] — matching
  Data/mimic_cxr.py (transforms.Normalize(mean=[0.5], std=[0.5])).
- The feature extractor is reused from Eval_metric/fid.py when available
  (build_feature_extractor: 'inception' -> 2048-d, 'xrv' -> 1024-d).  A local
  fallback is provided so this module also works on a branch whose fid.py does
  not yet expose build_feature_extractor.

`torch` is imported lazily inside extract_features() so that the pure-numpy
consumers (FDS math, T-SNE) can import this module without a torch install.
"""

import os
import glob
import hashlib

import numpy as np


IMG_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')


# ── image discovery / loading ────────────────────────────────────────────────
def list_images(image_dir, recursive=True):
    """Return a sorted list of image paths under image_dir."""
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f'image dir not found: {image_dir}')
    paths = []
    for ext in IMG_EXTS:
        pat = os.path.join(image_dir, '**', ext) if recursive else os.path.join(image_dir, ext)
        paths.extend(glob.glob(pat, recursive=recursive))
    # Skip obvious grid/montage previews so they don't pollute the feature set.
    paths = [p for p in paths
             if 'grid' not in os.path.basename(p).lower()]
    return sorted(paths)


def load_image_as_tensor(path, image_size):
    """
    Load one image -> torch.FloatTensor [1, 1, image_size, image_size] in [-1, 1].
    Grayscale (chest X-ray convention); matches training normalisation.
    """
    import torch
    from PIL import Image
    img = Image.open(path).convert('L').resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0        # [H, W] in [0, 1]
    t = torch.from_numpy(arr)[None, None]                  # [1, 1, H, W]
    return t * 2.0 - 1.0                                   # -> [-1, 1]


# ── feature extractor (reuse fid.py, else fallback) ──────────────────────────
def build_feature_extractor(eval_model, device):
    """
    Return (feature_extractor: nn.Module, preprocess: callable).
    preprocess maps [B, 1, H, W] in [-1, 1] to the extractor's expected input.

    Prefers Eval_metric.fid.build_feature_extractor (single source of truth).
    Falls back to local definitions when that symbol is absent.
    """
    try:
        from Eval_metric.fid import build_feature_extractor as _bfe
        return _bfe(eval_model, device)
    except (ImportError, AttributeError):
        return _build_feature_extractor_fallback(eval_model, device)


def _build_feature_extractor_fallback(eval_model, device):
    import torch  # noqa: F401
    if eval_model == 'inception':
        from Eval_metric.fid import InceptionV3Features, to_float_rgb
        return InceptionV3Features().to(device).eval(), to_float_rgb
    elif eval_model == 'xrv':
        return _XRVDenseNetFeatures().to(device).eval(), _to_xrv_input
    raise ValueError(f"Unknown eval_model '{eval_model}' (choices: 'inception', 'xrv')")


def _to_xrv_input(x):
    """[-1, 1]  ->  [-1024, 1024]  (torchxrayvision normalize convention)."""
    return x.clamp(-1., 1.) * 1024.


def _XRVDenseNetFeatures():
    import torch.nn as nn

    class XRVDenseNetFeatures(nn.Module):
        """1024-dim features from a torchxrayvision DenseNet-121 (CXR-pretrained)."""
        def __init__(self, weights='densenet121-res224-all'):
            super().__init__()
            try:
                import torchxrayvision as xrv
            except ImportError as e:
                raise ImportError(
                    'torchxrayvision is required for the XRV DenseNet-121 backbone.\n'
                    '  pip install torchxrayvision') from e
            self.model = xrv.models.DenseNet(weights=weights)
            self.model.eval()

        def forward(self, x):
            feats = self.model.features(x)                 # [B, 1024, h, w]
            import torch.nn.functional as F
            feats = F.relu(feats, inplace=True)
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            return feats.view(feats.shape[0], -1)          # [B, 1024]

    return XRVDenseNetFeatures()


# ── extraction with .npy cache ───────────────────────────────────────────────
def _cache_key(image_paths, eval_model, image_size):
    h = hashlib.md5()
    h.update(f'{eval_model}|{image_size}|{len(image_paths)}'.encode())
    for p in image_paths:
        h.update(os.path.basename(p).encode())
    return h.hexdigest()[:12]


def extract_features_from_tensors(images, eval_model, device='cpu',
                                  batch_size=16, cache_path=None, cache_id=None,
                                  verbose=True):
    """
    Extract features from an iterable of image tensors (each [1, H, W] or
    [1, 1, H, W] in [-1, 1]) — e.g. real images pulled straight from a Dataset
    whose split is chosen by args (train/val/test), so no PNG dump is needed.

    Caching is keyed by `cache_id` (a stable string, e.g. "test:200:256"); when
    it matches an existing cache the network is not re-run.

    Returns
    -------
    feats : np.ndarray  [N, D]
    """
    images = list(images)
    if len(images) == 0:
        raise RuntimeError('no images provided to extract_features_from_tensors')

    key = f'{eval_model}|{cache_id}|{len(images)}' if cache_id else None
    if cache_path and key and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            if str(data['key']) == key:
                if verbose:
                    print(f'[features] cache hit  {cache_path}  ({data["feats"].shape})')
                return data['feats']
        except Exception:
            pass

    import torch
    model, preprocess = build_feature_extractor(eval_model, device)

    def _as_bchw(t):
        if t.dim() == 3:            # [1, H, W] -> [1, 1, H, W]
            t = t.unsqueeze(0)
        return t

    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.cat([_as_bchw(images[j]) for j in range(i, min(i + batch_size, len(images)))], 0)
            batch = preprocess(batch.to(device))
            feats.append(model(batch).float().cpu().numpy())
            if verbose:
                print(f'[features] {eval_model}  {min(i + batch_size, len(images))}/{len(images)}',
                      end='\r')
    feats = np.concatenate(feats, 0)
    if verbose:
        print(f'\n[features] extracted {feats.shape} from {len(images)} tensors')

    if cache_path and key:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez(cache_path, feats=feats, key=key)
        if verbose:
            print(f'[features] cached -> {cache_path}')
    return feats


def extract_features(image_dir, eval_model, device='cpu', image_size=256,
                     batch_size=16, cache_path=None, recursive=True, verbose=True):
    """
    Extract features for every image under image_dir.

    Returns
    -------
    feats : np.ndarray  [N, D]
    paths : list[str]   the N image paths (same order as rows of feats)

    Caching
    -------
    If cache_path is given and a valid cache exists (same eval_model / image_size
    / file set), it is loaded instead of re-running the network.
    """
    paths = list_images(image_dir, recursive=recursive)
    if len(paths) == 0:
        raise RuntimeError(f'no images found under {image_dir}')

    key = _cache_key(paths, eval_model, image_size)
    if cache_path and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            if str(data['key']) == key:
                if verbose:
                    print(f'[features] cache hit  {cache_path}  ({data["feats"].shape})')
                return data['feats'], list(data['paths'])
        except Exception:
            pass  # stale/corrupt cache -> recompute

    import torch
    model, preprocess = build_feature_extractor(eval_model, device)
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            batch = torch.cat([load_image_as_tensor(p, image_size) for p in batch_paths], 0)
            batch = preprocess(batch.to(device))
            f = model(batch).float().cpu().numpy()
            feats.append(f)
            if verbose:
                print(f'[features] {eval_model}  {min(i + batch_size, len(paths))}/{len(paths)}',
                      end='\r')
    feats = np.concatenate(feats, 0)
    if verbose:
        print(f'\n[features] extracted {feats.shape} from {image_dir}')

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez(cache_path, feats=feats, paths=np.array(paths), key=key)
        if verbose:
            print(f'[features] cached -> {cache_path}')
    return feats, paths
