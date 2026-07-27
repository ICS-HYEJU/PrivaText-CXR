"""
Data/mimic_cxr.py  ?  MIMIC-CXR Map-Style Dataset
---------------------------------------------------
Map-style dataset compatible with DP-SGD (Opacus UniformWithReplacementSampler).
Each sample returns (image_tensor, report_str).

Loads from pre-built split directories produced by prepare_split_dirs():
    <prebuilt_split_dir>/
    ¦§¦¡¦¡ train/
    ¦¢   ¦§¦¡¦¡ p10/
    ¦¢   ¦¢   ¦¦¦¡¦¡ p10000032/
    ¦¢   ¦¢       ¦§¦¡¦¡ s50414267/
    ¦¢   ¦¢       ¦¢   ¦¦¦¡¦¡ <dicom_id>.dcm
    ¦¢   ¦¢       ¦¦¦¡¦¡ s50414267.txt
    ¦¢   ¦§¦¡¦¡ p11/ ... p19/
    ¦§¦¡¦¡ validate/
    ¦¦¦¡¦¡ test/

External usage (training scripts):
    from Data.mimic_cxr import dataset_loader
    train_loader = dataset_loader(args, embedder, split='train')
"""

import os
import re
import sys
import json
import argparse

import numpy as np
import pydicom
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    sys.path.insert(0, '/home/hjchoi/PycharmProjects/PrivaText-CXR')
    from Modules.BioBERT_embedder import BioBERTEmbedder
except ImportError:
    BioBERTEmbedder = None
    print("[mimic_cxr] WARNING: BioBERTEmbedder not found. "
          "dataset_loader() will not be available.")


# =============================================================================
# Args
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="MIMIC-CXR Dataset")
    parser.add_argument("--device_id",          type=int, default=1)

    # Paths
    parser.add_argument("--prebuilt_split_dir", type=str,
                        default='/storage/hjchoi/mimic/split',
                        help="root of pre-built split dirs (train/validate/test)")

    # Dataset
    parser.add_argument("--split",       type=str, default="train",
                        choices=["train", "validate", "test"])
    parser.add_argument("--image_size",  type=int, default=256)
    parser.add_argument("--max_length",  type=int, default=512)

    # DICOM integrity filtering
    parser.add_argument("--validate_dicom", action="store_true", default=True,
                        help="drop corrupted/unreadable DICOM files at index build")
    parser.add_argument("--no_validate_dicom", dest="validate_dicom",
                        action="store_false",
                        help="disable DICOM validation (keep every file)")
    parser.add_argument("--dicom_cache", type=str, default=None,
                        help="path to the DICOM validation cache "
                             "(default: <split_dir>/.dicom_valid_cache.json)")

    # DataLoader
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--drop_last",   type=bool, default=True)

    # BioBERT
    parser.add_argument("--biobert_path", type=str, default="/storage/hjchoi")

    return parser.parse_args()


# =============================================================================
# MIMICCXRDataset
# =============================================================================

class MIMICCXRDataset(Dataset):
    """
    MIMIC-CXR Map-Style Dataset.

    Scans <prebuilt_split_dir>/<split>/ at init to build an idx-mapped sample
    list (lightweight path strings only).  Each __getitem__ call loads one
    DICOM file on demand.

    Args:
        args : Namespace ? requires prebuilt_split_dir, split, image_size,
                           max_length
    """
    def __init__(self, args):
        super().__init__()

        self.split      = args.split
        self.image_size = args.image_size
        self.max_length = getattr(args, "max_length", 512)

        self.validate_dicom = getattr(args, "validate_dicom", True)
        self.dicom_cache    = getattr(args, "dicom_cache", None)

        self.scan_root = os.path.join(args.prebuilt_split_dir, self.split)
        if not os.path.isdir(self.scan_root):
            raise FileNotFoundError(
                f"[MIMICCXRDataset] split dir not found: {self.scan_root}"
            )

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.samples = self._build_index()
        n_raw = len(self.samples)

        if self.validate_dicom:
            self.samples = self._filter_corrupted(self.samples)

        n_dropped = n_raw - len(self.samples)
        print(f"[MIMICCXRDataset] split='{self.split}'  "
              f"total={len(self.samples)}  (dropped {n_dropped} corrupted)")

    # -------------------------------------------------------------------------
    # Map-style interface
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns:
            image  : Tensor [1, H, W]  normalised to [-1, 1]
            report : str  ? "FINDINGS: <...> IMPRESSION: <...>"
        """
        meta = self.samples[idx]

        try:
            image = self._load_dcm(meta["dcm_path"])
        except Exception as e:
            print(f"[MIMICCXRDataset] load error idx={idx}: {e}")
            image = self._blank_image()

        report = self._load_report(meta["report_path"])
        return image, report

    # -------------------------------------------------------------------------
    # Index build
    # -------------------------------------------------------------------------

    def _build_index(self) -> list:
        """
        Walk scan_root and collect (dcm_path, report_path) for every .dcm file.
        Structure: <scan_root>/p1X/pXXXXXXXX/sYYYYYYYY/*.dcm
        """
        samples = []

        for prefix in sorted(os.listdir(self.scan_root)):
            prefix_dir = os.path.join(self.scan_root, prefix)
            if not os.path.isdir(prefix_dir) or not prefix.startswith("p"):
                continue

            for patient_id in sorted(os.listdir(prefix_dir)):
                patient_dir = os.path.join(prefix_dir, patient_id)
                if not os.path.isdir(patient_dir):
                    continue

                for study in sorted(os.listdir(patient_dir)):
                    study_dir = os.path.join(patient_dir, study)
                    if not os.path.isdir(study_dir) or not study.startswith("s"):
                        continue

                    report_path = os.path.join(patient_dir, f"{study}.txt")

                    for fname in sorted(os.listdir(study_dir)):
                        if not fname.endswith(".dcm"):
                            continue
                        samples.append({
                            "dcm_path"   : os.path.join(study_dir, fname),
                            "report_path": report_path,
                            "study_id"   : study,
                            "patient_id" : patient_id,
                        })

        return samples

    # -------------------------------------------------------------------------
    # DICOM integrity filtering
    # -------------------------------------------------------------------------

    def _cache_path(self) -> str:
        if self.dicom_cache:
            return self.dicom_cache
        return os.path.join(self.scan_root, ".dicom_valid_cache.json")

    @staticmethod
    def _is_readable_dcm(path: str) -> bool:
        """
        Return True only if the pixel data can actually be decoded.

        Reading `pixel_array` forces the pixel-data decode, which is exactly
        the step that raises for truncated / corrupted files
        ("number of bytes of pixel data is less than expected").
        """
        try:
            dcm = pydicom.dcmread(path)
            _ = dcm.pixel_array
            return True
        except Exception:
            return False

    def _filter_corrupted(self, samples: list) -> list:
        """
        Drop samples whose DICOM pixel data cannot be decoded.

        Results are cached to a JSON manifest keyed by path + file size, so the
        (expensive) full decode is only paid once; subsequent runs re-validate
        only files that are new or whose size changed.
        """
        cache_path = self._cache_path()
        cache = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cache = json.load(f)
            except Exception as e:
                print(f"[MIMICCXRDataset] cache read failed ({e}); revalidating")
                cache = {}

        kept, dropped = [], []
        dirty = False
        n_total = len(samples)

        for i, meta in enumerate(samples):
            path = meta["dcm_path"]
            try:
                size = os.path.getsize(path)
            except OSError:
                dropped.append(path)
                continue

            entry = cache.get(path)
            if entry is not None and entry.get("size") == size:
                valid = entry.get("valid", False)
            else:
                valid = self._is_readable_dcm(path)
                cache[path] = {"size": size, "valid": valid}
                dirty = True

            if valid:
                kept.append(meta)
            else:
                dropped.append(path)

            if dirty and (i + 1) % 2000 == 0:
                print(f"[MIMICCXRDataset] validating DICOMs "
                      f"{i + 1}/{n_total} ...")

        if dirty:
            try:
                tmp = cache_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(cache, f)
                os.replace(tmp, cache_path)
            except Exception as e:
                print(f"[MIMICCXRDataset] cache write failed ({e})")

        if dropped:
            print(f"[MIMICCXRDataset] dropped {len(dropped)} corrupted file(s); "
                  f"e.g. {dropped[0]}")

        return kept

    # -------------------------------------------------------------------------
    # Loaders
    # -------------------------------------------------------------------------

    def _load_dcm(self, dcm_path: str) -> torch.Tensor:
        dcm = pydicom.dcmread(dcm_path)
        arr = dcm.pixel_array.astype(np.float32)

        if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr

        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        arr = np.clip(arr, 0, 255)

        image = Image.fromarray(arr.astype(np.uint8)).convert("L")
        return self.transform(image)

    def _blank_image(self) -> torch.Tensor:
        return torch.zeros(1, self.image_size, self.image_size)

    def _load_report(self, report_path: str) -> str:
        if not os.path.exists(report_path):
            return ""

        with open(report_path, "r", encoding="utf-8") as f:
            raw = f.read()

        report = self._parse_report(raw)

        if self.max_length and len(report) > self.max_length:
            report = report[: self.max_length]

        return report

    def _parse_report(self, text: str) -> str:
        """
        Extract FINDINGS and IMPRESSION sections.
        Falls back to full text if neither section is found.
        """
        pattern = re.compile(
            r"(FINDINGS|IMPRESSION)\s*:(.*?)(?=\n[A-Z ]+:|$)",
            re.IGNORECASE | re.DOTALL,
        )
        sections = {
            name.lower(): content.strip()
            for name, content in pattern.findall(text)
        }

        if not sections:
            return text.strip()

        parts = []
        for key in ("findings", "impression"):
            if key in sections and sections[key]:
                parts.append(f"{key.upper()}: {sections[key]}")

        return " ".join(parts).strip()


# =============================================================================
# dataset_loader  ?  DataLoader with BioBERT collate (used by training scripts)
# =============================================================================

def dataset_loader(
    args,
    embedder,
    split: str = None,
) -> DataLoader:
    """
    Build a DataLoader for MIMIC-CXR that yields (images, context).

    Each batch contains:
        images  : Tensor [B, 1, H, W]
        context : Tensor [B, seq_len, output_dim]  (BioBERT embedding)

    Args:
        args     : Namespace from parse_args() (or compatible Namespace)
        embedder : BioBERTEmbedder instance already moved to target device
        split    : override args.split if provided

    Note:
        Keep num_workers=0 when embedder is on GPU to avoid CUDA fork errors.

    Returns:
        DataLoader yielding (images [B,1,H,W], context [B,seq_len,output_dim])

    Example:
        from Data.mimic_cxr import dataset_loader
        train_loader = dataset_loader(args, embedder, split='train')
    """
    if BioBERTEmbedder is None:
        raise ImportError("BioBERTEmbedder could not be imported. "
                          "Check Modules/BioBERT_embedder.py path.")

    import argparse as _argparse
    loader_args = _argparse.Namespace(**vars(args))
    if split is not None:
        loader_args.split = split

    dataset = MIMICCXRDataset(loader_args)

    def collate_fn(batch):
        images, reports = zip(*batch)
        images  = torch.stack(images)
        context = embedder(list(reports))
        return images, context

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = (loader_args.split == "train"),
        num_workers = args.num_workers,
        collate_fn  = collate_fn,
        drop_last   = args.drop_last,
        pin_memory  = False,
    )

    print(f"[dataset_loader] split={loader_args.split}  "
          f"samples={len(dataset)}  bs={args.batch_size}  "
          f"steps={len(loader)}")
    return loader


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    args = parse_args()

    if BioBERTEmbedder is None:
        raise ImportError(
            "BioBERTEmbedder not found. "
            "Check /home/hjchoi/PycharmProjects/PrivaText-CXR/Modules/BioBERT_embedder.py"
        )

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    embedder = BioBERTEmbedder(model_path=args.biobert_path).to(device)

    dataloader = dataset_loader(args, embedder)

    for batch_id, (image, context) in enumerate(dataloader):
        if batch_id == 1:
            break
        print(f"image  : {image.shape}  min={image.min():.3f}  max={image.max():.3f}")
        print(f"context: {context.shape}  dtype={context.dtype}")