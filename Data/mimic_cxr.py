"""
Data/mimic_cxr.py  –  MIMIC-CXR Map-Style Dataset
---------------------------------------------------
Map-style dataset compatible with DP-SGD (Opacus UniformWithReplacementSampler).
Each sample returns (image_tensor, report_str).

Directory structure (original):
    <root_path>/
    ├── files/
    │   └── p10/
    │       └── p10000032/
    │           ├── s50414267/
    │           │   └── <dicom_id>.dcm
    │           └── s50414267.txt
    ├── mimic-cxr-2.0.0-split.csv      (Mode A – required)
    └── mimic-cxr-2.0.0-metadata.csv   (Mode A – optional)

Modes:
    Mode A  split.csv found  → official train / validate / test labels  [default]
    Mode C  split.csv absent → folder scan + patient-level 80/10/10     [fallback]

prepare_split_dirs (--make_split_dir):
    One-time preprocessing step. Reads Mode A/C sample list and creates:
        <split_dir>/train/  validate/  test/
    mirroring the original p10/pXXX/sYYY/ hierarchy via symlinks or copies.
    Run once after data download completes.

External usage (training scripts):
    from Data.mimic_cxr import dataset_loader
    train_loader = dataset_loader(args, embedder, split='train')
"""

import os
import re
import sys
import random
import shutil
import time
import argparse
from typing import Optional, Callable

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

import torch
import torch.nn as nn
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

SPLIT_CSV = "mimic-cxr-2.0.0-split.csv"
META_CSV  = "mimic-cxr-2.0.0-metadata.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="MIMIC-CXR Dataset")

    # ── Paths ─────────────────────────────────────────────────────────────────
    parser.add_argument("--root_path",  type=str, required=True,
                        help="MIMIC-CXR dataset root directory")
    parser.add_argument("--split_csv",  type=str, default=SPLIT_CSV,
                        help="split CSV filename (relative to root_path)")
    parser.add_argument("--meta_csv",   type=str, default=META_CSV,
                        help="metadata CSV filename (relative to root_path)")

    # ── Dataset ───────────────────────────────────────────────────────────────
    parser.add_argument("--split",       type=str, default="train",
                        choices=["train", "validate", "test"],
                        help="dataset split to load")
    parser.add_argument("--image_size",  type=int, default=256,
                        help="resize both sides to this value")
    parser.add_argument("--max_length",  type=int, default=512,
                        help="max character length of report text")
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--view_filter", type=str, nargs="+",
                        default=["PA", "AP"],
                        help="ViewPosition values to keep (requires metadata.csv). "
                             "e.g. --view_filter PA AP  |  --view_filter PA  |  pass empty to disable")

    # ── BioBERT ───────────────────────────────────────────────────────────────
    parser.add_argument("--biobert_path", type=str,
                        default="dmis-lab/biobert-base-cased-v1.2",
                        help="local path or HuggingFace model ID for BioBERT")

    # ── prepare_split_dirs ────────────────────────────────────────────────────
    parser.add_argument("--make_split_dir", action="store_true",
                        help="organise files into train/validate/test folders "
                             "(run once after data download completes)")
    parser.add_argument("--split_dir",   type=str, default=None,
                        help="output root for split folders "
                             "(default: <root_path>/split)")
    parser.add_argument("--use_symlink", action="store_true", default=True,
                        help="use symlinks instead of file copies in split dirs")

    return parser.parse_args()


# =============================================================================
# MIMICCXRDataset
# =============================================================================

class MIMICCXRDataset(Dataset):
    """
    MIMIC-CXR Map-Style Dataset.

    Args:
        args : Namespace from parse_args()
               Required fields: root_path, split, image_size, max_length,
                                split_csv, meta_csv
    """

    SPLIT_CSV = SPLIT_CSV
    META_CSV  = META_CSV

    def __init__(self, args):
        super().__init__()

        self.root_dir   = args.root_path
        self.files_dir  = os.path.join(args.root_path, "files")
        self.split      = args.split
        self.image_size = args.image_size
        self.max_length = getattr(args, "max_length", 512)

        self.split_csv   = os.path.join(args.root_path, args.split_csv)
        self.meta_csv    = os.path.join(args.root_path, args.meta_csv)
        self.view_filter = getattr(args, "view_filter", ["PA", "AP"])

        self.transform  = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        # ── Mode A (primary) ──────────────────────────────────────────────────
        if os.path.exists(self.split_csv):
            self._mode   = "A"
            print(f"[MIMICCXRDataset] Mode A: split CSV found → {self.split_csv}")
            self.samples = self._build_from_csv()

        # ── Mode C (fallback) ─────────────────────────────────────────────────
        else:
            self._mode   = "C"
            print(f"[MIMICCXRDataset] Mode C: split CSV not found → folder scan")
            self.samples = self._build_from_folder()

        print(f"[MIMICCXRDataset] mode={self._mode}  split='{self.split}'  "
              f"total={len(self.samples)}")

    # -------------------------------------------------------------------------
    # Map-style interface
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns:
            image  : Tensor [1, H, W]  normalised to [-1, 1]
            report : str  – "FINDINGS: <...> IMPRESSION: <...>"
        """
        meta = self.samples[idx]

        try:
            image = self._load_dcm(meta["dcm_path"])
        except Exception as e:
            print(f"[MIMICCXRDataset] WARNING: failed to load "
                  f"{meta['dcm_path']}: {e}")
            image = self._blank_image()

        report = self._load_report(meta["report_path"])
        return image, report

    # -------------------------------------------------------------------------
    # Mode A – official CSV split
    # -------------------------------------------------------------------------

    def _build_from_csv(self) -> list:
        split_df = pd.read_csv(self.split_csv)

        if self.split is not None:
            split_df = split_df[
                split_df["split"] == self.split
            ].reset_index(drop=True)

        if os.path.exists(self.meta_csv):
            meta_df  = pd.read_csv(self.meta_csv)
            split_df = pd.merge(
                split_df,
                meta_df[["dicom_id", "ViewPosition"]],
                on="dicom_id",
                how="left",
            )
            if self.view_filter and "ViewPosition" in split_df.columns:
                before = len(split_df)
                split_df = split_df[
                    split_df["ViewPosition"].isin(self.view_filter)
                ].reset_index(drop=True)
                print(f"[MIMICCXRDataset] ViewPosition filter={self.view_filter}  "
                      f"{before} → {len(split_df)} samples")
        else:
            if self.view_filter:
                print(f"[MIMICCXRDataset] WARNING: --view_filter set but "
                      f"metadata.csv not found at {self.meta_csv}. Filter skipped.")

        samples = []
        for _, row in split_df.iterrows():
            subject_id = int(row["subject_id"])
            study_id   = int(row["study_id"])
            dicom_id   = str(row["dicom_id"])

            patient_dir = self._get_patient_dir(subject_id)
            dcm_path    = os.path.join(patient_dir, f"s{study_id}", f"{dicom_id}.dcm")
            report_path = os.path.join(patient_dir, f"s{study_id}.txt")

            samples.append({
                "dcm_path"   : dcm_path,
                "report_path": report_path,
                "study_id"   : f"s{study_id}",
                "patient_id" : f"p{subject_id}",
            })

        return samples

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_patient_dir(self, subject_id: int) -> str:
        """subject_id=10000032 → files/p10/p10000032"""
        pid_str = f"p{subject_id}"
        prefix  = pid_str[:3]
        return os.path.join(self.files_dir, prefix, pid_str)

    def _load_dcm(self, dcm_path: str):
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
        Output: "FINDINGS: <text> IMPRESSION: <text>"
        Falls back to full text if no sections are found.
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

    # =========================================================================
    # Legacy – Mode C (folder scan + patient-level split)
    # Used when mimic-cxr-2.0.0-split.csv is not available.
    # =========================================================================

    def _build_from_folder(self) -> list:
        all_samples = self._scan_all_folders()

        if self.split is None:
            return all_samples

        patient_ids = sorted(set(s["patient_id"] for s in all_samples))
        random.seed(42)
        random.shuffle(patient_ids)

        n       = len(patient_ids)
        n_train = int(n * 0.8)
        n_val   = int(n * 0.1)

        split_map = {
            "train"   : set(patient_ids[:n_train]),
            "validate": set(patient_ids[n_train : n_train + n_val]),
            "test"    : set(patient_ids[n_train + n_val:]),
        }

        target = split_map.get(self.split, set())
        return [s for s in all_samples if s["patient_id"] in target]

    def _scan_all_folders(self) -> list:
        samples = []

        for prefix in sorted(os.listdir(self.files_dir)):
            prefix_dir = os.path.join(self.files_dir, prefix)
            if not os.path.isdir(prefix_dir) or not prefix.startswith("p"):
                continue

            for patient_id in sorted(os.listdir(prefix_dir)):
                patient_dir = os.path.join(prefix_dir, patient_id)
                if not os.path.isdir(patient_dir):
                    continue

                for entry in sorted(os.listdir(patient_dir)):
                    study_dir = os.path.join(patient_dir, entry)
                    if not os.path.isdir(study_dir) or not entry.startswith("s"):
                        continue

                    report_path = os.path.join(patient_dir, f"{entry}.txt")

                    for fname in sorted(os.listdir(study_dir)):
                        if not fname.endswith(".dcm"):
                            continue
                        samples.append({
                            "dcm_path"   : os.path.join(study_dir, fname),
                            "report_path": report_path,
                            "study_id"   : entry,
                            "patient_id" : patient_id,
                        })

        return samples


# =============================================================================
# prepare_split_dirs  –  one-time file organisation (--make_split_dir)
# =============================================================================

def prepare_split_dirs(args) -> dict:
    """
    Physically organise MIMIC-CXR files into per-split subdirectories.

    Mirrors the original p10/pXXX/sYYY/ hierarchy under each split folder:
        <split_dir>/
        ├── train/    └── p10/ └── p10000032/ └── s50414267/ └── <id>.dcm
        ├── validate/
        └── test/

    Args:
        args : Namespace — uses root_path, split_dir, use_symlink

    Returns:
        dict  split → number of samples  e.g. {'train': 227827, ...}

    Raises:
        FileExistsError if any split subdirectory already exists.
    """
    split_dir = args.split_dir or os.path.join(args.root_path, "split")
    os.makedirs(split_dir, exist_ok=True)

    counts = {}
    for split in ["train", "validate", "test"]:
        dest_split = os.path.join(split_dir, split)
        if os.path.exists(dest_split):
            raise FileExistsError(
                f"'{dest_split}' already exists. "
                "Remove it manually before re-running prepare_split_dirs()."
            )

        print(f"\n[prepare_split_dirs] Building '{split}' sample list ...")

        split_args = argparse.Namespace(**vars(args))
        split_args.split = split
        ds = MIMICCXRDataset(split_args)

        print(f"[prepare_split_dirs] Writing {len(ds.samples)} entries → {dest_split}")
        n = 0
        for meta in ds.samples:
            src_dcm    = meta["dcm_path"]
            src_report = meta["report_path"]

            rel_dcm    = os.path.relpath(src_dcm,    os.path.join(args.root_path, "files"))
            rel_report = os.path.relpath(src_report, os.path.join(args.root_path, "files"))

            dst_dcm    = os.path.join(dest_split, rel_dcm)
            dst_report = os.path.join(dest_split, rel_report)

            os.makedirs(os.path.dirname(dst_dcm),    exist_ok=True)
            os.makedirs(os.path.dirname(dst_report), exist_ok=True)

            for src, dst in [(src_dcm, dst_dcm), (src_report, dst_report)]:
                if os.path.exists(dst) or os.path.islink(dst):
                    continue
                if not os.path.exists(src):
                    continue
                if args.use_symlink:
                    os.symlink(os.path.abspath(src), dst)
                else:
                    shutil.copy2(src, dst)

            n += 1
            if n % 10000 == 0:
                print(f"  ... {n}/{len(ds.samples)}")

        counts[split] = n
        print(f"[prepare_split_dirs] '{split}' done: {n} samples")

    print(f"\n[prepare_split_dirs] Complete → {split_dir}")
    return counts


# =============================================================================
# dataset_loader  –  DataLoader with BioBERT collate (used by training scripts)
# =============================================================================

def dataset_loader(
    args,
    embedder,
    split      : str  = None,
    num_workers: int  = 0,
    drop_last  : bool = True,
) -> DataLoader:
    """
    Build a DataLoader for MIMIC-CXR that yields (images, context).

    Wraps MIMICCXRDataset with a BioBERT collate_fn so each batch contains:
        images  : Tensor [B, 1, H, W]
        context : Tensor [B, seq_len, output_dim]  (BioBERT embedding)

    Args:
        args        : Namespace from parse_args() (or compatible Namespace)
        embedder    : BioBERTEmbedder instance already moved to target device
        split       : override args.split if provided
        num_workers : use 0 when embedder is on GPU to avoid CUDA fork errors
        drop_last   : True required for DP-SGD UniformWithReplacementSampler

    Returns:
        DataLoader yielding (images [B,1,H,W], context [B,seq_len,output_dim])

    Example (training script):
        from Data.mimic_cxr import dataset_loader
        train_loader = dataset_loader(args, embedder, split='train')
    """
    if BioBERTEmbedder is None:
        raise ImportError("BioBERTEmbedder could not be imported. "
                          "Check Modules/BioBERT_embedder.py path.")

    loader_args = argparse.Namespace(**vars(args))
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
        num_workers = num_workers,
        collate_fn  = collate_fn,
        drop_last   = drop_last,
        pin_memory  = False,
    )

    print(f"[dataset_loader] split={loader_args.split}  "
          f"samples={len(dataset)}  bs={args.batch_size}  "
          f"steps={len(loader)}")
    return loader


# =============================================================================
# Main  –  image / context 확인 (dataset.py 방식)
# =============================================================================

if __name__ == "__main__":
    args = parse_args()

    if args.make_split_dir:
        prepare_split_dirs(args)

    if BioBERTEmbedder is None:
        raise ImportError(
            "BioBERTEmbedder not found. "
            "Check /home/hjchoi/PycharmProjects/PrivaText-CXR/Modules/BioBERT_embedder.py"
        )

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = BioBERTEmbedder(model_path=args.biobert_path).to(device)

    dataloader = dataset_loader(args, embedder)

    for batch_id, data in enumerate(dataloader):
        if batch_id == 1:
            break
        image, context = data[0], data[1]
        print(f"image  : {image.shape}  "
              f"min={image.min():.3f}  max={image.max():.3f}")
        print(f"context: {context.shape}  "       # [B, seq_len, output_dim]
              f"dtype={context.dtype}")
