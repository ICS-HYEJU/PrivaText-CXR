"""
Data/mimic_cxr.py  –  MIMIC-CXR Map-Style Dataset
---------------------------------------------------
Map-style dataset compatible with DP-SGD (Opacus UniformWithReplacementSampler).
Each sample returns (image_tensor, report_str) for direct use with BioBERT embedder.

Directory structure:
    <root_dir>/
    ├── files/
    │   └── p10/
    │       └── p10000032/
    │           ├── s50414267/
    │           │   └── <dicom_id>.dcm
    │           └── s50414267.txt
    ├── mimic-cxr-2.0.0-split.csv      (Mode A – official split)
    ├── mimic-cxr-2.0.0-metadata.csv   (Mode A – optional ViewPosition filter)
    └── cxr-record-list.csv.gz         (Mode B – patient-level 80/10/10 split)

Three modes (tried in order):
    Mode A  mimic-cxr-2.0.0-split.csv found    → official train / validate / test split
    Mode B  cxr-record-list.csv.gz found        → record CSV, patient-level 80/10/10 split
    Mode C  neither found                       → folder scan, patient-level 80/10/10 split

Usage:
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    dataset = MIMICCXRDataset(root_dir='/path/to/mimic-cxr/2.1.0',
                              split='train', transform=transform)
    img, report = dataset[0]   # Tensor[1,256,256], str

LDM make_batch example:
    def make_batch():
        img, report = next(gen)
        img = img.to(device)
        ctx = biobert_embedder([report]).detach()   # [1, seq_len, 768]
        return {'image': img, 'context': ctx}
"""

import os
import re
import random
import time
from typing import Optional, Callable

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

import torch
import torch.nn as nn
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import AutoTokenizer, AutoModel


# =============================================================================
# MIMICCXRDataset
# =============================================================================

class MIMICCXRDataset(Dataset):
    """
    MIMIC-CXR Map-Style Dataset.

    Args:
        root_dir   : root path (e.g. /home/hjchoi/physionet.org/files/mimic-cxr/2.1.0)
        split      : 'train' | 'validate' | 'test' | None (None = full dataset)
        transform  : torchvision transform applied to the PIL image
        max_length : max character length of report text (None = no limit)
        image_size : fallback image size used for blank tensors on load error
    """

    SPLIT_CSV  = "mimic-cxr-2.0.0-split.csv"
    META_CSV   = "mimic-cxr-2.0.0-metadata.csv"
    RECORD_CSV = "cxr-record-list.csv.gz"

    def __init__(
        self,
        root_dir  : str,
        split     : Optional[str] = "train",
        transform : Optional[Callable] = None,
        max_length: Optional[int] = None,
        image_size: int = 256,
    ):
        super().__init__()

        self.root_dir   = root_dir
        self.files_dir  = os.path.join(root_dir, "files")
        self.split      = split
        self.transform  = transform
        self.max_length = max_length
        self.image_size = image_size

        if transform is None:
            print("[MIMICCXRDataset] WARNING: transform=None. "
                  "DataLoader will fail to collate PIL Images. "
                  "Pass a torchvision transform.")

        split_csv  = os.path.join(root_dir, self.SPLIT_CSV)
        meta_csv   = os.path.join(root_dir, self.META_CSV)
        record_csv = os.path.join(root_dir, self.RECORD_CSV)

        if os.path.exists(split_csv):
            self._mode   = "A"
            print("[MIMICCXRDataset] Mode A: official split CSV found")
            self.samples = self._build_from_csv(split_csv, meta_csv, split)
        elif os.path.exists(record_csv):
            self._mode   = "B"
            print("[MIMICCXRDataset] Mode B: cxr-record-list.csv.gz found → patient-level split")
            self.samples = self._build_from_record_csv(record_csv, split)
        else:
            self._mode   = "C"
            print("[MIMICCXRDataset] Mode C: no CSV found → folder scan + patient-level split")
            self.samples = self._build_from_folder(split)

        print(f"[MIMICCXRDataset] split='{split}'  total={len(self.samples)}")

    # -------------------------------------------------------------------------
    # Map-style interface
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns:
            image  : Tensor [1, H, W] in [-1, 1]  (after Normalize(0.5,0.5))
                     Falls back to zero tensor on DICOM load error.
            report : str  –  FINDINGS + IMPRESSION text for BioBERT embedding
        """
        meta = self.samples[idx]

        # ── Image ─────────────────────────────────────────────────────────────
        try:
            image = self._load_dcm(meta["dcm_path"])
        except Exception as e:
            print(f"  [MIMICCXRDataset] WARNING: failed to load {meta['dcm_path']}: {e}")
            image = self._blank_image()

        # ── Report ────────────────────────────────────────────────────────────
        report = self._load_report(meta["report_path"])

        return image, report

    # -------------------------------------------------------------------------
    # Mode A – official CSV split
    # -------------------------------------------------------------------------

    def _build_from_csv(self, split_csv: str, meta_csv: str,
                         split: Optional[str]) -> list:
        split_df = pd.read_csv(split_csv)

        if split is not None:
            split_df = split_df[split_df["split"] == split].reset_index(drop=True)

        if os.path.exists(meta_csv):
            meta_df  = pd.read_csv(meta_csv)
            split_df = pd.merge(
                split_df,
                meta_df[["dicom_id", "ViewPosition"]],
                on="dicom_id",
                how="left",
            )

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
    # Mode B – cxr-record-list.csv.gz + patient-level split
    # -------------------------------------------------------------------------

    def _build_from_record_csv(self, record_csv: str, split: Optional[str]) -> list:
        """
        Build sample list from cxr-record-list.csv.gz.

        Expected columns: subject_id, study_id, dicom_id, path
            path  e.g. "files/p10/p10000032/s50414267/<dicom_id>.dcm"

        Patient-level 80/10/10 random split (seed=42).
        """
        df = pd.read_csv(record_csv)

        if not {"subject_id", "study_id", "dicom_id", "path"}.issubset(df.columns):
            raise ValueError(
                f"cxr-record-list.csv.gz must contain columns "
                f"[subject_id, study_id, dicom_id, path]. "
                f"Found: {list(df.columns)}"
            )

        if split is not None:
            patient_ids = sorted(df["subject_id"].unique().tolist())
            random.seed(42)
            random.shuffle(patient_ids)
            n        = len(patient_ids)
            n_train  = int(n * 0.8)
            n_val    = int(n * 0.1)
            split_map = {
                "train"   : set(patient_ids[:n_train]),
                "validate": set(patient_ids[n_train : n_train + n_val]),
                "test"    : set(patient_ids[n_train + n_val :]),
            }
            keep = split_map.get(split, set())
            df   = df[df["subject_id"].isin(keep)].reset_index(drop=True)

        samples = []
        for _, row in df.iterrows():
            subject_id = int(row["subject_id"])
            study_id   = int(row["study_id"])
            rel_path   = str(row["path"])           # relative to root_dir

            dcm_path    = os.path.join(self.root_dir, rel_path)
            patient_dir = self._get_patient_dir(subject_id)
            report_path = os.path.join(patient_dir, f"s{study_id}.txt")

            samples.append({
                "dcm_path"   : dcm_path,
                "report_path": report_path,
                "study_id"   : f"s{study_id}",
                "patient_id" : f"p{subject_id}",
            })

        return samples

    # -------------------------------------------------------------------------
    # Mode C – folder scan + patient-level split
    # -------------------------------------------------------------------------

    def _build_from_folder(self, split: Optional[str]) -> list:
        all_samples = self._scan_all_folders()

        if split is None:
            return all_samples

        patient_ids = sorted(set(s["patient_id"] for s in all_samples))
        random.seed(42)
        random.shuffle(patient_ids)

        n       = len(patient_ids)
        n_train = int(n * 0.8)
        n_val   = int(n * 0.1)

        split_map = {
            "train"   : set(patient_ids[:n_train]),
            "validate": set(patient_ids[n_train: n_train + n_val]),
            "test"    : set(patient_ids[n_train + n_val:]),
        }

        target = split_map.get(split, set())
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

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_patient_dir(self, subject_id: int) -> str:
        """e.g. subject_id=10000032 → files/p10/p10000032"""
        pid_str = f"p{subject_id}"
        prefix  = pid_str[:3]          # "p10"
        return os.path.join(self.files_dir, prefix, pid_str)

    def _load_dcm(self, dcm_path: str):
        """
        Load DICOM → normalised PIL "L" → transform → Tensor [1, H, W].

        Fixes vs naive approach:
          - MONOCHROME1 inversion  (lung = bright → needs flip)
          - np.clip before uint8 cast to prevent overflow
        """
        dcm = pydicom.dcmread(dcm_path)
        arr = dcm.pixel_array.astype(np.float32)

        # Invert MONOCHROME1 so that lung fields are always dark
        if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr

        # Normalise to [0, 255] then clip to prevent uint8 overflow
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        arr = np.clip(arr, 0, 255)

        image = Image.fromarray(arr.astype(np.uint8)).convert("L")

        if self.transform:
            image = self.transform(image)

        return image

    def _blank_image(self):
        """Return a zero tensor (or blank PIL) when DICOM load fails."""
        if self.transform:
            return torch.zeros(1, self.image_size, self.image_size)
        return Image.fromarray(
            np.zeros((self.image_size, self.image_size), dtype=np.uint8), "L"
        )

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

        Output format:
            "FINDINGS: <text> IMPRESSION: <text>"

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


# =============================================================================
# BioBERTEmbedder
# =============================================================================

class BioBERTEmbedder(nn.Module):
    """
    BioBERT-based text embedder for radiology reports.

    Input  : list[str]  – report texts (FINDINGS: ~ IMPRESSION: ~)
    Output : Tensor [B, seq_len, output_dim]

    Args:
        model_path : local path or HuggingFace model ID
                     e.g. "/storage/hjchoi/biobert"
                          "dmis-lab/biobert-base-cased-v1.2"
        output_dim : projection output dim; must match UNet context_dim (default 512)
        freeze     : freeze BioBERT weights (True for DP fine-tuning)
        max_length : tokeniser max length in tokens (default 128)
    """

    def __init__(
        self,
        model_path : str,
        output_dim : int  = 512,
        freeze     : bool = True,
        max_length : int  = 128,
    ):
        super().__init__()
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bert      = AutoModel.from_pretrained(model_path)   # hidden_dim = 768
        self.proj      = nn.Linear(768, output_dim)

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

        n_proj = sum(p.numel() for p in self.proj.parameters())
        print(f"[BioBERTEmbedder] model={model_path}  "
              f"output_dim={output_dim}  freeze={freeze}  "
              f"proj_params={n_proj:,}")

    @property
    def device(self):
        return next(self.bert.parameters()).device

    def forward(self, texts: list) -> torch.Tensor:
        """
        Args:
            texts : list[str]
        Returns:
            Tensor [B, seq_len, output_dim]
        """
        enc = self.tokenizer(
            texts,
            return_tensors = "pt",
            padding        = True,
            truncation     = True,
            max_length     = self.max_length,
        ).to(self.device)

        ctx = nullcontext() if self.bert.training else torch.no_grad()
        with ctx:
            out = self.bert(**enc)              # last_hidden_state [B, seq_len, 768]

        return self.proj(out.last_hidden_state) # [B, seq_len, output_dim]


# =============================================================================
# DataLoader builder
# =============================================================================

def _default_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def build_mimic_loader(
    root_path  : str,
    embedder   : BioBERTEmbedder,
    split      : str  = "train",
    image_size : int  = 256,
    batch_size : int  = 8,
    num_workers: int  = 0,
    max_length : int  = 512,
    drop_last  : bool = True,
) -> DataLoader:
    """
    Build a DataLoader for MIMIC-CXR that yields
        (images, context)
    where context is the BioBERT embedding of each report.

    Args:
        root_path   : dataset root directory
        embedder    : BioBERTEmbedder instance (already .to(device))
        split       : 'train' | 'validate' | 'test'
        image_size  : spatial resolution after resize
        batch_size  : logical batch size
        num_workers : worker processes (use 0 when embedder is on GPU)
        max_length  : report text character truncation before tokenisation
        drop_last   : drop the last incomplete batch (required for DP-SGD)

    Returns:
        DataLoader yielding (images [B,1,H,W], context [B,seq_len,output_dim])

    Note:
        BioBERT embedding runs inside collate_fn, which executes in the
        MAIN process regardless of num_workers. GPU embedding is therefore
        safe, but set num_workers=0 if you encounter CUDA multiprocessing
        errors.
    """
    dataset = MIMICCXRDataset(
        root_dir   = root_path,
        split      = split,
        transform  = _default_transform(image_size),
        max_length = max_length,
        image_size = image_size,
    )

    def collate_fn(batch):
        images, reports = zip(*batch)
        images  = torch.stack(images)                       # [B, 1, H, W]
        context = embedder(list(reports))                   # [B, seq_len, output_dim]
        return images, context

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = (split == "train"),
        num_workers = num_workers,
        collate_fn  = collate_fn,
        drop_last   = drop_last,
        pin_memory  = False,   # moved to device inside collate_fn
    )

    print(f"[build_mimic_loader] split={split}  "
          f"samples={len(dataset)}  bs={batch_size}  "
          f"steps={len(loader)}")
    return loader


# =============================================================================
# Debug utilities
# =============================================================================

def _sep(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def debug_init(root_dir: str, split: str = "train") -> MIMICCXRDataset:
    _sep(f"Debug 1: init  split={split}")
    print(f"  root exists   : {os.path.exists(root_dir)}")
    print(f"  files exists  : {os.path.exists(os.path.join(root_dir, 'files'))}")
    print(f"  split CSV (A) : {os.path.exists(os.path.join(root_dir, MIMICCXRDataset.SPLIT_CSV))}")
    print(f"  record CSV (B): {os.path.exists(os.path.join(root_dir, MIMICCXRDataset.RECORD_CSV))}")

    ds = MIMICCXRDataset(root_dir=root_dir, split=split, max_length=512)
    print(f"  mode={ds._mode}  samples={len(ds)}")
    if ds.samples:
        for k, v in ds.samples[0].items():
            print(f"    {k}: {v}")
    return ds


def debug_single(ds: MIMICCXRDataset, idx: int = 0):
    _sep(f"Debug 2: single item  idx={idx}")
    meta = ds.samples[idx]
    print(f"  dcm exists    : {os.path.exists(meta['dcm_path'])}")
    print(f"  report exists : {os.path.exists(meta['report_path'])}")

    t0 = time.time()
    img, report = ds[idx]
    print(f"  load time : {time.time()-t0:.3f}s")

    if hasattr(img, "shape"):
        print(f"  img shape : {img.shape}  dtype={img.dtype}"
              f"  min={img.min():.3f}  max={img.max():.3f}")
    else:
        print(f"  img size  : {img.size}  mode={img.mode}")

    print(f"  report len: {len(report)} chars")
    print(f"  report[:200]: {repr(report[:200])}")


def debug_multi(ds: MIMICCXRDataset, n: int = 5):
    _sep(f"Debug 3: multi-sample  n={n}")
    indices       = random.sample(range(len(ds)), min(n, len(ds)))
    times, errors = [], []

    for i, idx in enumerate(indices):
        try:
            t0 = time.time()
            img, report = ds[idx]
            elapsed = time.time() - t0
            times.append(elapsed)
            print(f"  [{i+1}/{n}] idx={idx:6d}  "
                  f"patient={ds.samples[idx]['patient_id']}  "
                  f"report={len(report):4d}c  t={elapsed:.3f}s")
        except Exception as e:
            errors.append((idx, str(e)))
            print(f"  [{i+1}/{n}] idx={idx:6d}  ERROR: {e}")

    if times:
        print(f"\n  avg={sum(times)/len(times):.3f}s  errors={len(errors)}/{n}")


def debug_loader(ds: MIMICCXRDataset, batch_size: int = 4):
    _sep(f"Debug 4: DataLoader  bs={batch_size}")
    has_tf = ds.transform is not None
    if not has_tf:
        print("  no transform → injecting minimal one for collation")
        ds.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    imgs, reports = next(iter(loader))

    print(f"  image shape : {imgs.shape}")
    print(f"  image dtype : {imgs.dtype}")
    print(f"  report lens : {[len(r) for r in reports]}")

    if not has_tf:
        ds.transform = None


def debug_reports(ds: MIMICCXRDataset, n: int = 3):
    _sep(f"Debug 5: report parsing  n={n}")
    shown = 0
    for meta in ds.samples:
        if not os.path.exists(meta["report_path"]):
            continue
        with open(meta["report_path"], "r", encoding="utf-8") as f:
            raw = f.read()
        parsed = ds._parse_report(raw)
        print(f"\n  {os.path.basename(meta['report_path'])}")
        print(f"  RAW    : {raw[:300].strip()!r}")
        print(f"  PARSED : {parsed[:200]!r}")
        shown += 1
        if shown >= n:
            break


# =============================================================================
# Main
# =============================================================================

def main():
    ROOT_DIR = "/home/hjchoi/physionet.org/files/mimic-cxr/2.1.0"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    ds = debug_init(ROOT_DIR, split="train")
    if len(ds) == 0:
        print("ERROR: no samples found.")
        return

    ds = MIMICCXRDataset(
        root_dir   = ROOT_DIR,
        split      = "train",
        transform  = transform,
        max_length = 512,
    )

    debug_single(ds, idx=0)
    debug_multi(ds,  n=5)
    debug_loader(ds, batch_size=4)
    debug_reports(ds, n=3)

    _sep("Done")
    print(f"  mode={ds._mode}  split={ds.split}  total={len(ds)}")


if __name__ == "__main__":
    main()
