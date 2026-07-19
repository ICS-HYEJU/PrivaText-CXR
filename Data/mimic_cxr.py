"""
Data/mimic_cxr.py  -  MIMIC-CXR Map-Style Dataset
---------------------------------------------------
Map-style dataset compatible with DP-SGD (Opacus UniformWithReplacementSampler).
Each sample returns (image_tensor, report_str).

Reads directly from the original MIMIC-CXR PhysioNet directory using the
official split CSV (mimic-cxr-2.0.0-split.csv).  No pre-built split
directories required.

Expected layout:
    <root_path>/
    ├── files/
    │   ├── p10/
    │   │   └── p10000032/
    │   │       ├── s50414267/
    │   │       │   └── <dicom_id>.dcm
    │   │       └── s50414267.txt
    │   └── p11/ ... p19/
    └── mimic-cxr-2.0.0-split.csv

The CSV has columns: dicom_id, subject_id, study_id, split
  split values: 'train', 'validate', 'test'

DICOM files that do not yet exist on disk are silently skipped so the
dataset works with a partial download (e.g. only p10 + p11 downloaded).

External usage (training scripts):
    from Data.mimic_cxr import dataset_loader
    train_loader = dataset_loader(args, embedder, split='train')
"""

import os
import re
import sys
import argparse

import numpy as np
import pandas as pd
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
    parser.add_argument("--device_id", type=int, default=1)

    # Paths
    parser.add_argument("--root_path", type=str,
                        default='/storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0',
                        help="Root of original MIMIC-CXR PhysioNet download "
                             "(contains files/ and mimic-cxr-2.0.0-split.csv)")
    parser.add_argument("--split_csv", type=str,
                        default='mimic-cxr-2.0.0-split.csv',
                        help="Split CSV filename (relative to root_path, or absolute path)")

    # Dataset
    parser.add_argument("--split",       type=str, default="train",
                        choices=["train", "validate", "test"])
    parser.add_argument("--image_size",  type=int, default=256)
    parser.add_argument("--max_length",  type=int, default=512)

    # Prompt / conditioning source
    parser.add_argument("--prompt_mode", type=str, default="report",
                        choices=["report", "full", "label"],
                        help="'report'=FINDINGS+IMPRESSION (truncated), "
                             "'full'=entire .txt, 'label'=CheXpert pathologies")
    parser.add_argument("--chexpert_csv", type=str,
                        default="mimic-cxr-2.0.0-chexpert.csv",
                        help="CheXpert label CSV (relative to root_path or "
                             "absolute); used only when prompt_mode='label'")

    # DataLoader
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--drop_last",   type=bool, default=True)

    # BioBERT
    parser.add_argument("--biobert_path", type=str, default="/storage/hjchoi")

    return parser.parse_args()


# =============================================================================
# Prompt / label configuration
# =============================================================================

# Default set of CheXpert pathology columns used when prompt_mode='label'.
# These are the classifiable findings requested for alignment evaluation.
# Column names must match mimic-cxr-2.0.0-chexpert.csv exactly.
DEFAULT_LABEL_SET = [
    "Pleural Effusion",
    "Cardiomegaly",
    "Edema",
    "Pneumothorax",
    "Lung Opacity",
    "No Finding",
]


# =============================================================================
# MIMICCXRDataset
# =============================================================================

class MIMICCXRDataset(Dataset):
    """
    MIMIC-CXR Map-Style Dataset.

    Reads the official split CSV to build an idx-mapped sample list at init
    (lightweight path strings only).  Each __getitem__ call loads one DICOM
    file on demand.  DICOM files not present on disk are silently skipped,
    so the dataset works with a partial download.

    Args:
        args : Namespace - requires root_path, split_csv, split, image_size,
                           max_length
    """
    def __init__(self, args):
        super().__init__()

        self.split      = args.split
        self.image_size = args.image_size
        self.max_length = getattr(args, "max_length", 512)

        self.root_path = args.root_path
        self.files_dir = os.path.join(self.root_path, "files")

        # Resolve split CSV path (absolute or relative to root_path)
        split_csv = getattr(args, "split_csv", "mimic-cxr-2.0.0-split.csv")
        if os.path.isabs(split_csv):
            self.split_csv = split_csv
        else:
            self.split_csv = os.path.join(self.root_path, split_csv)

        if not os.path.isfile(self.split_csv):
            raise FileNotFoundError(
                f"[MIMICCXRDataset] split CSV not found: {self.split_csv}"
            )
        if not os.path.isdir(self.files_dir):
            raise FileNotFoundError(
                f"[MIMICCXRDataset] files/ directory not found: {self.files_dir}"
            )

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        # Optional patient-level filter for DP budget-isolated splits
        # (D_search / D_train / D_test).  When set, only samples whose
        # patient_id (e.g. "p10000032") is in this set are kept.  Patient-level
        # filtering guarantees the image-level disjointness required by
        # parallel composition.  See docs/DP_LDM_FRAMEWORK_PLAN.md §2.
        wl = getattr(args, "patient_whitelist", None)
        self.patient_whitelist = set(wl) if wl is not None else None

        # ── Prompt mode ───────────────────────────────────────────────────────
        # 'report' : FINDINGS + IMPRESSION sections, truncated to max_length
        #            characters (default, matches training conditioning).
        # 'full'   : entire report .txt (whitespace-collapsed). NOTE: BioBERT
        #            still caps at max_length TOKENS at tokenization time.
        # 'label'  : classifiable pathology names from the CheXpert CSV
        #            (e.g. "Pleural Effusion, Cardiomegaly"), for alignment eval.
        self.prompt_mode = getattr(args, "prompt_mode", "report")
        self.label_set = list(getattr(args, "label_set", None) or DEFAULT_LABEL_SET)
        self.study_labels = None
        if self.prompt_mode == "label":
            self.study_labels = self._load_chexpert_labels(args)

        self.samples = self._build_index()
        if self.patient_whitelist is not None:
            self.samples = [s for s in self.samples
                            if s["patient_id"] in self.patient_whitelist]
        print(f"[MIMICCXRDataset] split='{self.split}'  total={len(self.samples)}"
              + f"  prompt_mode='{self.prompt_mode}'"
              + (f"  (patient_whitelist={len(self.patient_whitelist)} patients)"
                 if self.patient_whitelist is not None else ""))

    # -------------------------------------------------------------------------
    # Map-style interface
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns:
            image  : Tensor [1, H, W]  normalised to [-1, 1]
            report : str  - "FINDINGS: <...> IMPRESSION: <...>"
        """
        meta = self.samples[idx]

        try:
            image = self._load_dcm(meta["dcm_path"])
        except Exception as e:
            print(f"[MIMICCXRDataset] load error idx={idx}: {e}")
            image = self._blank_image()

        report = self._get_prompt(meta)
        return image, report

    # -------------------------------------------------------------------------
    # Index build
    # -------------------------------------------------------------------------

    def _build_index(self) -> list:
        """
        Read the split CSV, construct DICOM and report paths, and keep only
        rows whose DICOM file exists on disk.
        """
        df = pd.read_csv(self.split_csv)

        if self.split:
            df = df[df["split"] == self.split].reset_index(drop=True)

        samples = []
        missing = 0

        for _, row in df.iterrows():
            subject_id = int(row["subject_id"])
            study_id   = int(row["study_id"])
            dicom_id   = str(row["dicom_id"]).strip()

            pid_str    = f"p{subject_id}"
            prefix     = pid_str[:3]            # e.g. "p10"

            patient_dir = os.path.join(self.files_dir, prefix, pid_str)
            study_dir   = os.path.join(patient_dir, f"s{study_id}")
            dcm_path    = os.path.join(study_dir, f"{dicom_id}.dcm")
            report_path = os.path.join(patient_dir, f"s{study_id}.txt")

            if not os.path.exists(dcm_path):
                missing += 1
                continue

            samples.append({
                "dcm_path"    : dcm_path,
                "report_path" : report_path,
                "study_id"    : f"s{study_id}",
                "patient_id"  : pid_str,
                "subject_id"  : subject_id,   # numeric, for CheXpert join
                "study_id_num": study_id,      # numeric, for CheXpert join
            })

        if missing:
            print(f"[MIMICCXRDataset] {missing} DICOM files not yet downloaded, skipped.")

        return samples

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

    def _get_prompt(self, meta: dict) -> str:
        """
        Build the conditioning text for one sample according to prompt_mode:
            'label'  -> classifiable pathology names from CheXpert CSV
            'full'   -> entire report .txt (whitespace-collapsed)
            'report' -> FINDINGS + IMPRESSION, char-truncated (default)
        """
        if self.prompt_mode == "label":
            return self._label_prompt(meta)
        if self.prompt_mode == "full":
            return self._load_full_report(meta["report_path"])
        return self._load_report(meta["report_path"])

    def _label_prompt(self, meta: dict) -> str:
        """
        Comma-joined positive CheXpert findings for this study.
        Falls back to 'No Finding' when nothing (other than No Finding) is
        positive or the study is absent from the label CSV.
        """
        key = (meta["subject_id"], meta["study_id_num"])
        positives = self.study_labels.get(key, [])
        findings = [p for p in positives if p != "No Finding"]
        if findings:
            return ", ".join(findings)
        return "No Finding"

    def _load_full_report(self, report_path: str) -> str:
        """Entire report text with whitespace/newlines collapsed to spaces.

        No character truncation here; the BioBERT tokenizer still caps the
        sequence at max_length TOKENS, so extremely long reports are bounded
        at embedding time rather than silently cut mid-sentence.
        """
        if not os.path.exists(report_path):
            return ""
        with open(report_path, "r", encoding="utf-8") as f:
            raw = f.read()
        return " ".join(raw.split())

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

    # -------------------------------------------------------------------------
    # CheXpert labels (prompt_mode='label')
    # -------------------------------------------------------------------------

    def _load_chexpert_labels(self, args) -> dict:
        """
        Read mimic-cxr-2.0.0-chexpert.csv and return
            {(subject_id, study_id) -> [positive label names in self.label_set]}

        Only value == 1.0 counts as positive; uncertain (-1.0), negative (0.0)
        and blank are treated as absent.
        """
        csv = getattr(args, "chexpert_csv", "mimic-cxr-2.0.0-chexpert.csv")
        path = csv if os.path.isabs(csv) else os.path.join(self.root_path, csv)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[MIMICCXRDataset] prompt_mode='label' needs the CheXpert label "
                f"CSV but it was not found: {path}\n"
                f"  Download mimic-cxr-2.0.0-chexpert.csv into {self.root_path} "
                f"or pass --chexpert_csv <path>."
            )

        df = pd.read_csv(path)
        cols = [c for c in self.label_set if c in df.columns]
        missing_cols = [c for c in self.label_set if c not in df.columns]
        if missing_cols:
            print(f"[MIMICCXRDataset] WARNING: label columns not in CheXpert CSV, "
                  f"ignored: {missing_cols}")

        labels = {}
        for _, row in df.iterrows():
            key = (int(row["subject_id"]), int(row["study_id"]))
            labels[key] = [c for c in cols if row[c] == 1.0]

        print(f"[MIMICCXRDataset] CheXpert labels loaded: {len(labels)} studies  "
              f"label_set={cols}")
        return labels


# =============================================================================
# dataset_loader  -  DataLoader with BioBERT collate (used by training scripts)
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
