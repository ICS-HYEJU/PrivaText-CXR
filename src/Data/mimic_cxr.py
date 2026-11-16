"""
Data/mimic_cxr.py  -  MIMIC-CXR Map-Style Dataset
---------------------------------------------------
Map-style dataset compatible with DP-SGD (Opacus UniformWithReplacementSampler).
Each sample returns (image_tensor, report_str).

Two input modes are supported (auto-detected from the args passed in):

1. PhysioNet mode  (pass `root_path` [+ optional `split_csv`])
   Reads directly from the original MIMIC-CXR PhysioNet directory using the
   official split CSV (mimic-cxr-2.0.0-split.csv).  No pre-built split dirs.

       <root_path>/
       ├── files/
       │   ├── p10/p10000032/s50414267/<dicom_id>.dcm
       │   └── p10/p10000032/s50414267.txt
       └── mimic-cxr-2.0.0-split.csv

2. Pre-built split mode  (pass `prebuilt_split_dir`)
   Scans <prebuilt_split_dir>/<split>/ built by prepare_split_dirs().

       <prebuilt_split_dir>/
       ├── train/  ├── validate/  └── test/

DICOM files that are missing on disk are silently skipped (partial downloads).
DICOM files whose pixel data cannot be decoded (truncated / corrupted:
"number of bytes of pixel data is less than expected") are dropped at index
build when `validate_dicom` is enabled (default), with the validation result
cached to a JSON manifest so the (expensive) decode is paid only once.

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
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Modules.BioBERT_embedder import BioBERTEmbedder
except ImportError:
    BioBERTEmbedder = None
    print("[mimic_cxr] WARNING: BioBERTEmbedder not found. "
          "dataset_loader() will not be available.")


# =============================================================================
# Report section parsing (module-level, shared by the legacy path and the
# new text_mode path so both stay in sync)
# =============================================================================

_SECTION_RE = re.compile(
    r"(FINDINGS|IMPRESSION)\s*:(.*?)(?=\n[A-Z ]+:|$)",
    re.IGNORECASE | re.DOTALL,
)


def _split_sections(text: str) -> dict:
    """Map {'findings': ..., 'impression': ...} for whichever sections are present."""
    return {
        name.lower(): content.strip()
        for name, content in _SECTION_RE.findall(text)
    }


# =============================================================================
# CheXpert label loading (text_mode='LABEL' / 'LABEL+IMPRESSION')
# =============================================================================

def _load_chexpert_gt(csv_path: str):
    """
    Load CheXpert labels keyed by (subject_id, study_id).

    Returns:
        gt         : dict[(subject_id:int, study_id:int)] -> {label_col: float}
                      (blank cells -> NaN; -1 = uncertain, 1 = positive,
                      0 = negative, NaN = not mentioned)
        label_cols : list[str] - pathology columns, in the CSV's own column order
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    label_cols = [c for c in df.columns if c not in ("subject_id", "study_id")]
    subj = df["subject_id"].astype(int).to_numpy()
    stud = df["study_id"].astype(int).to_numpy()
    vals = df[label_cols].to_numpy(dtype=float)

    gt = {}
    for i in range(len(df)):
        gt[(int(subj[i]), int(stud[i]))] = {
            label_cols[j]: vals[i, j] for j in range(len(label_cols))
        }
    return gt, label_cols


def _labels_to_text(row: dict, label_cols) -> str:
    """
    Build 'LABEL: a, b, c' from a CheXpert GT row, keeping only columns that
    are positive (== 1.0). Returns None when no column is positive (caller
    drops the sample) - uncertain (-1) and missing (NaN) values are excluded,
    matching the project's existing CheXpert evaluation convention
    (see Eval_metric/downstream_cls.py).
    """
    positives = [col for col in label_cols if row.get(col) == 1.0]
    if not positives:
        return None
    return "LABEL: " + ", ".join(positives)


# =============================================================================
# Args
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="MIMIC-CXR Dataset")
    parser.add_argument("--device_id", type=int, default=1)

    # Paths  -  provide EITHER --root_path (PhysioNet mode) OR
    #           --prebuilt_split_dir (pre-built split mode)
    parser.add_argument("--root_path", type=str,
                        default='/storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0',
                        help="Root of original MIMIC-CXR PhysioNet download "
                             "(contains files/ and mimic-cxr-2.0.0-split.csv)")
    parser.add_argument("--split_csv", type=str,
                        default='mimic-cxr-2.0.0-split.csv',
                        help="Split CSV filename (relative to root_path, or absolute path)")
    parser.add_argument("--prebuilt_split_dir", type=str, default=None,
                        help="root of pre-built split dirs (train/validate/test); "
                             "when set, takes precedence over --root_path")

    # Dataset
    parser.add_argument("--split",       type=str, default="train",
                        choices=["train", "validate", "test"])
    parser.add_argument("--image_size",  type=int, default=256)
    parser.add_argument("--max_length",  type=int, default=512)

    # Conditioning text mode  -  None (default) keeps the legacy
    # FINDINGS+IMPRESSION concatenation for backward compatibility.
    parser.add_argument("--text_mode", type=str, default=None,
                        choices=["LABEL", "LABEL+IMPRESSION", "FINDINGS"],
                        help="Conditioning text source. Omit for the legacy "
                             "FINDINGS+IMPRESSION concatenation. 'LABEL' and "
                             "'LABEL+IMPRESSION' require --chexpert_csv "
                             "(or mimic-cxr-2.0.0-chexpert.csv under root_path).")
    parser.add_argument("--chexpert_csv", type=str, default=None,
                        help="CheXpert label CSV path (absolute, or relative "
                             "to root_path). Only used when --text_mode is "
                             "'LABEL' or 'LABEL+IMPRESSION'. Default: "
                             "mimic-cxr-2.0.0-chexpert.csv under root_path.")

    # DICOM integrity filtering
    parser.add_argument("--validate_dicom", action="store_true", default=True,
                        help="drop corrupted/unreadable DICOM files at index build")
    parser.add_argument("--no_validate_dicom", dest="validate_dicom",
                        action="store_false",
                        help="disable DICOM validation (keep every file)")
    parser.add_argument("--dicom_cache", type=str, default=None,
                        help="path to the DICOM validation cache JSON "
                             "(default: alongside the data / cwd)")

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

    Builds an idx-mapped sample list at init (lightweight path strings only).
    Each __getitem__ call loads one DICOM file on demand.

    Args:
        args : Namespace.  Provide EITHER
                 - root_path [+ split_csv]        (PhysioNet mode), OR
                 - prebuilt_split_dir             (pre-built split mode)
               plus split, image_size, max_length.
               Optional: patient_whitelist, validate_dicom, dicom_cache.
    """
    def __init__(self, args):
        super().__init__()

        self.split      = args.split
        self.image_size = args.image_size
        self.max_length = getattr(args, "max_length", 512)

        self.validate_dicom = getattr(args, "validate_dicom", True)
        self.dicom_cache    = getattr(args, "dicom_cache", None)
        self.manifest_csv   = getattr(args, "manifest_csv", None)

        # Conditioning text mode. None (default) = legacy FINDINGS+IMPRESSION
        # concatenation, unchanged for every existing caller that doesn't pass
        # text_mode explicitly (dataset_loader(), LDM_train.py, ...).
        self.text_mode = getattr(args, "text_mode", None)
        if self.text_mode not in (None, "LABEL", "LABEL+IMPRESSION", "FINDINGS"):
            raise ValueError(
                f"[MIMICCXRDataset] unknown text_mode={self.text_mode!r}; "
                "expected one of None, 'LABEL', 'LABEL+IMPRESSION', 'FINDINGS'"
            )
        self.chexpert_csv = getattr(args, "chexpert_csv", None)

        # Optional patient-level filter for DP budget-isolated splits
        # (D_search / D_train / D_test).  When set, only samples whose
        # patient_id (e.g. "p10000032") is in this set are kept.
        wl = getattr(args, "patient_whitelist", None)
        self.patient_whitelist = set(wl) if wl is not None else None

        # -- Resolve input mode ------------------------------------------------
        prebuilt  = getattr(args, "prebuilt_split_dir", None)
        root_path = getattr(args, "root_path", None)

        if self.manifest_csv:
            self.mode = "manifest"
            if not os.path.isfile(self.manifest_csv):
                raise FileNotFoundError(f"[MIMICCXRDataset] manifest CSV not found: {self.manifest_csv}")
            self.root_path = root_path
        elif prebuilt:
            self.mode      = "prebuilt"
            self.scan_root = os.path.join(prebuilt, self.split)
            if not os.path.isdir(self.scan_root):
                raise FileNotFoundError(
                    f"[MIMICCXRDataset] split dir not found: {self.scan_root}"
                )
        elif root_path:
            self.mode      = "physionet"
            self.root_path = root_path
            self.files_dir = os.path.join(self.root_path, "files")

            split_csv = getattr(args, "split_csv", "mimic-cxr-2.0.0-split.csv")
            self.split_csv = (split_csv if os.path.isabs(split_csv)
                              else os.path.join(self.root_path, split_csv))

            if not os.path.isfile(self.split_csv):
                raise FileNotFoundError(
                    f"[MIMICCXRDataset] split CSV not found: {self.split_csv}"
                )
            if not os.path.isdir(self.files_dir):
                raise FileNotFoundError(
                    f"[MIMICCXRDataset] files/ directory not found: {self.files_dir}"
                )
        else:
            raise ValueError(
                "[MIMICCXRDataset] provide either 'prebuilt_split_dir' or "
                "'root_path' in args."
            )

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.samples = self._build_index()
        if self.patient_whitelist is not None:
            self.samples = [s for s in self.samples
                            if s["patient_id"] in self.patient_whitelist]

        # LABEL / LABEL+IMPRESSION need a CheXpert row per sample. Do this
        # before DICOM validation (which decodes pixel data - expensive) so
        # samples without a usable label are dropped as cheaply as possible.
        if self.text_mode in ("LABEL", "LABEL+IMPRESSION"):
            self.samples = self._attach_chexpert_labels(self.samples)

        n_raw = len(self.samples)
        if self.validate_dicom:
            self.samples = self._filter_corrupted(self.samples)
        n_dropped = n_raw - len(self.samples)

        print(f"[MIMICCXRDataset] mode='{self.mode}'  split='{self.split}'  "
              f"total={len(self.samples)}  (dropped {n_dropped} corrupted)"
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
            report : str  - conditioning text; format depends on text_mode
                     (default: "FINDINGS: <...> IMPRESSION: <...>")
        """
        meta = self.samples[idx]

        try:
            image = self._load_image(meta.get("image_path", meta.get("dcm_path")))
        except Exception as e:
            if self.mode == "manifest":
                raise RuntimeError(f"[MIMICCXRDataset] load error idx={idx}: {e}") from e
            print(f"[MIMICCXRDataset] load error idx={idx}: {e}")
            image = self._blank_image()

        report = self._build_text(meta)
        return image, report

    # -------------------------------------------------------------------------
    # Index build
    # -------------------------------------------------------------------------

    def _build_index(self) -> list:
        if self.mode == "manifest":
            return self._build_index_manifest()
        if self.mode == "prebuilt":
            return self._build_index_prebuilt()
        return self._build_index_physionet()

    def _build_index_manifest(self) -> list:
        """Load an audited image manifest and retain the requested split."""
        import pandas as pd
        df = pd.read_csv(self.manifest_csv)
        required = {"dicom_id", "subject_id", "study_id", "dataset_split", "image_path", "report_path"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"[MIMICCXRDataset] manifest missing columns: {sorted(missing)}")
        df = df[df["dataset_split"].eq(self.split)].reset_index(drop=True)
        samples = []
        for row in df.itertuples(index=False):
            image_path, report_path = str(row.image_path), str(row.report_path)
            if not os.path.isfile(image_path) or not os.path.isfile(report_path):
                continue
            samples.append({"image_path": image_path, "dcm_path": image_path,
                            "report_path": report_path, "dicom_id": str(row.dicom_id),
                            "study_id": f"s{int(row.study_id)}",
                            "patient_id": f"p{int(row.subject_id)}"})
        return samples

    def _build_index_physionet(self) -> list:
        """
        Read the split CSV, construct DICOM and report paths, and keep only
        rows whose DICOM file exists on disk.
        """
        import pandas as pd

        df = pd.read_csv(self.split_csv)
        if self.split:
            df = df[df["split"] == self.split].reset_index(drop=True)

        samples = []
        missing = 0

        for _, row in df.iterrows():
            subject_id = int(row["subject_id"])
            study_id   = int(row["study_id"])
            dicom_id   = str(row["dicom_id"]).strip()

            pid_str = f"p{subject_id}"
            prefix  = pid_str[:3]            # e.g. "p10"

            patient_dir = os.path.join(self.files_dir, prefix, pid_str)
            study_dir   = os.path.join(patient_dir, f"s{study_id}")
            dcm_path    = os.path.join(study_dir, f"{dicom_id}.dcm")
            report_path = os.path.join(patient_dir, f"s{study_id}.txt")

            if not os.path.exists(dcm_path):
                missing += 1
                continue

            samples.append({
                "dcm_path"   : dcm_path,
                "report_path": report_path,
                "study_id"   : f"s{study_id}",
                "patient_id" : pid_str,
            })

        if missing:
            print(f"[MIMICCXRDataset] {missing} DICOM files not yet downloaded, skipped.")

        return samples

    def _build_index_prebuilt(self) -> list:
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
    # CheXpert label matching (text_mode='LABEL' / 'LABEL+IMPRESSION')
    # -------------------------------------------------------------------------

    def _resolve_chexpert_csv(self) -> str:
        csv_path = self.chexpert_csv or "mimic-cxr-2.0.0-chexpert.csv"
        if not os.path.isabs(csv_path) and not os.path.isfile(csv_path):
            root = getattr(self, "root_path", None)
            if root:
                csv_path = os.path.join(root, csv_path)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"[MIMICCXRDataset] CheXpert CSV not found: {csv_path} "
                "(required for text_mode='LABEL'/'LABEL+IMPRESSION'; pass an "
                "absolute --chexpert_csv path if not using root_path mode)"
            )
        return csv_path

    def _attach_chexpert_labels(self, samples: list) -> list:
        """
        Look up each sample's CheXpert row and precompute its 'LABEL: ...'
        text (stored as sample['label_text']). Samples with no positive label
        - all uncertain/missing, or no CheXpert row at all - are dropped.
        """
        csv_path = self._resolve_chexpert_csv()
        gt, label_cols = _load_chexpert_gt(csv_path)
        self.chexpert_label_cols = label_cols

        kept = []
        dropped = 0
        for meta in samples:
            subject_id = int(meta["patient_id"][1:])   # "p10000032" -> 10000032
            study_id   = int(meta["study_id"][1:])      # "s50414267" -> 50414267
            row = gt.get((subject_id, study_id))
            label_text = _labels_to_text(row, label_cols) if row is not None else None
            if label_text is None:
                dropped += 1
                continue
            meta = dict(meta)
            meta["label_text"] = label_text
            kept.append(meta)

        print(f"[MIMICCXRDataset] CheXpert label match ({csv_path}): "
              f"kept={len(kept)}  dropped={dropped} "
              "(no positive label / not found in CheXpert CSV)")
        return kept

    # -------------------------------------------------------------------------
    # DICOM integrity filtering
    # -------------------------------------------------------------------------

    def _cache_path(self) -> str:
        if self.dicom_cache:
            return self.dicom_cache
        if self.mode == "prebuilt":
            return os.path.join(self.scan_root, ".dicom_valid_cache.json")
        # PhysioNet root may be read-only; default to cwd.
        return os.path.join(os.getcwd(), f".dicom_valid_cache_{self.split}.json")

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
            path = meta.get("image_path", meta.get("dcm_path"))
            try:
                size = os.path.getsize(path)
            except OSError:
                dropped.append(path)
                continue

            entry = cache.get(path)
            if entry is not None and entry.get("size") == size:
                valid = entry.get("valid", False)
            else:
                valid = self._is_readable_image(path)
                cache[path] = {"size": size, "valid": valid}
                dirty = True

            if valid:
                kept.append(meta)
            else:
                dropped.append(path)

            if dirty and (i + 1) % 2000 == 0:
                print(f"[MIMICCXRDataset] validating DICOMs {i + 1}/{n_total} ...")

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


    @staticmethod
    def _is_readable_image(path: str) -> bool:
        if str(path).lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                with Image.open(path) as image:
                    image.verify()
                return True
            except Exception:
                return False
        return MIMICCXRDataset._is_readable_dcm(path)

    def _load_image(self, path: str) -> torch.Tensor:
        if str(path).lower().endswith((".jpg", ".jpeg", ".png")):
            with Image.open(path) as image:
                return self.transform(image.convert("L"))
        return self._load_dcm(path)

    def _blank_image(self) -> torch.Tensor:
        return torch.zeros(1, self.image_size, self.image_size)

    def _read_report_raw(self, report_path: str) -> str:
        if not os.path.exists(report_path):
            return ""
        with open(report_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_report(self, report_path: str) -> str:
        raw = self._read_report_raw(report_path)
        report = self._parse_report(raw)

        if self.max_length and len(report) > self.max_length:
            report = report[: self.max_length]

        return report

    def _parse_report(self, text: str) -> str:
        """
        Extract FINDINGS and IMPRESSION sections (legacy concatenation).
        Falls back to full text if neither section is found.
        """
        sections = _split_sections(text)
        if not sections:
            return text.strip()

        parts = [f"{key.upper()}: {sections[key]}"
                 for key in ("findings", "impression") if sections.get(key)]
        return " ".join(parts).strip()

    def _extract_section(self, text: str, keys) -> str:
        """
        Join whichever of `keys` sections are present, '' if none found.
        Unlike _parse_report, this never falls back to the full raw text -
        used by text_mode='FINDINGS'/'LABEL+IMPRESSION' where leaking an
        unrelated section back in would defeat the point of the mode.
        """
        sections = _split_sections(text)
        parts = [f"{key.upper()}: {sections[key]}" for key in keys if sections.get(key)]
        return " ".join(parts).strip()

    def _build_text(self, meta: dict) -> str:
        """
        Build the conditioning text for one sample according to self.text_mode:
            None (default)     -> legacy FINDINGS+IMPRESSION concatenation
            'FINDINGS'          -> "FINDINGS: ..." only; falls back to the
                                    full report text when the section is absent
            'LABEL'             -> "LABEL: a, b, c" from CheXpert (precomputed
                                    in meta['label_text'] by _attach_chexpert_labels)
            'LABEL+IMPRESSION'  -> "LABEL: a, b, c IMPRESSION: ..."; label only
                                    when the IMPRESSION section is absent
        """
        if self.text_mode is None:
            return self._load_report(meta["report_path"])

        raw = self._read_report_raw(meta["report_path"])

        if self.text_mode == "FINDINGS":
            text = self._extract_section(raw, ("findings",)) or raw.strip()
        elif self.text_mode == "LABEL":
            text = meta["label_text"]
        elif self.text_mode == "LABEL+IMPRESSION":
            impression = self._extract_section(raw, ("impression",))
            text = meta["label_text"] + (f" {impression}" if impression else "")
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"Unknown text_mode: {self.text_mode!r}")

        if self.max_length and len(text) > self.max_length:
            text = text[: self.max_length]
        return text


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
