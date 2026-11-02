"""Build an audited official-split manifest for one available MIMIC-CXR prefix."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image


ID_COLUMNS = ["subject_id", "study_id"]


def readable_jpg(path: str) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="Patient prefix such as p13")
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--jpg-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()

    if len(args.prefix) != 3 or not args.prefix.startswith("p") or not args.prefix[1:].isdigit():
        raise ValueError("--prefix must look like p13")

    report_root = Path(args.report_root).resolve()
    jpg_root = Path(args.jpg_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = pd.read_csv(report_root / "cxr-record-list.csv.gz", dtype={"dicom_id": str})
    split = pd.read_csv(report_root / "mimic-cxr-2.0.0-split.csv", dtype={"dicom_id": str})
    chexpert = pd.read_csv(report_root / "mimic-cxr-2.0.0-chexpert.csv")
    keys = ["dicom_id", "subject_id", "study_id"]
    if records.duplicated(keys).any() or split.duplicated(keys).any():
        raise RuntimeError("record-list and split keys must be unique")

    records["patient_prefix"] = "p" + records.subject_id.astype(str).str[:2]
    records = records[records.patient_prefix.eq(args.prefix)].copy()
    frame = records.merge(split[keys + ["split"]], on=keys, how="left", validate="one_to_one")
    if frame["split"].isna().any():
        raise RuntimeError("record-list rows are missing from the official split CSV")

    label_columns = [column for column in chexpert.columns if column not in ID_COLUMNS]
    chexpert["_chexpert_found"] = True
    frame = frame.merge(chexpert, on=ID_COLUMNS, how="left", validate="many_to_one")

    patient = "p" + frame.subject_id.astype(str)
    study = "s" + frame.study_id.astype(str)
    frame["image_path"] = [
        str(jpg_root / args.prefix / patient_id / study_id / f"{dicom_id}.jpg")
        for patient_id, study_id, dicom_id in zip(patient, study, frame.dicom_id)
    ]
    frame["report_path"] = [
        str(report_root / "files" / args.prefix / patient_id / f"{study_id}.txt")
        for patient_id, study_id in zip(patient, study)
    ]
    frame["image_exists"] = frame.image_path.map(os.path.isfile)
    frame["report_exists"] = frame.report_path.map(os.path.isfile)
    if args.verify_images:
        values = []
        for index, path in enumerate(frame.image_path, 1):
            values.append(readable_jpg(path) if os.path.isfile(path) else False)
            if index % 5000 == 0:
                print(f"[verify] {index}/{len(frame)}")
        frame["image_readable"] = values
    else:
        frame["image_readable"] = frame.image_exists

    positive = frame[label_columns].eq(1.0)
    frame["chexpert_found"] = frame["_chexpert_found"].fillna(False).astype(bool)
    frame["positive_label_count"] = positive.sum(axis=1)
    frame["positive_labels"] = positive.apply(
        lambda row: ";".join(row.index[row.to_numpy()].tolist()), axis=1)
    frame["single_positive_label"] = positive.idxmax(axis=1).where(
        frame.positive_label_count.eq(1), "")
    frame["dataset_split"] = frame["split"]
    frame["exclusion_reason"] = ""
    frame.loc[~frame.image_exists, "exclusion_reason"] = "missing_image"
    frame.loc[frame.image_exists & ~frame.image_readable, "exclusion_reason"] = "unreadable_image"
    frame.loc[frame.image_readable & ~frame.report_exists, "exclusion_reason"] = "missing_report"

    usable = frame.image_readable & frame.report_exists
    manifest = frame.loc[usable].copy()
    columns = [
        "dicom_id", "subject_id", "study_id", "patient_prefix", "split",
        "dataset_split", "image_path", "report_path", "image_exists",
        "image_readable", "report_exists", "chexpert_found",
        "positive_label_count", "positive_labels", "single_positive_label",
        "exclusion_reason",
    ]
    manifest[columns].to_csv(out_dir / f"{args.prefix}_paired_manifest.csv.gz", index=False)
    frame.loc[~frame.image_exists, columns].to_csv(
        out_dir / f"{args.prefix}_missing_images.csv", index=False)

    expected_studies = set(zip(frame.subject_id.astype(int), frame.study_id.astype(int)))
    paired_studies = set(zip(manifest.subject_id.astype(int), manifest.study_id.astype(int)))
    report_only = frame[
        frame.apply(lambda row: (int(row.subject_id), int(row.study_id)) not in paired_studies, axis=1)
    ][["subject_id", "study_id", "report_path"]].drop_duplicates()
    report_only.to_csv(out_dir / f"{args.prefix}_reports_without_jpg.csv", index=False)

    split_sets = {
        name: set(part.subject_id.astype(int))
        for name, part in manifest.groupby("dataset_split")
    }
    overlap = {
        "train_validate": len(split_sets.get("train", set()) & split_sets.get("validate", set())),
        "train_test": len(split_sets.get("train", set()) & split_sets.get("test", set())),
        "validate_test": len(split_sets.get("validate", set()) & split_sets.get("test", set())),
    }
    if any(overlap.values()):
        raise RuntimeError(f"patient leakage in official split: {overlap}")

    summary = {
        "prefix": args.prefix,
        "policy": "MIMIC-CXR official split; available readable JPG and existing TXT only",
        "image_decode_verified": args.verify_images,
        "expected": {
            "images": len(frame),
            "studies": len(expected_studies),
            "patients": int(frame.subject_id.nunique()),
        },
        "paired": {
            "images": len(manifest),
            "studies": len(paired_studies),
            "patients": int(manifest.subject_id.nunique()),
        },
        "missing_images": int((~frame.image_exists).sum()),
        "unreadable_images": int((frame.image_exists & ~frame.image_readable).sum()),
        "missing_reports": int((~frame.report_exists).sum()),
        "reports_without_any_jpg": len(report_only),
        "counts_by_official_split": manifest.dataset_split.value_counts().sort_index().to_dict(),
        "patient_overlap": overlap,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
