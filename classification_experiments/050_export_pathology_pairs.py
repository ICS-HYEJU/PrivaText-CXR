#!/usr/bin/env python3
"""Export one high-scoring synthetic/real image pair per CheXpert pathology."""
import csv
import gzip
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "RESULT_20261111_DISTRIBUTION_MATCHED_FILTERING" / "filtered" / "eps1" / "distribution_matched5500"
OUT = ROOT / "RESULT_20261109_120931" / "evaluation_balanced" / "images"
LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax",
    "Support Devices",
]


def read_gz_csv(path):
    with gzip.open(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def safe_name(label):
    return label.lower().replace(" ", "_")


def main():
    selected = {r["pool_index"]: r for r in read_gz_csv(BASE / "selection.csv.gz")}
    manifest = {r["pool_index"]: r for r in read_gz_csv(BASE / "pool_manifest.csv.gz")}
    chosen = []
    for label in LABELS:
        candidates = [manifest[k] for k in selected if k in manifest and manifest[k].get(label) == "1.0"]
        if not candidates:
            raise RuntimeError(f"No selected positive sample for {label}")
        row = max(candidates, key=lambda r: float(selected[r["pool_index"]]["selection_score"]))
        score = selected[row["pool_index"]]
        syn = Path(score["file"])
        if str(syn).startswith("/workspace/PrivaText-CXR/"):
            syn = ROOT / str(syn).split("/workspace/PrivaText-CXR/", 1)[1]
        if not syn.exists():
            raise FileNotFoundError(syn)
        real = Path(row["real_image_path"])
        if not real.exists():
            raise FileNotFoundError(real)
        stem = safe_name(label)
        syn_out = OUT / f"{stem}__synthetic.png"
        real_out = OUT / f"{stem}__real.png"
        syn_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(syn, syn_out)
        with Image.open(real) as im:
            im.convert("RGB").save(real_out, format="PNG")
        chosen.append({
            "pathology": label,
            "pool_index": row["pool_index"],
            "selection_score": score["selection_score"],
            "synthetic_png": syn_out.name,
            "real_png": real_out.name,
            "real_image_path": row["real_image_path"],
            "subject_id": row["subject_id"],
            "study_id": row["study_id"],
            "prompt_text": row["prompt_text"],
        })
    with (OUT / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=chosen[0].keys())
        w.writeheader()
        w.writerows(chosen)
    print(f"Exported {len(chosen)} pathology pairs to {OUT}")
    for r in chosen:
        print(f"{r['pathology']}: pool={r['pool_index']} score={float(r['selection_score']):.6f}")


if __name__ == "__main__":
    main()
