#!/usr/bin/env python3
"""Export one high-scoring synthetic/real image pair per CheXpert pathology."""
import csv
import gzip
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "RESULT_20261110_OFFICIAL_TRAIN_FILTERED_UTILITY"
OUT = ROOT / "RESULT_20261109_120931" / "evaluation_balanced" / "images"
LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax",
    "Support Devices",
]
# The highest aggregate score for effusion was visually subtle; this candidate
# has a clearly visible pleural fluid meniscus in the paired real radiograph.
VISUAL_OVERRIDES = {"Pleural Effusion": "3296"}


def read_gz_csv(path):
    with gzip.open(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def safe_name(label):
    return label.lower().replace(" ", "_")


def main():
    manifest = {r["pool_index"]: r for r in read_gz_csv(BASE / "official_train_pool_22000.csv.gz")}
    scores = {}
    for eps in ("1", "10"):
        scores[eps] = {r["source_index"]: r for r in read_gz_csv(BASE / f"filter_scores/eps{eps}_dual_consistency.csv.gz")}
    chosen = []
    for label in LABELS:
        candidates = [r for k, r in manifest.items()
                      if r.get(label) == "1.0" and k in scores["1"] and k in scores["10"]
                      and scores["1"][k].get("dual_consistency") and scores["10"][k].get("dual_consistency")]
        if not candidates:
            raise RuntimeError(f"No selected positive sample for {label}")
        if label in VISUAL_OVERRIDES:
            row = manifest[VISUAL_OVERRIDES[label]]
        else:
            row = max(candidates, key=lambda r: sum(float(scores[e][r["pool_index"]]["dual_consistency"]) for e in ("1", "10")) / 2)
        score = {e: scores[e][row["pool_index"]] for e in ("1", "10")}
        real = Path(row["real_image_path"])
        if not real.exists():
            raise FileNotFoundError(real)
        stem = safe_name(label)
        syn_out = {e: OUT / f"{stem}__eps{e}.png" for e in ("1", "10")}
        real_out = OUT / f"{stem}__real.png"
        OUT.mkdir(parents=True, exist_ok=True)
        for e in ("1", "10"):
            syn = BASE / f"generated/eps{e}/samples/train_p{int(row['pool_index']):05d}.png"
            if not syn.exists():
                raise FileNotFoundError(syn)
            shutil.copy2(syn, syn_out[e])
        with Image.open(real) as im:
            im.convert("RGB").save(real_out, format="PNG")
        chosen.append({
            "pathology": label,
            "pool_index": row["pool_index"],
            "eps1_dual_consistency": score["1"]["dual_consistency"],
            "eps10_dual_consistency": score["10"]["dual_consistency"],
            "eps1_png": syn_out["1"].name,
            "eps10_png": syn_out["10"].name,
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
        print(f"{r['pathology']}: pool={r['pool_index']} eps1={float(r['eps1_dual_consistency']):.6f} eps10={float(r['eps10_dual_consistency']):.6f}")


if __name__ == "__main__":
    main()
