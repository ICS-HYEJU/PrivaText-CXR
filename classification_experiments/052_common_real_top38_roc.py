#!/usr/bin/env python3
"""Recompute top-38 ROC curves with one common Real reference set."""
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "RESULT_20261109_120931"
OUT = RESULT / "evaluation_balanced" / "roc_coordinates_common_real"
LABELS = [
    "Atelectasis", "Consolidation", "Pneumothorax", "Edema", "Effusion",
    "Pneumonia", "Cardiomegaly", "Lung Lesion", "Fracture", "Lung Opacity",
    "Enlarged Cardiomediastinum",
]


def read_gz(path):
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(device="cuda:0", weights="densenet121-res224-all"):
    from Eval_metric.downstream_cls import (
        auroc_per_pathology, classify, load_chexpert_gt, load_xrv_classifier,
    )
    from Eval_metric.features import load_image_as_tensor

    combined = read_gz(RESULT / "combined_validate_test_manifest.csv.gz")
    by_source = {int(r["source_split_index"]): r for r in combined}
    selected = {}
    for eps in ("1", "10"):
        selected[eps] = read_gz(RESULT / f"filtered/eps{eps}/top038/selected_manifest.csv.gz")
    # Use one fixed reference for both epsilon arms. The full combined
    # validate+test manifest is required here so rare labels such as Fracture
    # still satisfy the min_pos=10 reliability rule.
    real_rows = combined
    real_paths = [r["image_path"] for r in real_rows]
    real_keys = [(int(r["subject_id"]), int(r["study_id"])) for r in real_rows]
    gt = load_chexpert_gt("/storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz")
    model = load_xrv_classifier(weights, device)
    labels = (LABELS if not weights.endswith("-nih") else
              ["Atelectasis", "Consolidation", "Pneumothorax", "Edema",
               "Effusion", "Pneumonia", "Cardiomegaly"])

    def score(paths, keys):
        tensors = [load_image_as_tensor(p, 224) for p in paths]
        raw = classify(model, tensors, device, batch_size=32)
        _, _, _, included, curves = auroc_per_pathology(raw, model, gt, keys, 10)
        if set(included) != set(labels):
            raise RuntimeError(f"Unexpected reliable labels: {included}")
        return curves

    curves = {"real": score(real_paths, real_keys)}
    for eps in ("1", "10"):
        rows = [r for r in selected[eps] if Path(r["file"]).exists()]
        curves[f"eps{eps}"] = score(
            [r["file"] for r in rows],
            [(int(r["subject_id"]), int(r["study_id"])) for r in rows],
        )

    out = RESULT / "evaluation_balanced" / (
        "roc_coordinates_common_real_nih" if weights.endswith("-nih")
        else "roc_coordinates_common_real"
    )
    out.mkdir(parents=True, exist_ok=True)
    global OUT
    OUT = out
    all_rows = []
    mat = {"pathologies": np.asarray(labels, dtype=object)}
    for pathology in labels:
        for arm in ("real", "eps1", "eps10"):
            y, s = curves[arm][pathology]
            fpr, tpr, _ = roc_curve(y, s)
            auroc = float(roc_auc_score(y, s))
            safe = pathology.lower().replace(" ", "_")
            with (OUT / f"{arm}_{safe}.csv").open("w", newline="") as f:
                w = csv.writer(f); w.writerow(["arm", "pathology", "point", "fpr", "tpr", "auroc"])
                for j, (x, z) in enumerate(zip(fpr, tpr)):
                    w.writerow([arm, pathology, j, f"{x:.12g}", f"{z:.12g}", f"{auroc:.12g}"])
                    all_rows.append({"arm": arm, "pathology": pathology, "point": j,
                                     "fpr": f"{x:.12g}", "tpr": f"{z:.12g}",
                                     "auroc": f"{auroc:.12g}"})
            mat.setdefault(f"{arm}_fpr", []).append(fpr)
            mat.setdefault(f"{arm}_tpr", []).append(tpr)
            mat.setdefault(f"{arm}_auroc", []).append(auroc)
    with (OUT / "top38_common_real_roc_coordinates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys()); w.writeheader(); w.writerows(all_rows)
    for k in ("real_fpr", "real_tpr", "eps1_fpr", "eps1_tpr", "eps10_fpr", "eps10_tpr"):
        mat[k] = np.asarray(mat[k], dtype=object)
    for k in ("real_auroc", "eps1_auroc", "eps10_auroc"):
        mat[k] = np.asarray(mat[k], dtype=float).reshape(-1, 1)
    savemat(OUT / "top38_common_real_roc_coordinates.mat", mat, do_compression=True)
    summary = {"common_real_n": len(real_rows), "eps1_synthetic_n": len(selected["1"]),
               "eps10_synthetic_n": len(selected["10"]), "pathologies": labels,
               "protocol": f"top38 XRV label agreement ({weights}); one common Real set from combined validate+test manifest",
               "xrv_weights": weights,
               "auroc": {p: {a: mat[f"{a}_auroc"][i] for a in ("real", "eps1", "eps10")} for i, p in enumerate(labels)}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps({"out": str(OUT), "common_real_n": len(real_rows), "eps1_n": len(selected["1"]), "eps10_n": len(selected["10"])}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="densenet121-res224-all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args.device, args.weights)
