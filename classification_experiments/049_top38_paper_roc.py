"""Render paper-ready top38 XRV AUROC ROC panels.

This intentionally uses only the XRV ``densenet121-res224-all`` weight set.
Each epsilon arm is rendered as a 3x4 grid for the 11 reliable pathologies,
with the final unused cell reserved for the real/generated line-style legend.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "RESULT_20261109_120931"
COMBINED = RESULT / "combined_validate_test_manifest.csv.gz"
ALL_LABELS = [
    "Atelectasis", "Consolidation", "Pneumothorax", "Edema", "Effusion",
    "Pneumonia", "Cardiomegaly", "Lung Lesion", "Fracture", "Lung Opacity",
    "Enlarged Cardiomediastinum",
]


def read_csv_gz(path: Path):
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_curves(collected, out_path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(3, 4, figsize=(8.0, 6.6), squeeze=False)
    axes_flat = axes.ravel()
    for ax, pathology in zip(axes_flat, ALL_LABELS):
        cg = collected["curves_gen"][pathology]
        cr = collected["curves_real"][pathology]
        fg, tg, _ = roc_curve(*cg)
        fr, tr, _ = roc_curve(*cr)
        # Solid/dashed lines are deliberately high-contrast and thick for
        # two-column paper reduction.  No per-panel legend is used.
        ax.plot(fr, tr, linestyle="--", color="#1455c0", linewidth=2.6)
        ax.plot(fg, tg, linestyle="-", color="#c62828", linewidth=2.6)
        ax.plot([0, 1], [0, 1], linestyle=":", color="#555555", linewidth=1.3)
        ax.set_title(pathology, pad=4, fontweight="bold")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
        ax.set_xlabel("FPR", labelpad=1); ax.set_ylabel("TPR", labelpad=1)
        ax.grid(alpha=0.16, linewidth=0.6)

    # The twelfth panel is reserved exclusively for the shared legend.
    leg_ax = axes_flat[len(ALL_LABELS)]
    leg_ax.axis("off")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#c62828", lw=2.8, linestyle="-", label="Synthetic"),
        Line2D([0], [0], color="#1455c0", lw=2.8, linestyle="--", label="Real"),
        Line2D([0], [0], color="#555555", lw=1.5, linestyle=":", label="Chance"),
    ]
    leg_ax.legend(handles=handles, loc="center", frameon=False, fontsize=13,
                  handlelength=2.8, borderaxespad=0.0)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.075, top=0.93,
                        wspace=0.16, hspace=0.34)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=450, facecolor="white")
    plt.close(fig)


def save_matlab_data(collected, out_path: Path):
    """Persist exact ROC coordinates in a MATLAB-compatible .mat file."""
    from scipy.io import savemat
    from sklearn.metrics import roc_curve
    real_fpr, real_tpr, gen_fpr, gen_tpr = [], [], [], []
    for pathology in ALL_LABELS:
        fr, tr, _ = roc_curve(*collected["curves_real"][pathology])
        fg, tg, _ = roc_curve(*collected["curves_gen"][pathology])
        real_fpr.append(fr); real_tpr.append(tr)
        gen_fpr.append(fg); gen_tpr.append(tg)
    savemat(str(out_path), {
        "pathologies": np.asarray(ALL_LABELS, dtype=object),
        "real_fpr": np.asarray(real_fpr, dtype=object),
        "real_tpr": np.asarray(real_tpr, dtype=object),
        "synthetic_fpr": np.asarray(gen_fpr, dtype=object),
        "synthetic_tpr": np.asarray(gen_tpr, dtype=object),
    }, do_compression=True)


def render(epsilon: str, device: str):
    # Imports are deliberately delayed: this script runs in medclip_env with
    # the preserved source tree on PYTHONPATH.
    from Eval_metric.downstream_cls import (
        XRV_TO_CHEXPERT, auroc_per_pathology, classify, load_chexpert_gt,
        load_xrv_classifier,
    )
    from Eval_metric.features import load_image_as_tensor

    arm = RESULT / "filtered" / epsilon / "top038"
    selected = read_csv_gz(arm / "selected_manifest.csv.gz")
    combined = read_csv_gz(COMBINED)
    by_source = {int(r["source_split_index"]): r for r in combined}
    gen_paths, real_paths, keys = [], [], []
    for row in selected:
        source_index = int(row["source_index"])
        ref = by_source[source_index]
        gp = Path(row["file"])
        if not gp.is_absolute():
            gp = arm / gp
        rp = Path(ref["image_path"])
        if not gp.exists() or not rp.exists():
            continue
        gen_paths.append(str(gp))
        real_paths.append(str(rp))
        keys.append((int(row["subject_id"]), int(row["study_id"])))

    gt = load_chexpert_gt(
        "/storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz"
    )
    gen_t = [load_image_as_tensor(p, 224) for p in gen_paths]
    real_t = [load_image_as_tensor(p, 224) for p in real_paths]
    model = load_xrv_classifier("densenet121-res224-all", device)
    p_gen = classify(model, gen_t, device, batch_size=32)
    p_real = classify(model, real_t, device, batch_size=32)
    _, _, _, included, curves_gen = auroc_per_pathology(p_gen, model, gt, keys, 10)
    _, _, _, _, curves_real = auroc_per_pathology(p_real, model, gt, keys, 10)
    if set(included) != set(ALL_LABELS):
        raise RuntimeError(f"Unexpected reliable labels: {included}")
    out = arm / "label_auroc_roc_all_paper.png"
    collected = {"curves_gen": curves_gen, "curves_real": curves_real}
    plot_curves(collected, out,
                f"XRV AUROC — top38, ε={epsilon.removeprefix('eps')}")
    mat = arm / "top38_xrv_roc_all.mat"
    save_matlab_data(collected, mat)
    print(json.dumps({"epsilon": epsilon, "n": len(keys), "output": str(out),
                      "matlab_data": str(mat)}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epsilon", nargs="+", default=["eps1", "eps10"])
    p.add_argument("--device", default="cuda:1")
    args = p.parse_args()
    for eps in args.epsilon:
        render(eps, args.device)


if __name__ == "__main__":
    main()
