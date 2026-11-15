#!/usr/bin/env python3
"""Export top-38 ROC coordinates for every pathology and epsilon arm."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "RESULT_20261109_120931" / "evaluation_balanced"
OUT = BASE / "roc_coordinates"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for eps in ("1", "10"):
        folder = BASE / f"eps{eps}" / "filtered_top38"
        mat = loadmat(folder / "top38_xrv_roc_all.mat", squeeze_me=True)
        paths = list(np.atleast_1d(mat["pathologies"]))
        agreement = json.loads((folder / "label_agreement.json").read_text())["all"]
        for i, pathology in enumerate(paths):
            pathology = str(pathology)
            safe = pathology.lower().replace(" ", "_")
            for curve, fpr_key, tpr_key, auroc_key in (
                ("real", "real_fpr", "real_tpr", "per_pathology_real"),
                ("synthetic", "synthetic_fpr", "synthetic_tpr", "per_pathology_gen"),
            ):
                fpr = np.atleast_1d(mat[fpr_key][i]).astype(float)
                tpr = np.atleast_1d(mat[tpr_key][i]).astype(float)
                auroc = agreement[auroc_key][pathology]
                rows = [{
                    "epsilon": eps,
                    "curve": curve,
                    "pathology": pathology,
                    "point": j,
                    "fpr": f"{x:.12g}",
                    "tpr": f"{y:.12g}",
                    "auroc": f"{float(auroc):.12g}",
                } for j, (x, y) in enumerate(zip(fpr, tpr))]
                all_rows.extend(rows)
                with (OUT / f"eps{eps}_{curve}_{safe}.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

    combined = OUT / "top38_roc_coordinates_all.csv"
    with combined.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    (OUT / "README.md").write_text(
        "# Top-38 ROC coordinates\n\n"
        "Each CSV contains `point`, `fpr`, `tpr`, and AUROC for one epsilon, "
        "curve (`real` or `synthetic`), and pathology. `real` is the real "
        "reference used by the corresponding epsilon evaluation. The combined "
        "file contains all 44 curves. Source files are the epsilon-specific "
        "`top38_xrv_roc_all.mat` and `label_agreement.json` files.\n"
    )
    print(f"Wrote {len(all_rows)} coordinate rows to {OUT}")


if __name__ == "__main__":
    main()
