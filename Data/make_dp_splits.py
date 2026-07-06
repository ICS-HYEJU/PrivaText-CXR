"""
Data/make_dp_splits.py  –  Patient-level DP budget-isolated split generator
===========================================================================
Partitions the MIMIC-CXR *train* split into two DISJOINT patient groups so DP
privacy budget is isolated (parallel composition; docs/DP_LDM_FRAMEWORK_PLAN.md §2):

    D_search : hyperparameter search   (ε spent here only)
    D_train  : final DP training       (final ε reported over this)

D_test is NOT carved out here — use the official split ('test' or 'validate')
from the same CSV, which is already disjoint from 'train'.

This reads the official split CSV (mimic-cxr-2.0.0-split.csv) used by
Data/mimic_cxr.py, so the assignment aligns exactly with the dataset's
patient_id (== "p{subject_id}").  Because MIMIC groups patients by prefix
(p10..p19) and a patient lives in exactly one prefix, D_search can be drawn
from PART of a single prefix (e.g. the first K patients of p10), with the rest
of that prefix joining D_train.

Determinism: subject_ids are sorted before slicing (no RNG).

Output JSON:
    {
      "config": {...},
      "counts": {"search": N1, "train": N2},
      "assignment": {"p10000032": "search", "p10000764": "train", ...}
    }

Usage:
    python Data/make_dp_splits.py \\
        --split_csv /storage/hjchoi/.../mimic-cxr-2.0.0-split.csv \\
        --base_split train \\
        --search_prefixes p10 \\
        --search_count 300 \\
        --out ./dp_splits.json

Consume in training/inference:
    import json, argparse
    from Data.mimic_cxr import MIMICCXRDataset
    m = json.load(open("dp_splits.json"))["assignment"]
    search_patients = [p for p, g in m.items() if g == "search"]
    ds_args = argparse.Namespace(root_path=..., split_csv=..., split="train",
                                 image_size=256, max_length=512,
                                 patient_whitelist=search_patients)
    D_search = MIMICCXRDataset(ds_args)     # only p10-subset patients kept
"""

import os
import json
import argparse

import pandas as pd


def build_assignment(subject_ids,
                     search_prefixes,
                     search_count):
    """
    subject_ids     : iterable of int subject_id (from split CSV, base_split)
    search_prefixes : list[str]  e.g. ["p10"]  (prefix = 'p' + first 2 digits)
    search_count    : max patients per search prefix (<=0 = whole prefix);
                      the remainder of a search prefix falls through to train.

    Returns {patient_id: "search"|"train"}.  patient_id == "p{subject_id}".
    """
    search_prefixes = set(search_prefixes or [])

    # patient_id and its prefix, deduplicated + sorted for determinism
    patients = sorted({f"p{int(sid)}" for sid in subject_ids})
    by_prefix = {}
    for pid in patients:
        by_prefix.setdefault(pid[:3], []).append(pid)   # pid[:3] e.g. "p10"

    assignment = {}
    for prefix, pids in by_prefix.items():
        if prefix in search_prefixes:
            k = len(pids) if search_count <= 0 else min(search_count, len(pids))
            for pid in pids[:k]:
                assignment[pid] = "search"
            for pid in pids[k:]:          # remainder of this prefix → train
                assignment[pid] = "train"
        else:
            for pid in pids:
                assignment[pid] = "train"
    return assignment


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate patient-level DP budget-isolated splits (CSV-based)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--split_csv", required=True,
                   help="Path to mimic-cxr-2.0.0-split.csv "
                        "(columns: dicom_id, subject_id, study_id, split)")
    p.add_argument("--base_split", default="train",
                   choices=["train", "validate", "test"],
                   help="Which official split to partition into search/train")
    p.add_argument("--search_prefixes", nargs="+", default=["p10"],
                   help="Prefixes to draw D_search from")
    p.add_argument("--search_count", type=int, default=-1,
                   help="Max patients per search prefix (-1 = whole prefix). "
                        "e.g. 300 = first 300 patients of p10 → D_search, "
                        "rest of p10 → D_train")
    p.add_argument("--out", default="./dp_splits.json")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.split_csv):
        raise FileNotFoundError(f"split CSV not found: {args.split_csv}")

    df = pd.read_csv(args.split_csv)
    df = df[df["split"] == args.base_split]
    if df.empty:
        raise RuntimeError(f"No rows with split=='{args.base_split}' in "
                           f"{args.split_csv}")

    subject_ids = df["subject_id"].unique()
    print(f"[csv] base_split='{args.base_split}'  "
          f"unique patients={len(subject_ids)}")

    assignment = build_assignment(
        subject_ids,
        search_prefixes=args.search_prefixes,
        search_count=args.search_count,
    )

    counts = {"search": 0, "train": 0}
    for g in assignment.values():
        counts[g] += 1

    # Safety net: verify disjointness (guaranteed by construction).
    search = {p for p, g in assignment.items() if g == "search"}
    train  = {p for p, g in assignment.items() if g == "train"}
    assert not (search & train), "search/train overlap!"

    out = {
        "config": {
            "split_csv": args.split_csv,
            "base_split": args.base_split,
            "search_prefixes": args.search_prefixes,
            "search_count": args.search_count,
        },
        "counts": counts,
        "assignment": assignment,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[split] search={counts['search']}  train={counts['train']}  "
          f"(patients, disjoint)")
    print("[split] D_test = official 'test'/'validate' split (not carved here)")
    print(f"[split] manifest → {args.out}")
    print("\nUse in training/inference:")
    print(f"  m = json.load(open('{args.out}'))['assignment']")
    print("  search_patients = [p for p,g in m.items() if g=='search']")
    print("  ds_args.patient_whitelist = search_patients  # → MIMICCXRDataset")


if __name__ == "__main__":
    main()
