"""
Data/make_dp_splits.py  –  Patient-level DP budget-isolated split generator
===========================================================================
Partitions MIMIC-CXR patients into three DISJOINT groups so that DP privacy
budget is isolated (parallel composition, see docs/DP_LDM_FRAMEWORK_PLAN.md §2):

    D_search : hyperparameter search   (ε spent here only)
    D_train  : final DP training       (final ε reported over this)
    D_test   : evaluation              (no gradients, no ε)

Because MIMIC's p10..p19 prefix folders partition patients (a patient lives in
exactly one prefix), splitting at the patient-folder level yields patient-level
disjointness, which in turn guarantees image-level disjointness required for
image-level (sample-level) DP.

Key feature: D_search can be drawn from just PART of a single prefix (e.g. the
first K patients of p10), leaving the rest of that prefix for D_train.

Output: a JSON manifest  {patient_id: "search"|"train"|"test", ...}  plus a
summary.  Consume it in training/inference via:

    import json, argparse
    from Data.mimic_cxr import MIMICCXRDataset
    manifest = json.load(open("dp_splits.json"))["assignment"]
    search_patients = [p for p, g in manifest.items() if g == "search"]
    ds_args = argparse.Namespace(prebuilt_split_dir=..., split="train",
                                 image_size=256, max_length=512,
                                 patient_whitelist=search_patients)
    D_search = MIMICCXRDataset(ds_args)

Usage:
    python Data/make_dp_splits.py \\
        --mimic_root /storage/hjchoi/mimic/split/train \\
        --search_prefixes p10 \\
        --search_count 300 \\
        --test_prefixes p19 \\
        --out ./dp_splits.json

Determinism: patients are sorted lexicographically before slicing, so the same
inputs always produce the same split (no RNG).
"""

import os
import re
import json
import argparse


def list_patients_by_prefix(scan_root: str) -> dict:
    """
    Return {prefix: [patient_id, ...]} for <scan_root>/pNN/pXXXXXXXX/.
    Patient ids are sorted for deterministic slicing.
    """
    by_prefix = {}
    for prefix in sorted(os.listdir(scan_root)):
        prefix_dir = os.path.join(scan_root, prefix)
        if not os.path.isdir(prefix_dir) or not re.fullmatch(r"p\d+", prefix):
            continue
        patients = sorted(
            pid for pid in os.listdir(prefix_dir)
            if os.path.isdir(os.path.join(prefix_dir, pid)) and pid.startswith("p")
        )
        by_prefix[prefix] = patients
    return by_prefix


def build_assignment(by_prefix: dict,
                     search_prefixes: list,
                     search_count: int,
                     test_prefixes: list) -> dict:
    """
    Assign every patient to 'search' | 'train' | 'test' (disjoint).

    Rules (applied per patient, first match wins):
      1. test_prefixes            → 'test'   (whole prefix)
      2. search_prefixes, and within them the first `search_count` patients
         (or all if search_count <= 0)  → 'search'
         The REMAINING patients of a search prefix fall through to 'train'.
      3. everything else          → 'train'

    This lets D_search be only PART of a prefix (e.g. first 300 of p10) while
    the rest of p10 joins D_train.
    """
    search_prefixes = set(search_prefixes or [])
    test_prefixes   = set(test_prefixes or [])
    overlap = search_prefixes & test_prefixes
    if overlap:
        raise ValueError(f"prefixes cannot be both search and test: {overlap}")

    assignment = {}
    for prefix, patients in by_prefix.items():
        if prefix in test_prefixes:
            for pid in patients:
                assignment[pid] = "test"
        elif prefix in search_prefixes:
            k = len(patients) if search_count <= 0 else min(search_count, len(patients))
            for pid in patients[:k]:
                assignment[pid] = "search"
            for pid in patients[k:]:          # remainder of this prefix → train
                assignment[pid] = "train"
        else:
            for pid in patients:
                assignment[pid] = "train"
    return assignment


def summarize(assignment: dict) -> dict:
    counts = {"search": 0, "train": 0, "test": 0}
    for g in assignment.values():
        counts[g] += 1
    return counts


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate patient-level DP budget-isolated splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mimic_root", required=True,
                   help="Directory containing pNN/ patient-prefix folders "
                        "(e.g. <split>/train)")
    p.add_argument("--search_prefixes", nargs="+", default=["p10"],
                   help="Prefixes to draw D_search from")
    p.add_argument("--search_count", type=int, default=-1,
                   help="Max patients per search prefix (-1 = whole prefix). "
                        "e.g. 300 = first 300 patients of p10 → D_search, "
                        "rest of p10 → D_train")
    p.add_argument("--test_prefixes", nargs="+", default=["p19"],
                   help="Prefixes reserved entirely for D_test")
    p.add_argument("--out", default="./dp_splits.json")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.mimic_root):
        raise FileNotFoundError(f"mimic_root not found: {args.mimic_root}")

    by_prefix = list_patients_by_prefix(args.mimic_root)
    if not by_prefix:
        raise RuntimeError(f"No pNN/ prefix folders under {args.mimic_root}")

    print("[scan] patients per prefix:")
    for prefix, patients in by_prefix.items():
        print(f"  {prefix}: {len(patients)}")

    assignment = build_assignment(
        by_prefix,
        search_prefixes=args.search_prefixes,
        search_count=args.search_count,
        test_prefixes=args.test_prefixes,
    )
    counts = summarize(assignment)

    # Disjointness is guaranteed by construction (each patient assigned once),
    # but verify no patient id collides across groups as a safety net.
    groups = {"search": set(), "train": set(), "test": set()}
    for pid, g in assignment.items():
        groups[g].add(pid)
    assert not (groups["search"] & groups["train"]), "search/train overlap!"
    assert not (groups["search"] & groups["test"]),  "search/test overlap!"
    assert not (groups["train"]  & groups["test"]),  "train/test overlap!"

    out = {
        "config": {
            "mimic_root": args.mimic_root,
            "search_prefixes": args.search_prefixes,
            "search_count": args.search_count,
            "test_prefixes": args.test_prefixes,
        },
        "counts": counts,
        "assignment": assignment,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n[split] search={counts['search']}  train={counts['train']}  "
          f"test={counts['test']}  (patients, disjoint)")
    print(f"[split] manifest → {args.out}")
    print("\nUse in training/inference:")
    print("  manifest = json.load(open('%s'))['assignment']" % args.out)
    print("  search_patients = [p for p,g in manifest.items() if g=='search']")
    print("  ds_args.patient_whitelist = search_patients  # → MIMICCXRDataset")


if __name__ == "__main__":
    main()
