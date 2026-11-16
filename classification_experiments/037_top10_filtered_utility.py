"""E037: evaluate S2.2k and R1.1k+S2.2k from E036 top-10% scores."""
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "RESULT_20261110_OFFICIAL_TRAIN_FILTERED_UTILITY"
OUT = ROOT / "RESULT_20261111_OFFICIAL_TRAIN_TOP10_UTILITY"
POOL = BASE / "official_train_pool_22000.csv.gz"
TEST = BASE / "official_test_manifest.csv.gz"
DATA_ROOT = "/storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0"
SOURCE = ROOT / ".claude/worktrees/synth-utility-improve/src"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def build_top10(epsilon):
    scored = pd.read_csv(BASE / "filter_scores" / f"eps{epsilon}_dual_consistency.csv.gz")
    n_keep = int(round(len(scored) * 0.10))
    selected = scored.sort_values(
        ["dual_consistency", "index", "sample"], ascending=[False, True, True]
    ).head(n_keep).copy()
    if len(scored) != 22000 or len(selected) != 2200 or not selected.split.eq("train").all():
        raise AssertionError(f"invalid eps{epsilon} top10 selection")
    destination = OUT / "filtered" / f"eps{epsilon}" / "top010"
    samples = destination / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    descriptions = []
    for ordinal, row in enumerate(selected.itertuples(index=False)):
        name = f"f{ordinal:06d}_p{int(row.index):05d}.png"
        link = samples / name
        if not link.exists():
            link.symlink_to(Path(row.file))
        descriptions.append({
            "index": int(row.index), "description": row.prompt,
            "file": str(Path("samples") / name),
        })
    pd.DataFrame(descriptions).to_csv(destination / "descriptions.csv", index=False)
    selected.to_csv(destination / "selected_manifest.csv.gz", index=False, compression="gzip")
    dump(destination / "filter_summary.json", {
        "epsilon": epsilon, "method": "reuse_e036_dual_consistency_top_percent",
        "top_pct": 10, "n_input": len(scored), "n_selected": len(selected),
        "dual_cutoff": float(selected.dual_consistency.min()),
        "dual_mean_input": float(scored.dual_consistency.mean()),
        "dual_mean_selected": float(selected.dual_consistency.mean()),
        "source_scores": str(BASE / "filter_scores" / f"eps{epsilon}_dual_consistency.csv.gz"),
    })
    return destination


def train(epsilon, with_real, device):
    label = "R1100_S2200" if with_real else "S2200"
    # Existing all-synthetic arm assembly is size-agnostic: top010 contains 2,200 rows.
    trainer_arm = "RS5500" if with_real else "S5500"
    out = OUT / "classification" / f"eps{epsilon}" / label
    out.mkdir(parents=True, exist_ok=True)
    command = [
        "/opt/medclip_env/bin/python", "Eval_metric/train_downstream_cls.py",
        "--pool_manifest", str(POOL), "--synth_dir",
        str(OUT / "filtered" / f"eps{epsilon}" / "top010"),
        "--arm", trainer_arm, "--test_manifest_csv", str(TEST),
        "--root_path", DATA_ROOT, "--device", f"cuda:{device}",
        "--epochs", "30", "--batch_size", "32", "--seed", "42",
        "--num_workers", "4", "--out_dir", str(out),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SOURCE}:{ROOT}"
    with (out / "train.log").open("w") as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    result = json.loads((out / "eval_summary.json").read_text())
    result.update({
        "arm": label, "display_arm": label, "epsilon": epsilon,
        "filter_top_pct": 10, "n_synthetic": 2200,
    })
    dump(out / "eval_summary.json", result)
    print(f"eps{epsilon} {label}: AUROC={result['auroc_macro']:.4f} "
          f"macro_ACC={result['accuracy_macro']:.4f}", flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for epsilon in (1, 10):
        build_top10(epsilon)

    def queue(epsilon, device):
        return [train(epsilon, False, device), train(epsilon, True, device)]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(queue, 1, 0), executor.submit(queue, 10, 1)]
        for future in futures:
            results.extend(future.result())
    dump(OUT / "classification_results.json", results)
    pd.DataFrame([{key: row.get(key) for key in (
        "epsilon", "display_arm", "n_train", "n_test", "auroc_macro",
        "accuracy_macro", "accuracy_micro", "accuracy_samplewise",
        "accuracy_exact_match_known")}
        for row in results]).to_csv(OUT / "classification_results.csv", index=False)
    (OUT / "PIPELINE_COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
