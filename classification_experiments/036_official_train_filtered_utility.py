"""E036: leakage-free official-train filtered synthetic classifier utility.

Build 22,000 official-train candidates, generate one DDIM50 sample per candidate
for E033 epsilon 1/10, retain the dual-consistency top 25% (5,500 images), and
train from-scratch DenseNet-121 classifiers. Official test is used only for the
final frozen classifier evaluation.
"""
import argparse
import concurrent.futures
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".claude/worktrees/synth-utility-improve/src"
if not SOURCE.is_dir():
    SOURCE = ROOT / ".claude/worktrees/ldm-dp-text-prompt3/src"
sys.path[:0] = [str(ROOT), str(SOURCE), str(ROOT / "classification_experiments")]

RESULT = ROOT / "RESULT_20261110_OFFICIAL_TRAIN_FILTERED_UTILITY"
MANIFEST = ROOT / "data_manifests/mimic_official_split_p10_p15/ldm_official_split_manifest.csv.gz"
DATA_ROOT = Path("/storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0")
CHEXPERT = DATA_ROOT / "mimic-cxr-2.0.0-chexpert.csv"
POOL = RESULT / "official_train_pool_22000.csv.gz"
TEST = RESULT / "official_test_manifest.csv.gz"
N_CANDIDATES = 22000
N_REAL = 1100
PCTS = 25
SEED = 42


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def load_034():
    spec = importlib.util.spec_from_file_location(
        "e036_filter_base", ROOT / "classification_experiments/034_filter_evaluate_combined.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_pool():
    from Data.mimic_cxr import MIMICCXRDataset, _labels_to_text
    from Eval_metric.downstream_cls import load_chexpert_gt

    source = pd.read_csv(MANIFEST)
    test = source[source.dataset_split.eq("test")].copy().reset_index(drop=True)
    test.to_csv(TEST, index=False, compression="gzip")
    ds = MIMICCXRDataset(argparse.Namespace(
        root_path=str(DATA_ROOT), manifest_csv=str(MANIFEST), split="train",
        image_size=256, max_length=512, patient_whitelist=None, text_mode=None))
    gt = load_chexpert_gt(str(CHEXPERT))
    labels = list(next(iter(gt.values())))

    # One radiograph per study prevents multi-view duplicates from dominating the pool.
    candidates = []
    seen_studies = set()
    for ds_index, meta in enumerate(ds.samples):
        subject = int(str(meta["patient_id"]).lstrip("p"))
        study = int(str(meta["study_id"]).lstrip("s"))
        if study in seen_studies:
            continue
        seen_studies.add(study)
        candidates.append((ds_index, subject, study, meta))
    if len(candidates) < N_CANDIDATES:
        raise RuntimeError(f"only {len(candidates)} unique train studies; need {N_CANDIDATES}")
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(candidates), N_CANDIDATES, replace=False)
    rows = []
    for pool_index, position in enumerate(chosen):
        ds_index, subject, study, meta = candidates[int(position)]
        values = gt.get((subject, study), {})
        label_text = _labels_to_text(values, labels)
        raw = ds._read_report_raw(meta["report_path"])
        impression = ds._extract_section(raw, ("impression",))
        prompt = ((label_text + (" " + impression if impression else ""))
                  if label_text else ds._build_text(meta))[:512]
        row = {
            "pool_index": pool_index, "in_pool_a": pool_index < N_REAL,
            "ds_index": ds_index, "subject_id": subject, "study_id": study,
            "real_image_path": meta["image_path"], "prompt_text": prompt,
        }
        row.update({label: values.get(label, np.nan) for label in labels})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(POOL, index=False, compression="gzip")
    train_subjects = set(frame.subject_id.astype(int))
    test_subjects = set(test.subject_id.astype(int))
    overlap = sorted(train_subjects & test_subjects)
    if overlap:
        raise AssertionError(f"official train/test patient overlap: {overlap[:5]}")
    write_json(RESULT / "pool_audit.json", {
        "official_split": True, "seed": SEED, "n_candidates": len(frame),
        "n_real_pool_a": int(frame.in_pool_a.sum()), "unique_studies": int(frame.study_id.nunique()),
        "unique_train_patients": int(frame.subject_id.nunique()),
        "official_test_images": len(test), "official_test_patients": int(test.subject_id.nunique()),
        "train_test_patient_overlap": 0, "one_image_per_study": True,
    })
    print(f"POOL built candidates={len(frame)} real={frame.in_pool_a.sum()} test={len(test)}", flush=True)


def generate(epsilon, device, steps=50, max_prompts=None):
    from diagnostics import inference_args, save_image
    from LDM_dp_inference import build_and_load_ldm
    from ablation_eval import ddim

    checkpoint = ROOT / f"restored_ckpts/e033/eps{epsilon}_ldm_dp_final.pt"
    args = inference_args(checkpoint, device)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = blob.get("args", {})
    if saved.get("text_encoder") != "StanfordAIMI/RadBERT":
        raise RuntimeError("checkpoint is not pinned RadBERT E033")
    args.biobert_path = saved["biobert_path"]
    args.lora_ckpt = str(checkpoint)
    model, embedder = build_and_load_ldm(args, torch.device(f"cuda:{device}"))
    model.eval()
    pool = pd.read_csv(POOL)
    if max_prompts is not None:
        pool = pool.iloc[:max_prompts].copy()
    out = RESULT / "generated" / f"eps{epsilon}"
    samples = out / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    mapping_path = out / "mapping.csv"
    rows = []
    for row in pool.itertuples(index=False):
        index = int(row.pool_index)
        path = samples / f"train_p{index:05d}.png"
        seed = 360000 + epsilon * 100000 + index
        if not path.exists():
            with torch.no_grad():
                context = embedder([row.prompt_text]).to(model.device)
                image = ddim(model, context, seed, steps=steps)
            save_image(image[0], path)
        rows.append({
            "index": index, "split": "train", "subject_id": int(row.subject_id),
            "study_id": int(row.study_id), "prompt": row.prompt_text,
            "sample": 0, "seed": seed, "file": str(path), "source_index": index,
        })
        if (index + 1) % 100 == 0 or index + 1 == len(pool):
            pd.DataFrame(rows).to_csv(mapping_path, index=False)
            print(f"eps{epsilon} train DDIM{steps} {index + 1}/{len(pool)}", flush=True)
    pd.DataFrame(rows).rename(columns={"prompt": "description"})[
        ["index", "description", "file"]].to_csv(out / "descriptions.csv", index=False)
    write_json(out / "generation.json", {
        "epsilon": epsilon, "split": "train", "n_prompts": len(pool), "n_images": len(rows),
        "samples_per_prompt": 1, "sampler": "DDIM", "steps": steps, "eta": 0,
        "checkpoint": str(checkpoint), "manifest": str(MANIFEST),
        "text_encoder": saved.get("text_encoder"),
        "text_encoder_revision": saved.get("text_encoder_revision"),
        "text_encoder_weight_sha256": saved.get("text_encoder_weight_sha256"),
    })


def filter_one(epsilon):
    out = load_034().filter_arm(RESULT, epsilon, PCTS)
    selected = pd.read_csv(out / "selected_manifest.csv.gz")
    if len(selected) != 5500 or not selected.split.eq("train").all():
        raise AssertionError(f"invalid filtered eps{epsilon}: n={len(selected)}")
    print(f"FILTER eps{epsilon} selected={len(selected)}", flush=True)


def train_one(arm, epsilon, device):
    label = "R1100" if arm == "R" else {
        "RS1100U": "R1100_S1100", "S1100": "S1100",
        "S5500": "S5500", "RS5500": "R1100_S5500",
    }[arm]
    out = RESULT / "classification" / ("shared" if epsilon is None else f"eps{epsilon}") / label
    command = [
        "/opt/medclip_env/bin/python", "Eval_metric/train_downstream_cls.py",
        "--pool_manifest", str(POOL), "--arm", arm,
        "--test_manifest_csv", str(TEST), "--root_path", str(DATA_ROOT),
        "--device", f"cuda:{device}", "--epochs", "30", "--batch_size", "32",
        "--seed", "42", "--num_workers", "4", "--out_dir", str(out),
    ]
    if epsilon is not None:
        command += ["--synth_dir", str(RESULT / "filtered" / f"eps{epsilon}" / "top025")]
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SOURCE}:{ROOT}"
    with (out / "train.log").open("w") as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    payload = json.loads((out / "eval_summary.json").read_text())
    payload["display_arm"] = label
    payload["epsilon"] = epsilon
    write_json(out / "eval_summary.json", payload)
    print(f"CLASSIFIER eps={epsilon} arm={label} AUROC={payload['auroc_macro']:.4f} "
          f"ACC={payload['accuracy_macro']:.4f}", flush=True)
    return payload


def train_all():
    results = [train_one("R", None, 0)]
    jobs = []
    for epsilon, device in ((1, 0), (10, 1)):
        for arm in ("RS1100U", "S1100", "S5500", "RS5500"):
            jobs.append((arm, epsilon, device))
    # One sequential queue per GPU avoids oversubscribing either card.
    def queue(epsilon, device):
        return [train_one(arm, epsilon, device)
                for arm in ("RS1100U", "S1100", "S5500", "RS5500")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(queue, 1, 0), ex.submit(queue, 10, 1)]
        for future in futures:
            results.extend(future.result())
    write_json(RESULT / "classification_results.json", results)
    pd.DataFrame([{k: row.get(k) for k in (
        "epsilon", "display_arm", "n_train", "n_test", "auroc_macro",
        "accuracy_macro", "accuracy_micro", "accuracy_samplewise",
        "accuracy_exact_match_known")}
        for row in results]).to_csv(RESULT / "classification_results.csv", index=False)
    (RESULT / "PIPELINE_COMPLETE").write_text("complete\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("build", "generate", "filter", "train"))
    parser.add_argument("--epsilon", type=int, choices=(1, 10))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--max-prompts", type=int, default=None)
    args = parser.parse_args()
    RESULT.mkdir(parents=True, exist_ok=True)
    if args.task == "build":
        build_pool()
    elif args.task == "generate":
        if args.epsilon is None:
            raise ValueError("--epsilon is required")
        generate(args.epsilon, args.device, args.steps, args.max_prompts)
    elif args.task == "filter":
        if args.epsilon is None:
            raise ValueError("--epsilon is required")
        filter_one(args.epsilon)
    else:
        train_all()


if __name__ == "__main__":
    main()
