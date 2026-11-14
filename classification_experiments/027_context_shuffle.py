"""E027 correct-vs-patient-shuffled validation denoising diagnostic."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "classification_experiments")]
from diagnostics import OUT, dump, inference_args
from structured_conditioning import load_model


def main(arm, device):
    from Data.mimic_cxr import MIMICCXRDataset

    rows = pd.read_csv(OUT / "ablation" / arm / "validation_mapping.csv")
    assert len(rows) == 802
    ds = MIMICCXRDataset(argparse.Namespace(
        root_path="/storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0",
        manifest_csv=str(ROOT / "data_manifests/mimic_p10_p12/ldm_dp_manifest.csv.gz"),
        split="validate", image_size=256, max_length=512, patient_whitelist=None))
    keys = [(int(str(m["patient_id"]).lstrip("p")),
             int(str(m["study_id"]).lstrip("s"))) for m in ds.samples]
    assert keys == list(zip(rows.subject_id, rows.study_id))

    rng = np.random.default_rng(800042)
    subjects = rows.subject_id.to_numpy()
    groups = [np.flatnonzero(subjects == s) for s in rng.permutation(np.unique(subjects))]
    largest = max(map(len, groups))
    assert largest * 2 <= len(rows)
    order = np.concatenate([rng.permutation(group) for group in groups])
    permutation = np.empty(len(rows), dtype=int)
    permutation[order] = np.roll(order, largest)
    assert len(np.unique(permutation)) == len(rows)
    assert np.all(subjects != subjects[permutation])

    out = OUT / "027_context_shuffle" / arm
    out.mkdir(parents=True, exist_ok=True)
    mapping = rows.copy()
    mapping["shuffle_index"] = permutation
    mapping.to_csv(out / "mapping.csv", index=False)

    dev = torch.device(f"cuda:{device}")
    checkpoint = OUT / "ablation" / arm / "ldm_dp_final.pt"
    model, _ = load_model(inference_args(checkpoint, device), dev)
    latent_blocks = []
    for start in range(0, len(ds), 8):
        images = torch.stack([ds[i][0] for i in range(start, min(start + 8, len(ds)))])
        with torch.no_grad():
            posterior = model.first_stage_model.encode(images.to(dev))
            latent_blocks.append((posterior.mode() * model.scale_factor).cpu())
    latents = torch.cat(latent_blocks)

    records = []
    for timestep in (50, 250, 500, 750):
        generator = torch.Generator(device=dev).manual_seed(810000 + timestep)
        for start in range(0, len(rows), 4):
            end = min(start + 4, len(rows))
            z = latents[start:end].to(dev)
            noise = torch.randn(z.shape, device=dev, generator=generator)
            t = torch.full((len(z),), timestep, device=dev, dtype=torch.long)
            with torch.no_grad():
                noisy = model.q_sample(z, t, noise=noise)
                correct = model.apply_model(
                    noisy, t, model.encode_condition(rows.prompt.iloc[start:end].tolist()))
                shuffled = model.apply_model(
                    noisy, t,
                    model.encode_condition(rows.prompt.iloc[permutation[start:end]].tolist()))
                losses = [(prediction - noise).square().flatten(1).mean(1).cpu().numpy()
                          for prediction in (correct, shuffled)]
            records.extend({
                "index": index, "subject_id": int(subjects[index]), "t": timestep,
                "correct": float(losses[0][index - start]),
                "shuffled": float(losses[1][index - start]),
            } for index in range(start, end))
        print(arm, "t", timestep, "done", flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv(out / "losses.csv", index=False)
    summary = {}
    for timestep, part in frame.groupby("t"):
        patient_groups = [group[["correct", "shuffled"]].to_numpy()
                          for _, group in part.groupby("subject_id")]
        rng = np.random.default_rng(42)
        deltas = []
        for _ in range(500):
            sample = np.concatenate([
                patient_groups[i] for i in rng.integers(
                    len(patient_groups), size=len(patient_groups))])
            deltas.append(float((sample[:, 1] - sample[:, 0]).mean()))
        correct_mean = float(part.correct.mean())
        shuffled_mean = float(part.shuffled.mean())
        summary[str(timestep)] = {
            "correct_mse": correct_mean, "shuffled_mse": shuffled_mean,
            "delta_shuffled_minus_correct": shuffled_mean - correct_mean,
            "relative_gain": (shuffled_mean - correct_mean) / shuffled_mean,
            "patient_ci95": np.quantile(deltas, [.025, .975]).tolist(),
        }
    result = {"arm": arm, "checkpoint": str(checkpoint), "n": len(rows),
              "timesteps": [50, 250, 500, 750], "results": summary}
    dump(out / "results.json", result)
    dump(ROOT / "classification_experiments/evidence/027_context_shuffle.json", result)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="counterfactual")
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    main(args.arm, args.device)
