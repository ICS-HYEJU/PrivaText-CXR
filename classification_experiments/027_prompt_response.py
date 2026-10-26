"""E027 prompt-response screen with paired noise and E004/E024 controls."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'classification_experiments')]
from ablation_eval import ddim
from diagnostics import dump, inference_args, save_image, score_paths, PROBES
from LDM_dp_inference import build_and_load_ldm

SENTENCES = {
    'Cardiomegaly': ('Cardiac silhouette is enlarged.', 'Cardiac silhouette is not enlarged.'),
    'Pleural Effusion': ('A left pleural effusion is present.', 'No pleural effusion is present.'),
    'Edema': ('Pulmonary edema is present.', 'No pulmonary edema is present.'),
    'Atelectasis': ('Left basilar atelectasis is present.', 'No atelectasis is present.'),
}
OUT = ROOT / 'classification_experiments/artifacts/027_prompt_response'


def generate(arm, device, seeds=32, steps=50):
    checkpoint = ROOT / 'classification_experiments/artifacts/ablation' / arm / 'ldm_dp_final.pt'
    args = inference_args(checkpoint, device)
    model, _ = build_and_load_ldm(args, torch.device(f'cuda:{device}'))
    folder = OUT / arm
    rows = []
    for pathology in PROBES:
        for seed in range(seeds):
            for target in (0, 1):
                prompt = ('LABEL: ' + (pathology if target else 'No Finding') +
                          ' IMPRESSION: ' + SENTENCES[pathology][0 if target else 1])
                path = folder / 'samples' / f'{pathology.replace(" ", "_")}_{seed:03d}_{target}.png'
                if not path.exists():
                    context = model.encode_condition([prompt])
                    image = ddim(model, context, 30000 + seed, steps=steps)
                    save_image(image[0], path)
                rows.append({'arm': arm, 'pathology': pathology, 'seed': seed,
                             'target': target, 'prompt': prompt, 'file': str(path)})
            if (seed + 1) % 8 == 0:
                print(f'{arm} {pathology} {seed + 1}/{seeds}', flush=True)
    pd.DataFrame(rows).to_csv(folder / 'mapping.csv', index=False)
    dump(folder / 'generation.json', {
        'arm': arm, 'checkpoint': str(checkpoint), 'n': len(rows), 'seeds': seeds,
        'sampler': 'DDIM', 'steps': steps, 'eta': 0, 't_range': [0, 999],
        'paired_noise_seed_base': 30000,
    })


def evaluate(arms):
    from sklearn.metrics import roc_auc_score

    result = {}
    for arm in arms:
        folder = OUT / arm
        rows = pd.read_csv(folder / 'mapping.csv')
        meta = json.loads((folder / 'generation.json').read_text())
        assert len(rows) == meta['n'] == 8 * meta['seeds']
        assert not rows.duplicated(['pathology', 'seed', 'target']).any()
        predictions = score_paths(rows.file.tolist())
        result[arm] = {}
        for encoder, scores in predictions.items():
            frame = rows[['pathology', 'seed', 'target']].copy()
            for pathology, values in scores.items():
                frame[pathology] = values
            frame.to_csv(folder / f'{encoder}_scores.csv', index=False)
            result[arm][encoder] = {}
            for pathology in PROBES:
                subset = frame[frame.pathology == pathology]
                pairs = subset.pivot(index='seed', columns='target', values=pathology)
                delta = (pairs[1] - pairs[0]).to_numpy()
                rng = np.random.default_rng(20261023)
                bootstrap = rng.choice(delta, (2000, len(delta)), replace=True).mean(1)
                result[arm][encoder][pathology] = {
                    'auroc': float(roc_auc_score(subset.target, subset[pathology])),
                    'paired_mean_score_delta': float(delta.mean()),
                    'delta_ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
                    'positive_direction_fraction': float((delta > 0).mean()),
                    'pairs': len(delta),
                }
    dump(OUT / 'results.json', result)
    dump(ROOT / 'classification_experiments/evidence/027_prompt_response.json', result)
    for arm in arms:
        print(arm, {encoder: {p: round(v['auroc'], 4) for p, v in values.items()}
                    for encoder, values in result[arm].items()}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['generate', 'evaluate'])
    parser.add_argument('--arm', default='counterfactual')
    parser.add_argument('--arms', nargs='+', default=['counterfactual'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seeds', type=int, default=32)
    parser.add_argument('--steps', type=int, default=50)
    args = parser.parse_args()
    if args.task == 'generate':
        generate(args.arm, args.device, args.seeds, args.steps)
    else:
        evaluate(args.arms)


if __name__ == '__main__':
    main()
