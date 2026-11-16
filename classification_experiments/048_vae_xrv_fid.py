"""DenseNet-121/XRV FID for deterministic reconstructions from the active VAE."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".claude/worktrees/synth-utility-improve/src"
sys.path[:0] = [str(ROOT), str(SOURCE)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-ckpt", default=str(ROOT / "checkpoints/vae/vae_ep0070.pt"))
    p.add_argument("--root-path", default="/storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0")
    p.add_argument("--manifest", default=str(ROOT / "RESULT_20261109_120931/combined_validate_test_manifest.csv.gz"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-model", choices=["xrv", "inception"], default="xrv",
                   help="feature extractor used for the FID calculation")
    p.add_argument("--output", default=str(ROOT / "RESULT_20261112_VAE_XRV_FID/vae_ep0070_xrv_fid.json"))
    args = p.parse_args()

    from Data.mimic_cxr import MIMICCXRDataset
    from Eval_metric.fid import build_feature_extractor, compute_fid, load_vae

    ds = MIMICCXRDataset(argparse.Namespace(
        root_path=args.root_path, manifest_csv=args.manifest, split="combined",
        image_size=256, max_length=512, patient_whitelist=None))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vae = load_vae(args.vae_ckpt, str(device), 256)
    vae.eval()
    feat_model, preprocess = build_feature_extractor(args.eval_model, device)
    real_features, recon_features = [], []
    mse_total, n_batches, n_images = 0.0, 0, 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            recon = vae.decode(vae.encode(imgs).mode()).clamp(-1, 1)
            mse_total += float(torch.nn.functional.mse_loss(recon, imgs))
            n_batches += 1
            real_features.append(feat_model(preprocess(imgs)).cpu().numpy())
            recon_features.append(feat_model(preprocess(recon)).cpu().numpy())
            n_images += len(imgs)
            if n_images % 256 < len(imgs):
                print(f"[VAE FID] {n_images}/{len(ds)}", flush=True)
    fr = np.concatenate(real_features, axis=0)
    fg = np.concatenate(recon_features, axis=0)
    result = {
        "vae_ckpt": str(Path(args.vae_ckpt).resolve()),
        "feature_extractor": ("torchxrayvision densenet121-res224-all"
                               if args.eval_model == "xrv"
                               else "torchvision InceptionV3 pool3"),
        "feature_dim": int(fr.shape[1]),
        "dataset_manifest": str(Path(args.manifest).resolve()),
        "dataset_split": "combined validate+test legacy evaluation manifest",
        "n_images": int(n_images),
        "reconstruction": "posterior.mode deterministic decode",
        "mean_batch_mse": float(mse_total / max(n_batches, 1)),
        ("fid_xrv_densenet121" if args.eval_model == "xrv" else "fid_inception_v3"): float(compute_fid(fr, fg)),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
