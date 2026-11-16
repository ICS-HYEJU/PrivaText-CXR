"""E027: DP Projection+LoRA training with per-record condition contrast."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "classification_experiments")]

COUNTERFACTUAL = "LABEL: No Finding IMPRESSION: No acute cardiopulmonary abnormality."


def module_digest(module):
    h = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def loss_components(ldm, z, correct_context, counterfactual_context, timesteps, noise,
                    margin, rank_weight):
    x_noisy = ldm.q_sample(x_start=z, t=timesteps, noise=noise)
    predicted_correct = ldm.apply_model(x_noisy, timesteps, correct_context)
    predicted_counterfactual = ldm.apply_model(x_noisy, timesteps, counterfactual_context)
    if ldm.parameterization != "eps":
        raise RuntimeError("E027 requires epsilon parameterization")

    correct_mse = ldm.get_loss(predicted_correct, noise, mean=False).mean((1, 2, 3))
    counterfactual_mse = ldm.get_loss(
        predicted_counterfactual, noise, mean=False).mean((1, 2, 3))
    logvar_t = ldm.logvar[timesteps.cpu()].to(ldm.device)
    diffusion = ldm.l_simple_weight * (correct_mse / torch.exp(logvar_t) + logvar_t)
    diffusion = diffusion + (
        ldm.original_elbo_weight * ldm.lvlb_weights[timesteps] * correct_mse)
    rank = F.relu(margin + correct_mse - counterfactual_mse)
    total = (diffusion + rank_weight * rank).mean()
    prefix = "train" if ldm.training else "val"
    loss_dict = {
        f"{prefix}/loss_simple": correct_mse.mean(),
        f"{prefix}/loss_diffusion": diffusion.mean(),
        f"{prefix}/loss_counterfactual": counterfactual_mse.mean(),
        f"{prefix}/condition_gap": (counterfactual_mse - correct_mse).mean(),
        f"{prefix}/loss_rank_raw": rank.mean(),
        f"{prefix}/loss_rank_weighted": rank_weight * rank.mean(),
        f"{prefix}/rank_active_fraction": (rank > 0).float().mean(),
        f"{prefix}/loss": total,
    }
    return total, loss_dict, rank


def install_and_train(argv):
    import cached_training
    import LDM_dp_finetune as ft
    from Model.Diffusion.LDM_dp import LatentDiffusionDP
    from opacus import PrivacyEngine

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rank-margin", type=float, default=0.01)
    parser.add_argument("--rank-weight", type=float, default=0.1)
    parser.add_argument("--experiment-id", default="027")
    parser.add_argument("--smoke-save-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    control, remaining = parser.parse_known_args(argv)
    if control.rank_margin <= 0 or control.rank_weight <= 0:
        raise ValueError("rank margin and weight must be positive")

    smoke_dir = control.smoke_save_dir or (
        ROOT / "classification_experiments/artifacts/counterfactual_smoke")
    runtime = {"ldm": None, "seed": 42, "vae_digest": None,
               "experiment_id": control.experiment_id, "smoke_dir": Path(smoke_dir),
               "rank_margin": control.rank_margin, "rank_weight": control.rank_weight}
    original_main = ft.main

    def intercepted_main():
        original_make_dataset = ft._make_dataset
        original_parse_args = ft.parse_args

        def make_dataset(*args, **kwargs):
            dataset = original_make_dataset(*args, **kwargs)
            if control.smoke and kwargs.get("split") == "train":
                dataset = torch.utils.data.Subset(dataset, range(min(32, len(dataset))))
            return dataset

        def parse_args():
            args = original_parse_args()
            args.condition_rank_margin = control.rank_margin
            args.condition_rank_weight = control.rank_weight
            args.condition_rank_negative = COUNTERFACTUAL
            runtime["seed"] = args.seed
            if control.smoke:
                args.logical_batch = 8
                args.physical_batch = 4
                args.epochs = 1
                args.warmup_steps = 0
                args.scheduler_type = "none"
                args.save_every = 1
                args.log_every = 1
                args.do_validation = False
                args.save_dir = str(smoke_dir)
            return args

        generators = {}

        def training_step(self, batch):
            device = self.device
            if str(device) not in generators:
                generators[str(device)] = torch.Generator(device=device).manual_seed(
                    200000 + runtime["seed"])
            generator = generators[str(device)]
            with torch.no_grad():
                posterior = self.first_stage_model.encode(batch["image"].to(device))
                z = self.scale_factor * (
                    posterior.mean + posterior.std * torch.randn(
                        posterior.mean.shape, device=device, generator=generator))
            correct = self.encode_condition(batch["reports"])
            counterfactual = self.encode_condition([COUNTERFACTUAL] * len(z))
            timesteps = torch.randint(
                0, self.num_timesteps, (len(z),), device=device, generator=generator)
            noise = torch.randn(z.shape, device=device, generator=generator)
            runtime["ldm"] = self
            if runtime["vae_digest"] is None:
                runtime["vae_digest"] = module_digest(self.first_stage_model)
            total, loss_dict, _ = loss_components(
                self, z, correct, counterfactual, timesteps, noise,
                control.rank_margin, control.rank_weight)
            return total, loss_dict

        original_make_private = PrivacyEngine.make_private
        preflight_done = {"value": False}

        def make_private(engine, **kwargs):
            if not preflight_done["value"]:
                module = kwargs["module"]
                dataset = kwargs["data_loader"].dataset
                samples = [dataset[i] for i in range(min(2, len(dataset)))]
                images = torch.stack([sample[0] for sample in samples]).to(module.device)
                reports = [sample[1] for sample in samples]
                with torch.no_grad():
                    posterior = module.first_stage_model.encode(images)
                    z = module.scale_factor * posterior.mean
                correct = module.encode_condition(reports)
                negative = module.encode_condition([COUNTERFACTUAL] * len(samples))
                timesteps = torch.full((len(samples),), 250, device=module.device,
                                       dtype=torch.long)
                generator = torch.Generator(device=module.device).manual_seed(
                    500000 + runtime["seed"])
                noise = torch.randn(z.shape, device=module.device, generator=generator)
                _, losses, rank = loss_components(
                    module, z, correct, negative, timesteps, noise,
                    control.rank_margin, control.rank_weight)
                if not torch.isfinite(rank).all() or not (rank > 0).any():
                    raise AssertionError(f"invalid/inactive rank loss: {rank}")
                named = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
                grads = torch.autograd.grad(rank.mean(), [p for _, p in named], allow_unused=True)
                projection_sq = 0.0
                lora_sq = 0.0
                for (name, _), grad in zip(named, grads):
                    if grad is None:
                        continue
                    value = float(grad.detach().float().pow(2).sum().cpu())
                    if "embedder.proj" in name:
                        projection_sq += value
                    if "lora_A" in name or "lora_B" in name:
                        lora_sq += value
                projection_norm = projection_sq ** 0.5
                lora_norm = lora_sq ** 0.5
                if not math.isfinite(projection_norm) or projection_norm <= 0:
                    raise AssertionError(f"projection rank gradient invalid: {projection_norm}")
                if not math.isfinite(lora_norm) or lora_norm <= 0:
                    raise AssertionError(f"LoRA rank gradient invalid: {lora_norm}")
                module.zero_grad(set_to_none=True)
                runtime["preflight"] = {
                    "correct_mse": float(losses["train/loss_simple"].detach().cpu()),
                    "counterfactual_mse": float(
                        losses["train/loss_counterfactual"].detach().cpu()),
                    "condition_gap": float(losses["train/condition_gap"].detach().cpu()),
                    "rank_loss": float(rank.mean().detach().cpu()),
                    "projection_grad_norm": projection_norm,
                    "lora_grad_norm": lora_norm,
                }
                runtime["vae_digest"] = module_digest(module.first_stage_model)
                print("E027_RANK_GRAD_OK", runtime["preflight"], flush=True)
                preflight_done["value"] = True
            return original_make_private(engine, **kwargs)

        ft._make_dataset = make_dataset
        ft.parse_args = parse_args
        LatentDiffusionDP.training_step_dp = training_step
        PrivacyEngine.make_private = make_private
        return original_main()

    ft.main = intercepted_main
    cached_training.train(remaining)
    if control.smoke:
        verify_smoke(runtime)


def verify_smoke(runtime):
    ldm = runtime["ldm"]
    if ldm is None:
        raise RuntimeError("smoke did not run")
    if module_digest(ldm.first_stage_model) != runtime["vae_digest"]:
        raise AssertionError("frozen VAE changed")
    checkpoint = torch.load(runtime["smoke_dir"] / "ldm_dp_final.pt",
                            map_location="cpu", weights_only=False)
    saved, current = checkpoint["model"], ldm.state_dict()
    selected = [k for k in current if "embedder.proj" in k or
                "lora_A" in k or "lora_B" in k]
    mismatched = [k for k in selected if k not in saved or not torch.equal(
        current[k].detach().cpu(), saved[k].detach().cpu())]
    if not selected or mismatched:
        raise AssertionError(f"checkpoint mismatch: {mismatched[:5]}")
    evidence = {
        "status": "passed", "dp_per_record_negative": True,
        "counterfactual": COUNTERFACTUAL, "rank_margin": runtime["rank_margin"],
        "rank_weight": runtime["rank_weight"], "vae_frozen": True,
        "checkpoint_projection_lora_exact": True, "physical_batch": 4,
        "opacus_two_forward_complete": True,
        "epsilon_spent": float(checkpoint["epsilon_spent"]),
        "preflight": runtime["preflight"],
    }
    path = ROOT / f"classification_experiments/evidence/{runtime['experiment_id']}_smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    print("E027_SMOKE_OK", path, flush=True)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "smoke"}:
        raise SystemExit("usage: counterfactual_training.py {smoke|train} [training args]")
    command, args = sys.argv[1], sys.argv[2:]
    if command == "smoke":
        args = ["--smoke"] + args
    install_and_train(args)


if __name__ == "__main__":
    main()
