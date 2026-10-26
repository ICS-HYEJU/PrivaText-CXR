"""E028 matched structured clipping-only training with pathology label flips."""
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
from structured_conditioning import LABELS, PREFIX, unpack

EXCLUDED = {LABELS.index("No Finding"), LABELS.index("Support Devices")}


def module_digest(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def flip_report(report):
    text, states = unpack(str(report))
    positives = [i for i, state in enumerate(states) if state == 1 and i not in EXCLUDED]
    if not positives:
        return str(report), False, None
    stable = int(hashlib.sha256(str(report).encode()).hexdigest()[:16], 16)
    index = positives[stable % len(positives)]
    changed = list(states)
    changed[index] = 0
    negative = PREFIX + json.dumps({"text": text, "states": changed}, ensure_ascii=False)
    check_text, check_states = unpack(negative)
    differences = [i for i, pair in enumerate(zip(states, check_states)) if pair[0] != pair[1]]
    assert check_text == text and differences == [index] and states[index] == 1 and check_states[index] == 0
    return negative, True, LABELS[index]


def flip_batch(reports, device):
    negatives, active, labels = [], [], []
    for report in reports:
        negative, enabled, label = flip_report(report)
        negatives.append(negative)
        active.append(enabled)
        labels.append(label)
    return negatives, torch.tensor(active, dtype=torch.bool, device=device), labels


def components(ldm, z, correct_context, negative_context, timesteps, noise,
               active, relative_margin, rank_weight):
    noisy = ldm.q_sample(x_start=z, t=timesteps, noise=noise)
    predicted_correct = ldm.apply_model(noisy, timesteps, correct_context)
    predicted_negative = ldm.apply_model(noisy, timesteps, negative_context)
    correct_mse = ldm.get_loss(predicted_correct, noise, mean=False).mean((1, 2, 3))
    negative_mse = ldm.get_loss(predicted_negative, noise, mean=False).mean((1, 2, 3))
    logvar_t = ldm.logvar[timesteps.cpu()].to(ldm.device)
    diffusion = ldm.l_simple_weight * (correct_mse / torch.exp(logvar_t) + logvar_t)
    diffusion = diffusion + ldm.original_elbo_weight * ldm.lvlb_weights[timesteps] * correct_mse
    relative_gap = (negative_mse - correct_mse) / correct_mse.detach().clamp_min(1e-6)
    raw_rank = F.relu(relative_margin - relative_gap)
    rank = torch.where(active, raw_rank, raw_rank * 0.0)
    total = (diffusion + rank_weight * rank).mean()
    prefix = "train" if ldm.training else "val"
    logs = {
        f"{prefix}/loss_simple": correct_mse.mean(),
        f"{prefix}/loss_diffusion": diffusion.mean(),
        f"{prefix}/loss_flip": negative_mse.mean(),
        f"{prefix}/relative_gap": relative_gap[active].mean() if active.any() else relative_gap.mean() * 0,
        f"{prefix}/loss_rank_raw": rank[active].mean() if active.any() else rank.mean(),
        f"{prefix}/loss_rank_weighted": rank_weight * rank.mean(),
        f"{prefix}/rank_active_fraction": active.float().mean(),
        f"{prefix}/loss": total,
    }
    return total, logs, diffusion, rank


def flatten_grads(grads, params):
    return torch.cat([(torch.zeros_like(param) if grad is None else grad).detach().float().flatten()
                      for grad, param in zip(grads, params)])


def gradient_audit(module, dataset, margin, weight, seed, clipping_norm):
    eligible = []
    for index in range(min(64, len(dataset))):
        sample = dataset[index]
        try:
            _, active, _ = flip_report(sample[1])
        except ValueError:
            continue
        if active:
            eligible.append(sample)
        if len(eligible) == 2:
            break
    if len(eligible) < 2:
        raise AssertionError("could not find two eligible label-flip records")
    params_named = [(name, param) for name, param in module.named_parameters() if param.requires_grad]
    params = [param for _, param in params_named]
    vectors = []
    details = []
    for record_index, (image, report) in enumerate(eligible):
        image = image.unsqueeze(0).to(module.device)
        with torch.no_grad():
            posterior = module.first_stage_model.encode(image)
            z = module.scale_factor * posterior.mean
        negative, active, pathology = flip_report(report)
        correct_context = module.encode_condition([report])
        negative_context = module.encode_condition([negative])
        timestep = torch.tensor([250], device=module.device)
        generator = torch.Generator(device=module.device).manual_seed(700000 + seed + record_index)
        noise = torch.randn(z.shape, device=module.device, generator=generator)
        _, logs, diffusion, rank = components(
            module, z, correct_context, negative_context, timestep, noise,
            torch.tensor([active], device=module.device), margin, weight)
        diffusion_grads = torch.autograd.grad(diffusion.mean(), params, retain_graph=True,
                                              allow_unused=True)
        rank_grads = torch.autograd.grad(rank.mean(), params, allow_unused=True)
        diffusion_vector = flatten_grads(diffusion_grads, params)
        rank_vector = flatten_grads(rank_grads, params)
        total_vector = diffusion_vector + weight * rank_vector
        norm = total_vector.norm().clamp_min(1e-20)
        clipped = total_vector * min(1.0, clipping_norm / float(norm.cpu()))
        cosine = float(F.cosine_similarity(diffusion_vector, rank_vector, dim=0).cpu())
        projection_mask = torch.tensor([
            "embedder.proj" in name for name, param in params_named for _ in range(param.numel())],
            device=total_vector.device)
        label_mask = torch.tensor([
            "label_tokens" in name for name, param in params_named for _ in range(param.numel())],
            device=total_vector.device)
        lora_mask = ~(projection_mask | label_mask)
        details.append({
            "pathology": pathology,
            "correct_mse": float(logs["train/loss_simple"].detach().cpu()),
            "flip_mse": float(logs["train/loss_flip"].detach().cpu()),
            "relative_gap": float(logs["train/relative_gap"].detach().cpu()),
            "diffusion_grad_norm": float(diffusion_vector.norm().cpu()),
            "rank_grad_norm": float(rank_vector.norm().cpu()),
            "diffusion_rank_cosine": cosine,
            "total_grad_norm": float(norm.cpu()),
            "projection_rank_grad_norm": float(rank_vector[projection_mask].norm().cpu()),
            "label_token_rank_grad_norm": float(rank_vector[label_mask].norm().cpu()),
            "other_lora_rank_grad_norm": float(rank_vector[lora_mask].norm().cpu()),
            "clip_factor": min(1.0, clipping_norm / float(norm.cpu())),
        })
        vectors.append((total_vector, clipped, diffusion_vector))
        module.zero_grad(set_to_none=True)
    aggregate = torch.stack([item[0] for item in vectors]).mean(0)
    clipped_aggregate = torch.stack([item[1] for item in vectors]).mean(0)
    diffusion_aggregate = torch.stack([item[2] for item in vectors]).mean(0)
    return {
        "records": details,
        "unclipped_vs_clipped_aggregate_cosine": float(
            F.cosine_similarity(aggregate, clipped_aggregate, dim=0).cpu()),
        "diffusion_vs_clipped_aggregate_cosine": float(
            F.cosine_similarity(diffusion_aggregate, clipped_aggregate, dim=0).cpu()),
    }


def install_and_train(argv):
    import cached_training
    import LDM_dp_finetune as ft
    from Model.Diffusion.LDM_dp import LatentDiffusionDP
    from opacus import PrivacyEngine

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rank-weight", type=float, required=True)
    parser.add_argument("--relative-margin", type=float, default=0.01)
    parser.add_argument("--experiment-id", default="028")
    parser.add_argument("--smoke-save-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    control, remaining = parser.parse_known_args(argv)
    smoke_dir = control.smoke_save_dir or ROOT / "classification_experiments/artifacts/label_flip_smoke"
    runtime = {"ldm": None, "vae_digest": None, "seed": 42, "smoke_dir": Path(smoke_dir),
               "experiment_id": control.experiment_id, "weight": control.rank_weight,
               "margin": control.relative_margin}
    original_main = ft.main

    def intercepted_main():
        original_dataset = ft._make_dataset
        original_parse = ft.parse_args

        def make_dataset(*args, **kwargs):
            dataset = original_dataset(*args, **kwargs)
            if control.smoke and kwargs.get("split") == "train":
                dataset = torch.utils.data.Subset(dataset, range(min(64, len(dataset))))
            return dataset

        def parse_args():
            args = original_parse()
            if not getattr(args, "structured_labels", False):
                raise AssertionError("E028 requires --structured-labels")
            if getattr(args, "diagnostic_privacy_mode", None) != "clip":
                raise AssertionError("E028 first stage requires --privacy-control clip")
            args.label_flip_relative_margin = control.relative_margin
            args.label_flip_rank_weight = control.rank_weight
            runtime["seed"] = args.seed
            runtime["clip"] = args.max_grad_norm
            if control.smoke:
                args.logical_batch, args.physical_batch, args.epochs = 8, 4, 1
                args.warmup_steps, args.scheduler_type = 0, "none"
                args.save_every, args.log_every, args.do_validation = 1, 1, False
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
                z = self.scale_factor * (posterior.mean + posterior.std * torch.randn(
                    posterior.mean.shape, device=device, generator=generator))
            reports = batch["reports"]
            negatives, active, _ = flip_batch(reports, device)
            correct_context = self.encode_condition(reports)
            negative_context = self.encode_condition(negatives)
            timesteps = torch.randint(0, self.num_timesteps, (len(z),),
                                      device=device, generator=generator)
            noise = torch.randn(z.shape, device=device, generator=generator)
            runtime["ldm"] = self
            if runtime["vae_digest"] is None:
                runtime["vae_digest"] = module_digest(self.first_stage_model)
            total, logs, _, _ = components(
                self, z, correct_context, negative_context, timesteps, noise, active,
                control.relative_margin, control.rank_weight)
            return total, logs

        original_private = PrivacyEngine.make_private
        audited = {"done": False}

        def make_private(engine, **kwargs):
            if not audited["done"]:
                runtime["gradient_audit"] = gradient_audit(
                    kwargs["module"], kwargs["data_loader"].dataset,
                    control.relative_margin, control.rank_weight, runtime["seed"],
                    kwargs["max_grad_norm"])
                print("E028_GRADIENT_AUDIT", runtime["gradient_audit"], flush=True)
                audited["done"] = True
            return original_private(engine, **kwargs)

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
    if module_digest(ldm.first_stage_model) != runtime["vae_digest"]:
        raise AssertionError("frozen VAE changed")
    checkpoint = torch.load(runtime["smoke_dir"] / "ldm_dp_final.pt",
                            map_location="cpu", weights_only=False)
    selected = [key for key in ldm.state_dict() if "embedder.proj" in key or
                "label_tokens" in key or "lora_A" in key or "lora_B" in key]
    mismatch = [key for key in selected if key not in checkpoint["model"] or not torch.equal(
        ldm.state_dict()[key].detach().cpu(), checkpoint["model"][key].detach().cpu())]
    if mismatch:
        raise AssertionError(f"checkpoint mismatch: {mismatch[:5]}")
    evidence = {
        "status": "passed", "privacy_control": "clip", "noise_multiplier": 0.0,
        "epsilon_infinite": math.isinf(float(checkpoint["epsilon_spent"])),
        "relative_margin": runtime["margin"], "rank_weight": runtime["weight"],
        "vae_frozen": True, "checkpoint_exact": True, "gradient_audit": runtime["gradient_audit"]}
    path = ROOT / f"classification_experiments/evidence/{runtime['experiment_id']}_smoke.json"
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    print("E028_SMOKE_OK", path, flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "smoke"}:
        raise SystemExit("usage: label_flip_training.py {train|smoke} [args]")
    command, args = sys.argv[1], sys.argv[2:]
    if command == "smoke":
        args = ["--smoke"] + args
    install_and_train(args)
