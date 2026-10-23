"""E024: DP Projection+LoRA training with a frozen XRV label-preservation loss.

This is an experiment-only wrapper around cached_training.py. It preserves the
E004 data order, latent/timestep RNG, DP noise RNG and optimizer settings while
adding a differentiable predicted-x0 -> frozen VAE -> frozen XRV path.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "classification_experiments")]

# Explicit name mapping. The tuple order is the target tensor order and is
# resolved against model.pathologies at runtime; XRV/CSV column order is never assumed.
AUX_LABEL_MAP = (
    ("Atelectasis", "Atelectasis"),
    ("Cardiomegaly", "Cardiomegaly"),
    ("Consolidation", "Consolidation"),
    ("Edema", "Edema"),
    ("Effusion", "Pleural Effusion"),
    ("Pneumonia", "Pneumonia"),
    ("Pneumothorax", "Pneumothorax"),
    ("Lung Opacity", "Lung Opacity"),
    ("Lung Lesion", "Lung Lesion"),
    ("Fracture", "Fracture"),
    ("Enlarged Cardiomediastinum", "Enlarged Cardiomediastinum"),
)


class LabeledReport(str):
    """A tokenizer-compatible string carrying its private label target."""

    def __new__(cls, text, labels, label_mask):
        obj = str.__new__(cls, text)
        obj.labels = labels
        obj.label_mask = label_mask
        return obj

    def __reduce_ex__(self, protocol):
        return (type(self), (str(self), self.labels, self.label_mask))

    def __deepcopy__(self, memo):
        copied = type(self)(str(self), self.labels.clone(), self.label_mask.clone())
        memo[id(self)] = copied
        return copied


def make_targets(row):
    labels = torch.zeros(len(AUX_LABEL_MAP), dtype=torch.float32)
    mask = torch.zeros(len(AUX_LABEL_MAP), dtype=torch.bool)
    if row is None:
        return labels, mask
    for i, (_, csv_name) in enumerate(AUX_LABEL_MAP):
        value = row.get(csv_name, np.nan)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value in (0.0, 1.0):
            labels[i] = value
            mask[i] = True
    return labels, mask


class XRVLabelDataset:
    """Adds mapped CheXpert targets to the cached-posterior training dataset."""

    def __init__(self, cached_dataset):
        from Data.mimic_cxr import _load_chexpert_gt

        if not hasattr(cached_dataset, "base"):
            raise TypeError("E024 requires cached_training.CachedDataset")
        self.cached = cached_dataset
        self.base = cached_dataset.base
        gt_path = self.base._resolve_chexpert_csv()
        self.gt, _ = _load_chexpert_gt(gt_path)
        self.gt_path = gt_path
        self.valid_counts = torch.zeros(len(AUX_LABEL_MAP), dtype=torch.long)
        for meta in self.base.samples:
            row = self.gt.get(self._key(meta))
            _, mask = make_targets(row)
            self.valid_counts += mask.long()
        print("[XRV-AUX] labels", gt_path, "valid_counts=", self.valid_counts.tolist(), flush=True)

    @staticmethod
    def _key(meta):
        return (int(str(meta["patient_id"]).lstrip("p")),
                int(str(meta["study_id"]).lstrip("s")))

    def __len__(self):
        return len(self.cached)

    def __getitem__(self, index):
        latent, text = self.cached[index]
        labels, mask = make_targets(self.gt.get(self._key(self.base.samples[index])))
        return latent, LabeledReport(text, labels, mask)


def module_digest(module):
    h = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def get_teacher(holder, device):
    key = str(device)
    if holder.get("device") != key:
        import torchxrayvision as xrv

        teacher = xrv.models.DenseNet(weights="densenet121-res224-all").to(device).eval()
        teacher.requires_grad_(False)
        head_indices = []
        for xrv_name, _ in AUX_LABEL_MAP:
            if xrv_name not in teacher.pathologies:
                raise RuntimeError(f"XRV pathology missing: {xrv_name}")
            head_indices.append(teacher.pathologies.index(xrv_name))
        holder.clear()
        holder.update(teacher=teacher, head_indices=head_indices, device=key,
                      initial_digest=module_digest(teacher))
        print("[XRV-AUX] frozen teacher loaded", key,
              "heads=", [(n, i) for (n, _), i in zip(AUX_LABEL_MAP, head_indices)], flush=True)
    return holder["teacher"], holder["head_indices"]


def targets_from_reports(reports, device):
    if not all(isinstance(r, LabeledReport) for r in reports):
        raise TypeError("E024 batch contains a report without attached labels")
    labels = torch.stack([r.labels for r in reports]).to(device)
    masks = torch.stack([r.label_mask for r in reports]).to(device)
    return labels, masks


def xrv_auxiliary_per_sample(ldm, x_noisy, predicted_noise, timesteps, reports,
                             teacher_holder, timestep_max):
    """Return per-record masked BCE; inactive records are exactly zero."""
    device = x_noisy.device
    labels, label_mask = targets_from_reports(reports, device)
    active = (timesteps <= timestep_max) & label_mask.any(dim=1)
    aux_per_sample = predicted_noise.sum(dim=(1, 2, 3)) * 0.0
    if active.any():
        indices = active.nonzero(as_tuple=False).flatten()
        x0 = ldm.predict_start_from_noise(
            x_noisy[indices], timesteps[indices], predicted_noise[indices])
        offload_device = getattr(ldm, "offload_device", device)
        decoded = ldm.differentiable_decode_first_stage(x0.to(offload_device)).to(device)
        if not torch.isfinite(x0).all() or not torch.isfinite(decoded).all():
            print("E024_NONFINITE_DECODE", {
                "x0_finite": bool(torch.isfinite(x0).all()),
                "x0_min": float(torch.nan_to_num(x0).min().detach().cpu()),
                "x0_max": float(torch.nan_to_num(x0).max().detach().cpu()),
                "decoded_finite": bool(torch.isfinite(decoded).all()),
                "decoded_min": float(torch.nan_to_num(decoded).min().detach().cpu()),
                "decoded_max": float(torch.nan_to_num(decoded).max().detach().cpu()),
            }, flush=True)
        xrv_input = F.interpolate(decoded, size=(224, 224), mode="bilinear", align_corners=False)
        xrv_input = xrv_input.clamp(-1.0, 1.0) * 1024.0
        teacher, head_indices = get_teacher(teacher_holder, device)
        # DenseNet.forward applies sigmoid/op_norm for pretrained weights.
        # Use the classifier head directly so BCEWithLogits receives logits.
        logits = teacher.classifier(teacher.features2(xrv_input))[:, head_indices]
        if not torch.isfinite(logits).all():
            print("E024_NONFINITE_XRV", {
                "input_finite": bool(torch.isfinite(xrv_input).all()),
                "input_min": float(torch.nan_to_num(xrv_input).min().detach().cpu()),
                "input_max": float(torch.nan_to_num(xrv_input).max().detach().cpu()),
                "logits_finite": bool(torch.isfinite(logits).all()),
                "logits_min": float(torch.nan_to_num(logits).min().detach().cpu()),
                "logits_max": float(torch.nan_to_num(logits).max().detach().cpu()),
            }, flush=True)
        selected_labels = labels[indices]
        selected_mask = label_mask[indices]
        element = F.binary_cross_entropy_with_logits(logits, selected_labels, reduction="none")
        record_loss = (element * selected_mask).sum(dim=1) / selected_mask.sum(dim=1).clamp_min(1)
        aux_per_sample = aux_per_sample.index_copy(0, indices, record_loss)
    return aux_per_sample, active, label_mask


def combined_loss(ldm, z, context, timesteps, noise, reports, teacher_holder,
                  aux_weight, timestep_max):
    """Exact existing diffusion loss plus prespecified per-record XRV loss."""
    x_noisy = ldm.q_sample(x_start=z, t=timesteps, noise=noise)
    predicted_noise = ldm.apply_model(x_noisy, timesteps, context)
    if ldm.parameterization != "eps":
        raise RuntimeError("E024 was designed for epsilon-parameterized diffusion")

    loss_simple = ldm.get_loss(predicted_noise, noise, mean=False).mean((1, 2, 3))
    logvar_t = ldm.logvar[timesteps.cpu()].to(ldm.device)
    diffusion_per_sample = ldm.l_simple_weight * (
        loss_simple / torch.exp(logvar_t) + logvar_t)
    diffusion_per_sample = diffusion_per_sample + (
        ldm.original_elbo_weight * ldm.lvlb_weights[timesteps] * loss_simple)

    aux_per_sample, active, label_mask = xrv_auxiliary_per_sample(
        ldm, x_noisy, predicted_noise, timesteps, reports,
        teacher_holder, timestep_max)
    total = (diffusion_per_sample + aux_weight * aux_per_sample).mean()
    prefix = "train" if ldm.training else "val"
    loss_dict = {
        f"{prefix}/loss_simple": loss_simple.mean(),
        f"{prefix}/loss_vlb": (ldm.lvlb_weights[timesteps] * loss_simple).mean(),
        f"{prefix}/loss_diffusion": diffusion_per_sample.mean(),
        f"{prefix}/loss_xrv_raw": aux_per_sample[active].mean() if active.any() else aux_per_sample.mean(),
        f"{prefix}/loss_xrv_weighted": aux_weight * aux_per_sample.mean(),
        f"{prefix}/xrv_active_fraction": active.float().mean(),
        f"{prefix}/xrv_valid_labels": label_mask.float().sum(dim=1).mean(),
        f"{prefix}/loss": total,
    }
    if ldm.learn_logvar:
        loss_dict[f"{prefix}/loss_gamma"] = (
            loss_simple / torch.exp(logvar_t) + logvar_t).mean()
        loss_dict["logvar"] = ldm.logvar.data.mean()
    return total, loss_dict


def install_and_train(argv):
    import cached_training
    import LDM_dp_finetune as ft
    from Model.Diffusion.LDM_dp import LatentDiffusionDP

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--aux-weight", type=float, default=0.01)
    parser.add_argument("--aux-timestep-max", type=int, default=250)
    parser.add_argument("--smoke", action="store_true")
    control, remaining = parser.parse_known_args(argv)
    if control.aux_weight <= 0:
        raise ValueError("--aux-weight must be positive")
    if not 0 <= control.aux_timestep_max < 1000:
        raise ValueError("--aux-timestep-max must be in [0, 999]")

    teacher_holder = {}
    runtime = {"seed": 42, "ldm": None, "vae_digest": None, "smoke": control.smoke}
    original_main = ft.main

    def intercepted_main():
        cached_make_dataset = ft._make_dataset
        cached_parse_args = ft.parse_args

        def make_dataset(*args, **kwargs):
            dataset = cached_make_dataset(*args, **kwargs)
            if kwargs.get("split") == "train":
                dataset = XRVLabelDataset(dataset)
                if control.smoke:
                    dataset = torch.utils.data.Subset(dataset, range(min(32, len(dataset))))
            return dataset

        def parse_args():
            args = cached_parse_args()
            args.xrv_aux_weight = control.aux_weight
            args.xrv_aux_timestep_max = control.aux_timestep_max
            args.xrv_aux_teacher = "densenet121-res224-all"
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
                args.save_dir = str(ROOT / "classification_experiments/artifacts/xrv_aux_smoke")
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
            context = self.encode_condition(batch["reports"])
            timesteps = torch.randint(
                0, self.num_timesteps, (len(z),), device=device, generator=generator)
            noise = torch.randn(z.shape, device=device, generator=generator)
            runtime["ldm"] = self
            if control.smoke and runtime["vae_digest"] is None:
                runtime["vae_digest"] = module_digest(self.first_stage_model)
            return combined_loss(
                self, z, context, timesteps, noise, batch["reports"], teacher_holder,
                control.aux_weight, control.aux_timestep_max)

        # Run one real auxiliary-only backward pass before Opacus wraps the
        # trainable module. This proves that the frozen teacher path reaches
        # both projection and LoRA without contaminating per-sample hooks.
        from opacus import PrivacyEngine
        cached_make_private = PrivacyEngine.make_private
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
                context = module.encode_condition(reports)
                timesteps = torch.full((len(samples),), min(100, control.aux_timestep_max),
                                       device=module.device, dtype=torch.long)
                generator = torch.Generator(device=module.device).manual_seed(500000 + runtime["seed"])
                noise = torch.randn(z.shape, device=module.device, generator=generator)
                x_noisy = module.q_sample(x_start=z, t=timesteps, noise=noise)
                predicted_noise = module.apply_model(x_noisy, timesteps, context)
                aux_per, active, _ = xrv_auxiliary_per_sample(
                    module, x_noisy, predicted_noise, timesteps, reports,
                    teacher_holder, control.aux_timestep_max)
                if not active.any() or not torch.isfinite(aux_per).all():
                    print("E024_PREFLIGHT_INVALID", {
                        "active": active.detach().cpu().tolist(),
                        "aux": aux_per.detach().cpu().tolist(),
                    }, flush=True)
                    raise AssertionError("E024 auxiliary preflight produced invalid loss")
                named = [(name, param) for name, param in module.named_parameters()
                         if param.requires_grad]
                grads = torch.autograd.grad(aux_per.mean(), [param for _, param in named],
                                            allow_unused=True)
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
                    raise AssertionError(f"projection auxiliary gradient invalid: {projection_norm}")
                if not math.isfinite(lora_norm) or lora_norm <= 0:
                    raise AssertionError(f"LoRA auxiliary gradient invalid: {lora_norm}")
                module.zero_grad(set_to_none=True)
                runtime["preflight"] = {
                    "aux_loss": float(aux_per.mean().detach().cpu()),
                    "projection_grad_norm": projection_norm,
                    "lora_grad_norm": lora_norm,
                    "active_records": int(active.sum().item()),
                }
                runtime["vae_digest"] = module_digest(module.first_stage_model)
                print("E024_AUX_GRAD_OK", runtime["preflight"], flush=True)
                preflight_done["value"] = True
            return cached_make_private(engine, **kwargs)

        ft._make_dataset = make_dataset
        ft.parse_args = parse_args
        LatentDiffusionDP.training_step_dp = training_step
        PrivacyEngine.make_private = make_private
        return original_main()

    # cached_training installs the exact E004 cache/RNG/Opacus protocol, then
    # calls ft.main; intercepted_main adds only the label and auxiliary path.
    ft.main = intercepted_main
    cached_training.train(remaining)

    if control.smoke:
        verify_smoke(runtime, teacher_holder)


def verify_smoke(runtime, teacher_holder):
    ldm = runtime["ldm"]
    if ldm is None or not teacher_holder:
        raise RuntimeError("smoke training did not execute the auxiliary path")
    vae_after = module_digest(ldm.first_stage_model)
    teacher_after = module_digest(teacher_holder["teacher"])
    if vae_after != runtime["vae_digest"]:
        raise AssertionError("frozen VAE changed during smoke training")
    if teacher_after != teacher_holder["initial_digest"]:
        raise AssertionError("frozen XRV changed during smoke training")

    save_dir = ROOT / "classification_experiments/artifacts/xrv_aux_smoke"
    checkpoint = torch.load(save_dir / "ldm_dp_final.pt", map_location="cpu", weights_only=False)
    saved = checkpoint["model"]
    current = ldm.state_dict()
    selected = [k for k in current if "embedder.proj" in k or "lora_A" in k or "lora_B" in k]
    if not selected:
        raise AssertionError("no projection/LoRA tensors found for checkpoint check")
    mismatched = [k for k in selected if k not in saved or not torch.equal(
        current[k].detach().cpu(), saved[k].detach().cpu())]
    if mismatched:
        raise AssertionError(f"checkpoint round-trip mismatch: {mismatched[:5]}")
    if any("xrv" in k.lower() for k in saved):
        raise AssertionError("frozen XRV was unexpectedly stored in the checkpoint")

    # Unit assertions for the uncertain/missing mask convention.
    fake = {csv: (-1.0 if i == 0 else (np.nan if i == 1 else float(i % 2)))
            for i, (_, csv) in enumerate(AUX_LABEL_MAP)}
    _, mask = make_targets(fake)
    if mask[0] or mask[1] or int(mask.sum()) != len(AUX_LABEL_MAP) - 2:
        raise AssertionError("uncertain/missing label mask failed")

    evidence = {
        "status": "passed",
        "vae_frozen": True,
        "xrv_frozen": True,
        "checkpoint_projection_lora_exact": True,
        "xrv_absent_from_checkpoint": True,
        "uncertain_missing_mask": True,
        "physical_batch": 4,
        "aux_weight": 0.01,
        "aux_timestep_max": 250,
        "teacher": "densenet121-res224-all",
        "opacus_training_complete": True,
        "epsilon_spent": float(checkpoint["epsilon_spent"]),
        "auxiliary_preflight": runtime.get("preflight"),
    }
    evidence_path = ROOT / "classification_experiments/evidence/024_smoke.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    print("E024_SMOKE_OK", evidence_path, flush=True)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "smoke"}:
        raise SystemExit("usage: xrv_auxiliary_training.py {smoke|train} [training args]")
    command = sys.argv[1]
    args = sys.argv[2:]
    if command == "smoke":
        args = ["--smoke"] + args
    install_and_train(args)


if __name__ == "__main__":
    main()
