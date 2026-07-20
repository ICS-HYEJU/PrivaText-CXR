"""
Eval_metric/clipscore.py  —  Text-image alignment via domain CLIP (Eval step 4)
===============================================================================
Concept
-------
A CLIP-style model has an image encoder f_img and a text encoder f_txt trained
(contrastively) into a SHARED embedding space.  Alignment of an image x with a
text t is their cosine similarity, and CLIPScore reports:

        CLIPScore(x, t) = w * max(cos(f_img(x), f_txt(t)), 0)   (w = 2.5)

Generic CLIP is trained on natural images and is unreliable on chest X-rays, so
we use DOMAIN encoders:
    - biovil-t : Microsoft BioViL-T (CXR image + report, temporal-aware)
    - medclip  : MedCLIP (image-report semantic matching)
    - cxr-clip : CXR-specific CLIP, loaded through open_clip (--clip_model/
                 --clip_pretrained) or any open_clip checkpoint.

Our LDM has NO CLIP/contrastive training loss (see docs), so these encoders are
purely for EVALUATION — no information leakage, and different from the BioBERT
text encoder used for conditioning.

Why also a real baseline
------------------------
Absolute cosine scales differ per encoder, so a raw generated CLIPScore is hard
to read.  We also measure the encoder's alignment on REAL (image, report) pairs
from the same split — an empirical ceiling — and report the GAP:
        gap = real_clipscore_mean - gen_clipscore_mean      (smaller = better)

Usage
-----
    # generated images vs their prompts (descriptions.csv), + real baseline
    python Eval_metric/clipscore.py \\
        --gen_dir  ./EVAL/gen_out/eps1/samples \\
        --backend  biovil-t --device cuda:0 \\
        --root_path /storage/.../mimic-cxr/2.1.0 --eval_split test --max_real 361 \\
        --output   ./eval/eps1/clipscore.json
"""

import os
import csv
import glob
import json
import argparse

import numpy as np


# ── encoder adapters ─────────────────────────────────────────────────────────
def _l2norm(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


class _MedCLIP:
    name = 'medclip'

    def __init__(self, device):
        import torch  # noqa: F401
        from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
        self.torch = __import__('torch')
        self.device = device
        self.model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        self.model.from_pretrained()           # downloads weights
        self.model.to(device).eval()
        self.proc = MedCLIPProcessor()

    def encode_image(self, paths):
        from PIL import Image
        embs = []
        with self.torch.no_grad():
            for p in paths:
                img = Image.open(p).convert('RGB')
                inp = self.proc(images=img, return_tensors='pt')
                v = self.model.encode_image(inp['pixel_values'].to(self.device))
                embs.append(v.cpu().numpy())
        return _l2norm(np.concatenate(embs, 0))

    def encode_text(self, texts):
        embs = []
        with self.torch.no_grad():
            for t in texts:
                inp = self.proc(text=[t], return_tensors='pt', padding=True, truncation=True)
                v = self.model.encode_text(inp['input_ids'].to(self.device),
                                           inp['attention_mask'].to(self.device))
                embs.append(v.cpu().numpy())
        return _l2norm(np.concatenate(embs, 0))


class _BioViLT:
    name = 'biovil-t'

    def __init__(self, device):
        # health_multimodal (Microsoft) provides BioViL-T image + text inference.
        from health_multimodal.text import get_bert_inference
        from health_multimodal.text.utils import BertEncoderType
        from health_multimodal.image import get_image_inference
        from health_multimodal.image.utils import ImageModelType
        self.text_inf = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
        self.img_inf = get_image_inference(ImageModelType.BIOVIL_T)

    def encode_image(self, paths):
        from pathlib import Path
        embs = [self.img_inf.get_projected_global_embedding(Path(p)) for p in paths]
        embs = np.stack([e.detach().cpu().numpy() if hasattr(e, 'detach') else np.asarray(e)
                         for e in embs], 0)
        return _l2norm(embs)

    def encode_text(self, texts):
        emb = self.text_inf.get_embeddings_from_prompt(list(texts), normalize=False)
        emb = emb.detach().cpu().numpy() if hasattr(emb, 'detach') else np.asarray(emb)
        return _l2norm(emb)


class _OpenCLIP:
    """Generic open_clip backend — for CXR-CLIP checkpoints or any open_clip model."""
    name = 'openclip'

    def __init__(self, device, model_name='ViT-B-32', pretrained='openai'):
        import torch  # noqa: F401
        import open_clip
        self.torch = __import__('torch')
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_image(self, paths):
        from PIL import Image
        with self.torch.no_grad():
            imgs = self.torch.stack([self.preprocess(Image.open(p).convert('RGB'))
                                     for p in paths]).to(self.device)
            v = self.model.encode_image(imgs).cpu().numpy()
        return _l2norm(v)

    def encode_text(self, texts):
        with self.torch.no_grad():
            tok = self.tokenizer(list(texts)).to(self.device)
            v = self.model.encode_text(tok).cpu().numpy()
        return _l2norm(v)


def load_encoder(backend, device='cpu', clip_model='ViT-B-32', clip_pretrained='openai'):
    if backend == 'medclip':
        return _MedCLIP(device)
    if backend == 'biovil-t':
        return _BioViLT(device)
    if backend in ('cxr-clip', 'openclip'):
        return _OpenCLIP(device, clip_model, clip_pretrained)
    raise ValueError(f"unknown backend '{backend}' "
                     "(choices: biovil-t, medclip, cxr-clip/openclip)")


# ── scoring ──────────────────────────────────────────────────────────────────
def _paired_cos(img_emb, txt_emb):
    """Row-wise cosine of already-L2-normed embeddings -> [N]."""
    d = min(img_emb.shape[1], txt_emb.shape[1])
    return np.sum(img_emb[:, :d] * txt_emb[:, :d], axis=1)


def clipscore(image_paths, prompts, encoder, w=2.5, verbose=True):
    if verbose:
        print(f'[clip] {encoder.name}: encoding {len(image_paths)} image/text pair(s)')
    img_emb = encoder.encode_image(image_paths)
    txt_emb = encoder.encode_text(prompts)
    cos = _paired_cos(img_emb, txt_emb)
    score = w * np.clip(cos, 0, None)
    return {'clipscore_mean': float(score.mean()), 'clipscore_std': float(score.std()),
            'cos_mean': float(cos.mean()), 'cos_std': float(cos.std()), 'n': int(len(cos))}


def load_pairs_from_csv(gen_dir):
    """
    Pair generated images with their prompts using descriptions.csv written by
    the inference script.  descriptions.csv lives in gen_dir or its parent and
    has columns index, description, file (file relative to the csv's folder).
    """
    for base in (gen_dir, os.path.dirname(os.path.abspath(gen_dir))):
        csv_path = os.path.join(base, 'descriptions.csv')
        if os.path.isfile(csv_path):
            paths, prompts = [], []
            with open(csv_path, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    fp = os.path.join(base, row['file'])
                    if os.path.isfile(fp):
                        paths.append(fp)
                        prompts.append(row['description'])
            if paths:
                return paths, prompts
    # fallback: no csv -> cannot pair prompts
    raise FileNotFoundError(
        f'descriptions.csv not found near {gen_dir}; cannot pair images to prompts.')


def real_baseline(root_path, split_csv, eval_split, image_size, max_length,
                  encoder, max_real=None, w=2.5, tmp_dir=None):
    """
    Encoder's alignment on REAL (image, report) pairs from the split — an
    empirical ceiling.  Real images are dumped to PNG (encoders read from path).
    """
    import argparse as _a
    from PIL import Image
    from Data.mimic_cxr import MIMICCXRDataset
    ds = MIMICCXRDataset(_a.Namespace(
        root_path=root_path, split_csv=split_csv, split=eval_split,
        image_size=image_size, max_length=max_length, patient_whitelist=None))
    n = len(ds) if max_real in (None, 0) else min(max_real, len(ds))
    tmp_dir = tmp_dir or os.path.join('.', '_clip_real_tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    paths, prompts = [], []
    for i in range(n):
        img, report = ds[i]                    # img [1,H,W] in [-1,1]
        arr = ((img.squeeze(0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype('uint8')
        fp = os.path.join(tmp_dir, f'real_{i:05d}.png')
        Image.fromarray(arr, mode='L').save(fp)
        paths.append(fp)
        prompts.append(report)
    return clipscore(paths, prompts, encoder, w=w)


def parse_args():
    p = argparse.ArgumentParser(description='CLIPScore with domain encoders')
    p.add_argument('--gen_dir', required=True, help='folder of generated pngs (samples/)')
    p.add_argument('--backend', default='biovil-t',
                   choices=['biovil-t', 'medclip', 'cxr-clip', 'openclip'])
    p.add_argument('--clip_model', default='ViT-B-32', help='open_clip model name (cxr-clip/openclip)')
    p.add_argument('--clip_pretrained', default='openai', help='open_clip weights or ckpt path')
    p.add_argument('--w', default=2.5, type=float, help='CLIPScore scale')
    p.add_argument('--device', default='cpu')
    # optional real baseline
    p.add_argument('--root_path', default=None, help='enable real baseline from dataset split')
    p.add_argument('--split_csv', default='mimic-cxr-2.0.0-split.csv')
    p.add_argument('--eval_split', default='test')
    p.add_argument('--image_size', default=256, type=int)
    p.add_argument('--max_length', default=512, type=int)
    p.add_argument('--max_real', default=None, type=int)
    p.add_argument('--output', default=None)
    return p.parse_args()


def run_clipscore(args):
    enc = load_encoder(args.backend, args.device, args.clip_model, args.clip_pretrained)
    paths, prompts = load_pairs_from_csv(args.gen_dir)
    res = {'backend': enc.name, 'w': args.w}
    gen = clipscore(paths, prompts, enc, w=args.w)
    res.update({f'gen_{k}': v for k, v in gen.items()})
    print(f'[clip] gen clipscore={gen["clipscore_mean"]:.4f} (cos={gen["cos_mean"]:.4f}, n={gen["n"]})')
    if args.root_path:
        real = real_baseline(args.root_path, args.split_csv, args.eval_split,
                             args.image_size, args.max_length, enc,
                             max_real=args.max_real, w=args.w)
        res.update({f'real_{k}': v for k, v in real.items()})
        res['clipscore_gap'] = float(real['clipscore_mean'] - gen['clipscore_mean'])
        print(f'[clip] real baseline={real["clipscore_mean"]:.4f}  '
              f'gap(real-gen)={res["clipscore_gap"]:.4f}')
    return res


def main():
    args = parse_args()
    res = run_clipscore(args)
    print(json.dumps(res, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(res, f, indent=2)
        print(f'[save] -> {args.output}')


if __name__ == '__main__':
    main()
