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


def _shim_clip_feature_extractor():
    """
    MedCLIP does `from transformers import CLIPFeatureExtractor`, a name removed
    from newer transformers (now CLIPImageProcessor). Alias it on the transformers
    module BEFORE importing medclip so that import resolves. Robust to lazy
    modules / missing names; warns if it cannot resolve a replacement.
    """
    import transformers
    def _try(getter):
        try:
            return getter()
        except Exception:
            return None
    if _try(lambda: transformers.CLIPFeatureExtractor) is not None:
        return
    cls = (_try(lambda: transformers.CLIPImageProcessor)
           or _try(lambda: transformers.CLIPImageProcessorFast)
           or _try(lambda: __import__(
               'transformers.models.clip.image_processing_clip',
               fromlist=['CLIPImageProcessor']).CLIPImageProcessor))
    if cls is not None:
        transformers.CLIPFeatureExtractor = cls
        print(f'[medclip] shimmed transformers.CLIPFeatureExtractor -> {cls.__name__}')
    else:
        print('[medclip] WARNING: could not alias CLIPFeatureExtractor; medclip '
              'import may fail. Try: pip install "transformers<4.36"')


class _MedCLIP:
    name = 'medclip'

    def __init__(self, device, batch_size=32):
        import torch  # noqa: F401
        _shim_clip_feature_extractor()
        from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
        self.torch = __import__('torch')
        self.device = device
        self.batch_size = batch_size
        self.model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        self.model.from_pretrained()           # downloads weights
        self.model.to(device).eval()
        self.proc = MedCLIPProcessor()

    def encode_image(self, paths):
        # CXR are grayscale; MedCLIPProcessor normalizes single-channel, so we
        # feed 'L' (no RGB conversion). Batched for speed.
        from PIL import Image
        embs = []
        with self.torch.no_grad():
            for i in range(0, len(paths), self.batch_size):
                imgs = [Image.open(p).convert('L') for p in paths[i:i + self.batch_size]]
                inp = self.proc(images=imgs, return_tensors='pt')
                v = self.model.encode_image(inp['pixel_values'].to(self.device))
                embs.append(v.cpu().numpy())
        return _l2norm(np.concatenate(embs, 0))

    def encode_text(self, texts):
        texts = list(texts)
        embs = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                inp = self.proc(text=texts[i:i + self.batch_size],
                                return_tensors='pt', padding=True, truncation=True)
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

    def __init__(self, device, model_name='ViT-B-32', pretrained='openai', batch_size=32):
        import torch  # noqa: F401
        import open_clip
        self.torch = __import__('torch')
        self.device = device
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_image(self, paths):
        from PIL import Image
        embs = []
        with self.torch.no_grad():
            for i in range(0, len(paths), self.batch_size):
                imgs = self.torch.stack([self.preprocess(Image.open(p).convert('RGB'))
                                         for p in paths[i:i + self.batch_size]]).to(self.device)
                embs.append(self.model.encode_image(imgs).cpu().numpy())
        return _l2norm(np.concatenate(embs, 0))

    def encode_text(self, texts):
        texts = list(texts)
        embs = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                tok = self.tokenizer(texts[i:i + self.batch_size]).to(self.device)
                embs.append(self.model.encode_text(tok).cpu().numpy())
        return _l2norm(np.concatenate(embs, 0))


def load_encoder(backend, device='cpu', clip_model='ViT-B-32', clip_pretrained='openai',
                 batch_size=32):
    if backend == 'medclip':
        return _MedCLIP(device, batch_size)
    if backend == 'biovil-t':
        return _BioViLT(device)
    if backend in ('cxr-clip', 'openclip'):
        return _OpenCLIP(device, clip_model, clip_pretrained, batch_size)
    raise ValueError(f"unknown backend '{backend}' "
                     "(choices: biovil-t, medclip, cxr-clip/openclip)")


# ── scoring ──────────────────────────────────────────────────────────────────
def _paired_cos(img_emb, txt_emb):
    """Row-wise cosine of already-L2-normed embeddings -> [N]."""
    d = min(img_emb.shape[1], txt_emb.shape[1])
    return np.sum(img_emb[:, :d] * txt_emb[:, :d], axis=1)


def retrieval_metrics(img_emb, txt_emb, prompts, ks=(1, 5, 10)):
    """
    Cross retrieval on the shared embeddings.
      i2t: each image ranks all texts; hit if a top-k text has the SAME prompt.
      t2i: each text ranks all images; hit if a top-k image's prompt matches.
    Correctness is by TEXT EQUALITY (not index) so duplicate prompts — e.g.
    n_samples>1 per report — don't spuriously miss. Returns R@k + median/mean rank
    for both directions, plus duplicate count.
    """
    S = img_emb @ txt_emb.T                       # [N_img, N_txt] cosine (L2-normed)
    prompts = list(prompts)
    N = len(prompts)

    def _side(sim):
        Rhit = {k: 0 for k in ks}
        ranks = []
        for i in range(sim.shape[0]):
            order = np.argsort(-sim[i])
            pos = next((p for p, j in enumerate(order) if prompts[j] == prompts[i]), N - 1)
            r = pos + 1
            ranks.append(r)
            for k in ks:
                if r <= k:
                    Rhit[k] += 1
        ranks = np.asarray(ranks)
        out = {f'R@{k}': float(Rhit[k] / len(ranks)) for k in ks}
        out['median_rank'] = float(np.median(ranks))
        out['mean_rank'] = float(ranks.mean())
        return out

    res = {}
    for k, v in _side(S).items():
        res[f'{k}_i2t'] = v
    for k, v in _side(S.T).items():
        res[f'{k}_t2i'] = v
    res['retrieval_n'] = N
    res['duplicate_prompts'] = int(N - len(set(prompts)))
    return res


def clipscore(image_paths, prompts, encoder, w=2.5, text_mode='findings',
              retrieval_ks=(1, 5, 10), verbose=True):
    """
    Primary metric: cos_mean (raw cosine). clipscore_mean = w*max(cos,0) is a
    scaled convenience value (w=2.5 is calibrated for OpenAI CLIP, not MedCLIP).
    text_mode='findings' keeps only FINDINGS/IMPRESSION (shared with generation);
    'full' uses the prompt text as-is. retrieval_ks!=() adds R@k retrieval.
    """
    prompts = list(prompts)
    if text_mode == 'findings':
        from Eval_metric.text_utils import extract_findings_impression
        text_in = [extract_findings_impression(p) for p in prompts]
    else:
        text_in = prompts
    if verbose:
        print(f'[clip] {encoder.name}: encoding {len(image_paths)} image/text pair(s) '
              f'(text_mode={text_mode})')
    img_emb = encoder.encode_image(image_paths)
    txt_emb = encoder.encode_text(text_in)
    cos = _paired_cos(img_emb, txt_emb)
    score = w * np.clip(cos, 0, None)
    out = {'clipscore_mean': float(score.mean()), 'clipscore_std': float(score.std()),
           'cos_mean': float(cos.mean()), 'cos_std': float(cos.std()), 'n': int(len(cos))}
    if retrieval_ks:
        out.update(retrieval_metrics(img_emb, txt_emb, text_in, tuple(retrieval_ks)))
    return out


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
                  encoder, max_real=None, w=2.5, tmp_dir=None,
                  text_mode='findings', retrieval_ks=(1, 5, 10)):
    """
    Encoder's alignment on REAL (image, report) pairs from the split — an
    empirical ceiling.  Real images are dumped to PNG (encoders read from path).
    text_mode is applied inside clipscore(), identical to the generated path.
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
    return clipscore(paths, prompts, encoder, w=w, text_mode=text_mode,
                     retrieval_ks=retrieval_ks)


def parse_args():
    p = argparse.ArgumentParser(description='CLIPScore with domain encoders')
    p.add_argument('--gen_dir', required=True, help='folder of generated pngs (samples/)')
    p.add_argument('--backend', default='medclip',
                   choices=['biovil-t', 'medclip', 'cxr-clip', 'openclip'])
    p.add_argument('--clip_model', default='ViT-B-32', help='open_clip model name (cxr-clip/openclip)')
    p.add_argument('--clip_pretrained', default='openai', help='open_clip weights or ckpt path')
    p.add_argument('--w', default=2.5, type=float, help='CLIPScore scale (clipscore_mean only)')
    p.add_argument('--text_mode', default='findings', choices=['findings', 'full'],
                   help="'findings': keep FINDINGS/IMPRESSION only (default); 'full': as-is")
    p.add_argument('--retrieval_ks', nargs='+', type=int, default=[1, 5, 10],
                   help='R@k cutoffs; pass nothing to disable retrieval')
    p.add_argument('--batch_size', default=32, type=int, help='encoder batch size')
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
    ks = tuple(args.retrieval_ks or ())
    enc = load_encoder(args.backend, args.device, args.clip_model, args.clip_pretrained,
                       batch_size=args.batch_size)
    paths, prompts = load_pairs_from_csv(args.gen_dir)
    res = {'backend': enc.name, 'w': args.w, 'text_mode': args.text_mode}
    gen = clipscore(paths, prompts, enc, w=args.w, text_mode=args.text_mode, retrieval_ks=ks)
    res.update({f'gen_{k}': v for k, v in gen.items()})
    print(f'[clip] gen cos_mean={gen["cos_mean"]:.4f} (primary)  '
          f'clipscore={gen["clipscore_mean"]:.4f}  n={gen["n"]}')
    if args.root_path:
        real = real_baseline(args.root_path, args.split_csv, args.eval_split,
                             args.image_size, args.max_length, enc,
                             max_real=args.max_real, w=args.w,
                             text_mode=args.text_mode, retrieval_ks=ks)
        res.update({f'real_{k}': v for k, v in real.items()})
        res['gap_cos'] = float(real['cos_mean'] - gen['cos_mean'])
        res['gap_clipscore'] = float(real['clipscore_mean'] - gen['clipscore_mean'])
        print(f'[clip] real cos_mean={real["cos_mean"]:.4f}  '
              f'gap_cos(real-gen)={res["gap_cos"]:.4f}')
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
