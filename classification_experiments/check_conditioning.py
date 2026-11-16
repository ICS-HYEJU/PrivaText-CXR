"""Meaningful invariance checks on a trained UNet; CPU only, no experiment data mutation."""
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'classification_experiments')]
from diagnostics import inference_args,dump
from LDM_dp_inference import build_unet
from Model.attention_module_dp import disable_checkpointing
from Modules.BioBERT_embedder import BioBERTEmbedder

torch.set_num_threads(4); torch.manual_seed(12)
a=inference_args(ROOT/'checkpoints/ldm/ldm_epoch0100.pt',0)
u=build_unet(a).eval(); disable_checkpointing(u)
raw=torch.load(a.dp_ckpt,map_location='cpu',weights_only=False)['model']
u.load_state_dict({k.removeprefix('model.diffusion_model.'):v for k,v in raw.items() if k.startswith('model.diffusion_model.')},strict=True); del raw
x=torch.randn(1,1,16,16); t=torch.tensor([500]); c=torch.randn(1,5,512); padded=torch.cat([c,torch.randn(1,7,512)*5],1)
with torch.no_grad():
 y=u(x,t,context=c); z=u(x,t,context=padded,context_mask=torch.tensor([[True]*5+[False]*7])); unmasked=u(x,t,context=padded)
 maskdiff=float((y-z).abs().max()); nodiff=float((y-unmasked).abs().max())
assert maskdiff<1e-4 and nodiff>1e-4,(maskdiff,nodiff)
del u
m=BioBERTEmbedder('/storage/hjchoi'); m.frozen_eval=True; m.train(); assert not m.bert.training and m.proj.training
p,mask=m(['LABEL: Cardiomegaly','LABEL: No Finding IMPRESSION: No acute cardiopulmonary abnormality.'],return_mask=True)
p.square().mean().backward(); assert m.proj.weight.grad.abs().sum()>0 and all(x.grad is None for x in m.bert.parameters())
r={'trained_unet_masked_padding_max_diff':maskdiff,'unmasked_padding_max_diff':nodiff,'bert_eval_while_projection_training':True,'projection_gradient_nonzero':True,'mask_shape':list(mask.shape)}
dump(ROOT/'classification_experiments/evidence/conditioning_checks.json',r); print(r)
