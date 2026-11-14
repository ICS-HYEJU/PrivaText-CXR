"""One synthetic-data Opacus update verifies the opt-in mask/projection path."""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'classification_experiments')]
from diagnostics import inference_args,dump
from LDM_dp_inference import build_and_load_ldm
from Model.attention_module_dp import disable_checkpointing
from opacus import PrivacyEngine

torch.manual_seed(17)
a=inference_args(ROOT/'checkpoints/ldm/ldm_epoch0100.pt',0); a.lora_ckpt=None
model,_=build_and_load_ldm(a,torch.device('cuda:0')); disable_checkpointing(model)
model.context_masking=True; model.embedder.frozen_eval=True
params=model.configure_lora_params(rank=4,alpha=4,finetune_biobert=True)
loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.zeros(2,1)),batch_size=2)
model.train()
opt=torch.optim.AdamW(params,lr=1e-4)
private,opt,_=PrivacyEngine().make_private(module=model,optimizer=opt,data_loader=loader,noise_multiplier=1.,max_grad_norm=.001,poisson_sampling=False)
private.train(); before=model.embedder.proj.weight.detach().clone()
loss,_=private._module.training_step_dp({'image':torch.rand(2,1,256,256,device='cuda:0')*2-1,
 'reports':['LABEL: Cardiomegaly','LABEL: No Finding IMPRESSION: No acute cardiopulmonary disease.']})
opt.zero_grad(); loss.backward(); gs=model.embedder.proj.weight.grad_sample
assert tuple(gs.shape)==(2,512,768) and torch.isfinite(gs).all() and gs.abs().sum()>0
norms=gs.flatten(1).norm(dim=1).detach().cpu().tolist(); opt.step()
assert not torch.equal(before,model.embedder.proj.weight) and not model.embedder.bert.training
r={'synthetic_input_only':True,'opacus_projection_grad_sample_shape':list(gs.shape),'projection_grad_sample_norm':norms,'projection_updated':True,'bert_training':False,'loss':float(loss)}
dump(ROOT/'classification_experiments/evidence/opacus_checks.json',r); print(r)
