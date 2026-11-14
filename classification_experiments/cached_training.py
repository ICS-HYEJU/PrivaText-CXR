"""Exact float32 frozen-VAE posterior cache. Text encoding and latent sampling remain online."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'classification_experiments')]
from diagnostics import inference_args, dump

def prepare(device):
    from Data.mimic_cxr import MIMICCXRDataset
    from LDM_dp_inference import build_vae, _load_vae_ckpt
    ds=MIMICCXRDataset(argparse.Namespace(root_path='/storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0',
         manifest_csv=str(ROOT/'data_manifests/mimic_p10_p12/ldm_dp_manifest.csv.gz'),split='train',text_mode='LABEL+IMPRESSION',image_size=256,max_length=512,patient_whitelist=None))
    directory=ROOT/'classification_experiments/artifacts/train_cache'; directory.mkdir(parents=True,exist_ok=True)
    a=inference_args(ROOT/'finetune_dp/JPG_LABEL_IMPRESSION/eps10_ep30/ldm_dp_final.pt',device)
    vae=build_vae(a).to(f'cuda:{device}').eval(); _load_vae_ckpt(vae,a.vae_ckpt,f'cuda:{device}')
    cache=np.lib.format.open_memmap(directory/'posterior.npy',mode='w+',dtype='float32',shape=(len(ds),2,16,16))
    with torch.no_grad():
        for i in range(0,len(ds),16):
            x=torch.stack([ds[j][0] for j in range(i,min(i+16,len(ds)))]).to(f'cuda:{device}')
            q=vae.encode(x); cache[i:i+len(x)]=torch.cat([q.mean,q.logvar],dim=1).cpu().numpy()
            if i%1024==0: print(f'cache {i}/{len(ds)}',flush=True)
    cache.flush()
    # Save exact dataset ordering; cache must never be silently reused after manifest changes.
    keys=[(str(s['patient_id']),str(s['study_id']),str(s.get('image_path',s.get('dcm_path')))) for s in ds.samples]
    dump(directory/'metadata.json',{'n':len(ds),'keys':keys,'vae_ckpt':a.vae_ckpt,'dtype':'float32','posterior_sample_cached':False})
    print('CACHE_DONE',len(ds),flush=True)

class CachedDataset:
    def __init__(self,base,folder):
        self.base=base; self.cache=np.load(folder/'posterior.npy',mmap_mode='r')
        meta=json.loads((folder/'metadata.json').read_text())
        keys=[(str(s['patient_id']),str(s['study_id']),str(s.get('image_path',s.get('dcm_path')))) for s in base.samples]
        assert keys==[tuple(x) for x in meta['keys']] and len(base)==len(self.cache)
    def __len__(self): return len(self.base)
    def __getitem__(self,i): return torch.from_numpy(self.cache[i].copy()),self.base._build_text(self.base.samples[i])

def train(argv):
    import LDM_dp_finetune as ft
    from Model.VAE.Autoencoder import VAE
    from Model.VAE.Encoder import DiagonalGaussianDistribution
    folder=ROOT/'classification_experiments/artifacts/train_cache'
    original_dataset=ft._make_dataset
    def dataset(*args,**kwargs):
        base=original_dataset(*args,**kwargs)
        return CachedDataset(base,folder) if kwargs.get('split')=='train' else base
    ft._make_dataset=dataset
    original_encode=VAE.encode
    def encode(self,x):
        if tuple(x.shape[1:])==(2,16,16): return DiagonalGaussianDistribution(x)
        return original_encode(self,x)
    VAE.encode=encode
    run_config={}
    original_parse=ft.parse_args
    def parse():
        a=original_parse(); a.latent_cache=str(folder); a.experiment_protocol='full-cohort-3epochs-v1'; a.independent_noise_rng=True
        run_config["seed"]=a.seed
        return a
    ft.parse_args=parse
    # Separate DP noise and latent/timestep randomness from BERT dropout consumption.
    from opacus import PrivacyEngine
    original_private=PrivacyEngine.make_private
    def make_private(engine, **kwargs):
        dev=next(p.device for p in kwargs['module'].parameters() if p.requires_grad)
        kwargs['noise_generator']=torch.Generator(device=dev).manual_seed(300000+run_config["seed"])
        return original_private(engine,**kwargs)
    PrivacyEngine.make_private=make_private
    from Model.Diffusion.LDM_dp import LatentDiffusionDP
    generators={}
    def step(self,batch):
        dev=self.device
        if str(dev) not in generators: generators[str(dev)]=torch.Generator(device=dev).manual_seed(200000+run_config["seed"])
        gen=generators[str(dev)]
        with torch.no_grad():
            q=self.first_stage_model.encode(batch['image'].to(dev))
            z=self.scale_factor*(q.mean+q.std*torch.randn(q.mean.shape,device=dev,generator=gen))
        context=self.encode_condition(batch['reports'])
        t=torch.randint(0,self.num_timesteps,(len(z),),device=dev,generator=gen)
        noise=torch.randn(z.shape,device=dev,generator=gen)
        return self.p_losses(z,context,t,noise=noise)
    LatentDiffusionDP.training_step_dp=step
    sys.argv=['LDM_dp_finetune.py']+argv
    ft.main()

if __name__=='__main__':
    if sys.argv[1]=='prepare':
        p=argparse.ArgumentParser(); p.add_argument('task'); p.add_argument('--device',type=int,default=0); a=p.parse_args(); prepare(a.device)
    elif sys.argv[1]=='train': train(sys.argv[2:])
    else: raise SystemExit('prepare or train required')
