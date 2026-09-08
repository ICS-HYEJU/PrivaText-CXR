"""Validation screening with a full-range DDIM schedule; not truncated DDPM."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'classification_experiments')]
from diagnostics import OUT, dump, inference_args, save_image, score_paths, metrics, report

@torch.no_grad()
def ddim(model,context,seed,steps=50):
    assert 2<=steps<=model.num_timesteps
    schedule=np.linspace(0,model.num_timesteps-1,steps,dtype=int)[::-1].copy()
    gen=torch.Generator(device=model.device).manual_seed(seed)
    z=torch.randn((1,model.channels,model.image_size,model.image_size),device=model.device,generator=gen)
    for j,timestep in enumerate(schedule):
        t=torch.tensor([int(timestep)],device=model.device)
        eps=model.apply_model(z,t,context)
        alpha=model.alphas_cumprod[int(timestep)]
        previous=model.alphas_cumprod[int(schedule[j+1])] if j+1<len(schedule) else z.new_tensor(1.)
        x0=(z-torch.sqrt(1-alpha)*eps)/torch.sqrt(alpha)
        z=torch.sqrt(previous)*x0+torch.sqrt(1-previous)*eps # eta=0, DDIM eq.12
    return model.decode_first_stage(z)

def generate(arm,device,steps):
    from LDM_dp_inference import build_and_load_ldm
    from Data.mimic_cxr import MIMICCXRDataset,_labels_to_text
    from Eval_metric.downstream_cls import load_chexpert_gt
    directory=OUT/'ablation'/arm; ckpt=directory/'ldm_dp_final.pt'
    a=inference_args(ckpt,device); model,_=build_and_load_ldm(a,torch.device(f'cuda:{device}'))
    ds=MIMICCXRDataset(argparse.Namespace(root_path='/storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0',
       manifest_csv=str(ROOT/'data_manifests/mimic_p10_p12/ldm_dp_manifest.csv.gz'),split='validate',image_size=256,max_length=512,patient_whitelist=None))
    gt=load_chexpert_gt('/storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-2.0.0-chexpert.csv')
    cols=list(next(iter(gt.values())))
    rows=[]
    for i,meta in enumerate(ds.samples):
        subject=int(str(meta['patient_id']).lstrip('p')); study=int(str(meta['study_id']).lstrip('s'))
        label_text=_labels_to_text(gt.get((subject,study),{}),cols)
        if label_text:
            raw=ds._read_report_raw(meta['report_path']); impression=ds._extract_section(raw,('impression',))
            prompt=(label_text+(' '+impression if impression else ''))[:512]
        else: prompt=ds._build_text(meta)
        fp=directory/'validation'/f'{i:05d}.png'
        if not fp.exists():
            with torch.no_grad():
                context=model.encode_condition([prompt]); img=ddim(model,context,20000+i,steps)
            save_image(img[0],fp)
        rows.append({'index':i,'subject_id':subject,'study_id':study,'prompt':prompt,'file':str(fp)})
        if i%50==0: print(f'{arm} DDIM{steps} {i}/{len(ds)}',flush=True)
    pd.DataFrame(rows).to_csv(directory/'validation_mapping.csv',index=False)
    dump(directory/'validation_generation.json',{'n':len(rows),'sampler':'DDIM','eta':0,'steps':steps,'t_range':[0,999],
      'seed_base':20000,'checkpoint':str(ckpt),'context_masking':model.context_masking,'cohort':'all 802 validation images, no label filtering'})

def evaluate(arm):
    directory=OUT/'ablation'/arm; rows=pd.read_csv(directory/'validation_mapping.csv')
    assert len(rows)==802 and rows['index'].nunique()==802
    gt=pd.read_csv('/storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-2.0.0-chexpert.csv')
    labels=rows[['subject_id','study_id']].merge(gt,on=['subject_id','study_id'],how='left',validate='many_to_one')
    pred=score_paths(rows.file.tolist()); result={}
    for encoder,scores in pred.items():
        result[encoder]=metrics(scores,labels)
        f=rows[['index','subject_id','study_id']].copy()
        for p,v in scores.items(): f[p]=v
        f.to_csv(directory/f'validation_{encoder}_scores.csv',index=False)
    dump(directory/'validation_results.json',result)
    dump(ROOT/f'classification_experiments/evidence/{arm}_validation.json',result)
    print(arm,{e:v['macro'] for e,v in result.items()},flush=True)

def compare(a,b,number):
    import json
    from sklearn.metrics import roc_auc_score
    res={arm:json.loads((OUT/'ablation'/arm/'validation_results.json').read_text()) for arm in [a,b]}
    rows=pd.read_csv(OUT/'ablation'/a/'validation_mapping.csv')
    other=pd.read_csv(OUT/'ablation'/b/'validation_mapping.csv')
    assert rows[['index','subject_id','study_id','prompt']].equals(other[['index','subject_id','study_id','prompt']])
    gt=pd.read_csv('/storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-2.0.0-chexpert.csv')
    lab=rows[['subject_id','study_id']].merge(gt,on=['subject_id','study_id'],how='left',validate='many_to_one')
    subjects=lab.subject_id.unique(); groups={s:np.flatnonzero(lab.subject_id.to_numpy()==s) for s in subjects}; ci={}
    for enc in ['xrv','medclip']:
        score={arm:pd.read_csv(OUT/'ablation'/arm/f'validation_{enc}_scores.csv') for arm in [a,b]}
        included=[p for p,v in res[a][enc]['per_label'].items() if v['reliable']]
        rng=np.random.default_rng(42); deltas=[]
        for _ in range(500):
            ix=np.concatenate([groups[s] for s in rng.choice(subjects,len(subjects),replace=True)]); ds=[]
            for p in included:
                y=lab[p].to_numpy()[ix]; valid=np.isin(y,[0,1]); yy=y[valid]
                if len(np.unique(yy))<2: break
                ds.append(roc_auc_score(yy,score[b][p].to_numpy()[ix][valid])-roc_auc_score(yy,score[a][p].to_numpy()[ix][valid]))
            else: deltas.append(float(np.mean(ds)))
        ci[enc]={'delta':res[b][enc]['macro']-res[a][enc]['macro'],'ci95':np.quantile(deltas,[.025,.975]).tolist()}
    dump(ROOT/f'classification_experiments/evidence/{number}_comparison.json',{'a':a,'b':b,'results':res,'paired_patient_ci':ci})
    p=ROOT/f'classification_experiments/{number}.md'; text=p.read_text(); start=text.index('## 4. 실험 결과'); text=text[:start]
    text+='## 4. 실험 결과\n| arm | XRV AUROC | MedCLIP AUROC |\n|---|---:|---:|\n'
    for arm in [a,b]: text+=f"| {arm} | {res[arm]['xrv']['macro']:.4f} | {res[arm]['medclip']['macro']:.4f} |\n"
    for enc,v in ci.items(): text+=f"\n{enc}: {b}−{a} Δ={v['delta']:+.4f}, patient bootstrap 95% CI={v['ci95']}.\n"
    text+='\n## 5. 실험 결과 분석\n'
    text+=('XRV에서 개선 방향이 관측됐다. ' if ci['xrv']['delta']>0 else 'XRV 개선은 관측되지 않았다. ')
    text+='신뢰구간과 encoder별 방향을 함께 판단한다. 이 결과는 train 전체를 사용한 3-epoch/단일 training seed/50-step DDIM screening이다. 30-epoch/full DDPM 결과나 여러 seed에서의 확정적 개선으로 해석하지 않는다. 최종 후보는 full DDPM 및 독립 real-test utility 검증이 필요하다.\n'
    p.write_text(text)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('task',choices=['generate','evaluate','compare']); p.add_argument('--arm',default='baseline'); p.add_argument('--other',default='corrected'); p.add_argument('--number',default='003_conditioning_path'); p.add_argument('--device',type=int,default=0); p.add_argument('--steps',type=int,default=50); a=p.parse_args()
    if a.task=='generate': generate(a.arm,a.device,a.steps)
    elif a.task=='evaluate': evaluate(a.arm)
    else: compare(a.arm,a.other,a.number)
