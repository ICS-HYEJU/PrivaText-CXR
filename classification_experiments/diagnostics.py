"""Sequential classification diagnostics. Run generate in conda, score in medclip_env."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
OUT = ROOT/'classification_experiments/artifacts'
LABELS = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','Pleural Effusion','Pleural Other','Pneumonia','Pneumothorax']
PROBES = ['Cardiomegaly','Pleural Effusion','Edema','Atelectasis']

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def report(number, title, rationale, method, result, analysis):
    (ROOT/f'classification_experiments/{number}.md').write_text(
        f'# {number}\n\n## 1. 실험 주제\n{title}\n\n## 2. 실험 선정 근거\n{rationale}\n\n## 3. 실험 방식\n{method}\n\n## 4. 실험 결과\n{result}\n\n## 5. 실험 결과 분석\n{analysis}\n')

def inference_args(ckpt, device):
    import LDM_dp_inference as inf
    old=sys.argv
    sys.argv=['inference','--dp_ckpt',str(ckpt),'--lora_ckpt',str(ckpt),
              '--vae_ckpt',str(ROOT/'checkpoints/vae/vae_ep0070.pt'),
              '--biobert_path','/storage/hjchoi','--descriptions','diagnostic','--device_id',str(device)]
    a=inf.parse_args(); sys.argv=old
    return a

def save_image(t,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(((t.detach().cpu().squeeze().clamp(-1,1).numpy()+1)*127.5).astype('uint8')).save(path)

def reconstruct(device):
    from LDM_dp_inference import build_vae, _load_vae_ckpt
    from Data.mimic_cxr import MIMICCXRDataset
    directory=OUT/'001_vae'; directory.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'data_manifests/mimic_p10_p12/ldm_dp_manifest.csv.gz'
    ds=MIMICCXRDataset(argparse.Namespace(root_path='/storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0',
        manifest_csv=str(manifest),split='validate',image_size=256,max_length=512,patient_whitelist=None))
    rows=pd.read_csv(manifest); rows=rows[rows.dataset_split=='validate'].reset_index(drop=True)
    assert len(rows)==len(ds)
    rows.to_csv(directory/'manifest.csv',index=False)
    a=inference_args(ROOT/'finetune_dp/JPG_LABEL_IMPRESSION/eps10_ep30/ldm_dp_final.pt',device)
    vae=build_vae(a).to(f'cuda:{device}').eval(); _load_vae_ckpt(vae,a.vae_ckpt,f'cuda:{device}')
    torch.manual_seed(20260908); torch.cuda.manual_seed_all(20260908)
    mse={'mode':[],'sample':[]}; latent_shape=None
    with torch.no_grad():
        for i in range(0,len(ds),8):
            x=torch.stack([ds[j][0] for j in range(i,min(i+8,len(ds)))]).to(f'cuda:{device}')
            posterior=vae.encode(x); latent_shape=list(posterior.mode().shape[1:])
            for mode in ['mode','sample']:
                y=vae.decode(getattr(posterior,mode)()); mse[mode].extend(((y-x)**2).mean((1,2,3)).cpu().tolist())
                for k,img in enumerate(y): save_image(img,directory/mode/f'{i+k:05d}.png')
            for k,img in enumerate(x): save_image(img,directory/'real'/f'{i+k:05d}.png')
            print(f'reconstructed {min(i+8,len(ds))}/{len(ds)}',flush=True)
    dump(directory/'generation.json',{'n':len(ds),'latent_shape':latent_shape,'seed':20260908,'mse':{k:float(np.mean(v)) for k,v in mse.items()},'vae_ckpt':a.vae_ckpt})

def response(device,checkpoint,seeds):
    from LDM_dp_inference import build_and_load_ldm
    ckpt=ROOT/f'finetune_dp/JPG_LABEL_IMPRESSION/{checkpoint}/ldm_dp_final.pt'
    directory=OUT/'002_response'/checkpoint; directory.mkdir(parents=True,exist_ok=True)
    a=inference_args(ckpt,device); model,embedder=build_and_load_ldm(a,torch.device(f'cuda:{device}'))
    sentences={'Cardiomegaly':('Cardiac silhouette is enlarged.','Cardiac silhouette is not enlarged.'),
       'Pleural Effusion':('A left pleural effusion is present.','No pleural effusion is present.'),
       'Edema':('Pulmonary edema is present.','No pulmonary edema is present.'),
       'Atelectasis':('Left basilar atelectasis is present.','No atelectasis is present.')}
    rows=[]
    for pathology in PROBES:
        for seed in range(seeds):
            for target in [0,1]:
                # Match training's LABEL+IMPRESSION format; negative arm explicitly requests No Finding.
                prompt='LABEL: '+(pathology if target else 'No Finding')+' IMPRESSION: '+sentences[pathology][0 if target else 1]
                fp=directory/'samples'/f'{pathology.replace(" ","_")}_{seed:03d}_{target}.png'
                if not fp.exists():
                    with torch.no_grad():
                        c=embedder([prompt]); torch.manual_seed(10000+seed); torch.cuda.manual_seed_all(10000+seed)
                        img=model.sample(c,batch_size=1,verbose=False) # full trained 1000-step schedule
                    save_image(img[0],fp)
                rows.append({'pathology':pathology,'seed':seed,'target':target,'prompt':prompt,'file':str(fp)})
                pd.DataFrame(rows).to_csv(directory/'mapping.csv',index=False)
            print(f'{checkpoint} {pathology} seed {seed+1}/{seeds}',flush=True)
    dump(directory/'generation.json',{'n':len(rows),'seeds':seeds,'timesteps':model.num_timesteps,'checkpoint':str(ckpt),'noise_paired':True})

def score_paths(paths,device=0):
    from Eval_metric.downstream_cls import load_xrv_classifier, classify, XRV_TO_CHEXPERT, _valid_pathology
    from Eval_metric.features import load_image_as_tensor
    from Eval_metric.clipscore import load_encoder
    model=load_xrv_classifier('densenet121-res224-all',f'cuda:{device}')
    cols=[(i,XRV_TO_CHEXPERT[p]) for i,p in enumerate(model.pathologies) if p in XRV_TO_CHEXPERT and _valid_pathology(model,i)]
    scores=[]
    for i in range(0,len(paths),32):
        scores.append(classify(model,[load_image_as_tensor(p,224) for p in paths[i:i+32]],f'cuda:{device}',32))
    raw=np.concatenate(scores); xrv={p:raw[:,i] for i,p in cols}
    del model; torch.cuda.empty_cache()
    enc=load_encoder('medclip','cuda:0',batch_size=16)
    embs=enc.encode_image(paths); text=enc.encode_text(LABELS)
    clip={p:embs@text[j] for j,p in enumerate(LABELS)}
    return {'xrv':xrv,'medclip':clip}

def metrics(scores,labels):
    from sklearn.metrics import roc_auc_score, average_precision_score
    result={}
    for p,s in scores.items():
        y=labels[p].to_numpy(); valid=np.isin(y,[0,1]); y=y[valid].astype(int); s=s[valid]
        pos=int(y.sum()); neg=int(len(y)-pos)
        if min(pos,neg)<1: continue
        result[p]={'auroc':float(roc_auc_score(y,s)),'auprc':float(average_precision_score(y,s)),
                   'positive':pos,'negative':neg,'reliable':min(pos,neg)>=10}
    vals=[r['auroc'] for r in result.values() if r['reliable']]
    return {'per_label':result,'macro':float(np.mean(vals)) if vals else None}

def eval_recon():
    directory=OUT/'001_vae'; rows=pd.read_csv(directory/'manifest.csv')
    gt=pd.read_csv('/storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-2.0.0-chexpert.csv')
    # Select ID columns before merge: manifests may already contain label metadata.
    labels=rows[['subject_id','study_id']].merge(gt,on=['subject_id','study_id'],how='left',validate='many_to_one')
    result={'n':len(rows),'split':'validate','metrics':{}}
    for mode in ['real','mode','sample']:
        paths=[str(directory/mode/f'{i:05d}.png') for i in range(len(rows))]
        pred=score_paths(paths); result['metrics'][mode]={}
        for encoder,score in pred.items():
            result['metrics'][mode][encoder]=metrics(score,labels)
            frame=labels[['subject_id','study_id']].copy()
            for p,v in score.items(): frame[p]=v
            frame.to_csv(directory/f'{mode}_{encoder}_scores.csv',index=False)
        dump(directory/'results.json',result)
        print(mode,{k:v['macro'] for k,v in result['metrics'][mode].items()},flush=True)
    # Paired patient-cluster bootstrap, same resampled patients for original/reconstruction.
    from sklearn.metrics import roc_auc_score
    rng=np.random.default_rng(20260908); subjects=labels.subject_id.unique()
    groups={s:np.flatnonzero(labels.subject_id.to_numpy()==s) for s in subjects}
    result['paired_ci']={}
    for encoder in ['xrv','medclip']:
        scored=[p for p,v in result['metrics']['real'][encoder]['per_label'].items() if v['reliable']]
        arr={m:pd.read_csv(directory/f'{m}_{encoder}_scores.csv') for m in ['real','mode','sample']}
        deltas={m:[] for m in ['mode','sample']}
        for _ in range(500):
            ix=np.concatenate([groups[s] for s in rng.choice(subjects,len(subjects),replace=True)])
            aucs={m:[] for m in arr}
            for p in scored:
                y=labels[p].to_numpy()[ix]; valid=np.isin(y,[0,1]); yy=y[valid]
                if len(np.unique(yy))<2: break
                for m in arr: aucs[m].append(roc_auc_score(yy,arr[m][p].to_numpy()[ix][valid]))
            else:
                for m in deltas: deltas[m].append(float(np.mean(aucs[m])-np.mean(aucs['real'])))
        result['paired_ci'][encoder]={m:{'delta_ci95':np.quantile(v,[.025,.975]).tolist(),'valid_bootstraps':len(v)} for m,v in deltas.items() if v}
    dump(directory/'results.json',result)
    lines=['| 입력 | XRV AUROC | MedCLIP AUROC |','|---|---:|---:|']
    for mode in result['metrics']:
        r=result['metrics'][mode]; lines.append(f"| {mode} | {r['xrv']['macro']:.4f} | {r['medclip']['macro']:.4f} |")
    x=result['metrics']; delta=x['mode']['xrv']['macro']-x['real']['xrv']['macro']
    report('001_vae_reconstruction','VAE 재구성이 병변 분류 신호를 보존하는가?',
      '현재 latent 1×16×16의 병변 정보 손실과 diffusion 조건 학습 실패를 분리한다.',
      f'validation 전체 {len(rows)}장. 원본/결정론적 posterior.mode/seed=20260908 posterior.sample 비교. 동일 256 PNG→224 XRV 전처리. XRV-all, MedCLIP bare-label cosine의 multilabel AUROC/AUPRC. uncertain/missing 제외, 양성·음성 각각 10개 이상인 동일 label 집합의 macro. 기존 single-abnormality CLIP metric과 구별. 환자 단위 paired bootstrap 500회. 실행: conda diagnostics.py reconstruct; medclip_env diagnostics.py eval-recon. 모든 실행은 CXR_dp_medclip 내부.',
      '\n'.join(lines)+'\n\n상세 수치·support·paired CI: `artifacts/001_vae/results.json`. 원본/재구성 및 raw score CSV 보존.',
      f'XRV mode−real ΔAUROC={delta:+.4f}. 이 차이는 재구성 전처리와 VAE를 통과할 때의 병변 분리력 변화를 나타내며 생성 모델의 성능 상한을 엄밀히 증명하지는 않는다. '+
      ('재구성 단계의 손실이 커서 VAE도 개선 대상으로 유지한다. ' if delta < -.05 else '관측된 재구성 손실만으로 기존 합성 AUROC 약 0.50을 설명할 수 있는지 조건 반응 실험과 함께 판단한다. ')+
      'validation 결과이며 최종 p12 test와 혼합하지 않는다. 다음 별도 실험은 동일 난수의 병변 조건 반응이다.')

def eval_response():
    from sklearn.metrics import roc_auc_score
    directory=OUT/'002_response'; result={}
    for checkpoint in ['eps1_ep30','eps10_ep30']:
        folder=directory/checkpoint; rows=pd.read_csv(folder/'mapping.csv')
        done=json.loads((folder/'generation.json').read_text()); assert len(rows)==done['n']==8*done['seeds']
        assert not rows.duplicated(['pathology','seed','target']).any()
        pred=score_paths(rows.file.tolist()); result[checkpoint]={}
        for encoder,scores in pred.items():
            frame=rows[['pathology','seed','target']].copy()
            for p,v in scores.items(): frame[p]=v
            frame.to_csv(folder/f'{encoder}_scores.csv',index=False)
            result[checkpoint][encoder]={}
            for p in PROBES:
                subset=frame[frame.pathology==p]; pair=subset.pivot(index='seed',columns='target',values=p)
                delta=(pair[1]-pair[0]).to_numpy(); rng=np.random.default_rng(20260908)
                bs=rng.choice(delta,(2000,len(delta)),replace=True).mean(1)
                result[checkpoint][encoder][p]={'auroc':float(roc_auc_score(subset.target,subset[p])),
                   'paired_mean_score_delta':float(delta.mean()),'delta_ci95':np.quantile(bs,[.025,.975]).tolist(),
                   'positive_direction_fraction':float((delta>0).mean()),'pairs':len(delta)}
        dump(directory/'results.json',result)
    lines=['| checkpoint | 병변 | XRV AUROC | XRV paired Δscore | MedCLIP AUROC |','|---|---|---:|---:|---:|']
    for c,r in result.items():
        for p in PROBES:
            x=r['xrv'][p]; m=r['medclip'][p]
            lines.append(f"| {c} | {p} | {x['auroc']:.4f} | {x['paired_mean_score_delta']:+.6f} | {m['auroc']:.4f} |")
    report('002_prompt_response','동일 DDPM 난수에서 병변 조건의 존재·부재 반응',
      '생성기가 조건을 무시하는지, morphology 변화가 목표 병변 score 변화로 연결되는지 확인한다.',
      '두 JPG ep30 checkpoint, 4병변×존재/부재×32 seeds=256장씩. 존재는 LABEL: 병변+positive impression, 부재는 LABEL: No Finding+negative impression. 각 pair의 전체 DDPM 난수 seed를 동일하게 재설정, full 1000 steps. checkpoint/LoRA 동시 로딩. XRV-all 및 bare-label MedCLIP. paired score Δ/seed bootstrap 2000회와 intent-label AUROC. 이 AUROC는 실제 환자 라벨 평가와 별개이며 부재 arm의 No Finding은 다른 소견도 줄일 수 있다.',
      '\n'.join(lines)+'\n\nraw predictions와 CI: artifacts/002_response/results.json 및 checkpoint별 scores.csv.',
      'paired Δ가 0 부근이거나 부호가 불안정하면 조건 반영이 약한 증거다. 양의 Δ라도 작은 값/낮은 AUROC이면 임상적으로 충분한 분리라고 결론내리지 않는다. 프롬프트 intent는 영상에 실제 병변이 있다는 검증이 아니며 teacher score 자체도 ground truth가 아니다. 최종 utility와 다중 병변 데이터 평가를 따로 유지한다.')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('task',choices=['reconstruct','eval-recon','response','eval-response'])
    p.add_argument('--device',type=int,default=0); p.add_argument('--checkpoint',default='eps10_ep30'); p.add_argument('--seeds',type=int,default=32)
    a=p.parse_args()
    if a.task=='reconstruct': reconstruct(a.device)
    elif a.task=='eval-recon': eval_recon()
    elif a.task=='eval-response': eval_response()
    else: response(a.device,a.checkpoint,a.seeds)
