"""Independent CPU replay and acceptance of the finite crop/Mixup experiment."""
import argparse, csv, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy import stats
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.research_sweep import runtime
from experiments.research_sweep.combinations.methods import build
STUDY=ROOT/'experiments/research_sweep/deadline_sprint_20261106'
CONTROL=ROOT/'experiments/research_sweep/confirmation'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,x):runtime.atomic_json(p,x)

def inputs():
    freeze=read(STUDY/'freeze.json')
    assert sha(STUDY/'manifest.json')==freeze['manifest_sha256']
    for p,h in freeze['sources'].items():assert sha(ROOT/p)==h,p
    assert sha(ROOT/'.cache/baseline/train.npy')=='4a5241847d2fdc2d50a9ad6e386b47f26d121db6cb2e43b24e72190724314e48'
    split=rows(ROOT/'baseline/splits.csv');cache=np.load(ROOT/'.cache/baseline/train.npy',mmap_mode='r')
    val=[(i,r) for i,r in enumerate(split) if r['split']=='validation']
    assert len(val)==len({r['id'] for i,r in val})==5000
    for i,r in val:assert hashlib.sha256(cache[i].tobytes()).hexdigest()==r['pixel_sha256']
    a=read(ROOT/'baseline/data_audit.json')
    norm=[torch.tensor(a[k]).view(1,3,1,1) for k in ['train_rgb_mean','train_rgb_std']]
    return np.array(cache[[i for i,r in val]]),np.array([int(r['label']) for i,r in val]),[r['id'] for i,r in val],norm

@torch.inference_mode()
def audit_run(run,data):
    folder=STUDY/'runs'/run['run_id'];m=read(folder/'metrics.json');inf=read(folder/'inference.json')
    assert read(folder/'config.json')==run and m['config_sha256']==run['config_sha256']
    assert m['epochs_completed']==80 and m['validation_n']==5000 and m['parameters']==2436346
    for p,h in m['artifact_sha256'].items():assert sha(folder/p)==h,p
    assert inf['checkpoint_sha256']==sha(folder/'best.pt') and inf['inference_sha256']==sha(folder/'inference.npz')
    history=rows(folder/'history.csv')
    assert [int(r['epoch']) for r in history]==list(range(1,81))
    assert all(int(r['examples'])==40000 and int(r['updates'])==313 for r in history)
    best=max(int(r['validation_correct']) for r in history)
    earliest=next(int(r['epoch']) for r in history if int(r['validation_correct'])==best)
    assert m['best_epoch']==earliest and m['validation_correct']==best
    control=read(CONTROL/f"runs/confirm_000_seed{run['seed']}/config.json")
    cfg=run['spec']['recipe'];base={k:v for k,v in cfg.items() if k not in ['crop_padding','mixup_alpha','mixup_probability']}
    assert base==control['spec']['recipe']
    assert cfg['crop_padding']==4 and cfg['mixup_probability']==.5
    assert cfg['mixup_alpha']==(.2 if run['spec']['name']=='crop_mixup' else 0.)
    if (folder/'audit_cpu.json').exists():
        old=read(folder/'audit_cpu.json')
        assert old['passed'] and old['checkpoint_sha256']==inf['checkpoint_sha256'] and old['auditor_sha256']==sha(__file__)
        return old
    images,labels,ids,norm=data
    pred=rows(folder/'validation_predictions.csv')
    assert [r['id'] for r in pred]==ids
    cp=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    assert cp['config']==run and cp['epoch']==earliest and cp['validation_correct']==best
    model=build(cfg).eval().to(memory_format=torch.channels_last);model.load_state_dict(cp['model_state'],strict=True)
    raw=[];flip=[];start=time.monotonic()
    for pos in range(0,5000,128):
        x=torch.from_numpy(images[pos:pos+128]).permute(0,3,1,2).float()/255
        x=((x-norm[0])/norm[1]).contiguous(memory_format=torch.channels_last)
        raw.append(model(x).numpy());flip.append(model(x.flip(-1).contiguous(memory_format=torch.channels_last)).numpy())
    raw=np.concatenate(raw);flip=np.concatenate(flip)
    probs=((torch.from_numpy(raw).softmax(1)+torch.from_numpy(flip).softmax(1))/2).numpy()
    saved=np.load(folder/'inference.npz')
    assert np.array_equal(saved['labels'],labels)
    checks={}
    for name,actual,expected in [('raw',raw,saved['raw_logits']),('flip',flip,saved['flip_logits']),('tta',probs,saved['tta_probabilities'])]:
        assert np.isfinite(actual).all() and actual.shape==(5000,10)
        assert np.array_equal(actual.argmax(1),expected.argmax(1)),name+' CPU/GPU label disagreement'
        assert np.allclose(actual,expected,atol=2e-4,rtol=2e-4),name+' CPU/GPU numeric disagreement'
        checks[name+'_max_abs_error']=float(np.max(np.abs(actual-expected)))
    assert int((raw.argmax(1)==labels).sum())==inf['raw_correct']==best
    assert int((probs.argmax(1)==labels).sum())==inf['tta_correct']
    result=dict(passed=True,run_id=run['run_id'],checkpoint_sha256=sha(folder/'best.pt'),auditor_sha256=sha(__file__),n=5000,
        raw_correct=inf['raw_correct'],tta_correct=inf['tta_correct'],cpu_gpu_predictions_identical=True,elapsed_seconds=time.monotonic()-start,checked_at_utc=runtime.utc_now(),**checks)
    put(folder/'audit_cpu.json',result);print(json.dumps(result),flush=True);return result

def finish(manifest):
    assert read(CONTROL/'ACCEPTANCE.json')['passed'] and read(CONTROL/'REPLAY_ACCEPTANCE.json')['passed']
    values={}
    for seed in [142,143,144]:
        folder=CONTROL/f'runs/confirm_000_seed{seed}'
        m=read(folder/'metrics.json');t=read(folder/'tta_metrics.json')
        for obj in [m,t]:
            for name,h in obj['artifact_sha256'].items():assert sha(folder/name)==h
        values['control',seed,'raw']=m['validation_correct']/5000
        values['control',seed,'tta']=t['validation_correct']/5000
    for run in manifest['runs']:
        a=read(STUDY/'runs'/run['run_id']/'audit_cpu.json');assert a['passed']
        for policy in ['raw','tta']:values[run['spec']['name'],run['seed'],policy]=a[policy+'_correct']/5000
    summaries=[];comparisons=[]
    for name in ['control','crop','crop_mixup']:
        for policy in ['raw','tta']:
            scores=np.array([values[name,s,policy] for s in [142,143,144]])
            summaries.append(dict(configuration=name,policy=policy,n=3,mean_accuracy=float(scores.mean()),sd=float(scores.std(ddof=1)),seeds=scores.tolist()))
    for a,b in [('crop','control'),('crop_mixup','control'),('crop_mixup','crop')]:
        for policy in ['raw','tta']:
            diff=np.array([values[a,s,policy]-values[b,s,policy] for s in [142,143,144]])*100
            margin=float(stats.t.ppf(.975,2)*diff.std(ddof=1)/math.sqrt(3))
            p=float(stats.ttest_1samp(diff,0).pvalue) if diff.std() else (1. if diff.mean()==0 else 0.)
            comparisons.append(dict(candidate=a,reference=b,policy=policy,mean_delta_pp=float(diff.mean()),ci95_low_pp=float(diff.mean()-margin),ci95_high_pp=float(diff.mean()+margin),p_two_sided=p))
    adjusted=0.
    for rank,i in enumerate(sorted(range(6),key=lambda i:comparisons[i]['p_two_sided'])):
        adjusted=max(adjusted,min(1.,(6-rank)*comparisons[i]['p_two_sided']))
        comparisons[i]['p_holm_six_comparisons']=adjusted
    decision=read(STUDY/'DECISION.json')
    best=sorted([r for r in summaries if r['configuration']!='control'],key=lambda r:(-r['mean_accuracy'],r['policy']!='raw',r['configuration']!='crop'))[0]
    assert best['configuration']==decision['configuration'] and best['policy']==decision['policy'] and abs(best['mean_accuracy']-decision['mean_accuracy'])<1e-12
    evidence=dict(summaries=summaries,paired_comparisons=comparisons,limitations='Three seeds on the same repeatedly used validation set; intervals quantify seed variation, not independent data replication. 240-epoch incumbent is a different compute budget. No new Kaggle score.')
    put(STUDY/'COMPARISON.json',evidence)
    hashes={str(p.relative_to(STUDY)):sha(p) for p in [STUDY/'manifest.json',STUDY/'freeze.json',STUDY/'DECISION.json',STUDY/'COMPARISON.json',*[STUDY/'runs'/r['run_id']/'audit_cpu.json' for r in manifest['runs']]]}
    put(STUDY/'ACCEPTANCE.json',dict(passed=True,completed_verified=6,planned_runs=6,validated_at_utc=runtime.utc_now(),manifest_sha256=sha(STUDY/'manifest.json'),auditor_sha256=sha(__file__),artifact_sha256=hashes))
    print(json.dumps(dict(accepted=6,selection=decision)),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--final-fit',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);data=inputs();manifest=read(STUDY/'manifest.json')
    while True:
        done=0
        for run in manifest['runs']:
            if (STUDY/'runs'/run['run_id']/'inference.json').exists():audit_run(run,data);done+=1
        if done==6 and (STUDY/'DECISION.json').exists():finish(manifest);break
        if not args.watch:return
        time.sleep(10)
    if args.final_fit:
        import subprocess
        subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'delivery_tools/train_sprint_full50k.py')],cwd=ROOT,check=True)
if __name__=='__main__':main()
