"""Reproduce one registered40k/5k research run using the original training engine."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
TASK=Path(__file__).resolve().parents[1];REPO=TASK.parents[1]
sys.path.insert(0,str(REPO))
import operators,local_runtime
from experiments.research_sweep.combinations import runner as engine

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    runs=json.loads((TASK/'evidence/study_manifest.json').read_text())['runs']
    run=next(r for r in runs if r['run_id']==args.run_id)
    for name,digest in run['input_source_hashes'].items():
        normalized=name.replace('\\','/')
        path=(TASK/'source_archive'/normalized) if 'deadline_sprint_20261106' in normalized else (REPO/normalized)
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    out=args.output_dir.resolve();assert not out.is_relative_to(TASK);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);data,audit=engine.load_data()
    assert len(data['train'][0])==40000 and len(data['validation'][0])==5000
    cfg=run['spec']['recipe'];model=operators.build(cfg)
    assert sum(p.numel() for p in model.parameters())==2436346
    for epoch in range(1,81):
        order,counts,_=operators.sampler(data['train'][1],run['seed'],epoch,128)
        assert np.array_equal(np.sort(order),np.arange(40000)) and counts.min()>0
        operators.augmentation_plan(run['seed'],epoch,1)
    check=dict(passed=True,run_id=run['run_id'],training_images=40000,validation_images=5000,epochs_checked=80,parameters=2436346)
    local_runtime.atomic_json(out/'preflight.json',check)
    if args.check_only:print(json.dumps(check));return
    local_runtime.OUTPUT=out;engine.HERE=out;engine.CATEGORY='sprint_research';engine.runtime=local_runtime
    engine.write_json(out/'manifest.json',dict(runs=[run]))
    engine.methods.augmentation_plan=operators.augmentation_plan;engine.methods.augment=operators.augment;engine.methods.objective=operators.objective
    with local_runtime.gpu_lock('sprint_research',run['run_id']):engine.train(run,data,audit)
    cp=torch.load(out/'runs'/run['run_id']/'best.pt',map_location='cpu',weights_only=True)
    torch.save(dict(model_state=cp['model_state'],recipe=cfg,seed=run['seed'],epoch=cp['epoch'],parameters=2436346,mean=audit['train_rgb_mean'],std=audit['train_rgb_std'],training_images=40000,policy='raw'),out/'model.pt')
    print(json.dumps(dict(completed=True,run_id=run['run_id'],checkpoint=str(out/'model.pt'))))
if __name__=='__main__':main()
