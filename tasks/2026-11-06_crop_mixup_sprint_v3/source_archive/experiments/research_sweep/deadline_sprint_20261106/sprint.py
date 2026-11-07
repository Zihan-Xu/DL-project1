"""Execute the finite, three-seed crop/Mixup study using the frozen training engine."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.research_sweep import runtime
from experiments.research_sweep.combinations import runner as engine
from experiments.research_sweep.deadline_sprint_20261106 import operators


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(target=HERE):
    torch.set_num_threads(4)
    engine.HERE = target
    target.mkdir(parents=True, exist_ok=True)
    engine.CATEGORY = 'deadline_sprint'
    engine.methods.augmentation_plan = operators.augmentation_plan
    engine.methods.augment = operators.augment
    engine.methods.objective = operators.objective


def make_run(name, seed, epochs=80):
    source = read(ROOT / f'experiments/research_sweep/confirmation/runs/confirm_000_seed{seed}/config.json')
    spec = copy.deepcopy(source['spec'])
    spec.update(name=name, control='confirm_000', seeds=[142,143,144])
    spec['recipe'].update(epochs=epochs, crop_padding=4, mixup_alpha=.2 if name=='crop_mixup' else 0., mixup_probability=.5)
    assert spec['recipe']['dropout'] == spec['recipe']['drop_path'] == spec['recipe']['rotation'] == 0
    run = {'run_id':f'{name}_seed{seed}', 'seed':seed, 'spec':spec,
           'input_source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), HERE/'operators.py', ROOT/'experiments/research_sweep/combinations/runner.py',ROOT/'experiments/research_sweep/combinations/methods.py',ROOT/'experiments/research_sweep/architecture/models.py',ROOT/'experiments/research_sweep/training/methods.py',ROOT/'baseline/splits.csv',ROOT/'baseline/data_audit.json']}}
    run['config_sha256'] = engine.digest(run)
    return run


def register():
    manifest = {'created_at_utc':runtime.utc_now(),'seeds':[142,143,144], 'epochs':80,
                'runs':[make_run(name, seed) for seed in [142,143,144] for name in ['crop','crop_mixup']],
                'selection':'Complete all six runs. Rank mean validation correct for raw and horizontal-flip probability average. Ties prefer raw, no Mixup, then name. Do not pick highest single seed.',
                'benchmark_control':'Reuse accepted confirm_000 seed142/143/144 (same core recipe, optimizer, schedule, batch sampler, split and seeds; crop absent).',
                'current_best_recipe_validation_mean':.9604,'current_reported_kaggle_score':.879,
                'deadline_utc':'2026-11-07T04:59:59Z','final_fit_seed':42,'final_fit_training_images':50000,
                'final_fit_epochs':80,'kaggle_auto_submit':False}
    if (HERE/'manifest.json').exists():
        old=read(HERE/'manifest.json')
        assert old['runs']==manifest['runs'];return old
    engine.write_json(HERE/'manifest.json',manifest)
    engine.write_json(HERE/'freeze.json', {'manifest_sha256':sha(HERE/'manifest.json'), 'sources':manifest['runs'][0]['input_source_hashes']})
    return manifest


@torch.inference_mode()
def inference(run, data, audit):
    folder = HERE/'runs'/run['run_id']
    cp = torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    assert cp['config']==run
    engine.configure_gpu(run['seed'])
    net=engine.methods.build(run['spec']['recipe']).cuda().to(memory_format=torch.channels_last).eval()
    net.load_state_dict(cp['model_state'],strict=True)
    norm=[torch.tensor(audit[k],device='cuda').view(1,3,1,1) for k in ['train_rgb_mean','train_rgb_std']]
    images=torch.from_numpy(data['validation'][0]).cuda()
    labels=data['validation'][1]
    raw=engine.methods.predict(net,images,norm).cpu().numpy()
    assert np.array_equal(raw,np.load(folder/'validation_logits.npy'))
    mirror=engine.methods.predict(net,images.flip(2),norm).cpu().numpy()
    probabilities=(torch.from_numpy(raw).softmax(1)+torch.from_numpy(mirror).softmax(1)).numpy()/2
    np.savez_compressed(folder/'inference.npz',raw_logits=raw,flip_logits=mirror,tta_probabilities=probabilities,labels=labels)
    results={'run_id':run['run_id'],'raw_correct':int((raw.argmax(1)==labels).sum()),'tta_correct':int((probabilities.argmax(1)==labels).sum()),'n':len(labels),'checkpoint_sha256':sha(folder/'best.pt'),'inference_sha256':sha(folder/'inference.npz'),'raw_logits_replay_identical':True}
    engine.write_json(folder/'inference.json',results)
    del net,images
    torch.cuda.empty_cache()
    return results


def run_all():
    manifest=register()
    freeze=read(HERE/'freeze.json')
    assert sha(HERE/'manifest.json')==freeze['manifest_sha256']
    for name,digest in freeze['sources'].items():assert sha(ROOT/name)==digest,name
    data,audit=engine.load_data()
    for run in manifest['runs']:
        runtime.check_control('deadline_sprint')
        with runtime.gpu_lock('deadline_sprint',run['run_id']):
            engine.train(run,data,audit)
            inference(run,data,audit)
        report(False)
    report(True)


def report(final):
    manifest=read(HERE/'manifest.json')
    rows=[]
    for run in manifest['runs']:
        folder=HERE/'runs'/run['run_id']
        if not (folder/'inference.json').exists():continue
        m=read(folder/'metrics.json'); inf=read(folder/'inference.json')
        rows.append(dict(configuration=run['spec']['name'],seed=run['seed'],epochs=m['epochs_completed'],parameters=m['parameters'],raw_correct=inf['raw_correct'],tta_correct=inf['tta_correct'],validation_n=inf['n'],compute_seconds=m['gpu_compute_seconds']))
    engine.write_csv(HERE/'results.csv',rows)
    summary=[]
    for name in ['crop','crop_mixup']:
        selected=[r for r in rows if r['configuration']==name]
        for policy in ['raw','tta']:
            if selected:
                scores=np.array([r[policy+'_correct']/r['validation_n'] for r in selected])
                summary.append(dict(configuration=name,policy=policy,n=len(scores),planned_n=3,mean_accuracy=float(scores.mean()),sd=float(scores.std(ddof=1)) if len(scores)>1 else None))
    engine.write_csv(HERE/'summary.csv',summary)
    if final:
        assert len(rows)==6
        best=sorted(summary,key=lambda r:(-r['mean_accuracy'],r['policy']!='raw',r['configuration']!='crop'))[0]
        decision={**best,'deployment_seed':42,'full_training_images':50000,'epochs':80,'exceeds_previous_best_validation_mean':best['mean_accuracy']>.9604,'official_kaggle_score':None}
        engine.write_json(HERE/'DECISION.json',decision)
    engine.status('awaiting_independent_audit' if final else 'running',active_run=None,message=f'{len(rows)}/6 training and inference results')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['benchmark','register','run','report']);args=parser.parse_args()
    setup()
    if args.action=='register':
        print(json.dumps({'registered':len(register()['runs'])}));return
    if args.action=='benchmark':
        target=HERE/'benchmark'
        setup(target)
        data,audit=engine.load_data()
        run=make_run('crop_mixup',142,2)
        with runtime.gpu_lock('deadline_sprint','two_epoch_benchmark'):
            start=time.monotonic();engine.train(run,data,audit)
        print(json.dumps({'elapsed_seconds':time.monotonic()-start,'benchmark_only':True}));return
    if args.action=='run':run_all()
    else:report(False)


if __name__=='__main__':
    import msvcrt
    with (HERE/'worker.lock').open('a+b') as stream:
        stream.seek(0,2)
        if not stream.tell():stream.write(b'0');stream.flush()
        stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
