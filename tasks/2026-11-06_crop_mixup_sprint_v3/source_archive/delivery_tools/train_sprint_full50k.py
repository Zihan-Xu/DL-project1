"""Run the accepted 80-epoch recipe on all 50k images with prescribed seed 42."""
import importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Reuse the already verified deterministic full-data engine, without editing it.
LEGACY=ROOT/'experiments/research_sweep/final_full50k'
sys.path.insert(0,str(LEGACY))
spec=importlib.util.spec_from_file_location('frozen_full_engine',LEGACY/'train.py')
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
from experiments.research_sweep.deadline_sprint_20261106 import operators as new
from experiments.research_sweep.combinations import methods
from experiments.research_sweep import runtime as project_runtime
from delivery_tools.audit_sprint import sha,read,rows,put
STUDY=ROOT/'experiments/research_sweep/deadline_sprint_20261106'
HERE=STUDY/'final_full50k'

def accepted():
    a=read(STUDY/'ACCEPTANCE.json')
    assert a['passed'] and a['completed_verified']==a['planned_runs']==6
    for p,h in a['artifact_sha256'].items():assert sha(STUDY/p)==h
    return True

def prepare():
    accepted();decision=read(STUDY/'DECISION.json');manifest=read(STUDY/'manifest.json')
    run=next(r for r in manifest['runs'] if r['spec']['name']==decision['configuration'])
    cfg=run['spec']['recipe'];assert cfg['epochs']==80
    old=read(LEGACY/'config.json')
    source=rows(ROOT/'data/competition/train_labels.csv');splits=rows(ROOT/'baseline/splits.csv')
    assert len(source)==50000 and [r['id'] for r in source]==[r['id'] for r in splits]
    cache=np.load(ROOT/'.cache/baseline/train.npy');labels=np.array([int(r['label']) for r in source],dtype=np.int64)
    assert cache.shape==(50000,32,32,3) and cache.dtype==np.uint8 and np.array_equal(np.bincount(labels),np.full(10,5000))
    for p in ['.cache/baseline/train.npy','data/competition/train_labels.csv']:
        oldhash=next(h for name,h in old['source_hashes'].items() if name.replace('\\','/')==p)
        assert sha(ROOT/p)==oldhash
    for i,r in enumerate(splits):assert __import__('hashlib').sha256(cache[i].tobytes()).hexdigest()==r['pixel_sha256'] and int(r['label'])==labels[i]
    sums=np.zeros(3);sq=np.zeros(3)
    for chunk in np.array_split(cache,100):
        x=chunk.astype(np.float64)/255;sums+=x.sum((0,1,2));sq+=(x*x).sum((0,1,2))
    mean=sums/(50000*1024);std=np.sqrt(sq/(50000*1024)-mean**2)
    config=dict(seed=42,configuration=decision['configuration'],recipe=cfg,policy=decision['policy'],epochs=80,training_images=50000,mean=mean.tolist(),std=std.tolist(),parameters=2436346,
        checkpoint_rule='last prescribed epoch; no validation or early stopping',selection_validation_mean=decision['mean_accuracy'],selection_note='Research models trained on 40k; not independent accuracy of this final 50k model',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [STUDY/'ACCEPTANCE.json',STUDY/'DECISION.json',STUDY/'manifest.json',ROOT/'data/competition/train_labels.csv',ROOT/'.cache/baseline/train.npy']},
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),LEGACY/'train.py',STUDY/'operators.py',ROOT/'experiments/research_sweep/combinations/methods.py',ROOT/'experiments/research_sweep/training/methods.py',ROOT/'experiments/research_sweep/architecture/models.py']})
    if (HERE/'config.json').exists():assert config==read(HERE/'config.json')
    else:put(HERE/'config.json',config)
    for epoch in range(1,81):
        order,counts,_=methods.sampler(labels,42,epoch,cfg['batch_size'])
        assert np.array_equal(np.sort(order),np.arange(50000)) and counts.min()>0
        f,a,c=new.augmentation_plan(42,epoch,1,n=50000)
        assert f.shape==(1,50000) and new.CROPS.shape==(50000,2)
    put(HERE/'preflight.json',dict(passed=True,images_verified=50000,epochs_checked=80,config_sha256=sha(HERE/'config.json')))
    return config,cache,labels

def main():
    HERE.mkdir(parents=True,exist_ok=True)
    engine.HERE=HERE;engine.STUDY=STUDY;engine.accepted=accepted;engine.prepare=prepare
    engine.runtime=SimpleNamespace(atomic_json=project_runtime.atomic_json,utc_now=project_runtime.utc_now,
        check_control=lambda _category:project_runtime.check_control('deadline_sprint'),
        gpu_lock=lambda _category,run_id:project_runtime.gpu_lock('deadline_sprint',run_id))
    engine.operators=SimpleNamespace(build=methods.build,sampler=methods.sampler,make_scheduler=methods.make_scheduler,predict=methods.predict,augmentation_plan=new.augmentation_plan,augment=new.augment,objective=new.objective)
    import msvcrt
    with (HERE/'worker.lock').open('a+b') as lock:
        lock.seek(0,2)
        if not lock.tell():lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:engine.main()
        except Exception as exc:engine.status('failed',error=repr(exc));raise
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
if __name__=='__main__':main()
