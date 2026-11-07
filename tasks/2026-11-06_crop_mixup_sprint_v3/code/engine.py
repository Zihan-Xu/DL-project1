import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import csv, hashlib, json, random, sys, time, platform
from pathlib import Path
import numpy as np
import torch
from PIL import Image
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import local_runtime as runtime
import operators
STUDY = ROOT / 'experiments/research_sweep/confirmation'
DATA = ROOT / 'data/competition'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def rows(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def status(phase, **kw): runtime.atomic_json(HERE/'status.json', dict(phase=phase,pid=os.getpid(),updated_at_utc=runtime.utc_now(),**kw))
def save(p, value):
    temp=p.with_suffix('.tmp')
    with temp.open('wb') as f: torch.save(value,f); f.flush(); os.fsync(f.fileno())
    if p.exists(): os.replace(p,p.with_name('previous.pt'))
    os.replace(temp,p)
def write_csv(p, values):
    temp=p.with_suffix('.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    os.replace(temp,p)
def accepted():
    for name in ['ACCEPTANCE.json','REPLAY_ACCEPTANCE.json','SELECTION.json']:
        if not (STUDY/name).exists(): return False
    a,b=read(STUDY/'ACCEPTANCE.json'),read(STUDY/'REPLAY_ACCEPTANCE.json')
    assert a['passed'] and b['passed']
    assert a['completed_verified']==a['tta_completed_verified']==30
    assert a['manifest_sha256']==b['manifest_sha256']==sha(STUDY/'manifest.json')
    for name,digest in a['report_artifact_sha256'].items(): assert sha(STUDY/name)==digest,name
    return True

def prepare():
    selection=read(STUDY/'SELECTION.json');manifest=read(STUDY/'manifest.json')
    ranking=sorted(rows(STUDY/'inference_summary.csv'),key=lambda r:(-int(r['total_correct']),r['policy']!='raw',int(r['epochs']),r['configuration']))
    assert len(ranking)==20 and all(int(r['n'])==int(r['planned_n'])==3 for r in ranking)
    assert ranking[0]['configuration']==selection['selected_configuration'] and ranking[0]['policy']==selection['selected_policy']
    spec=next(c for c in manifest['configs'] if c['name']==selection['selected_configuration'])
    recipe=spec['recipe'];assert recipe['views']==1 and recipe['kd_temperature']==0
    source=rows(DATA/'train_labels.csv');splits=rows(ROOT/'baseline/splits.csv')
    assert len(source)==50000 and len({r['id'] for r in source})==50000
    assert [r['id'] for r in source]==[r['id'] for r in splits]
    labels=np.array([int(r['label']) for r in source],dtype=np.int64)
    assert np.array_equal(np.bincount(labels),np.full(10,5000))
    cache=np.load(ROOT/'.cache/baseline/train.npy')
    assert cache.shape==(50000,32,32,3) and cache.dtype==np.uint8
    for i,r in enumerate(source):
        with Image.open(DATA/'train_images'/r['filename']) as im:
            assert im.mode=='RGB' and im.size==(32,32)
            pixels=np.array(im)
        assert np.array_equal(pixels,cache[i]) and hashlib.sha256(pixels.tobytes()).hexdigest()==splits[i]['pixel_sha256']
        assert r['label']==splits[i]['label']
    sums=np.zeros(3);sq=np.zeros(3)
    for chunk in np.array_split(cache,100):
        x=chunk.astype(np.float64)/255;sums+=x.sum((0,1,2));sq+=(x*x).sum((0,1,2))
    count=50000*32*32;mean=sums/count;std=np.sqrt(sq/count-mean**2)
    config=dict(seed=42,configuration=spec['name'],recipe=recipe,policy=selection['selected_policy'],
        epochs=recipe['epochs'],training_images=50000,mean=mean.tolist(),std=std.tolist(),parameters=spec['parameters'],
        checkpoint_rule='last prescribed epoch; no validation or early stopping',
        selection_validation_mean_percent=selection['mean_accuracy_percent'],
        selection_note='40k-trained research models only; not accuracy of the final 50k model',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [STUDY/'SELECTION.json',STUDY/'ACCEPTANCE.json',STUDY/'REPLAY_ACCEPTANCE.json',STUDY/'manifest.json',DATA/'train_labels.csv',ROOT/'.cache/baseline/train.npy']},
        code_hashes={p.name:sha(p) for p in [HERE/'train.py',HERE/'operators.py',HERE/'architecture_model.py',HERE/'PLAN.txt']})
    if (HERE/'config.json').exists(): assert config==read(HERE/'config.json')
    else: runtime.atomic_json(HERE/'config.json',config)
    # Full-size sampling/augmentation checks precede any optimization.
    for epoch in range(1,recipe['epochs']+1):
        order,counts,_=operators.sampler(labels,42,epoch,recipe['batch_size'])
        assert len(order)==50000 and counts.min()>0
        f,a,c=operators.augmentation_plan(42,epoch,1,n=50000)
        assert f.shape==(1,50000) and c[:,:,1:].min()>=0 and c[:,:,1:].max()<32
    runtime.atomic_json(HERE/'preflight.json',dict(passed=True,images_verified=50000,epochs_checked=recipe['epochs'],config_sha256=sha(HERE/'config.json')))
    return config,cache,labels

def main():
    torch.set_num_threads(4)
    status('waiting_for_confirmation_acceptance')
    while not accepted():
        runtime.check_control('confirmation');time.sleep(10)
    config,cache,labels=prepare();cfg=config['recipe'];config_sha=sha(HERE/'config.json')
    status('queued',epochs=cfg['epochs'],configuration=config['configuration'],policy=config['policy'])
    with runtime.gpu_lock('confirmation','final_full50k_seed42'):
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        images=torch.from_numpy(cache).cuda();y=torch.from_numpy(labels).cuda()
        norm=[torch.tensor(config[k],device='cuda',dtype=torch.float32).view(1,3,1,1) for k in ['mean','std']]
        net=operators.build(cfg).cuda().to(memory_format=torch.channels_last)
        assert sum(p.numel() for p in net.parameters())==config['parameters']<=5000000
        opt=torch.optim.SGD(net.parameters(),lr=cfg['lr'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
        sch=operators.make_scheduler(opt,cfg)
        torch.manual_seed(30042);torch.cuda.manual_seed_all(30042)
        history=[]
        if (HERE/'last.pt').exists():
            cp=torch.load(HERE/'last.pt',map_location='cpu',weights_only=False);assert cp['config_sha256']==config_sha
            net.load_state_dict(cp['model_state']);opt.load_state_dict(cp['optimizer_state']);sch.load_state_dict(cp['scheduler_state']);history=cp['history']
            random.setstate(cp['python_rng']);np.random.set_state(cp['numpy_rng']);torch.set_rng_state(cp['torch_rng']);torch.cuda.set_rng_state_all(cp['cuda_rng']);del cp
        runtime.atomic_json(HERE/'environment.json',dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),bf16_training=True,fp32_inference=True))
        for epoch in range(len(history)+1,cfg['epochs']+1):
            runtime.check_control('confirmation');torch.cuda.synchronize();start=time.monotonic()
            order_np,counts,swaps=operators.sampler(labels,42,epoch,cfg['batch_size'])
            flip,angle,cut=operators.augmentation_plan(42,epoch,1,n=50000)
            order=torch.from_numpy(order_np).cuda();fp,ang,cu=[torch.as_tensor(v,device='cuda') for v in (flip,angle,cut)]
            net.train();totals=torch.zeros(2,dtype=torch.float64,device='cuda');lr=opt.param_groups[0]['lr']
            for pos in range(0,50000,cfg['batch_size']):
                ids=order[pos:pos+cfg['batch_size']];x=operators.augment(images,ids,norm,fp[0,ids],ang[0,ids],cu[0,ids],cfg)
                opt.zero_grad(set_to_none=True);loss,logits=operators.objective(net,x,y[ids],cfg)
                if not torch.isfinite(loss).item():raise FloatingPointError('Nonfinite training loss')
                loss.backward();opt.step();totals[0]+=loss.detach()*len(ids);totals[1]+=(logits.detach().argmax(1)==y[ids]).sum()
            sch.step();torch.cuda.synchronize();values=(totals/50000).tolist()
            history.append(dict(epoch=epoch,lr=lr,train_loss=values[0],train_accuracy=values[1],examples=50000,updates=len(counts),seconds=time.monotonic()-start,order_sha256=hashlib.sha256(order_np.tobytes()).hexdigest(),sampler_swaps=swaps))
            save(HERE/'last.pt',dict(config_sha256=config_sha,epoch=epoch,model_state=net.state_dict(),optimizer_state=opt.state_dict(),scheduler_state=sch.state_dict(),history=history,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
            write_csv(HERE/'history.csv',history);status('training',epochs=cfg['epochs'],**history[-1])
            if epoch==1 or epoch%10==0:print(json.dumps(history[-1]),flush=True)
        runtime.check_control('confirmation');net.eval()
        save(HERE/'model.pt',dict(model_state={k:v.detach().cpu() for k,v in net.state_dict().items()},recipe=cfg,policy=config['policy'],mean=config['mean'],std=config['std'],seed=42,epoch=cfg['epochs'],parameters=config['parameters'],training_images=50000,config_sha256=config_sha))
        test=rows(DATA/'test.csv');sample=rows(DATA/'sample_submission.csv');assert [r['id'] for r in test]==[r['id'] for r in sample] and len(test)==2000
        pixels=[]
        for r in test:
            with Image.open(DATA/'test_images'/r['filename']) as im:
                assert im.mode=='RGB' and im.size==(32,32);pixels.append(np.array(im))
        test_images=torch.from_numpy(np.stack(pixels)).cuda()
        raw=operators.predict(net,test_images,norm)
        scores=raw
        if config['policy']=='tta':scores=.5*(raw.softmax(1)+operators.predict(net,test_images.flip(2),norm).softmax(1))
        pred=scores.argmax(1).cpu().tolist()
        write_csv(HERE/'submission.csv',[dict(id=r['id'],label=p) for r,p in zip(test,pred)])
        np.save(HERE/'competition_scores.npy',scores.cpu().numpy())
        runtime.atomic_json(HERE/'RESULT.json',dict(completed_at_utc=runtime.utc_now(),configuration=config['configuration'],policy=config['policy'],epoch=cfg['epochs'],parameters=config['parameters'],training_images=50000,optimizer_updates=len(history)*len(counts),total_training_presentations=50000*cfg['epochs'],validation_accuracy=None,kaggle_score=None,uploaded=False,artifact_sha256={n:sha(HERE/n) for n in ['model.pt','submission.csv','competition_scores.npy','history.csv','config.json']}))
        status('awaiting_independent_replay',epoch=cfg['epochs'],epochs=cfg['epochs'])

if __name__=='__main__':
    import msvcrt
    with (HERE/'worker.lock').open('a+b') as lock:
        lock.seek(0,2)
        if not lock.tell():lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        except Exception as exc:
            status('paused' if isinstance(exc,runtime.Paused) else 'cancelled' if isinstance(exc,runtime.Cancelled) else 'failed',error=repr(exc));raise
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
