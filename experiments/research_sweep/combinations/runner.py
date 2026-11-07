"""Freeze, validate, execute and resume the explicitly authorized combination study."""
import argparse
import copy
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback
import uuid

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.research_sweep import runtime
from experiments.research_sweep.architecture.runner import load_data, sha, digest, configure_gpu as configure_base, cpu_state, rng_state, restore_rng, SPLIT, CACHE, AUDIT
from experiments.research_sweep.architecture.models import build as original_build
from experiments.research_sweep.combinations import methods
from experiments.research_sweep.combinations.register import design

CATEGORY = 'combinations'
TEACHER = ROOT/'experiments/width_sweep/3x/resnet20.pt'
SOURCE_PATHS = [HERE/'runner.py',HERE/'methods.py',HERE/'register.py',
    HERE/'report.py',
    ROOT/'model.py', ROOT/'experiments/research_sweep/runtime.py',
    ROOT/'experiments/research_sweep/architecture/models.py',
    ROOT/'experiments/research_sweep/architecture/runner.py',
    ROOT/'experiments/research_sweep/training/methods.py']


def configure_gpu(seed):
    configure_base(seed)
    torch.use_deterministic_algorithms(True)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def replace(source, target):
    deadline = time.monotonic()+5
    while True:
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.05)


def atomic(path, writer, binary=False, previous=False):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('wb' if binary else 'w', **({} if binary else dict(encoding='utf-8',newline=''))) as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    if previous and path.exists():
        prior = path.with_name('previous.pt')
        link = prior.with_name('previous.'+uuid.uuid4().hex+'.tmp')
        try:
            os.link(path,link)
        except OSError:
            shutil.copyfile(path,link)
        replace(link,prior)
    replace(temp,path)


def write_json(path,value):
    atomic(path,lambda f:json.dump(value,f,ensure_ascii=False,indent=2))


def save(path,value):
    atomic(path,lambda f:torch.save(value,f),binary=True,previous=Path(path).name=='last.pt')


def write_csv(path,records):
    if not records:
        return
    def writer(f):
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    atomic(path,writer)


def event(message,**fields):
    value=dict(at_utc=runtime.utc_now(),message=message,**fields)
    with (HERE/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush()
    print(json.dumps(value,ensure_ascii=False),flush=True)


def status(phase,**fields):
    old=read(HERE/'status.json') if (HERE/'status.json').exists() else {}
    total=len(read(HERE/'manifest.json')['runs']) if (HERE/'manifest.json').exists() else 86
    done=len(list((HERE/'runs').glob('*/metrics.json')))
    old.update(category=CATEGORY,phase=phase,total_runs=total,completed_runs=done,
               updated_at_utc=runtime.utc_now(),pid=os.getpid(),**fields)
    write_json(HERE/'status.json',old)


def verify_freeze():
    freeze=read(HERE/'freeze.json')
    for rel,expected in freeze['hashes'].items():
        assert sha(ROOT/rel)==expected, 'Frozen input/source changed: '+rel
    assert sha(HERE/'manifest.json')==freeze['manifest_sha256']
    return read(HERE/'manifest.json')


def load_teacher(device='cuda'):
    from model import ResNet20
    teacher=ResNet20(3).to(device=device,memory_format=torch.channels_last)
    checkpoint=torch.load(TEACHER,map_location='cpu',weights_only=True)
    teacher.load_state_dict(checkpoint['model_state'])
    teacher.eval();teacher.requires_grad_(False)
    return teacher


def prepare():
    if (HERE/'freeze.json').exists():
        verify_freeze();event('Existing freeze verified');return
    status('implementing',active_run='cpu_preflight',epoch=0)
    configs, comparisons=design()
    assert len(configs)==22
    data,audit=load_data()
    assert sha(TEACHER)=='aa64543eba16b580a5a6f6bb8b88d31afeff1156eefcd7bd7b62f8e948b0d02a'
    checks=[]
    # Exact old-backbone parity when extra modules have zero strength.
    for b in (False,True):
        cfg=dict(configs[0]['recipe'],blur=b)
        torch.manual_seed(42);a=methods.build(cfg).eval()
        torch.manual_seed(42);bnet=original_build(dict(widths=[48,96,192],blocks=3,variant='blur' if b else 'plain')).eval()
        assert all(torch.equal(v,bnet.state_dict()[k]) for k,v in a.state_dict().items())
        x=torch.randn(2,3,32,32)
        with torch.no_grad():assert torch.equal(a(x),bnet(x))
    checks.append('exact original plain/Blur initialization and output parity')
    for spec in configs:
        cfg=spec['recipe'];spec.update(methods.cost(cfg))
        spec['teacher_parameters']=2436346 if cfg['kd_temperature'] else 0
        assert spec['parameters']+spec['teacher_parameters']<=5_000_000
        model=methods.build(cfg)
        loss=F.cross_entropy(model(torch.randn(2,3,32,32)),torch.tensor([0,1]))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    checks.append('22 model configurations: shape, parameter budget and finite connected gradients')
    # Full epoch coverage and ten-class checks, including every long-run epoch.
    plans=[]
    for seed in range(42,47):
        for epoch in range(1,241 if seed<45 else 81):
            f,a,c=methods.augmentation_plan(seed,epoch,1)
            f2,a2,c2=methods.augmentation_plan(seed,epoch,2)
            assert np.array_equal(f[0],f2[0]) and np.array_equal(a[0],a2[0]) and np.array_equal(c[0],c2[0])
            for batch in (128,64):
                if batch==64 and (seed>=45 or epoch>80):continue
                order,counts,swaps=methods.sampler(data['train'][1],seed,epoch,batch)
                assert np.array_equal(np.sort(order),np.arange(40000)) and counts.min()>=1
                plans.append(dict(seed=seed,epoch=epoch,batch_size=batch,swaps=swaps,
                    order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
                    augmentation_sha256=hashlib.sha256(f.tobytes()+a.tobytes()+c.tobytes()).hexdigest()))
    write_csv(HERE/'sampling_preflight.csv',plans)
    checks.append('all registered seed/epoch/batch sampler coverage; common view-zero augmentation')
    # Direct pixel assertions catch clipped Cutout edge/size/broadcast errors.
    images=torch.full((2,32,32,3),255,dtype=torch.uint8);ids=torch.tensor([0,1])
    norm=[torch.zeros(1,3,1,1),torch.ones(1,3,1,1)]
    cfg=dict(configs[0]['recipe'],cutout=12)
    out=methods.augment(images,ids,norm,torch.tensor([False,False]),torch.zeros(2),torch.tensor([[1,16,16],[1,0,0]]),cfg)
    assert (out[0,0]==0).sum()==144 and (out[1,0]==0).sum()==36
    checks.append('Cutout center 12x12 and clipped corner 6x6 exact mask')
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in SOURCE_PATHS}
    hashes={**sources,**{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [SPLIT,CACHE,AUDIT,TEACHER,HERE/'PLAN.md']}}
    runs=[]
    for seed in range(42,47):
        for spec in configs:
            if seed not in spec['seeds']:continue
            r=dict(run_id=spec['name']+f'_seed{seed}',seed=seed,spec=spec,input_source_hashes=hashes)
            r['config_sha256']=digest(r);runs.append(r)
    assert len(runs)==86 and sum(r['spec']['recipe']['epochs'] for r in runs)==7360
    manifest=dict(category=CATEGORY,created_at_utc=runtime.utc_now(),configs=configs,runs=runs,
        comparisons=comparisons,total_runs=86,epochs=7360,normalization=dict(mean=audit['train_rgb_mean'],std=audit['train_rgb_std']))
    write_json(HERE/'manifest.json',manifest)
    write_json(HERE/'cpu_preflight.json',dict(passed=True,checks=checks,at_utc=runtime.utc_now()))
    frozen=HERE/'frozen_sources';frozen.mkdir(exist_ok=True)
    for p in SOURCE_PATHS:
        target=frozen/str(p.relative_to(ROOT)).replace('\\','__').replace('/','__')
        target.write_bytes(p.read_bytes())
    write_json(HERE/'freeze.json',dict(frozen_at_utc=runtime.utc_now(),hashes=hashes,manifest_sha256=sha(HERE/'manifest.json')))
    status('queued',active_run=None,epoch=0,message='CPU preflight passed; finite 86-run manifest frozen')
    event('Plan frozen',runs=86,configs=22,comparisons=len(comparisons))


def smoke():
    manifest=verify_freeze()
    if (HERE/'gpu_preflight.json').exists():
        assert read(HERE/'gpu_preflight.json')['passed'];return
    status('queued',active_run='gpu_preflight',epoch=0)
    with runtime.gpu_lock(CATEGORY,'gpu_preflight'):
        configure_gpu(4242)
        data,audit=load_data()
        images=torch.from_numpy(data['train'][0][:8]).cuda()
        labels=torch.from_numpy(data['train'][1][:8]).cuda()
        norm=[torch.tensor(v,device='cuda').view(1,3,1,1) for v in (audit['train_rgb_mean'],audit['train_rgb_std'])]
        checks=[]
        for spec in manifest['configs']:
            runtime.check_control(CATEGORY)
            cfg=spec['recipe'];net=methods.build(cfg).cuda().to(memory_format=torch.channels_last)
            opt=torch.optim.SGD(net.parameters(),lr=.1,momentum=.9,weight_decay=cfg['weight_decay'])
            teacher=load_teacher() if cfg['kd_temperature'] else None
            flip,angle,cut=methods.augmentation_plan(42,1,cfg['views'],8)
            fp,ang,cu=[torch.as_tensor(v,device='cuda') for v in (flip,angle,cut)]
            ids=torch.arange(8,device='cuda')
            x=torch.cat([methods.augment(images,ids,norm,fp[v],ang[v],cu[v],cfg) for v in range(cfg['views'])])
            loss,_=methods.objective(net,x,labels.repeat(cfg['views']),cfg,teacher)
            loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters())
            opt.step();opt.zero_grad(set_to_none=True)
            assert methods.predict(net,images,norm).shape==(8,10)
            checks.append(dict(config=spec['name'],passed=True))
            del net,opt,teacher,x,loss
        # Real GPU dropout branch save/restore reproduces the next optimizer step.
        cfg=next(s['recipe'] for s in manifest['configs'] if s['name']=='core_drop1')
        configure_gpu(999)
        net=methods.build(cfg).cuda().to(memory_format=torch.channels_last)
        opt=torch.optim.SGD(net.parameters(),lr=.1,momentum=.9,weight_decay=cfg['weight_decay'])
        sch=methods.make_scheduler(opt,cfg)
        a,b=np.random.default_rng(1),np.random.default_rng(2)
        x=(images.permute(0,3,1,2).float()/255-norm[0])/norm[1]
        def step():
            net.train();opt.zero_grad(set_to_none=True)
            loss,_=methods.objective(net,x,labels,cfg);loss.backward();opt.step();sch.step()
        step()
        buf=io.BytesIO();torch.save(dict(model=net.state_dict(),opt=opt.state_dict(),sch=sch.state_dict(),rng=rng_state(a,b)),buf)
        step();expected=cpu_state(net)
        buf.seek(0);state=torch.load(buf,map_location='cpu',weights_only=False)
        net.load_state_dict(state['model']);opt.load_state_dict(state['opt']);sch.load_state_dict(state['sch']);restore_rng(state['rng'],a,b)
        step();assert all(torch.equal(v,net.state_dict()[k].cpu()) for k,v in expected.items())
        checks.append(dict(config='GPU_exact_resume_dropout_optimizer_scheduler_RNG',passed=True))
        write_json(HERE/'gpu_preflight.json',dict(passed=True,checks=checks,at_utc=runtime.utc_now(),device=torch.cuda.get_device_name()))
        del net,opt,sch,x,images,labels,norm,state
        gc.collect();torch.cuda.empty_cache()
    event('GPU preflight passed',checks=len(checks))


def verify_completed(folder,run):
    m=read(folder/'metrics.json')
    assert m['config_sha256']==run['config_sha256'] and m['epochs_completed']==run['spec']['recipe']['epochs'] and m['validation_n']==5000
    for name,h in m['artifact_sha256'].items():
        assert sha(folder/name)==h, 'Artifact changed: '+str(folder/name)
    return m


def train(run,data,audit):
    folder=HERE/'runs'/run['run_id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():verify_completed(folder,run);return
    if (folder/'config.json').exists():assert read(folder/'config.json')==run
    else:write_json(folder/'config.json',run)
    cfg=run['spec']['recipe'];configure_gpu(run['seed'])
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    images,y=[torch.from_numpy(v).cuda() for v in data['train'][:2]]
    val_images,val_y=[torch.from_numpy(v).cuda() for v in data['validation'][:2]]
    norm=[torch.tensor(v,device='cuda',dtype=torch.float32).view(1,3,1,1) for v in (audit['train_rgb_mean'],audit['train_rgb_std'])]
    net=methods.build(cfg).cuda().to(memory_format=torch.channels_last)
    opt=torch.optim.SGD(net.parameters(),lr=cfg['lr'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    sch=methods.make_scheduler(opt,cfg)
    teacher=load_teacher() if cfg['kd_temperature'] else None
    # Re-seed network stochastic layers after all architecture/teacher creation.
    torch.manual_seed(run['seed']+30000);torch.cuda.manual_seed_all(run['seed']+30000)
    a,b=np.random.default_rng(run['seed']+10000),np.random.default_rng(run['seed']+20000)
    history=[];best_state=None;best_correct=-1;best_epoch=0;prior_seconds=0.;prior_peak=0
    info=dict(run_id=run['run_id'],phase='running',pid=os.getpid(),started_at_utc=runtime.utc_now(),
        config_sha256=run['config_sha256'],python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),device=torch.cuda.get_device_name(),
        bf16_training=True,fp32_validation=True,tf32=False,cudnn_benchmark=False,cudnn_deterministic=True,
        deterministic_algorithms=True,cublas_workspace_config=':4096:8')
    if (folder/'last.pt').exists():
        cp=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        assert cp['config_sha256']==run['config_sha256']
        net.load_state_dict(cp['model_state']);opt.load_state_dict(cp['optimizer_state']);sch.load_state_dict(cp['scheduler_state'])
        history=cp['history'];best_state=cp['best_state'];best_correct=cp['best_correct'];best_epoch=cp['best_epoch']
        prior_seconds=cp['lock_seconds'];prior_peak=cp['peak_allocated_bytes']
        restore_rng(cp['rng_state'],a,b)
        info.update(started_at_utc=read(folder/'run.json')['started_at_utc'],resumed_at_utc=runtime.utc_now())
        write_csv(folder/'history.csv',history)
        save(folder/'best.pt',dict(model_state=best_state,config=run,epoch=best_epoch,validation_correct=best_correct))
        del cp
        event('Resumed committed checkpoint',run_id=run['run_id'],epoch=len(history))
    write_json(folder/'run.json',info)
    for epoch in range(len(history)+1,cfg['epochs']+1):
        runtime.check_control(CATEGORY)
        status('running',active_run=run['run_id'],epoch=epoch-1,message='Training with shared GPU lock')
        torch.cuda.synchronize();t0=time.monotonic();lr=opt.param_groups[0]['lr']
        order_np,counts,swaps=methods.sampler(data['train'][1],run['seed'],epoch,cfg['batch_size'])
        flip,angle,cut=methods.augmentation_plan(run['seed'],epoch,cfg['views'])
        order=torch.from_numpy(order_np).cuda();fp,ang,cu=[torch.as_tensor(v,device='cuda') for v in (flip,angle,cut)]
        net.train();totals=torch.zeros(2,dtype=torch.float64,device='cuda')
        for pos in range(0,40000,cfg['batch_size']):
            ids=order[pos:pos+cfg['batch_size']]
            x=torch.cat([methods.augment(images,ids,norm,fp[v,ids],ang[v,ids],cu[v,ids],cfg) for v in range(cfg['views'])])
            labels=y[ids].repeat(cfg['views']);opt.zero_grad(set_to_none=True)
            loss,logits=methods.objective(net,x,labels,cfg,teacher)
            if not torch.isfinite(loss).item():raise FloatingPointError('Nonfinite training loss')
            loss.backward();opt.step()
            totals[0]+=loss.detach()*len(labels);totals[1]+=(logits.detach().argmax(1)==labels).sum()
        val_logits=methods.predict(net,val_images,norm);correct=int((val_logits.argmax(1)==val_y).sum())
        val_loss=float(F.cross_entropy(val_logits,val_y));train_loss,train_acc=(totals/(40000*cfg['views'])).tolist()
        assert np.isfinite([train_loss,val_loss]).all()
        torch.cuda.synchronize();compute=time.monotonic()-t0
        if correct>best_correct:
            best_state,best_correct,best_epoch=cpu_state(net),correct,epoch
            save(folder/'best.pt',dict(model_state=best_state,config=run,epoch=best_epoch,validation_correct=best_correct))
        sch.step()
        history.append(dict(epoch=epoch,lr=lr,train_loss=train_loss,train_accuracy=train_acc,
            validation_loss=val_loss,validation_correct=correct,validation_accuracy=correct/5000,
            compute_seconds=compute,examples=40000,views=cfg['views'],updates=len(counts),sampler_swaps=swaps,
            order_sha256=hashlib.sha256(order_np.tobytes()).hexdigest(),
            view0_sha256=hashlib.sha256(flip[0].tobytes()+angle[0].tobytes()+cut[0].tobytes()).hexdigest()))
        save(folder/'last.pt',dict(config=run,config_sha256=run['config_sha256'],epoch=epoch,
            model_state=net.state_dict(),optimizer_state=opt.state_dict(),scheduler_state=sch.state_dict(),
            rng_state=rng_state(a,b),best_state=best_state,best_correct=best_correct,best_epoch=best_epoch,history=history,
            lock_seconds=prior_seconds+time.monotonic()-start,peak_allocated_bytes=max(prior_peak,torch.cuda.max_memory_allocated())))
        write_csv(folder/'history.csv',history)
        status('running',active_run=run['run_id'],epoch=epoch,message=f'Committed {epoch}/{cfg["epochs"]}; best {best_correct}/5000')
        if epoch%10==0 or epoch==1:event('Epoch committed',run_id=run['run_id'],epoch=epoch,correct=correct,best=best_correct,compute_seconds=round(compute,3))
        runtime.check_control(CATEGORY)
    net.load_state_dict(best_state);logits=methods.predict(net,val_images,norm);pred=logits.argmax(1)
    confusion=torch.bincount(val_y*10+pred,minlength=100).reshape(10,10).cpu().numpy()
    assert int(confusion.trace())==best_correct
    confidence=logits.softmax(1).amax(1).cpu().tolist()
    write_csv(folder/'validation_predictions.csv',[dict(id=row['id'],label=int(row['label']),prediction=p,confidence=c)
        for row,p,c in zip(data['validation'][2],pred.cpu().tolist(),confidence)])
    atomic(folder/'validation_logits.npy',lambda f:np.save(f,logits.cpu().numpy()),binary=True)
    tp=confusion.diagonal();recall=tp/confusion.sum(1);f1=2*tp/(confusion.sum(0)+confusion.sum(1))
    torch.cuda.synchronize()
    info.update(phase='completed',completed_at_utc=runtime.utc_now(),epochs_completed=cfg['epochs'],
        gpu_compute_seconds=sum(h['compute_seconds'] for h in history),lock_seconds=prior_seconds+time.monotonic()-start)
    write_json(folder/'run.json',info)
    artifacts=['config.json','run.json','history.csv','best.pt','last.pt','validation_predictions.csv','validation_logits.npy']
    metrics=dict(run_id=run['run_id'],configuration=run['spec']['name'],seed=run['seed'],config_sha256=run['config_sha256'],
        epochs_completed=cfg['epochs'],validation_n=5000,validation_correct=best_correct,validation_accuracy=best_correct/5000,
        best_epoch=best_epoch,validation_nll=float(F.cross_entropy(logits,val_y)),macro_f1=float(f1.mean()),
        class_recall=recall.tolist(),class_f1=f1.tolist(),worst_class_recall=float(recall.min()),confusion_matrix=confusion.tolist(),
        parameters=run['spec']['parameters'],macs=run['spec']['macs'],teacher_parameters=run['spec']['teacher_parameters'],
        gpu_compute_seconds=info['gpu_compute_seconds'],lock_seconds=info['lock_seconds'],
        peak_allocated_bytes=max(prior_peak,torch.cuda.max_memory_allocated()),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        training_source_presentations=40000*cfg['epochs'],training_views=40000*cfg['epochs']*cfg['views'],
        optimizer_updates=sum(h['updates'] for h in history),artifact_sha256={name:sha(folder/name) for name in artifacts})
    write_json(folder/'metrics.json',metrics)
    event('Run completed',run_id=run['run_id'],correct=best_correct,best_epoch=best_epoch,gpu_compute_seconds=round(info['gpu_compute_seconds'],2))


def run_all():
    manifest=verify_freeze();smoke();data,audit=load_data()
    for run in manifest['runs']:
        runtime.check_control(CATEGORY)
        folder=HERE/'runs'/run['run_id']
        if (folder/'metrics.json').exists():verify_completed(folder,run);continue
        status('queued',active_run=run['run_id'],epoch=0)
        with runtime.gpu_lock(CATEGORY,run['run_id']):
            try:train(run,data,audit)
            finally:gc.collect();torch.cuda.empty_cache()
        from experiments.research_sweep.combinations.report import report
        report(final=False)
    from experiments.research_sweep.combinations.report import report
    report(final=True)
    status('awaiting_audit',active_run=None,epoch=0,message='86 training results present; independent acceptance required')


def main():
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','smoke','run','report'])
    args=parser.parse_args();HERE.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (HERE/'worker.lock').open('a+b') as h:
        h.seek(0,2)
        if not h.tell():h.write(b'0');h.flush()
        h.seek(0);msvcrt.locking(h.fileno(),msvcrt.LK_NBLCK,1)
        try:
            if args.action=='prepare':prepare()
            elif args.action=='smoke':smoke()
            elif args.action=='run':run_all()
            else:
                from experiments.research_sweep.combinations.report import report
                report(final=False)
        except (runtime.Paused,runtime.Cancelled) as exc:
            status('paused' if isinstance(exc,runtime.Paused) else 'cancelled',message=str(exc))
            event('User control stopped worker',reason=str(exc));return 20
        except Exception as exc:
            write_json(HERE/'failures'/f'{time.time_ns()}.json',dict(at_utc=runtime.utc_now(),action=args.action,error=repr(exc),traceback=traceback.format_exc()))
            status('failed',message=repr(exc));event('Worker failed',error=repr(exc));raise
        finally:
            h.seek(0);msvcrt.locking(h.fileno(),msvcrt.LK_UNLCK,1)
    return 0


if __name__=='__main__':
    sys.exit(main())
