"""Numerical, sampling, budget, and resumability checks; never experimental results."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[2])]
import copy
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from runner import read,load_cpu,source_fingerprint,atomic_torch,rng_state,set_rng
from methods import *
from experiments.research_sweep.runtime import gpu_lock,check_control,atomic_json,utc_now,write_status,Paused,Cancelled


def main():
    torch.set_num_threads(4)
    manifest=read(HERE/'manifest.json'); by_id={c['id']:c for c in manifest['configs']}
    audit,cpu=load_cpu();records=[]
    for c in manifest['configs']:
        cost=model_cost(c['recipe'])
        assert cost['trainable_parameters']==c['parameters']
        assert c['parameters']+c['teacher_parameters']<=5_000_000
        records.append(dict(id=c['id'],**cost))
    # Initialization equality to historical model: wrapper introduces no hidden parameter changes.
    torch.manual_seed(42);reference=ResNet20(3)
    torch.manual_seed(42);wrapped=Classifier(by_id['base']['recipe'])
    assert all(torch.equal(v,wrapped.net.state_dict()[k]) for k,v in reference.state_dict().items())
    sampler_records=[]
    for seed in range(42,47):
        for batch in [64,128,256]:
            maximum=0
            for epoch in range(1,241):
                order,counts,swaps=sampler(cpu['train']['labels'],seed,epoch,batch)
                assert len(order)==40000 and counts.min()>=1
                maximum=max(maximum,swaps)
            sampler_records.append(dict(seed=seed,batch_size=batch,epochs_checked=240,max_swaps=maximum))
    for seed in range(42,47):
        a=augmentation_plan(seed,7,40000,2,.5);b=augmentation_plan(seed,7,40000,2,.5)
        assert all(np.array_equal(x,y) for x,y in zip(a,b))
        assert not np.array_equal(a[0][0],a[0][1])
    # All-equal embeddings have analytic contrastive loss log(N-1).
    z=torch.ones(20,128,requires_grad=True);labels=torch.arange(10).repeat(2)
    loss=supcon_loss(z,labels)
    assert abs(loss.item()-math.log(19))<1e-5
    loss.backward();assert torch.isfinite(z.grad).all()
    schedules={}
    for key in ['base','warmup5','multistep','sgdr','swa_noavg']:
        cfg=by_id[key]['recipe'];m=Classifier(cfg);opt=make_optimizer(m,cfg);scheduler=make_scheduler(opt,cfg)
        lrs=[]
        for epoch in range(80):
            lrs.append(opt.param_groups[0]['lr']);opt.step();scheduler.step()
        schedules[key]=lrs
    assert np.allclose(schedules['warmup5'][:5],[.02,.04,.06,.08,.1])
    assert np.isclose(schedules['multistep'][40],.01) and np.isclose(schedules['multistep'][60],.001)
    assert np.isclose(schedules['sgdr'][20],.1) and np.isclose(schedules['sgdr'][60],.1)
    assert np.allclose(schedules['swa_noavg'][60:],.01)
    atomic_json(HERE/'preflight_cpu.json',dict(passed=True,at_utc=utc_now(),models=records,samplers=sampler_records,schedules=schedules))
    write_status('training',phase='queued',active_run='preflight',epoch=None,message='CPU preflight passed; waiting for GPU smoke lock')
    print('CPU PREFLIGHT PASSED; GPU smoke queued',flush=True)
    results=[]
    with gpu_lock('training','preflight'):
        check_control('training')
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        images=torch.from_numpy(cpu['train']['images'][:256]).to('cuda')
        mean,std=[torch.tensor(audit[k],device='cuda').view(1,3,1,1) for k in ('train_rgb_mean','train_rgb_std')]
        y=torch.from_numpy(cpu['train']['labels'][:32]).to('cuda')
        idx=torch.arange(32,device='cuda')
        flips=torch.zeros(32,dtype=torch.bool,device='cuda')
        x=augment(images,idx,mean,std,flips)
        assert torch.equal(x,normalized(images[:32],mean,std))
        teacher=ResNet20(3).to('cuda').eval().requires_grad_(False)
        teacher.load_state_dict(torch.load(HERE.parents[1]/'width_sweep/3x/resnet20.pt',map_location='cpu',weights_only=True)['model_state'])
        for c in manifest['configs']:
            check_control('training')
            cfg=c['recipe'];torch.manual_seed(42)
            model=Classifier(cfg).to(device='cuda',memory_format=torch.channels_last).train()
            optimizer=make_optimizer(model,cfg)
            xx=torch.cat([x,x.flip(-1)]) if cfg['views']==2 else x
            yy=y.repeat(cfg['views'])
            before=[m.num_batches_tracked.item() for m in model.modules() if isinstance(m,nn.BatchNorm2d)]
            loss,logits,ce,aux=train_step(model,optimizer,xx,yy,cfg,teacher if cfg['kd_temperature'] else None)
            after=[m.num_batches_tracked.item() for m in model.modules() if isinstance(m,nn.BatchNorm2d)]
            assert all(b+1==a for b,a in zip(before,after)), 'BN updated twice during SAM'
            assert logits.shape==(len(yy),10) and torch.isfinite(loss)
            if c['output'].startswith('ema') or c['output']=='swa':
                shadow=copy.deepcopy(model).requires_grad_(False)
                update_average(shadow,model,.99);recompute_bn(shadow,images,mean,std)
                assert all(m.num_batches_tracked.item()==2 for m in shadow.modules() if isinstance(m,nn.BatchNorm2d))
            results.append(dict(id=c['id'],loss=loss.item(),ce=ce.item(),auxiliary=aux.item(),bn_update_once=True))
            print('GPU SMOKE '+c['id'],flush=True)
        # Same second step after on-disk restore, including dropout RNG and optimizer momentum.
        cfg=by_id['dropout01']['recipe'];torch.manual_seed(812)
        model=Classifier(cfg).to('cuda').train();optimizer=make_optimizer(model,cfg)
        scheduler=make_scheduler(optimizer,cfg)
        train_step(model,optimizer,x,y,cfg);scheduler.step()
        path=HERE/'preflight_resume.pt'
        atomic_torch(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state()))
        train_step(model,optimizer,x,y,cfg);expected=copy.deepcopy(model.state_dict())
        saved=torch.load(path,map_location='cpu',weights_only=False)
        restored=Classifier(cfg).to('cuda').train();restored.load_state_dict(saved['model'])
        restored_optimizer=make_optimizer(restored,cfg);restored_optimizer.load_state_dict(saved['optimizer'])
        restored_scheduler=make_scheduler(restored_optimizer,cfg);restored_scheduler.load_state_dict(saved['scheduler'])
        set_rng(saved['rng']);train_step(restored,restored_optimizer,x,y,cfg)
        assert all(torch.equal(v,restored.state_dict()[k]) for k,v in expected.items()),'Resume is not numerically identical'
        assert all(p.grad is None for p in teacher.parameters())
        torch.cuda.synchronize()
    atomic_json(HERE/'preflight.json',dict(passed=True,at_utc=utc_now(),source_fingerprint=source_fingerprint(),
                gpu_smoke=results,resume_bitwise_equal=True,teacher_frozen=True,not_experimental_results=True))
    print('ALL PREFLIGHT CHECKS PASSED',flush=True)


if __name__=='__main__':
    try:main()
    except (Paused,Cancelled) as exc:
        write_status('training',phase='paused',active_run='preflight',message=str(exc),cancelled=isinstance(exc,Cancelled))
        print(str(exc),flush=True)
    except Exception as exc:
        import traceback
        atomic_json(HERE/'preflight_failure.json',dict(at_utc=utc_now(),error=repr(exc),traceback=traceback.format_exc()))
        write_status('training',phase='failed',active_run='preflight',message=repr(exc))
        raise
