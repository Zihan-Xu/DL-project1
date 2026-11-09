"""Lock-aware, epoch-resumable execution of the full 71-run data matrix."""
import argparse
import gc
import hashlib
import json
import math
import marshal
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from study import (HERE,ROOT,ARMS,DEFAULTS,read,records,write_csv,digest,inputs,verify_prepared,sha256)
from model import ResNet20
from experiments.research_sweep.runtime import (atomic_json,utc_now,write_status,check_control,gpu_lock,Paused,Cancelled)


LOADED_RUNTIME_SHA256 = sha256(ROOT/'experiments/research_sweep/runtime.py')
LOADED_ATOMIC_JSON_SHA256 = hashlib.sha256(marshal.dumps(atomic_json.__code__)).hexdigest()


def runtime_identity():
    amendment = HERE/'runtime_io_amendment.json'
    return dict(source_sha256=LOADED_RUNTIME_SHA256,
                atomic_json_code_sha256=LOADED_ATOMIC_JSON_SHA256,
                amendment_sha256=sha256(amendment) if amendment.exists() else None)


def atomic_save(obj,path):
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    torch.save(obj,temp)
    os.replace(temp,path)


def log(folder,message):
    line=f'[{utc_now()}] {message}'
    print(line,flush=True)
    with (folder/'run.log').open('a',encoding='utf-8') as f:f.write(line+'\n')


def progress(phase,**kwargs):
    complete=sum(1 for p in (HERE/'runs').glob('*/metrics.json') if read(p).get('complete'))
    failed=sum(1 for p in (HERE/'runs').glob('*/run.json') if read(p).get('phase')=='failed')
    write_status('data',phase=phase,total_runs=71,completed_runs=complete,failed_runs=failed,
                 report_path=str(HERE/'REPORT.md'),**kwargs)


def code_hashes():
    paths=[HERE/'study.py',HERE/'runner.py',HERE/'analyze.py',HERE/'import_prior.py',HERE/'check_prior.py',ROOT/'model.py',ROOT/'run_color_balance.py',ROOT/'run_baseline.py',ROOT/'prepare_data.py',ROOT/'experiments/research_sweep/runtime.py']
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in paths}


def freeze():
    verify_prepared()
    assert read(HERE/'gpu_smoke.json')['passed']
    assert any(c.get('bitwise_equal_model_and_BN') for c in read(HERE/'gpu_smoke.json')['checks'])
    assert read(HERE/'historical_compatibility.json')['passed']
    frozen=dict(plan_sha256=sha256(HERE/'PLAN.md'),manifest_sha256=sha256(HERE/'manifest.json'),
                prepared_sha256=sha256(HERE/'prepared.json'),code_sha256=code_hashes(),
                audit_sha256={n:sha256(HERE/n) for n in ['preflight.json','gpu_smoke.json','historical_compatibility.json']},
                created_at_utc=utc_now())
    if (HERE/'freeze.json').exists():
        old=read(HERE/'freeze.json')
        for key in frozen:
            if key!='created_at_utc':assert old[key]==frozen[key],key
    else:
        for name,h in frozen['code_sha256'].items():
            destination=HERE/'frozen_source'/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,destination);assert sha256(destination)==h
        atomic_json(HERE/'freeze.json',frozen)
    return read(HERE/'freeze.json')


def verify_freeze():
    frozen=read(HERE/'freeze.json')
    assert frozen['plan_sha256']==sha256(HERE/'PLAN.md')
    assert frozen['manifest_sha256']==sha256(HERE/'manifest.json')
    assert frozen['prepared_sha256']==sha256(HERE/'prepared.json')
    expected=frozen['code_sha256']
    amendment_path=HERE/'implementation_amendment.json'
    if amendment_path.exists():
        amendment=read(amendment_path)
        assert amendment['base_freeze_sha256']==sha256(HERE/'freeze.json')
        assert amendment['prior_code_sha256']==expected
        assert amendment['affected_completed_runs']==[]
        assert amendment['prepared_sha256']==sha256(HERE/'prepared.json')
        assert amendment['audit_script_sha256']==sha256(HERE/'audit_cutout_fix.py')
        for name,h in amendment['snapshot_sha256'].items():assert sha256(HERE/name)==h,name
        assert sha256(HERE/amendment['audit_file'])==amendment['audit_sha256']
        assert read(HERE/amendment['audit_file'])['passed']
        expected=amendment['current_code_sha256']
    io_path = HERE/'runtime_io_amendment.json'
    if io_path.exists():
        io = read(io_path)
        assert io['base_freeze_sha256'] == sha256(HERE/'freeze.json')
        assert io['prior_amendment_sha256'] == sha256(amendment_path)
        assert io['prior_code_sha256'] == expected
        assert io['prepared_sha256'] == sha256(HERE/'prepared.json')
        for name, fingerprint in io['evidence_sha256'].items():
            assert sha256(HERE/name) == fingerprint, name
        assert read(HERE/io['audit_file'])['passed']
        assert read(HERE/io['shared_verification_file'])['passed']
        assert LOADED_RUNTIME_SHA256 == io['runtime_after_sha256']
        assert LOADED_ATOMIC_JSON_SHA256 == io['loaded_atomic_json_code_sha256']
        expected = io['current_code_sha256']
    assert expected==code_hashes(),'Code differs from frozen version and documented implementation amendment'
    for name,h in frozen['audit_sha256'].items():assert sha256(HERE/name)==h,name
    return frozen


def verify_checkpoint_code(previous_code, current_code, config):
    if previous_code == current_code:
        return
    if previous_code == read(HERE/'freeze.json')['code_sha256']:
        assert 'cutout' not in config, 'Original Cutout implementation cannot resume'
        return
    repair = read(HERE/'runtime_io_amendment.json')
    assert previous_code == repair['prior_code_sha256'], 'Unregistered checkpoint implementation'


def configuration(arm,seed):
    audit=read(ROOT/'baseline/data_audit.json')
    norm=read(HERE/'prepared'/f'seed_{seed}'/'normalization.json') if arm['series']=='C' else dict(mean=audit['train_rgb_mean'],std=audit['train_rgb_std'])
    return dict(DEFAULTS,**{k:v for k,v in arm.items() if k!='seeds'},seed=seed,normalization=norm,
                split_sha256=audit['split_sha256'],freeze_sha256=sha256(HERE/'freeze.json'),
                sampler_sha256=sha256(HERE/'prepared'/f'seed_{seed}'/f"{arm['sampler']}_plans.npz"))


def save_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all())


def restore_rng(obj):
    random.setstate(obj['python']);np.random.set_state(obj['numpy']);torch.set_rng_state(obj['torch_cpu'])
    torch.cuda.set_rng_state_all(obj['torch_cuda'])


def initialize(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True


class Augmentation:
    def __init__(self,cfg,device):
        self.cfg,self.device=cfg,device
        self.mean=torch.tensor(cfg['normalization']['mean'],device=device).view(1,3,1,1)
        self.std=torch.tensor(cfg['normalization']['std'],device=device).view(1,3,1,1)
        self.swaps=None
        if 'background' in cfg:
            self.swaps=torch.from_numpy(np.load(ROOT/'.cache/augmentations/swaps.npy')).to(device)
            selected=np.load(HERE/'prepared'/f"seed_{cfg['seed']}"/f"background_{cfg['background']}.npy")
            mask=np.zeros(40000,dtype=bool);mask[selected]=True
            self.background_mask=torch.as_tensor(mask,device=device)
        self.axis=torch.arange(32,device=device)

    def epoch(self,epoch):
        # One table per stream, keyed by occurrence and train ID. Generating a new
        # stream cannot alter initialization, batch order or another augmentation.
        def table(k):
            values=np.random.default_rng(np.random.SeedSequence([self.cfg['seed'],200,epoch,k])).random((4,40000)).astype(np.float32)
            return torch.as_tensor(values,device=self.device)
        self.tables={}
        if 'rotation' in self.cfg:self.tables={1:table(1),2:table(2)}
        if 'cutout' in self.cfg:self.tables={3:table(3),4:table(4),5:table(5)}
        if 'grayscale' in self.cfg:self.tables={6:table(6)}

    def __call__(self,images,idx,occ,flip):
        raw=images[idx]
        if self.swaps is not None:
            raw=torch.where(self.background_mask[idx,None,None,None],self.swaps[idx],raw)
        raw=torch.where(flip[:,None,None,None],raw.flip((2,)),raw)
        x=raw.permute(0,3,1,2).float().div_(255)
        if 'rotation' in self.cfg:
            apply=self.tables[1][occ,idx]<.5
            angle=(self.tables[2][occ,idx]*2-1)*math.radians(self.cfg['rotation'])
            c,s,zero=angle.cos(),angle.sin(),torch.zeros_like(angle)
            theta=torch.stack((c,-s,zero,s,c,zero),1).view(-1,2,3)
            grid=F.affine_grid(theta,x.shape,align_corners=False)
            rotated=F.grid_sample(x,grid,mode='bilinear',padding_mode='reflection',align_corners=False)
            x=torch.where(apply[:,None,None,None],rotated,x)
        if 'grayscale' in self.cfg:
            apply=self.tables[6][occ,idx]<self.cfg['grayscale']
            gray=(x*x.new_tensor([.2989,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
            x=torch.where(apply[:,None,None,None],gray.expand_as(x),x)
        x=(x-self.mean)/self.std
        if 'cutout' in self.cfg:
            apply=self.tables[3][occ,idx]<.5
            # A uniform float64 draw can round to exactly 1 on float32 conversion.
            # Clamp the integer center to the registered image domain [0,31].
            y=(self.tables[4][occ,idx]*32).long().clamp_max_(31)
            cx=(self.tables[5][occ,idx]*32).long().clamp_max_(31)
            half=self.cfg['cutout']//2
            mask=((self.axis[None,:,None]>=y[:,None,None]-half)&(self.axis[None,:,None]<y[:,None,None]+half)&
                  (self.axis[None,None,:]>=cx[:,None,None]-half)&(self.axis[None,None,:]<cx[:,None,None]+half)&apply[:,None,None])
            x=x.masked_fill(mask[:,None,:,:],0)
        return x.contiguous(memory_format=torch.channels_last)


@torch.inference_mode()
def predict(model,images,mean,std):
    model.eval()
    out=[]
    for start in range(0,len(images),128):
        x=(images[start:start+128].permute(0,3,1,2).float()/255-mean)/std
        out.append(model(x.contiguous(memory_format=torch.channels_last)))
    return torch.cat(out)


def emit_history(folder,history):
    fields=['epoch','lr','train_loss','train_accuracy','validation_loss','validation_correct','validation_accuracy','epoch_seconds']
    write_csv(folder/'history.csv',fields,[[row[k] for k in fields] for row in history])


def verify_complete(folder,cfg=None):
    required=['config.json','run.json','history.csv','best.pt','last.pt','validation_predictions.csv','metrics.json',
              'sampling_epochs.csv','batch_label_counts.csv']
    assert all((folder/n).exists() for n in required),f'Incomplete artifacts: {folder}'
    metrics=read(folder/'metrics.json');config=read(folder/'config.json')
    if cfg is not None:assert config==cfg
    assert metrics['complete'] and metrics['epochs']==80 and metrics['training_presentations']==3200000
    hist=records(folder/'history.csv');pred=records(folder/'validation_predictions.csv')
    assert len(hist)==80 and len(pred)==5000
    best=max(hist,key=lambda r:int(r['validation_correct']))
    assert int(best['epoch'])==metrics['best_epoch'] and int(best['validation_correct'])==metrics['validation_correct']
    assert sum(int(r['label']==r['prediction']) for r in pred)==metrics['validation_correct']
    assert sha256(folder/'best.pt')==metrics['best_checkpoint_sha256']
    assert sha256(folder/'last.pt')==metrics['last_checkpoint_sha256']
    assert sha256(folder/'validation_predictions.csv')==metrics['predictions_sha256']
    return metrics


def save_sampling(folder,seed,sampler,plans):
    summaries=read(HERE/'prepared'/f'seed_{seed}'/f'{sampler}_sampling.json')
    fields=[k for k in summaries[0] if k!='class_counts']
    write_csv(folder/'sampling_epochs.csv',fields+['label_'+str(c) for c in range(10)],
              [[*[r[k] for k in fields],*r['class_counts']] for r in summaries])
    write_csv(folder/'batch_label_counts.csv',['epoch','batch',*[f'label_{c}' for c in range(10)]],
              ([e+1,b+1,*cnt.tolist()] for e,cs in enumerate(plans['counts']) for b,cnt in enumerate(cs)))


def run_one(arm_name,seed):
    verify_freeze();check_control('data')
    arm=next(a for a in ARMS if a['name']==arm_name)
    assert seed in arm['seeds']
    cfg=configuration(arm,seed);run_id=f'{arm_name}_seed{seed}'
    folder=HERE/'runs'/run_id;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():
        verify_complete(folder,cfg);log(folder,'Verified complete; skip');return
    if (folder/'config.json').exists():assert read(folder/'config.json')==cfg
    else:atomic_json(folder/'config.json',cfg)
    plans=dict(np.load(HERE/'prepared'/f'seed_{seed}'/f"{arm['sampler']}_plans.npz"))
    save_sampling(folder,seed,arm['sampler'],plans)
    progress('queued',active_run=run_id,epoch=0,message='Waiting for FIFO GPU lock')
    log(folder,'QUEUED')
    with gpu_lock('data',run_id):
        started=time.monotonic();initialize(seed)
        if not torch.cuda.is_bf16_supported():raise RuntimeError('Required CUDA BF16 unavailable')
        device=torch.device('cuda')
        torch.cuda.reset_peak_memory_stats()
        audit,rows,source,train,val,labels=inputs()
        images=torch.from_numpy(source[train]).to(device)
        y=torch.as_tensor(labels,device=device)
        val_images=torch.from_numpy(source[val]).to(device)
        val_y=torch.as_tensor([int(rows[i]['label']) for i in val],device=device)
        aug=Augmentation(cfg,device)
        model=ResNet20(3).to(device=device,memory_format=torch.channels_last)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad)==cfg['trainable_parameters']
        optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=80)
        history=[];best_correct=-1;best_epoch=0;prior_seconds=0.;prior_peak=0;begin=1
        meta=dict(config=cfg,phase='running',run_id=run_id,pid=os.getpid(),created_at_utc=utc_now(),
                  code_sha256=code_hashes(),
                  code_amendment_sha256=sha256(HERE/'implementation_amendment.json') if (HERE/'implementation_amendment.json').exists() else None,
                  runtime_identity=runtime_identity(),
                  torch=str(torch.__version__),numpy=np.__version__,python=platform.python_version(),
                  cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),device=torch.cuda.get_device_name(),
                  training_autocast='bfloat16',evaluation_autocast='off',cudnn_deterministic=True,cudnn_benchmark=False,
                  initial_model_sha256=digest(torch.cat([p.detach().cpu().reshape(-1) for p in model.parameters()]).numpy()))
        if (folder/'last.pt').exists():
            last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
            assert last['config']==cfg and last['freeze_sha256']==sha256(HERE/'freeze.json')
            previous_code=last.get('code_sha256',read(HERE/'freeze.json')['code_sha256'])
            verify_checkpoint_code(previous_code, meta['code_sha256'], cfg)
            model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer']);scheduler.load_state_dict(last['scheduler'])
            restore_rng(last['rng']);history=last['history'];best_correct=last['best_correct'];best_epoch=last['best_epoch']
            begin=last['epoch']+1;prior_seconds=last['gpu_seconds'];prior_peak=last['peak_gpu_bytes']
            emit_history(folder,history);log(folder,f'RESUME at epoch {begin}')
            del last
        atomic_json(folder/'run.json',meta)
        try:
            for epoch in range(begin,81):
                check_control('data')
                progress('running',active_run=run_id,epoch=epoch-1,message=f'Training {arm_name} seed {seed}')
                t0=time.monotonic();lr=optimizer.param_groups[0]['lr'];model.train();aug.epoch(epoch)
                ix=torch.as_tensor(plans['indices'][epoch-1].astype(np.int64),device=device)
                oc=torch.as_tensor(plans['occurrences'][epoch-1].astype(np.int64),device=device)
                fl=torch.as_tensor(plans['flips'][epoch-1],device=device)
                total=torch.zeros(2,dtype=torch.float64,device=device)
                for start in range(0,40000,128):
                    idx=ix[start:start+128];occ=oc[start:start+128];flip=fl[start:start+128]
                    x=aug(images,idx,occ,flip);target=y[idx]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(x);loss=F.cross_entropy(logits,target)
                    loss.backward();optimizer.step()
                    total[0]+=loss.detach()*len(idx);total[1]+=(logits.detach().argmax(1)==target).sum()
                val_logits=predict(model,val_images,aug.mean,aug.std)
                val_loss=F.cross_entropy(val_logits,val_y).item()
                correct=int((val_logits.argmax(1)==val_y).sum().item())
                train_loss,train_acc=(total/40000).tolist()
                assert np.isfinite([train_loss,val_loss]).all(),f'Nonfinite epoch {epoch}'
                if correct>best_correct:
                    best_correct,best_epoch=correct,epoch
                    atomic_save(dict(config=cfg,epoch=epoch,validation_correct=correct,model=model.state_dict(),
                                     code_sha256=meta['code_sha256'],code_amendment_sha256=meta['code_amendment_sha256'],
                                     runtime_identity=meta['runtime_identity'],
                                     freeze_sha256=sha256(HERE/'freeze.json')),folder/'best.pt')
                scheduler.step()
                elapsed=time.monotonic()-t0
                history.append(dict(epoch=epoch,lr=lr,train_loss=train_loss,train_accuracy=train_acc,
                                    validation_loss=val_loss,validation_correct=correct,validation_accuracy=correct/5000,epoch_seconds=elapsed))
                state=dict(config=cfg,freeze_sha256=sha256(HERE/'freeze.json'),epoch=epoch,model=model.state_dict(),
                           code_sha256=meta['code_sha256'],code_amendment_sha256=meta['code_amendment_sha256'],
                           runtime_identity=meta['runtime_identity'],
                           optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=save_rng(),history=history,
                           best_correct=best_correct,best_epoch=best_epoch,gpu_seconds=prior_seconds+time.monotonic()-started,
                           peak_gpu_bytes=max(prior_peak,torch.cuda.max_memory_allocated()))
                atomic_save(state,folder/'last.pt');emit_history(folder,history)
                log(folder,f'{epoch:02d}/80 train={train_acc:.4f} val={correct}/5000 best={best_correct} seconds={elapsed:.2f}')
                progress('running',active_run=run_id,epoch=epoch,message=f'Best validation {best_correct}/5000; checkpoint saved')
                check_control('data')
            best=torch.load(folder/'best.pt',map_location=device,weights_only=False)
            model.load_state_dict(best['model'])
            result=predict(model,val_images,aug.mean,aug.std)
            predicted=result.argmax(1).cpu().numpy();truth=val_y.cpu().numpy()
            confidence=result.softmax(1).amax(1).cpu().numpy()
            assert int((predicted==truth).sum())==best_correct==best['validation_correct']
            confusion=np.bincount(truth*10+predicted,minlength=100).reshape(10,10)
            recalls=confusion.diagonal()/confusion.sum(1)
            f1=2*confusion.diagonal()/np.maximum(confusion.sum(0)+confusion.sum(1),1)
            write_csv(folder/'validation_predictions.csv',['id','label','prediction','confidence'],
                      ([rows[i]['id'],int(t),int(p),float(c)] for i,t,p,c in zip(val,truth,predicted,confidence)))
            gpu_seconds=prior_seconds+time.monotonic()-started
            peak=max(prior_peak,torch.cuda.max_memory_allocated())
            metrics=dict(complete=True,arm=arm_name,seed=seed,epochs=80,best_epoch=best_epoch,validation_n=5000,
                         validation_correct=best_correct,validation_accuracy=best_correct/5000,validation_loss=F.cross_entropy(result,val_y).item(),
                         macro_f1=float(f1.mean()),class_recall=recalls.tolist(),worst_class_recall=float(recalls.min()),
                         confusion_matrix=confusion.tolist(),training_presentations=3200000,updates=25040,
                         trainable_parameters=cfg['trainable_parameters'],macs_per_image=cfg['macs_per_image'],
                         gpu_seconds=gpu_seconds,peak_gpu_bytes=peak,best_checkpoint_sha256=sha256(folder/'best.pt'),
                         last_checkpoint_sha256=sha256(folder/'last.pt'),predictions_sha256=sha256(folder/'validation_predictions.csv'),
                         initial_model_sha256=meta['initial_model_sha256'],freeze_sha256=sha256(HERE/'freeze.json'))
            metrics.update(code_sha256=meta['code_sha256'],code_amendment_sha256=meta['code_amendment_sha256'])
            metrics['runtime_identity'] = meta['runtime_identity']
            meta.update(phase='completed',completed_at_utc=utc_now(),epochs=80,gpu_seconds=gpu_seconds,peak_gpu_bytes=peak)
            atomic_json(folder/'run.json',meta);atomic_json(folder/'metrics.json',metrics)
            verify_complete(folder,cfg);log(folder,f'COMPLETED: best epoch {best_epoch}, {best_correct}/5000')
        except (Paused,Cancelled) as exc:
            meta.update(phase='paused' if isinstance(exc,Paused) else 'cancelled',message=str(exc),updated_at_utc=utc_now())
            atomic_json(folder/'run.json',meta)
            progress('paused',active_run=run_id,epoch=history[-1]['epoch'] if history else 0,message=str(exc))
            log(folder,str(exc));raise
        except BaseException as exc:
            meta.update(phase='failed',message=str(exc),updated_at_utc=utc_now())
            atomic_json(folder/'run.json',meta);log(folder,traceback.format_exc())
            progress('failed',active_run=run_id,message=str(exc));raise
        finally:
            # Process exit is an additional safeguard; all GPU work stays in lock.
            del model,optimizer,scheduler,aug,images,y,val_images,val_y
            gc.collect();torch.cuda.empty_cache()
    progress('queued',active_run=None,epoch=0,message=f'{run_id} completed; next run will requeue')


def smoke():
    verify_prepared()
    progress('queued',active_run='data_smoke',epoch=0,message='GPU smoke waiting for shared lock')
    with gpu_lock('data','data_smoke'):
        initialize(42)
        assert torch.cuda.is_bf16_supported()
        _,rows,source,train,val,labels=inputs()
        images=torch.from_numpy(source[train]).cuda();y=torch.as_tensor(labels,device='cuda')
        plan=dict(np.load(HERE/'prepared/seed_42/A_plans.npz'))
        idx=torch.as_tensor(plan['indices'][0,:128].astype(np.int64),device='cuda')
        occ=torch.zeros(128,dtype=torch.int64,device='cuda');flip=torch.as_tensor(plan['flips'][0,:128],device='cuda')
        audit=read(ROOT/'baseline/data_audit.json');checks=[]
        for arm in [ARMS[10],*ARMS[11:]]:
            check_control('data')
            cfg=dict(arm,seed=42,normalization=dict(mean=audit['train_rgb_mean'],std=audit['train_rgb_std']))
            aug=Augmentation(cfg,torch.device('cuda'));aug.epoch(1)
            model=ResNet20(3).to(device='cuda',memory_format=torch.channels_last)
            x=aug(images,idx,occ,flip)
            assert x.shape==(128,3,32,32) and torch.isfinite(x).all()
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(model(x),y[idx])
            loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            model.eval()
            with torch.inference_mode():
                assert model(x).dtype==torch.float32
            checks.append(dict(arm=arm['name'],shape=list(x.shape),finite_forward_and_gradients=True))
            del model,aug,x,loss
        # Verify an actual save/reload reproduces the next SGD update, including
        # momentum and BN buffers, without treating this two-batch probe as a run.
        initialize(42)
        model=ResNet20(3).to(device='cuda',memory_format=torch.channels_last)
        optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=80)
        aug=Augmentation(dict(ARMS[10],seed=42,normalization=dict(mean=audit['train_rgb_mean'],std=audit['train_rgb_std'])),torch.device('cuda'))
        aug.epoch(1);x=aug(images,idx,occ,flip)
        def step():
            model.train();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(model(x),y[idx])
            loss.backward();optimizer.step()
        step();scheduler.step()
        atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=save_rng()),HERE/'smoke_resume.pt')
        step();expected={k:v.detach().clone() for k,v in model.state_dict().items()}
        saved=torch.load(HERE/'smoke_resume.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        # CPU RNG tensors must be loaded on CPU for torch.set_rng_state.
        saved_cpu=torch.load(HERE/'smoke_resume.pt',map_location='cpu',weights_only=False)
        restore_rng(saved_cpu['rng']);step()
        assert all(torch.equal(expected[k],v) for k,v in model.state_dict().items()),'Resume differs from uninterrupted SGD update'
        checks.append(dict(check='save_reload_next_update',bitwise_equal_model_and_BN=True,optimizer_momentum_restored=True,scheduler_restored=True))
        del model,optimizer,scheduler,aug,x,expected,saved,saved_cpu
        del images,y,idx,occ,flip
        gc.collect();torch.cuda.empty_cache()
        atomic_json(HERE/'gpu_smoke.json',dict(passed=True,checks=checks,created_at_utc=utc_now(),experimental_result=False))
    progress('implementing',active_run=None,epoch=0,message='GPU smoke passed; freezing plan before full training')


def all_runs():
    verify_freeze();verify_prepared()
    for entry in read(HERE/'manifest.json')['run_order']:
        check_control('data')
        folder=HERE/'runs'/entry['run_id']
        if (folder/'metrics.json').exists():
            verify_complete(folder);continue
        if entry['arm'] in ('B','B_tone'):
            cmd=[sys.executable,'-B','-u',str(HERE/'import_prior.py'),'--arm',entry['arm'],'--seed',str(entry['seed'])]
        else:
            cmd=[sys.executable,'-B','-u',str(HERE/'runner.py'),'one','--arm',entry['arm'],'--seed',str(entry['seed'])]
        completed=subprocess.run(cmd,cwd=ROOT)
        if completed.returncode!=0:
            check_control('data')
            raise RuntimeError(f"Run {entry['run_id']} exited {completed.returncode}; inspect preserved log/checkpoint")
        from analyze import analyze
        analyze(final=False)
    from analyze import analyze
    analyze(final=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['smoke','freeze','all','one'])
    parser.add_argument('--arm');parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    try:
        if args.action=='smoke':smoke()
        elif args.action=='freeze':freeze()
        elif args.action=='all':all_runs()
        else:run_one(args.arm,args.seed)
    except (Paused,Cancelled) as exc:
        progress('paused',message=str(exc));print(str(exc),flush=True);sys.exit(20 if isinstance(exc,Paused) else 21)
    except Exception:
        progress('failed',message=traceback.format_exc()[-1500:]);raise
