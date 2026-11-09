"""Resumable, FIFO GPU-locked executor for the frozen E21--E34 matrix."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE))
import argparse
import copy
import csv
import gc
import hashlib
import json
import os
import platform
import random
import time
import traceback
import uuid
import numpy as np
import torch
from torch.nn import functional as F
from model import ResNet20
from prepare_data import CACHE, OUT, sha256
from experiments.research_sweep.runtime import atomic_json,check_control,gpu_lock,utc_now,write_status,Paused,Cancelled
from methods import (Classifier,sampler,augmentation_plan,augment,make_optimizer,make_scheduler,
                     train_step,update_average,recompute_bn,predict,model_cost)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def table(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))


def atomic_csv(path,rows,fields=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(temp,path)


def atomic_torch(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    torch.save(value,temp); os.replace(temp,path)


def mirror_last(source,destination):
    """Each output has a durable full resume state without multiplying disk writes."""
    if source==destination: return
    temp=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.tmp')
    os.link(source,temp); os.replace(temp,destination)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def set_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def source_fingerprint():
    names=['PLAN.md','manifest.json','methods.py','runner.py','preflight.py','report.py']
    values={n:sha256(HERE/n) for n in names}
    values.update(original_model=sha256(ROOT/'model.py'),data_paths=sha256(ROOT/'prepare_data.py'),split=sha256(OUT/'splits.csv'),
                  teacher=sha256(ROOT/'experiments/width_sweep/3x/resnet20.pt'))
    return values


def freeze():
    path=HERE/'freeze.json'; hashes=source_fingerprint()
    assert hashes['teacher']=='aa64543eba16b580a5a6f6bb8b88d31afeff1156eefcd7bd7b62f8e948b0d02a', 'Teacher changed from registered provenance'
    if path.exists():
        saved=read(path)
        if saved['hashes']!=hashes:
            raise RuntimeError('Frozen source changed; preserve failure evidence and record a reviewed implementation amendment')
        return saved
    preflight=read(HERE/'preflight.json')
    assert preflight['passed'] and preflight['source_fingerprint']==hashes
    value=dict(created_at_utc=utc_now(),hashes=hashes,plan_sha256=hashes['PLAN.md'],manifest_sha256=hashes['manifest.json'])
    atomic_json(path,value); return value


def load_cpu():
    audit=read(OUT/'data_audit.json')
    assert sha256(OUT/'splits.csv')==audit['split_sha256']
    rows=table(OUT/'splits.csv')
    raw=np.load(CACHE/'train.npy',mmap_mode='r')
    result={}
    for split,n in [('train',40000),('validation',5000)]:
        indices=[i for i,r in enumerate(rows) if r['split']==split]
        rr=[rows[i] for i in indices]
        labels=np.array([int(r['label']) for r in rr],dtype=np.int64)
        assert len(rr)==n and np.all(np.bincount(labels,minlength=10)==n//10)
        result[split]=dict(images=np.array(raw[indices]),labels=labels,rows=rr)
    assert not ({r['id'] for r in result['train']['rows']} & {r['id'] for r in result['validation']['rows']})
    return audit,result


def completeness(manifest):
    complete=[];failed=[]
    for run in manifest['runs']:
        folder=HERE/run['run_id']
        if (folder/'metrics.json').exists():
            m=read(folder/'metrics.json')
            required=['config.json','run.json','history.csv','best.pt','last.pt','validation_predictions.csv']
            if m.get('complete') and all((folder/f).exists() for f in required): complete.append(run['run_id'])
        if (folder/'failure.json').exists() and run['run_id'] not in complete: failed.append(run['run_id'])
    return complete,failed


def status(manifest,phase,**kwargs):
    done,failed=completeness(manifest)
    write_status('training',phase=phase,total_runs=manifest['total_runs'],completed_runs=len(done),
                 failed_runs=len(failed),total_trajectories=manifest['total_trajectories'],
                 report_path=str(HERE/'REPORT.md'),**kwargs)


def metrics_from_logits(logits,labels):
    pred=logits.argmax(1)
    counts=torch.bincount(labels*10+pred,minlength=100).reshape(10,10).cpu().numpy()
    tp=np.diag(counts); recall=tp/np.maximum(counts.sum(1),1)
    f1=2*tp/np.maximum(counts.sum(1)+counts.sum(0),1)
    return dict(validation_n=int(counts.sum()),validation_correct=int(tp.sum()),
                validation_accuracy=float(tp.sum()/counts.sum()),nll=float(F.cross_entropy(logits,labels).item()),
                macro_f1=float(f1.mean()),class_recall=recall.tolist(),worst_class_recall=float(recall.min()),
                confusion_matrix=counts.tolist())


def run_trajectory(trajectory,manifest,frozen,audit,cpu):
    seed=trajectory['seed']; by_id={c['id']:c for c in manifest['configs']}
    spec=by_id[trajectory['config']]; cfg=spec['recipe']; name=spec['id']
    run_id=f'{name}/seed_{seed}'; folder=HERE/run_id
    outputs=[by_id[x] for x in trajectory['outputs']]
    output_folders={c['id']:HERE/c['id']/f'seed_{seed}' for c in outputs}
    done,_=completeness(manifest)
    if all(f"{c['id']}/seed_{seed}" in done for c in outputs): return
    check_control('training')
    status(manifest,'queued',active_run=run_id,epoch=0,message='Waiting for FIFO GPU ticket')
    print(f'QUEUED {run_id}',flush=True)
    with gpu_lock('training',run_id):
        check_control('training')
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        # Keep the same FP32 policy as the historical project default.
        torch.backends.cuda.matmul.allow_tf32=False
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        assert torch.cuda.is_bf16_supported()
        torch.cuda.reset_peak_memory_stats()
        images=torch.from_numpy(cpu['train']['images']).to('cuda')
        labels=torch.from_numpy(cpu['train']['labels']).to('cuda')
        val_images=torch.from_numpy(cpu['validation']['images']).to('cuda')
        val_labels=torch.from_numpy(cpu['validation']['labels']).to('cuda')
        mean,std=[torch.tensor(audit[k],device='cuda').view(1,3,1,1) for k in ('train_rgb_mean','train_rgb_std')]
        model=Classifier(cfg).to(device='cuda',memory_format=torch.channels_last)
        cost=model_cost(cfg)
        assert cost['trainable_parameters']==spec['parameters'] and spec['parameters']+spec['teacher_parameters']<=5_000_000
        optimizer=make_optimizer(model,cfg); scheduler=make_scheduler(optimizer,cfg)
        teacher=None
        if cfg['kd_temperature']:
            teacher=ResNet20(3).to(device='cuda',memory_format=torch.channels_last)
            ckpt=torch.load(ROOT/'experiments/width_sweep/3x/resnet20.pt',map_location='cpu',weights_only=True)
            assert ckpt['split_sha256']==audit['split_sha256']
            teacher.load_state_dict(ckpt['model_state']); teacher.eval(); teacher.requires_grad_(False)
            assert not any(p.requires_grad for p in teacher.parameters())
        averages={}
        for c in outputs:
            if c['output'].startswith('ema') or c['output']=='swa':
                averages[c['id']]=copy.deepcopy(model).requires_grad_(False).eval()
        eval_model=copy.deepcopy(model).requires_grad_(False).eval() if averages else None
        histories={c['id']:[] for c in outputs}
        bests={c['id']:dict(correct=-1,epoch=0) for c in outputs}
        times=dict(training=0.,online_validation=0.,average_update=0.,checkpoint_io=0.)
        for c in outputs: times[c['id']+'_bn']=0.; times[c['id']+'_validation']=0.
        epoch0=0;swa_count=0;state=None
        cfg_hash=digest(dict(spec=spec,seed=seed,outputs=trajectory['outputs']))
        meta=dict(config=cfg,seed=seed,config_sha256=cfg_hash,split_sha256=audit['split_sha256'],
                  code_hashes=frozen['hashes'],plan_sha256=frozen['plan_sha256'],
                  mean=audit['train_rgb_mean'],std=audit['train_rgb_std'],**cost,
                  python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
                  cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),device=torch.cuda.get_device_name(),
                  training_autocast='bfloat16',evaluation_autocast='off',cudnn_deterministic=True,cudnn_benchmark=False,
                  trajectory=run_id,outputs=trajectory['outputs'],teacher_parameters=spec['teacher_parameters'],
                  teacher_sha256=frozen['hashes']['teacher'] if teacher is not None else None,
                  started_at_utc=utc_now(),peak_memory_bytes=0)
        for c in outputs:
            out=output_folders[c['id']];out.mkdir(parents=True,exist_ok=True)
            config=dict(c,seed=seed)
            if (out/'config.json').exists(): assert read(out/'config.json')==config
            else: atomic_json(out/'config.json',config)
        # Global CUDA RNG is solely for stochastic network layers after initialization.
        torch.manual_seed(seed+5005)
        if (folder/'last.pt').exists():
            ckpt=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
            assert ckpt['config_sha256']==cfg_hash and ckpt['code_hashes']==frozen['hashes']
            assert ckpt['split_sha256']==audit['split_sha256']
            model.load_state_dict(ckpt['model_state']); optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            for key,avg in averages.items(): avg.load_state_dict(ckpt['averages'][key])
            set_rng(ckpt['rng']);epoch0=ckpt['epoch'];swa_count=ckpt['swa_count']
            histories=ckpt['histories'];bests=ckpt['bests'];times=ckpt['times'];meta=ckpt['metadata']
            print(f'RESUMED {run_id} at epoch {epoch0}',flush=True)
        for c in outputs:
            atomic_json(output_folders[c['id']]/'run.json',dict(meta,output=c['id'],resume_state_source=run_id+'/last.pt'))
            if histories[c['id']]:atomic_csv(output_folders[c['id']]/'history.csv',histories[c['id']])
        status(manifest,'running',active_run=run_id,epoch=epoch0,message='Training owns GPU lock')
        for epoch in range(epoch0+1,cfg['epochs']+1):
            check_control('training')
            epoch_started=time.monotonic()
            order,counts,swaps=sampler(cpu['train']['labels'],seed,epoch,cfg['batch_size'])
            flips,angles=augmentation_plan(seed,epoch,len(labels),cfg['views'],cfg['rotation_probability'])
            sample_hash=hashlib.sha256(order.tobytes()).hexdigest()
            aug_hash=hashlib.sha256(flips.tobytes()+angles.tobytes()).hexdigest()
            idx_all=torch.from_numpy(order).to('cuda'); flip_gpu=torch.from_numpy(flips).to('cuda')
            angle_gpu=torch.from_numpy(angles).to('cuda')
            model.train(); lr=optimizer.param_groups[0]['lr']
            totals=torch.zeros(4,dtype=torch.float64,device='cuda'); examples=0
            torch.cuda.synchronize(); t0=time.monotonic();average_seconds=0.
            for idx in idx_all.split(cfg['batch_size']):
                views=[augment(images,idx,mean,std,flip_gpu[v,idx],angle_gpu[v,idx] if cfg['rotation_degrees'] else None)
                       for v in range(cfg['views'])]
                x=views[0] if len(views)==1 else torch.cat(views)
                y=labels[idx].repeat(cfg['views'])
                loss,logits,ce,aux=train_step(model,optimizer,x,y,cfg,teacher)
                if name=='base':
                    ta=time.monotonic()
                    for key,avg in averages.items():update_average(avg,model,.99 if key=='ema099' else .999)
                    average_seconds+=time.monotonic()-ta
                totals[0]+=loss*len(y);totals[1]+=(logits.argmax(1)==y).sum()
                totals[2]+=ce*len(y);totals[3]+=aux*len(y);examples+=len(y)
            torch.cuda.synchronize();train_seconds=time.monotonic()-t0
            assert examples==40000*cfg['views']
            times['training']+=train_seconds;times['average_update']+=average_seconds
            train_loss,train_acc,train_ce,train_aux=(totals/examples).tolist()
            if not np.isfinite([train_loss,train_acc,train_ce,train_aux]).all(): raise FloatingPointError('Non-finite epoch')
            if cfg['scheduler']=='swa' and epoch>=61:
                swa_count+=1
                update_average(averages['swa'],model,1-1/swa_count)
            eval_results={}
            # Online trajectory always has an original-image validation record, including SWA warm phase.
            torch.cuda.synchronize();t0=time.monotonic()
            online_logits=predict(model,val_images,mean,std)
            torch.cuda.synchronize();online_seconds=time.monotonic()-t0
            times['online_validation']+=online_seconds
            online_metrics=metrics_from_logits(online_logits,val_labels)
            for c in outputs:
                key=c['id'];kind=c['output'];bn_seconds=0.;validation_seconds=0.
                eligible=cfg['scheduler']!='swa' or epoch>=61
                if kind=='online' or (cfg['scheduler']=='swa' and epoch<61):
                    candidate=model; logits=online_logits; values=online_metrics
                else:
                    candidate=eval_model
                    candidate.load_state_dict(model.state_dict() if kind=='swa_control' else averages[key].state_dict())
                    torch.cuda.synchronize(); t0=time.monotonic()
                    recompute_bn(candidate,images,mean,std)
                    torch.cuda.synchronize(); bn_seconds=time.monotonic()-t0
                    t0=time.monotonic();logits=predict(candidate,val_images,mean,std)
                    torch.cuda.synchronize();validation_seconds=time.monotonic()-t0
                    values=metrics_from_logits(logits,val_labels)
                times[key+'_bn']+=bn_seconds;times[key+'_validation']+=validation_seconds
                row=dict(epoch=epoch,lr=lr,train_loss=train_loss,train_accuracy=train_acc,ce_loss=train_ce,auxiliary_loss=train_aux,
                         validation_accuracy=values['validation_accuracy'],validation_loss=values['nll'],
                         validation_correct=values['validation_correct'],selection_eligible=eligible,
                         online_validation_accuracy=online_metrics['validation_accuracy'],train_seconds=train_seconds,
                         bn_seconds=bn_seconds,extra_validation_seconds=validation_seconds,online_validation_seconds=online_seconds,
                         updates=len(counts),training_presentations=examples,min_batch_class_count=int(counts.min()),
                         sampler_swaps=swaps,sample_plan_sha256=sample_hash,augmentation_plan_sha256=aug_hash,
                         epoch_seconds=time.monotonic()-epoch_started)
                histories[key].append(row)
                if eligible and values['validation_correct']>bests[key]['correct']:
                    bests[key]=dict(correct=values['validation_correct'],epoch=epoch)
                    atomic_torch(output_folders[key]/'best.pt',dict(model_state=candidate.state_dict(),epoch=epoch,
                                 validation_accuracy=values['validation_accuracy'],output=key,**meta))
                eval_results[key]=values['validation_accuracy']
            scheduler.step()
            meta['peak_memory_bytes']=max(meta['peak_memory_bytes'],torch.cuda.max_memory_allocated())
            t0=time.monotonic()
            state=dict(model_state=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                       averages={k:v.state_dict() for k,v in averages.items()},swa_count=swa_count,rng=rng_state(),
                       epoch=epoch,histories=histories,bests=bests,times=times,metadata=meta,
                       config_sha256=cfg_hash,split_sha256=audit['split_sha256'],code_hashes=frozen['hashes'])
            atomic_torch(folder/'last.pt',state)
            for c in outputs:
                key=c['id'];out=output_folders[key]
                mirror_last(folder/'last.pt',out/'last.pt');atomic_csv(out/'history.csv',histories[key])
                atomic_json(out/'run.json',dict(meta,output=key,epoch=epoch,times=times,resume_state_source=run_id+'/last.pt'))
            times['checkpoint_io']+=time.monotonic()-t0
            status(manifest,'running',active_run=run_id,epoch=epoch,
                   message='; '.join(f'{k} val={v:.4%}' for k,v in eval_results.items()))
            print(f"{run_id} {epoch}/{cfg['epochs']} loss={train_loss:.4f} "+' '.join(f'{k}={v:.2%}' for k,v in eval_results.items())+f' {time.monotonic()-epoch_started:.1f}s',flush=True)
            check_control('training')
        # Best checkpoint replay on all 5000 validation originals, kept under the same GPU lock.
        for c in outputs:
            key=c['id'];out=output_folders[key]
            best=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(best['model_state']);logits=predict(model,val_images,mean,std)
            metrics=metrics_from_logits(logits,val_labels)
            assert metrics['validation_correct']==bests[key]['correct'] and best['epoch']==bests[key]['epoch']
            history=histories[key]
            eligible=[r for r in history if r['selection_eligible']]
            selected=max(eligible,key=lambda r:r['validation_correct'])
            assert selected['epoch']==best['epoch'] and len(history)==cfg['epochs']
            confidence=logits.softmax(1).amax(1).tolist();predictions=logits.argmax(1).tolist()
            prediction_rows=[dict(id=r['id'],label=int(r['label']),prediction=p,confidence=confidence[i])
                             for i,(r,p) in enumerate(zip(cpu['validation']['rows'],predictions))]
            atomic_csv(out/'validation_predictions.csv',prediction_rows)
            checkpoint_hash=sha256(out/'best.pt')
            metrics.update(complete=True,config_id=key,seed=seed,best_epoch=best['epoch'],
                           final_epoch_accuracy=history[-1]['validation_accuracy'],
                           **cost,teacher_parameters=spec['teacher_parameters'],
                           training_seconds=times['training'],online_validation_seconds=times['online_validation'],
                           bn_seconds=times[key+'_bn'],extra_validation_seconds=times[key+'_validation'],
                           shared_trajectory_seconds=sum(v for k,v in times.items() if k!='average_update'),
                           average_update_cpu_enqueue_seconds=times['average_update'],
                           peak_memory_bytes=meta['peak_memory_bytes'],training_updates=sum(r['updates'] for r in history),
                           training_presentations=sum(r['training_presentations'] for r in history),
                           trajectory=run_id,checkpoint_sha256=checkpoint_hash,split_sha256=audit['split_sha256'],
                           config_sha256=cfg_hash,completed_at_utc=utc_now())
            atomic_json(out/'run.json',dict(meta,output=key,epoch=cfg['epochs'],times=times,completed_at_utc=utc_now(),
                                          resume_state_source=run_id+'/last.pt'))
            atomic_json(out/'metrics.json',metrics)
            if (out/'failure.json').exists():
                os.replace(out/'failure.json',out/('resolved_failure_'+str(time.time_ns())+'.json'))
        del model,optimizer,scheduler,teacher,averages,eval_model,images,labels,val_images,val_labels,state
        gc.collect();torch.cuda.empty_cache()
    status(manifest,'queued',active_run=None,epoch=None,message=f'Completed {run_id}; GPU released, next trajectory requeues')
    print('COMPLETE '+run_id,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',nargs='*');parser.add_argument('--seed',type=int)
    parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args()
    manifest=read(HERE/'manifest.json')
    frozen=freeze()
    if args.freeze_only: print(json.dumps(frozen,indent=2));return
    audit,cpu=load_cpu()
    active=None
    try:
        for trajectory in manifest['trajectories']:
            if args.only and trajectory['config'] not in args.only:continue
            if args.seed is not None and trajectory['seed']!=args.seed:continue
            active=f"{trajectory['config']}/seed_{trajectory['seed']}"
            run_trajectory(trajectory,manifest,frozen,audit,cpu)
            from report import generate
            generate(final=False)
        done,failed=completeness(manifest)
        if len(done)==manifest['total_runs']:
            from report import generate
            generate(final=True)
            status(manifest,'completed',active_run=None,epoch=None,message='All registered outputs audited; report and paired statistics complete')
        else:
            status(manifest,'queued',active_run=None,epoch=None,message='Filtered executor finished; full matrix remains incomplete')
    except (Paused,Cancelled) as exc:
        status(manifest,'paused',active_run=active,message=str(exc),cancelled=isinstance(exc,Cancelled))
        print(str(exc),flush=True)
        return
    except BaseException as exc:
        if active:
            folder=HERE/active;folder.mkdir(parents=True,exist_ok=True)
            failure=dict(at_utc=utc_now(),error=repr(exc),traceback=traceback.format_exc())
            atomic_json(folder/('failure_evidence_'+str(time.time_ns())+'.json'),failure)
            atomic_json(folder/'failure.json',failure)
        status(manifest,'failed',active_run=active,message=repr(exc))
        traceback.print_exc();raise


if __name__=='__main__':main()
