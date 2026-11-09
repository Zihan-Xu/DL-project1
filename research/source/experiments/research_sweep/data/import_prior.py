"""Explicitly provenance-labeled import of six protocol-identical completed runs."""
import argparse
import gc
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from study import HERE,ROOT,ARMS,read,records,sha256,digest,inputs,write_csv,atomic_json,utc_now
from runner import (initialize,configuration,verify_freeze,verify_complete,progress,log,atomic_save,predict,save_sampling,emit_history)
from runner import code_hashes, runtime_identity
from model import ResNet20
from experiments.research_sweep.runtime import gpu_lock,check_control
from run_color_balance import epoch_plan as prior_plan


def import_one(arm,seed):
    verify_freeze();check_control('data')
    assert arm in ('B','B_tone') and seed in (42,43,44)
    cfg=configuration(next(a for a in ARMS if a['name']==arm),seed)
    run_id=f'{arm}_seed{seed}';folder=HERE/'runs'/run_id;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():verify_complete(folder,cfg);return
    prior_name='B_label' if arm=='B' else arm
    origin=ROOT/'experiments/balanced_sampling/color_tone'/f'seed_{seed}'/prior_name
    source_meta=read(origin/'run.json');source_cfg=read(origin/'config.json')
    assert source_meta['training_examples_seen']==3200000 and source_meta['training_seconds']>0
    assert source_meta['config']==source_cfg
    for key in ('seed','epochs','batch_size','learning_rate','momentum','weight_decay','width_multiplier','presentations_per_epoch'):
        assert source_cfg[key]==cfg[key],key
    assert source_cfg['crop_padding']==0 and source_cfg['horizontal_flip_probability']==.5
    assert source_cfg['augmentation']=={'name':'paired_horizontal_flip','probability':.5}
    assert source_meta['mean']==cfg['normalization']['mean'] and source_meta['std']==cfg['normalization']['std']
    assert source_meta['training_autocast']=='bfloat16' and source_meta['evaluation_autocast']=='off'
    assert source_meta['cudnn_deterministic'] and not source_meta['cudnn_benchmark']
    assert source_meta['split_sha256']==cfg['split_sha256'] and source_meta['trainable_parameters']==2436346
    prior_code=read(origin.parents[1]/'execution_sources.json')
    for name,h in prior_code.items():assert sha256(ROOT/name)==h
    plans=dict(np.load(HERE/'prepared'/f'seed_{seed}'/f'{arm}_plans.npz'))
    _,rows,source,train,val,labels=inputs()
    chosen=np.load(origin/'indices.npy')
    prior_sampling=records(origin/'sampling_epochs.csv')
    prior_counts=records(origin/'batch_label_counts.csv')
    assert len(prior_sampling)==80 and len(prior_counts)==80*313
    prior_total_flips=0
    for epoch in range(1,81):
        idx,occ,flips,counts,rejected=prior_plan(chosen,labels,seed,epoch)
        assert np.array_equal(idx,plans['indices'][epoch-1]) and np.array_equal(occ,plans['occurrences'][epoch-1])
        assert np.array_equal(flips,plans['flips'][epoch-1]) and np.array_equal(counts,plans['counts'][epoch-1])
        logged=prior_sampling[epoch-1]
        assert logged['indices_sha256']==digest(idx)
        assert logged['augmentation_sha256']==digest(np.column_stack((idx,occ,flips)))
        assert int(logged['presentations'])==40000 and int(logged['unique_images'])==20000
        logged_counts=np.array([[int(row[f'label_{c}']) for c in range(10)] for row in prior_counts[(epoch-1)*313:epoch*313]])
        assert np.array_equal(counts,logged_counts)
        prior_total_flips+=int(flips.sum())
    assert prior_total_flips==source_meta['augmented_examples']
    historical=records(origin/'history.csv');assert len(historical)==80
    best_row=max(historical,key=lambda r:float(r['validation_accuracy']))
    checkpoint=torch.load(origin/'resnet20.pt',map_location='cpu',weights_only=True)
    assert checkpoint['config']==source_cfg and checkpoint['split_sha256']==cfg['split_sha256']
    assert checkpoint['epoch']==int(best_row['epoch']) and checkpoint['validation_accuracy']==float(best_row['validation_accuracy'])
    source_hashes={p.name:sha256(p) for p in origin.iterdir() if p.is_file() and p.name in (
        'run.json','config.json','history.csv','indices.npy','resnet20.pt','sampling_epochs.csv','batch_label_counts.csv','subset.csv')}
    provenance=dict(kind='verified_protocol_identical_completed_import',source=str(origin),source_sha256=source_hashes,
                    replay_code_sha256=code_hashes(),replay_runtime_identity=runtime_identity(),
                    execution_sources=prior_code,all_80_epoch_plans_identical=True,
                    absent_original_artifacts=['last_epoch_model','optimizer_state','scheduler_state','RNG_state','training_peak_memory'],
                    source_training_seconds=source_meta['training_seconds'],decision_basis='protocol identity; not validation result')
    source_dir=folder/'original';source_dir.mkdir(exist_ok=True)
    for name,h in source_hashes.items():
        if not(source_dir/name).exists():shutil.copy2(origin/name,source_dir/name)
        assert sha256(source_dir/name)==h
    atomic_json(folder/'provenance.json',provenance);atomic_json(folder/'config.json',cfg)
    save_sampling(folder,seed,arm,plans)
    history=[dict(epoch=int(r['epoch']),lr=float(r['lr']),train_loss=float(r['train_loss']),train_accuracy=float(r['train_accuracy']),
                  validation_loss=float(r['validation_loss']),validation_correct=round(float(r['validation_accuracy'])*5000),
                  validation_accuracy=float(r['validation_accuracy']),epoch_seconds=float(r['epoch_seconds'])) for r in historical]
    assert [r['epoch'] for r in history]==list(range(1,81))
    emit_history(folder,history)
    progress('queued',active_run=run_id,epoch=80,message='Protocol-identical historical run queued for fresh FP32 verification')
    log(folder,'IMPORT queued; full source and sampling compatibility passed')
    with gpu_lock('data',run_id+'_import_audit'):
        started=time.monotonic();initialize(seed);torch.cuda.reset_peak_memory_stats()
        model=ResNet20(3)
        initial_sha=digest(torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy())
        model=model.to(device='cuda',memory_format=torch.channels_last);model.load_state_dict(checkpoint['model_state'])
        images=torch.from_numpy(source[val]).cuda();truth=np.array([int(rows[i]['label']) for i in val])
        mean=torch.tensor(checkpoint['mean'],device='cuda').view(1,3,1,1)
        std=torch.tensor(checkpoint['std'],device='cuda').view(1,3,1,1)
        logits=predict(model,images,mean,std);predicted=logits.argmax(1).cpu().numpy()
        confidence=logits.softmax(1).amax(1).cpu().numpy();correct=int((predicted==truth).sum())
        assert correct/5000==checkpoint['validation_accuracy'], 'Imported score failed independent FP32 replay'
        confusion=np.bincount(truth*10+predicted,minlength=100).reshape(10,10)
        recalls=confusion.diagonal()/confusion.sum(1)
        f1=2*confusion.diagonal()/np.maximum(confusion.sum(0)+confusion.sum(1),1)
        write_csv(folder/'validation_predictions.csv',['id','label','prediction','confidence'],
                  ([rows[i]['id'],int(t),int(p),float(c)] for i,t,p,c in zip(val,truth,predicted,confidence)))
        atomic_save(dict(config=cfg,epoch=checkpoint['epoch'],validation_correct=correct,model=model.state_dict(),
                         freeze_sha256=sha256(HERE/'freeze.json'),provenance=provenance),folder/'best.pt')
        # Receipt only. It explicitly cannot be resumed and does not pretend to
        # contain a last-epoch model or unavailable historical optimizer state.
        atomic_save(dict(kind='completed_source_receipt',resumable=False,completed_epochs=80,config=cfg,
                         selected_epoch=checkpoint['epoch'],source_checkpoint_sha256=source_hashes['resnet20.pt'],
                         provenance=provenance,freeze_sha256=sha256(HERE/'freeze.json')),folder/'last.pt')
        audit_seconds=time.monotonic()-started
        metrics=dict(complete=True,arm=arm,seed=seed,epochs=80,best_epoch=checkpoint['epoch'],validation_n=5000,
                     validation_correct=correct,validation_accuracy=correct/5000,
                     validation_loss=F.cross_entropy(logits,torch.as_tensor(truth,device='cuda')).item(),
                     macro_f1=float(f1.mean()),class_recall=recalls.tolist(),worst_class_recall=float(recalls.min()),
                     confusion_matrix=confusion.tolist(),training_presentations=3200000,updates=25040,
                     trainable_parameters=2436346,macs_per_image=364660608,
                     gpu_seconds=source_meta['training_seconds']+audit_seconds,source_training_seconds=source_meta['training_seconds'],
                     additional_audit_seconds=audit_seconds,peak_gpu_bytes=None,audit_peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                     best_checkpoint_sha256=sha256(folder/'best.pt'),last_checkpoint_sha256=sha256(folder/'last.pt'),
                     predictions_sha256=sha256(folder/'validation_predictions.csv'),initial_model_sha256=initial_sha,
                     freeze_sha256=sha256(HERE/'freeze.json'),provenance='verified historical import; training peak and last optimizer state unavailable',
                     replay_code_sha256=provenance['replay_code_sha256'],replay_runtime_identity=provenance['replay_runtime_identity'])
        atomic_json(folder/'run.json',dict(config=cfg,phase='completed',provenance=provenance,completed_at_utc=utc_now(),
                                          source_training_seconds=source_meta['training_seconds'],additional_audit_seconds=audit_seconds))
        atomic_json(folder/'metrics.json',metrics);verify_complete(folder,cfg)
        log(folder,f'IMPORT VERIFIED: {correct}/5000, source best epoch {checkpoint["epoch"]}')
        check_control('data')
        del model,images,mean,std,logits
        gc.collect();torch.cuda.empty_cache()
    progress('queued',active_run=None,epoch=0,message=f'{run_id}: compatible import independently verified')


if __name__=='__main__':
    torch.set_num_threads(2)
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',required=True,type=int)
    args=p.parse_args();import_one(args.arm,args.seed)
