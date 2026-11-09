"""Frozen train-only input and complete sampling plans for E09–E20 (CPU only)."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from model import ResNet20
from prepare_data import CACHE, sha256
from run_color_balance import choose_tone_subset, training_statistics
from experiments.research_sweep.runtime import atomic_json, utc_now, write_status

SEEDS = [42, 43, 44]
FIVE = [42, 43, 44, 45, 46]
ARMS = [
    dict(name='A', experiments=['E09', 'E12'], sampler='A', series='C', seeds=FIVE),
    dict(name='B', experiments=['E09', 'E10', 'E13', 'E14', 'E15'], sampler='B', series='C', seeds=SEEDS),
    dict(name='C', experiments=['E09'], sampler='C', series='C', seeds=SEEDS),
    dict(name='D', experiments=['E11'], sampler='D', series='C', seeds=SEEDS),
    dict(name='E', experiments=['E10', 'E11', 'E12'], sampler='E', series='C', seeds=SEEDS),
    dict(name='F', experiments=['E12'], sampler='F', series='C', seeds=SEEDS),
    dict(name='G', experiments=['E12'], sampler='G', series='C', seeds=FIVE),
    *[dict(name='B_'+name, experiments=[eid], sampler='B_'+name, series='C', seeds=SEEDS)
      for name, eid in [('tone', 'E13'), ('saturation', 'E14'), ('brightness', 'E15')]],
    dict(name='full_flip', experiments=['E16', 'E17', 'E18', 'E19', 'E20'], sampler='A', series='full', seeds=FIVE),
    *[dict(name='rotation'+str(v), experiments=['E16'], sampler='A', series='full', seeds=SEEDS, rotation=v) for v in (5,10)],
    *[dict(name='cutout'+str(v), experiments=['E17'], sampler='A', series='full', seeds=FIVE if v==12 else SEEDS, cutout=v) for v in (8,12)],
    *[dict(name='grayscale'+str(v), experiments=['E18'], sampler='A', series='full', seeds=SEEDS, grayscale=v/100) for v in (10,20)],
    *[dict(name='background_'+name, experiments=[eid], sampler='A', series='full', seeds=SEEDS, background=name)
      for name,eid in [('manual750','E19'),('auto750','E19'),('perclass40','E20'),('perclass60','E20')]],
]
COMPARISONS = [('E09','B','A'),('E09','C','A'),('E09','C','B'),('E10','E','B'),('E11','E','D'),
               ('E12','F','E'),('E12','G','A'),('E13','B_tone','B'),('E14','B_saturation','B'),('E15','B_brightness','B')]
COMPARISONS += [(arm['experiments'][0], arm['name'], 'full_flip') for arm in ARMS[11:]]
COMPARISONS += [('E19','background_manual750','background_auto750'),
                ('E20','background_perclass40','background_manual750'),
                ('E20','background_perclass60','background_manual750'),
                ('E20','background_perclass60','background_perclass40')]
DEFAULTS = dict(architecture='CIFAR-ResNet-20-projection', width_multiplier=3, epochs=80, batch_size=128,
                learning_rate=.1, momentum=.9, weight_decay=.0001, flip=.5, crop=0, mixup=False,
                trainable_parameters=2436346, macs_per_image=364660608,
                presentations_per_epoch=40000, updates_per_epoch=313,
                train_precision='BF16', validation_precision='FP32')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def records(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)
    temp.replace(path)


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def inputs(verify_pixels=False):
    audit = read(ROOT/'baseline/data_audit.json')
    assert sha256(ROOT/'baseline/splits.csv') == audit['split_sha256']
    rows = records(ROOT/'baseline/splits.csv')
    source = np.load(CACHE/'train.npy', mmap_mode='r')
    train = np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    val = np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    labels = np.array([int(rows[i]['label']) for i in train])
    assert np.array_equal(np.bincount(labels), np.full(10,4000))
    assert len(val)==5000 and len(set(train)&set(val))==0
    if verify_pixels:
        for i in np.concatenate((train,val)):
            assert digest(source[i])==rows[i]['pixel_sha256'], rows[i]['id']
    return audit, rows, source, train, val, labels


def color_features(images):
    result=[]
    for start in range(0,len(images),256):
        rgb=images[start:start+256].astype(np.float64)/255
        avg=rgb.mean(axis=(1,2))
        vmax,vmin=rgb.max(axis=3),rgb.min(axis=3)
        sat=np.divide(vmax-vmin,vmax,out=np.zeros_like(vmax),where=vmax>0).mean(axis=(1,2))
        result.append(np.column_stack((avg[:,0]-avg[:,2],sat,avg@np.array([.2126,.7152,.0722]))))
    return np.concatenate(result)


def permutation(seed,cycle,labels):
    rng=np.random.default_rng(seed if cycle==0 else np.random.SeedSequence([seed,10,cycle]))
    for attempt in range(1000):
        p=rng.permutation(40000)
        if all(np.bincount(labels[h],minlength=10).min()*2>=313 for h in (p[:20000],p[20000:])):
            return p
    raise RuntimeError('D half cannot cover all classes per batch')


def fixed_selections(seed,labels,bins):
    p=permutation(seed,0,labels)
    orders=[p[labels[p]==c] for c in range(10)]
    selections={'A':np.arange(40000), 'G':np.arange(40000),
                'B':np.concatenate([o[:2000] for o in orders]),'C':np.concatenate([o[:1000] for o in orders])}
    for col,name in enumerate(('tone','saturation','brightness')):
        selections['B_'+name]=np.concatenate([choose_tone_subset(o,bins[:,col]) for o in orders])
    return selections


def selection(sampler,seed,epoch,labels,fixed):
    if sampler in fixed:
        return fixed[sampler]
    cycle,half=divmod(epoch-1,2)
    p=permutation(seed,cycle,labels)
    if sampler=='D':
        return p[half*20000:(half+1)*20000]
    return np.concatenate([p[labels[p]==c][half*2000:(half+1)*2000] for c in range(10)])


def flips(seed,epoch):
    return np.random.default_rng(np.random.SeedSequence([seed,200,epoch,0])).random((4,40000))<.5


def epoch_plan(sampler,seed,epoch,labels,fixed):
    chosen=selection(sampler,seed,epoch,labels,fixed)
    repetitions=40000//len(chosen)
    # Canonical tokens are class-major, occurrence-major, original-order within class.
    ids=np.concatenate([np.tile(chosen[labels[chosen]==c],repetitions) for c in range(10)])
    occurrences=np.concatenate([np.repeat(np.arange(repetitions,dtype=np.uint8),np.sum(labels[chosen]==c)) for c in range(10)])
    yl=labels[ids]
    rng=np.random.default_rng(np.random.SeedSequence([seed,100,epoch]))
    batch_ids=np.arange(40000)//128
    attempt=0
    if sampler in ('F','G'):
        class_slots=np.tile(rng.permutation(10),4000)
        order=np.empty(40000,dtype=np.int64)
        for c in range(10):
            order[class_slots==c]=rng.permutation(np.flatnonzero(yl==c))
        for start in range(0,40000,128):
            order[start:start+128]=rng.permutation(order[start:start+128])
        counts=np.bincount(batch_ids*10+yl[order],minlength=3130).reshape(313,10)
        assert np.all(counts.max(1)-counts.min(1)<=1)
    else:
        for attempt in range(1000):
            order=rng.permutation(40000)
            counts=np.bincount(batch_ids*10+yl[order],minlength=3130).reshape(313,10)
            if counts.min()>0:
                break
        else:
            raise RuntimeError('Could not make ten-class batches')
    idx=ids[order].astype(np.int32)
    occ=occurrences[order]
    f=flips(seed,epoch)[occ,idx]
    assert counts.min()>0 and counts[-1].sum()==64
    unique,freq=np.unique(idx,return_counts=True)
    assert len(unique)==len(chosen) and (freq==repetitions).all()
    assert ((idx>=0)&(idx<40000)).all()
    if sampler!='D':
        assert (counts.sum(0)==4000).all()
    return idx,occ,f,counts.astype(np.int16),attempt


def model_profile():
    model=ResNet20(3).eval()
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs=[]
    hooks=[]
    for m in model.modules():
        if isinstance(m,torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(lambda m,x,y:macs.append(y.numel()*m.in_channels//m.groups*m.kernel_size[0]*m.kernel_size[1])))
        elif isinstance(m,torch.nn.Linear):
            hooks.append(m.register_forward_hook(lambda m,x,y:macs.append(y.numel()*m.in_features)))
    with torch.inference_mode(): model(torch.zeros(1,3,32,32))
    for hook in hooks:hook.remove()
    assert params==2436346 and sum(macs)==364660608 and params<=5000000
    return params,sum(macs)


def background_selections(rows,train,labels):
    pool=read(ROOT/'output/augmentations/background_qa/pool.json')
    audit=read(ROOT/'baseline/data_audit.json')
    assert pool['split_sha256']==audit['split_sha256']
    for name,h in pool['cache_sha256'].items():
        assert sha256(ROOT/'.cache/augmentations'/name)==h
    id_to_idx={int(rows[i]['id']):k for k,i in enumerate(train)}
    accepted=records(ROOT/'output/augmentations/background_qa/accepted.csv')
    by_source={int(r['source_id']):r for r in accepted}
    valid=np.load(ROOT/'.cache/augmentations/valid.npy')
    assert valid.shape==(40000,) and valid.dtype==np.bool_
    for r in accepted:
        i,j=id_to_idx[int(r['source_id'])],id_to_idx[int(r['donor_id'])]
        assert labels[i]==int(r['source_label']) and labels[j]==int(r['donor_label']) and valid[i]
    assert set(np.flatnonzero(valid))=={id_to_idx[i] for i in by_source}
    kept=[r for r in records(ROOT/'experiments/paired_posttraining/pair_review.csv') if r['keep']=='1']
    manual=np.array([id_to_idx[int(r['source_id'])] for r in kept])
    for r in kept:
        a=by_source[int(r['source_id'])]
        assert int(a['donor_id'])==int(r['donor_id']) and int(a['source_label'])==int(r['label'])
    counts=np.bincount(labels[manual],minlength=10)
    assert counts.tolist()==[77,84,63,64,61,71,79,82,88,81]
    results={}
    for seed in SEEDS:
        rng=np.random.default_rng(np.random.SeedSequence([seed,300]))
        auto=np.concatenate([rng.permutation(np.flatnonzero(valid&(labels==c)))[:counts[c]] for c in range(10)])
        rng=np.random.default_rng(np.random.SeedSequence([seed,301]))
        orders=[rng.permutation(manual[labels[manual]==c]) for c in range(10)]
        selections=dict(manual750=manual,auto750=auto,perclass40=np.concatenate([x[:40] for x in orders]),
                        perclass60=np.concatenate([x[:60] for x in orders]))
        assert np.array_equal(np.bincount(labels[auto],minlength=10),counts)
        assert set(selections['perclass40'])<=set(selections['perclass60'])<=set(manual)
        for name,idx in selections.items():
            assert len(np.unique(idx))==len(idx) and valid[idx].all()
            out=HERE/'prepared'/f'seed_{seed}'
            out.mkdir(parents=True,exist_ok=True)
            np.save(out/f'background_{name}.npy',idx)
            write_csv(out/f'background_{name}.csv',['train_index','source_id','label','donor_id','mask_iou'],
                      [[int(i),rows[train[i]]['id'],int(labels[i]),by_source[int(rows[train[i]]['id'])]['donor_id'],
                        by_source[int(rows[train[i]]['id'])]['mask_iou']] for i in idx])
        results[str(seed)]={name:dict(n=len(idx),counts=np.bincount(labels[idx],minlength=10).tolist(),
                                      sha256=digest(idx),manual_overlap=len(set(idx)&set(manual))) for name,idx in selections.items()}
    assert len({v['auto750']['sha256'] for v in results.values()})==3
    return dict(pool=pool,selections=results,accepted_csv_sha256=sha256(ROOT/'output/augmentations/background_qa/accepted.csv'),
                review_csv_sha256=sha256(ROOT/'experiments/paired_posttraining/pair_review.csv'))


def prepare():
    HERE.mkdir(parents=True,exist_ok=True)
    if (HERE/'prepared.json').exists():
        verify_prepared()
        print('Prepared inputs verified',flush=True)
        return
    write_status('data',phase='implementing',total_runs=71,completed_runs=0,failed_runs=0,active_run=None,epoch=0,
                 message='Preparing and verifying all sampling plans',report_path=str(HERE/'REPORT.md'))
    params,macs=model_profile()
    manifest=dict(version='data-v1',category='data',created_at_utc=utc_now(),defaults=DEFAULTS,configurations=ARMS,
                  comparisons=[dict(experiment=e,treatment=t,control=c) for e,t,c in COMPARISONS],
                  total_runs=sum(len(a['seeds']) for a in ARMS),
                  run_order=[dict(arm=a['name'],seed=s,run_id=f"{a['name']}_seed{s}") for s in FIVE for a in ARMS if s in a['seeds']],
                  references=['P01:1512.03385','P08:1708.04552','experiments/balanced_sampling/DECISIONS.md'])
    atomic_json(HERE/'manifest.json',manifest)
    audit,rows,source,train,val,labels=inputs(True)
    feature=color_features(source[train]); val_feature=color_features(source[val])
    edges=np.quantile(feature,[1/3,2/3],axis=0)
    bins=np.stack([np.searchsorted(edges[:,c],feature[:,c],side='right') for c in range(3)],axis=1)
    val_bins=np.stack([np.searchsorted(edges[:,c],val_feature[:,c],side='right') for c in range(3)],axis=1)
    prepared=HERE/'prepared';prepared.mkdir(exist_ok=True)
    np.savez_compressed(prepared/'colors.npz',train_features=feature,validation_features=val_feature,edges=edges,
                        train_bins=bins,validation_bins=val_bins,labels=labels,
                        validation_labels=np.array([int(rows[i]['label']) for i in val]))
    background=background_selections(rows,train,labels)
    checks=[]; color_counts=[]; subset_hashes={}
    for seed in FIVE:
        folder=prepared/f'seed_{seed}';folder.mkdir(exist_ok=True)
        fixed=fixed_selections(seed,labels,bins)
        assert set(fixed['C'])<set(fixed['B'])<set(fixed['A'])
        for name,idx in fixed.items():
            np.save(folder/f'{name}_subset.npy',idx)
            if name.startswith('B'):
                assert set(fixed['C'])<=set(idx) and len(np.unique(idx))==20000
                for col,feature_name in enumerate(('tone','saturation','brightness')):
                    for c in range(10):
                        count=np.bincount(bins[idx[labels[idx]==c],col],minlength=3)
                        color_counts.append([seed,name,feature_name,c,*count.tolist(),*(count-np.array([666,667,667])).tolist()])
        mean,std=training_statistics(source,train[fixed['C']])
        atomic_json(folder/'normalization.json',dict(mean=mean,std=std,common_sha256=digest(fixed['C'])))
        subset_hashes[str(seed)]={k:digest(v) for k,v in fixed.items()}
        wanted=[a['sampler'] for a in ARMS if seed in a['seeds']]
        wanted=list(dict.fromkeys(wanted))
        epoch_pairs={}; plan_arrays={name:([],[],[],[]) for name in wanted}; summaries={name:[] for name in wanted}
        covered={name:set() for name in wanted}
        for epoch in range(1,81):
            current={}
            for name in wanted:
                idx,occ,f,counts,rejected=epoch_plan(name,seed,epoch,labels,fixed)
                current[name]=(idx,occ,f,counts)
                covered[name].update(idx.tolist())
                token=occ.astype(np.int64)*40000+idx
                plan_arrays[name][0].append(idx);plan_arrays[name][1].append(occ)
                plan_arrays[name][2].append(f);plan_arrays[name][3].append(counts)
                summaries[name].append(dict(epoch=epoch,unique_ids=len(np.unique(idx)),presentations=40000,
                    indices_sha256=digest(idx),occurrence_sha256=digest(occ),augmentation_sha256=digest(np.column_stack((idx,occ,f))),
                    token_multiset_sha256=digest(np.sort(token)),batch_counts_sha256=digest(counts),
                    class_counts=counts.sum(0).tolist(),cumulative_unique=len(covered[name]),
                    flipped=int(f.sum()),rejected_permutations=rejected,min_batch_class=int(counts.min()),max_batch_class=int(counts.max())))
                if name in ('D','E','F'):
                    if epoch%2:
                        epoch_pairs[name]=set(idx)
                    else:
                        assert len(epoch_pairs[name]|set(idx))==40000 and not(epoch_pairs[name]&set(idx))
                if name.startswith('B_'):
                    assert np.array_equal(counts,current['B'][3])
            for left,right in [('A','G'),('E','F')]:
                if left not in current:continue
                li,lo,lf,_=current[left];ri,ro,rf,_=current[right]
                lt=lo.astype(np.int64)*40000+li;rt=ro.astype(np.int64)*40000+ri
                lsort,rsort=np.argsort(lt),np.argsort(rt)
                assert np.array_equal(lt[lsort],rt[rsort]) and np.array_equal(lf[lsort],rf[rsort])
            if epoch==1 and 'E' in current: assert set(current['E'][0])==set(fixed['B'])
        for name,(ids,occ,flip,batch_counts) in plan_arrays.items():
            np.savez_compressed(folder/f'{name}_plans.npz',indices=np.stack(ids),occurrences=np.stack(occ),flips=np.stack(flip),counts=np.stack(batch_counts))
            atomic_json(folder/f'{name}_sampling.json',summaries[name])
            checks.append(dict(seed=seed,sampler=name,epochs_verified=80,ten_class_batches=80*313,
                               two_epoch_coverage_verified=name in ('D','E','F'),tail=64))
        print(f'Prepared seed {seed}: {len(wanted)} samplers x 80 epochs',flush=True)
    assert len({v['B'] for v in subset_hashes.values()})==5
    assert len({v['C'] for v in subset_hashes.values()})==5
    write_csv(prepared/'subset_color_counts.csv',['seed','arm','feature','label','low','middle','high','low_minus666','middle_minus667','high_minus667'],color_counts)
    # Independently reproduce full-train statistics and confirm they use train only.
    full_mean,full_std=training_statistics(source,train)
    assert np.allclose(full_mean,audit['train_rgb_mean'],rtol=0,atol=1e-10)
    assert np.allclose(full_std,audit['train_rgb_std'],rtol=0,atol=1e-10)
    atomic_json(HERE/'preflight.json',dict(passed=True,created_at_utc=utc_now(),parameters=params,macs=macs,
                cache_pixels_verified=45000,checks=checks,subset_hashes=subset_hashes,background=background,
                color_edges=edges.tolist(),full_mean=full_mean,full_std=full_std,seed_streams=dict(subset=10,batch=100,augmentation=200,background=300),
                all_80_epochs_verified=True,independent_occurrence_flips=True,paired_multisets_and_flips_verified=True))
    source_paths=[ROOT/'baseline/splits.csv',ROOT/'baseline/data_audit.json',ROOT/'model.py',
                  ROOT/'experiments/balanced_sampling/DECISIONS.md',ROOT/'research/2026-11-04/EXPERIMENT_ROADMAP.md',
                  ROOT/'output/augmentations/background_qa/accepted.csv',ROOT/'experiments/paired_posttraining/pair_review.csv',
                  ROOT/'.cache/augmentations/swaps.npy',ROOT/'.cache/augmentations/valid.npy']
    atomic_json(HERE/'prepared.json',dict(created_at_utc=utc_now(),split_sha256=audit['split_sha256'],
                file_sha256={str(p.relative_to(HERE)).replace('\\','/'):sha256(p) for p in sorted(prepared.rglob('*')) if p.is_file()},
                source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in source_paths}))
    print('All CPU preparation and preflight checks passed',flush=True)


def verify_prepared():
    obj=read(HERE/'prepared.json')
    for name,h in obj['file_sha256'].items(): assert sha256(HERE/name)==h,name
    for name,h in obj['source_sha256'].items(): assert sha256(ROOT/name)==h,name
    return obj


if __name__=='__main__':
    torch.set_num_threads(2)
    prepare()
