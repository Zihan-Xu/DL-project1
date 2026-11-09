"""Audit protocol identity before importing separately authorized finished runs."""
import csv
import io

import numpy as np
import torch

from study import HERE,ROOT,read,inputs,sha256,atomic_json,utc_now
from runner import Augmentation
from run_color_balance import STUDY,epoch_plan,PairedSampling,tone_features
from run_baseline import normalizer


def check():
    _,rows,source,train,val,labels=inputs()
    image=torch.from_numpy(source[train])
    colors=np.load(HERE/'prepared/colors.npz')
    old_edges=read(STUDY/'prepared.json')['tone_bin_edges']
    assert np.array_equal(np.searchsorted(old_edges,tone_features(source,train),side='right'),colors['train_bins'][:,0])
    assert np.array_equal(np.searchsorted(old_edges,tone_features(source,val),side='right'),colors['validation_bins'][:,0])
    results=[]
    execution=read(STUDY/'execution_sources.json')
    for name,h in execution.items():assert sha256(ROOT/name)==h
    for seed in (42,43,44):
        prior_norm=read(STUDY/f'seed_{seed}'/'normalization.json')
        current_norm=read(HERE/'prepared'/f'seed_{seed}'/'normalization.json')
        assert prior_norm['mean']==current_norm['mean'] and prior_norm['std']==current_norm['std']
        for arm,prior in [('B','B_label'),('B_tone','B_tone')]:
            chosen=np.load(STUDY/f'seed_{seed}'/prior/'indices.npy')
            plan=np.load(HERE/'prepared'/f'seed_{seed}'/f'{arm}_plans.npz')
            for epoch in range(1,81):
                idx,occ,flip,counts,rejected=epoch_plan(chosen,labels,seed,epoch)
                assert np.array_equal(idx,plan['indices'][epoch-1])
                assert np.array_equal(occ,plan['occurrences'][epoch-1])
                assert np.array_equal(flip,plan['flips'][epoch-1])
                assert np.array_equal(counts,plan['counts'][epoch-1])
            old=PairedSampling(chosen,labels,seed,torch.device('cpu'),csv.writer(io.StringIO()),csv.writer(io.StringIO()))
            new=Augmentation(dict(seed=seed,normalization=current_norm),torch.device('cpu'))
            for epoch in (1,2,80):
                new.epoch(epoch);batch=old.batches(epoch,torch.device('cpu'),128)
                for start in (0,128,256):
                    ix=next(batch)
                    oc=torch.as_tensor(plan['occurrences'][epoch-1,start:start+128].astype(np.int64))
                    fl=torch.as_tensor(plan['flips'][epoch-1,start:start+128])
                    previous=old(image,ix,normalizer(prior_norm['mean'],prior_norm['std'],torch.device('cpu')))
                    current=new(image,ix,oc,fl)
                    assert torch.equal(previous,current)
            results.append(dict(seed=seed,arm=arm,all_80_indices_occurrences_flips_counts_identical=True,
                                normalization_identical=True,normalized_cpu_inputs_bitwise_identical=True))
    report=dict(checked_at_utc=utc_now(),passed=True,comparison=results,prior_execution_sources=execution,
                train_and_validation_tone_bins_identical=True,
                training_implementation_equivalence=dict(
                    reviewed_functions=['model.ResNet20','run_baseline.train','run_color_balance.PairedSampling','runner.run_one','runner.Augmentation'],
                    initialization='same torch.manual_seed(seed), same ResNet20(3), fan_out and default BN/linear',
                    optimization='same SGD lr .1 momentum .9 WD .0001; zero_grad set_to_none; ordinary CE',
                    schedule='same CosineAnnealingLR T_max80, step once after each epoch validation',
                    precision='FP32 parameters/momentum, BF16 training autocast, FP32 validation, channels_last',
                    augmentation='exact all-epoch ID/occurrence flips; bitwise equal normalized CPU input probes',
                    budget='same 80*40000 presentations and 80*313 updates, 128 batch and 64 tail',
                    selection='same original 5000-image validation, highest correct count, first tied epoch',
                    validation='source run metadata and fresh lock-protected FP32 replay remain required per import'),
                limitations=['old optimizer/RNG/final model and training peak memory absent; completed-source receipt only'],
                decision_basis='protocol identity, not observed scores')
    atomic_json(HERE/'historical_compatibility.json',report)
    print('All six prior/new protocols and normalized input probes are identical',flush=True)


if __name__=='__main__':
    torch.set_num_threads(2);check()
