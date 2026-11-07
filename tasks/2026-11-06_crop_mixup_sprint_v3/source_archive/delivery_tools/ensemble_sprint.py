"""Registered secondary inference experiment, using the fixed primary winner."""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy import stats
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from delivery_tools.audit_sprint import read,rows,sha,put,inputs
from experiments.research_sweep import runtime
from experiments.research_sweep.combinations import methods
STUDY=ROOT/'experiments/research_sweep/deadline_sprint_20261106'

def register():
    plan=dict(registered_at_utc=runtime.utc_now(),seeds=[142,143,144],new_branch='Use the primary complete-study winner and its selected raw/tta policy; never choose the new recipe by ensemble outcomes.',reference_branch='long_111,240epochs,TTA,matched seed',mixture_weights=[.5,.5],parameters=4872692,planned_replays=3,
        selection='Recommend only if all three independent replays pass and the complete mean exceeds both primary winner and long_111/TTA mean0.9604. No weight or temperature search.',
        timing='Secondary plan registered after four primary runs, before any combination was evaluated. All six primary runs must pass acceptance first.',
        rules_url='https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/rules',rules_reviewed_at_utc='2026-11-07T04:10:00Z',
        rule_interpretation='Custom residual CNN components are allowed; count every parameter of both from-scratch residual branches against5M. No explicit ensemble prohibition was present; this is an interpretation of the architecture rule, not special staff approval.',
        publication='Keep new checkpoints/predictions local until competition deadline to avoid sharing between teams during the competition.',
        sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'delivery_tools/dual_residual.py']})
    path=STUDY/'ensemble_plan.json';assert not path.exists();put(path,plan);print(json.dumps(dict(registered=True,parameters=4872692)))

@torch.inference_mode()
def evaluate():
    plan=read(STUDY/'ensemble_plan.json')
    for p,h in plan['sources'].items():assert sha(ROOT/p)==h
    assert read(STUDY/'ACCEPTANCE.json')['passed']
    d=read(STUDY/'DECISION.json');images,labels,ids,norm=inputs();results=[]
    torch.set_num_threads(2)
    for seed in plan['seeds']:
        new=STUDY/f"runs/{d['configuration']}_seed{seed}";old=ROOT/f'experiments/research_sweep/confirmation/runs/long_111_seed{seed}'
        old_meta=read(old/'tta_metrics.json')
        assert sha(old/'best.pt')==old_meta['checkpoint_sha256']
        assert sha(old/'tta_probabilities.npy')==old_meta['artifact_sha256']['tta_probabilities.npy']
        oldrows=rows(old/'tta_predictions.csv');assert [r['id'] for r in oldrows]==ids and np.array_equal(np.array([int(r['label']) for r in oldrows]),labels)
        original=np.load(old/'tta_probabilities.npy');saved=np.load(new/'inference.npz')
        fresh=saved['tta_probabilities'] if d['policy']=='tta' else torch.from_numpy(saved['raw_logits']).softmax(1).numpy()
        expected=(original+fresh)/2
        def load(folder):
            cp=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            net=methods.build(cp['config']['spec']['recipe']).to(memory_format=torch.channels_last).eval();net.load_state_dict(cp['model_state'],strict=True);return net
        a,b=load(new),load(old);assert sum(p.numel() for net in [a,b] for p in net.parameters())==4872692
        outputs=[];start=time.monotonic()
        for pos in range(0,5000,128):
            x=torch.from_numpy(images[pos:pos+128]).permute(0,3,1,2).float()/255;x=((x-norm[0])/norm[1]).contiguous(memory_format=torch.channels_last)
            pa=a(x).softmax(1)
            if d['policy']=='tta':pa=(pa+a(x.flip(-1)).softmax(1))/2
            pb=(b(x).softmax(1)+b(x.flip(-1)).softmax(1))/2
            outputs.append(((pa+pb)/2).numpy())
        actual=np.concatenate(outputs);assert np.allclose(actual,expected,atol=2e-4,rtol=2e-4)
        assert np.array_equal(actual.argmax(1),expected.argmax(1))
        record=dict(seed=seed,old_correct=int((original.argmax(1)==labels).sum()),new_correct=int((fresh.argmax(1)==labels).sum()),combined_correct=int((actual.argmax(1)==labels).sum()),n=5000,parameters=4872692,passed=True,new_checkpoint_sha256=sha(new/'best.pt'),reference_checkpoint_sha256=sha(old/'best.pt'),max_abs_error=float(np.max(np.abs(actual-expected))),elapsed_seconds=time.monotonic()-start)
        np.savez_compressed(STUDY/f'ensemble_seed{seed}.npz',probabilities=actual,labels=labels)
        record['probabilities_sha256']=sha(STUDY/f'ensemble_seed{seed}.npz');put(STUDY/f'ensemble_seed{seed}.json',record);results.append(record);print(json.dumps(record),flush=True)
    scores=np.array([r['combined_correct']/5000 for r in results]);differences=np.array([(r['combined_correct']-r['old_correct'])/50 for r in results])
    margin=float(stats.t.ppf(.975,2)*differences.std(ddof=1)/math.sqrt(3))
    result=dict(passed=True,completed_verified=3,planned_runs=3,parameters=4872692,new_configuration=d['configuration'],new_policy=d['policy'],reference_configuration='long_111',reference_policy='tta',mixture_weights=[.5,.5],mean_accuracy=float(scores.mean()),sd=float(scores.std(ddof=1)),paired_delta_pp=float(differences.mean()),paired_ci95_pp=[float(differences.mean()-margin),float(differences.mean()+margin)],recommended=bool(scores.mean()>max(.9604,d['mean_accuracy'])),runs=results,plan_sha256=sha(STUDY/'ensemble_plan.json'),validated_at_utc=runtime.utc_now(),official_kaggle_score=None)
    put(STUDY/'ENSEMBLE_ACCEPTANCE.json',result);print(json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','watch','evaluate']);a=p.parse_args()
    if a.action=='register':register();return
    if a.action=='watch':
        while not (STUDY/'ACCEPTANCE.json').exists():time.sleep(10)
    evaluate()
if __name__=='__main__':main()
