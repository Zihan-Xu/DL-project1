"""Reproduce one of the 30 research runs using the original 40k training loop."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import torch
from experiments.research_sweep import runtime
from experiments.research_sweep.confirmation import runner

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(4)
    task=Path(__file__).resolve().parents[1]
    run=next(r for r in json.loads((task/'evidence/research_runs.json').read_text()) if r['run_id']==args.run_id)
    assert run['spec']['recipe']['kd_temperature']==0
    net=runner.methods.build(run['spec']['recipe'])
    assert sum(p.numel() for p in net.parameters())==run['spec']['parameters']==2436346
    if args.check_only:print(json.dumps(dict(passed=True,run_id=args.run_id,parameters=2436346)));return
    assert not args.output_dir.resolve().is_relative_to(task)
    data,audit=runner.load_data()
    runner.HERE=args.output_dir.resolve();runner.HERE.mkdir(parents=True,exist_ok=True)
    with runtime.gpu_lock('confirmation',args.run_id):runner.train(run,data,audit)
    cp=torch.load(runner.HERE/'runs'/args.run_id/'best.pt',map_location='cpu',weights_only=True)
    torch.save(dict(model_state=cp['model_state'],recipe=run['spec']['recipe'],mean=audit['train_rgb_mean'],std=audit['train_rgb_std'],seed=run['seed'],epoch=cp['epoch'],parameters=2436346),runner.HERE/'model.pt')
    print(json.dumps(dict(model=str(runner.HERE/'model.pt'),validation_correct=cp['validation_correct'],validation_n=5000)))

if __name__=='__main__':main()
