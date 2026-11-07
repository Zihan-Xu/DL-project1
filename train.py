"""Run one released configuration through the original, unchanged training code.

Run from the curated repository. Requires course data prepared by prepare_data.py
and CUDA with BF16 support. Outputs are kept apart from the released evidence.
"""
import argparse
import json
from pathlib import Path

import torch
from experiments.research_sweep import runtime
from experiments.research_sweep.combinations import runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('retraining'))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run = json.loads(args.config.read_text(encoding='utf-8'))
    assert run['spec']['recipe']['kd_temperature'] == 0, 'This release requires no teacher.'
    model = runner.methods.build(run['spec']['recipe'])
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == run['spec']['parameters'] <= 5_000_000
    if args.check_only:
        print(json.dumps({'passed': True, 'parameters': parameters, 'run_id': run['run_id']}))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('The original training implementation requires CUDA; inference also supports CPU.')
    data, audit = runner.load_data()
    runner.HERE = args.output_dir.resolve()
    runner.HERE.mkdir(parents=True, exist_ok=True)
    runner.write_json(runner.HERE / 'manifest.json', {'runs': [run]})
    # Use the original optimizer, scheduler, augmentation, seed streams and checkpointing.
    with runtime.gpu_lock('combinations', run['run_id']):
        runner.train(run, data, audit)
    best_path = runner.HERE / 'runs' / run['run_id'] / 'best.pt'
    saved = torch.load(best_path, map_location='cpu', weights_only=True)
    torch.save({'model_state': saved['model_state'], 'recipe': run['spec']['recipe'],
                'mean': audit['train_rgb_mean'], 'std': audit['train_rgb_std'],
                'seed': run['seed'], 'best_epoch': saved['epoch'], 'parameters': parameters,
                'source_checkpoint_sha256': runner.sha(best_path)}, runner.HERE / 'model.pt')
    print(json.dumps({'portable_checkpoint': str(runner.HERE / 'model.pt')}))


if __name__ == '__main__':
    main()
