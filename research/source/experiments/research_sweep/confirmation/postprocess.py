"""Fixed two-view inference evaluation on selected raw-validation checkpoints."""
import time
import numpy as np
import torch
from experiments.research_sweep import runtime
from experiments.research_sweep.confirmation import runner, methods


def evaluate_all():
    manifest = runner.verify_freeze()
    assert len(list((runner.HERE / 'runs').glob('*/metrics.json'))) == 30
    data, audit = runner.load_data()
    with runtime.gpu_lock('confirmation', 'fixed_flip_tta'):
        runner.configure_gpu(20261106)
        images = torch.from_numpy(data['validation'][0]).cuda()
        labels = torch.from_numpy(data['validation'][1]).cuda()
        norm = [torch.tensor(v, device='cuda').view(1, 3, 1, 1)
                for v in (audit['train_rgb_mean'], audit['train_rgb_std'])]
        for index, run in enumerate(manifest['runs']):
            runtime.check_control('confirmation')
            folder = runner.HERE / 'runs' / run['run_id']
            metrics = runner.verify_completed(folder, run)
            target = folder / 'tta_metrics.json'
            if target.exists():
                old = runner.read(target)
                assert old['config_sha256'] == run['config_sha256']
                assert old['checkpoint_sha256'] == runner.sha(folder / 'best.pt')
                assert old['raw_metrics_sha256'] == runner.sha(folder / 'metrics.json')
                for name, expected in old['artifact_sha256'].items():
                    assert runner.sha(folder / name) == expected
                continue
            runner.status('evaluating_tta', active_run=run['run_id'], epoch=0,
                          message=f'Fixed inference evaluation {index + 1}/30; training complete')
            saved = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
            assert saved['config'] == run
            net = methods.build(run['spec']['recipe']).cuda().to(memory_format=torch.channels_last)
            net.load_state_dict(saved['model_state'], strict=True)
            torch.cuda.synchronize(); start = time.monotonic()
            raw_logits = methods.predict(net, images, norm)
            torch.cuda.synchronize(); raw_seconds = time.monotonic() - start
            assert np.array_equal(raw_logits.cpu().numpy(), np.load(folder / 'validation_logits.npy'))
            torch.cuda.synchronize(); start = time.monotonic()
            # NHWC width dimension; this is an image reflection, not a vertical flip.
            flip_logits = methods.predict(net, images.flip(2), norm)
            probs = .5 * (raw_logits.softmax(1) + flip_logits.softmax(1))
            torch.cuda.synchronize(); flip_seconds = time.monotonic() - start
            assert probs.shape == (5000, 10) and torch.isfinite(probs).all()
            assert torch.allclose(probs.sum(1), torch.ones(5000, device='cuda'), atol=5e-7, rtol=0)
            preds = probs.argmax(1)
            confusion = torch.bincount(labels * 10 + preds, minlength=100).reshape(10, 10).cpu().numpy()
            tp = confusion.diagonal(); recall = tp / confusion.sum(1)
            f1 = 2 * tp / (confusion.sum(0) + confusion.sum(1))
            confidence = probs.amax(1).cpu().tolist()
            runner.atomic(folder / 'validation_flip_logits.npy',
                          lambda f: np.save(f, flip_logits.cpu().numpy()), binary=True)
            runner.atomic(folder / 'tta_probabilities.npy',
                          lambda f: np.save(f, probs.cpu().numpy()), binary=True)
            runner.write_csv(folder / 'tta_predictions.csv', [
                dict(id=row['id'], label=int(row['label']), prediction=pred, confidence=conf)
                for row, pred, conf in zip(data['validation'][2], preds.cpu().tolist(), confidence)])
            result = dict(run_id=run['run_id'], configuration=run['spec']['name'], seed=run['seed'],
                          config_sha256=run['config_sha256'], policy='mean_softmax_original_horizontal_flip',
                          validation_n=5000, validation_correct=int(tp.sum()),
                          validation_accuracy=float(tp.sum() / 5000), macro_f1=float(f1.mean()),
                          class_recall=recall.tolist(), class_f1=f1.tolist(),
                          worst_class_recall=float(recall.min()), confusion_matrix=confusion.tolist(),
                          validation_nll=float(-probs[torch.arange(5000, device='cuda'), labels].log().mean()),
                          parameters=metrics['parameters'], inference_macs=2 * metrics['macs'], views=2,
                          raw_inference_seconds=raw_seconds, tta_inference_seconds=raw_seconds + flip_seconds,
                          raw_logits_replay_bitwise_identical=True,
                          checkpoint_sha256=runner.sha(folder / 'best.pt'),
                          raw_metrics_sha256=runner.sha(folder / 'metrics.json'),
                          evaluated_at_utc=runtime.utc_now(),
                          artifact_sha256={name: runner.sha(folder / name) for name in
                                           ['validation_flip_logits.npy', 'tta_probabilities.npy', 'tta_predictions.csv']})
            runner.write_json(target, result)
            runner.event('Fixed flip inference evaluated', run_id=run['run_id'],
                         raw_correct=metrics['validation_correct'], tta_correct=result['validation_correct'])
            del net, saved, raw_logits, flip_logits, probs
