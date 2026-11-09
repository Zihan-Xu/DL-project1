"""Preregister, verify, run, resume, and report E01-E08. GPU use is serialized."""
import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback
import uuid

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from experiments.research_sweep import runtime
from experiments.research_sweep.architecture.models import build, complexity, PreBlock, BlurPool

CATEGORY = 'architecture'
SPLIT = ROOT / 'baseline/splits.csv'
CACHE = ROOT / '.cache/baseline/train.npy'
AUDIT = ROOT / 'baseline/data_audit.json'
TRAIN_SOURCES = ['models.py', 'runner.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_csv(path, records, fields=None):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(temp, path)


def save(path, value):
    temp = Path(path).with_name(Path(path).name + '.' + uuid.uuid4().hex + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def event(message, **fields):
    obj = dict(at_utc=runtime.utc_now(), message=message, **fields)
    with (HERE / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def counts():
    manifest = read(HERE / 'manifest.json') if (HERE / 'manifest.json').exists() else {'runs': []}
    done = sum((HERE / 'runs' / r['run_id'] / 'metrics.json').exists() for r in manifest['runs'])
    failed = len(list((HERE / 'failures').glob('*.json'))) if (HERE / 'failures').exists() else 0
    return dict(total_runs=len(manifest['runs']), completed_runs=done, failed_runs=failed)


def status(phase, **fields):
    runtime.write_status(CATEGORY, phase=phase, report_path=str(HERE / 'REPORT.md'), **counts(), **fields)


def table():
    specs = [
        ('baseline', list(range(1, 9)), (48, 96, 192), 3, 'plain', 5),
        ('depth32', [1], (48, 96, 192), 5, 'plain', 3),
        ('equal32', [2], (37, 74, 148), 5, 'plain', 3),
        ('equal44', [2], (31, 62, 124), 7, 'plain', 3),
        ('early', [3], (64, 96, 184), 3, 'plain', 3),
        ('late', [3], (32, 96, 200), 3, 'plain', 3),
        ('preact', [4], (48, 96, 192), 3, 'preact', 5),
        ('se8', [5], (48, 96, 192), 3, 'se8', 5),
        ('eca3', [6], (48, 96, 192), 3, 'eca3', 3),
        ('group4', [7], (48, 96, 192), 3, 'group4', 3),
        ('group4_wide', [7], (64, 128, 256), 3, 'group4', 3),
        ('blur', [8], (48, 96, 192), 3, 'blur', 3),
    ]
    return [dict(name=n, directions=[f'E{i:02}' for i in ids], widths=list(w), blocks=b,
                 variant=v, seeds=list(range(42, 42 + ns))) for n, ids, w, b, v, ns in specs]


def load_data():
    with SPLIT.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    audit = read(AUDIT)
    assert sha(SPLIT) == audit['split_sha256']
    raw = np.load(CACHE, mmap_mode='r')
    assert raw.shape == (50000, 32, 32, 3) and raw.dtype == np.uint8
    data = {}
    for split, expected in [('train', 40000), ('validation', 5000)]:
        indices = [i for i, row in enumerate(rows) if row['split'] == split]
        selected = [rows[i] for i in indices]
        y = np.array([int(row['label']) for row in selected], dtype=np.int64)
        assert len(y) == expected and np.all(np.bincount(y, minlength=10) == expected // 10)
        data[split] = (np.array(raw[indices]), y, selected)
    return data, audit


def epoch_plan(labels, batch_rng, aug_rng):
    """Paired permutations and flip flags independent of model RNG consumption."""
    rejected = 0
    while True:
        order = batch_rng.permutation(len(labels))
        if all(np.unique(labels[order[i:i+128]]).size == 10 for i in range(0, len(order), 128)):
            break
        rejected += 1
    flips = aug_rng.random(len(labels)) < .5
    return order, flips, rejected


def cpu_preflight(configs, data):
    checks = []
    from model import ResNet20
    torch.random.default_generator.manual_seed(4242)
    reference = ResNet20(3).eval()
    torch.random.default_generator.manual_seed(4242)
    control = build(configs[0]).eval()
    assert list(reference.state_dict()) == list(control.state_dict())
    assert all(torch.equal(v, control.state_dict()[k]) for k, v in reference.state_dict().items())
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(reference(x), control(x))
    checks.append('exact historical control state/output parity at fixed initialization seed')
    for config in configs:
        net = build(config)
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                assert torch.equal(layer.weight, torch.ones_like(layer.weight))
                assert torch.equal(layer.bias, torch.zeros_like(layer.bias))
        loss = F.cross_entropy(net(x), torch.tensor([0, 1]))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    checks.append('all architectures BN defaults, finite forward/backward and connected gradients')
    same = PreBlock(4, 4, 1).eval()
    same.conv2.weight.data.zero_()
    inp = torch.randn(2, 4, 8, 8)
    assert torch.equal(same(inp), inp)
    proj = PreBlock(4, 8, 2).eval()
    proj.conv2.weight.data.zero_()
    assert torch.equal(proj(inp), proj.shortcut(F.relu(proj.bn1(inp))))
    first = PreBlock(4, 4, 1, first=True).eval()
    first.conv2.weight.data.zero_()
    assert torch.equal(first(inp), F.relu(first.bn1(inp)))
    checks.append('preact identity, shared first activation and preactivated projection semantics')
    blur = BlurPool(4)
    assert len(list(blur.parameters())) == 0
    assert torch.equal(blur(torch.ones(2, 4, 32, 32)), torch.ones(2, 4, 16, 16))
    dense = F.conv2d(F.pad(inp, (1, 1, 1, 1), mode='reflect'), blur.kernel, groups=4)
    assert torch.allclose(blur(inp), dense[:, :, ::2, ::2], atol=1e-6)
    assert blur(blur(torch.ones(1, 4, 32, 32))).shape == (1, 4, 8, 8)
    checks.append('blur normalized fixed kernel, constants, even sampling phase and 32→16→8')
    y = data['train'][1]
    for seed in range(42, 47):
        a, b = np.random.default_rng(seed+10000), np.random.default_rng(seed+20000)
        a2, b2 = np.random.default_rng(seed+10000), np.random.default_rng(seed+20000)
        for _ in range(3):
            order, flips, _ = epoch_plan(y, a, b)
            order2, flips2, _ = epoch_plan(y, a2, b2)
            assert np.array_equal(order, order2) and np.array_equal(flips, flips2)
            assert np.array_equal(np.sort(order), np.arange(40000))
    checks.append('paired independent permutation/flip streams, 40000 unique images and ten classes/batch')
    tiny = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))
    optim = torch.optim.SGD(tiny.parameters(), lr=.1, momentum=.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=80)
    def step():
        xx = torch.randn(4, 3)
        optim.zero_grad(set_to_none=True)
        tiny(xx).square().mean().backward()
        optim.step()
        sched.step()
    step()
    buffer = io.BytesIO()
    torch.save(dict(model=tiny.state_dict(), optim=optim.state_dict(), scheduler=sched.state_dict(),
                    rng=torch.get_rng_state()), buffer)
    step()
    expected = copy.deepcopy(tiny.state_dict())
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=False)
    tiny.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])
    sched.load_state_dict(checkpoint['scheduler'])
    torch.set_rng_state(checkpoint['rng'])
    step()
    assert all(torch.equal(v, tiny.state_dict()[k]) for k, v in expected.items())
    checks.append('exact serialized model/optimizer/scheduler/RNG next-step CPU resume')
    return checks


def prepare():
    if (HERE / 'freeze.json').exists():
        verify_freeze()
        event('Existing frozen plan verified; no overwrite')
        return
    status('implementing', active_run=None, epoch=0, message='CPU preflight and parameter measurement')
    configs = table()
    data, audit = load_data()
    checks = cpu_preflight(configs, data)
    for config in configs:
        config.update(complexity(config))
    assert configs[0]['parameters'] == 2436346 and configs[6]['parameters'] == 2435770
    runtime.atomic_json(HERE / 'candidate_parameters.json', configs)
    source_hashes = {name: sha(HERE / name) for name in TRAIN_SOURCES}
    recipe = dict(epochs=80, batch_size=128, learning_rate=.1, momentum=.9, weight_decay=.0001,
                  horizontal_flip_probability=.5, crop_padding=0, training_dtype='bfloat16',
                  validation_dtype='float32', scheduler='CosineAnnealingLR_epoch_end',
                  normalization_mean=audit['train_rgb_mean'], normalization_std=audit['train_rgb_std'])
    data_hashes = dict(split_sha256=sha(SPLIT), cache_sha256=sha(CACHE), audit_sha256=sha(AUDIT))
    runs = [dict(run_id=f"{c['name']}_seed{seed}", seed=seed, architecture=c, recipe=recipe,
                 data_hashes=data_hashes, plan_sha256=sha(HERE / 'PLAN.md'), source_hashes=source_hashes)
            for seed in range(42, 47) for c in configs if seed in c['seeds']]
    assert len(runs) == 42
    for run in runs:
        run['config_sha256'] = digest(run)
    manifest = dict(category=CATEGORY, version=1, created_at_utc=runtime.utc_now(),
                    configurations=configs, recipe=recipe, data_hashes=data_hashes,
                    plan_sha256=sha(HERE / 'PLAN.md'), source_hashes=source_hashes, runs=runs,
                    expected_runs=42, expected_epochs=3360, expected_updates=1051680,
                    primary_comparisons=[dict(treatment=c['name'], control='baseline',
                                              direction=c['directions'][0]) for c in configs[1:]],
                    secondary_comparisons=[dict(treatment=t, control=c, direction=e) for t,c,e in
                        [('equal44','equal32','E02'),('early','late','E03'),
                         ('eca3','se8','E06'),('group4_wide','group4','E07')]])
    runtime.atomic_json(HERE / 'manifest.json', manifest)
    runtime.atomic_json(HERE / 'cpu_preflight.json', dict(passed=True, checks=checks, at_utc=runtime.utc_now()))
    runtime.atomic_json(HERE / 'freeze.json', dict(frozen_at_utc=runtime.utc_now(),
        manifest_sha256=sha(HERE / 'manifest.json'), plan_sha256=sha(HERE / 'PLAN.md'),
        source_hashes=source_hashes, data_hashes=data_hashes))
    snapshots = HERE / 'frozen_sources'
    snapshots.mkdir(exist_ok=True)
    for name in TRAIN_SOURCES:
        (snapshots / name).write_bytes((HERE / name).read_bytes())
    status('queued', message='42-run plan frozen; GPU smoke test next', active_run=None, epoch=0)
    event('Plan frozen', total_runs=42, manifest_sha256=sha(HERE / 'manifest.json'))


def verify_freeze():
    frozen = read(HERE / 'freeze.json')
    assert frozen['plan_sha256'] == sha(HERE / 'PLAN.md'), 'Frozen PLAN changed'
    assert frozen['manifest_sha256'] == sha(HERE / 'manifest.json'), 'Frozen manifest changed'
    for name, expected in frozen['source_hashes'].items():
        assert sha(HERE / name) == expected, f'Frozen training source changed: {name}'
    for name, path in [('split_sha256', SPLIT), ('cache_sha256', CACHE), ('audit_sha256', AUDIT)]:
        assert sha(path) == frozen['data_hashes'][name], f'Frozen data changed: {name}'
    return read(HERE / 'manifest.json')


def configure_gpu(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def smoke():
    manifest = verify_freeze()
    if (HERE / 'gpu_preflight.json').exists():
        assert read(HERE / 'gpu_preflight.json')['passed']
        return
    status('queued', active_run='gpu_preflight', epoch=0, message='Queued BF16 numerical smoke tests')
    with runtime.gpu_lock(CATEGORY, 'gpu_preflight'):
        runtime.check_control(CATEGORY)
        configure_gpu(4242)
        device = torch.device('cuda')
        data, audit = load_data()
        x = torch.from_numpy(data['train'][0][:8]).to(device).permute(0,3,1,2).float()/255
        mean = torch.tensor(audit['train_rgb_mean'], device=device).view(1,3,1,1)
        std = torch.tensor(audit['train_rgb_std'], device=device).view(1,3,1,1)
        x = (x-mean)/std
        labels = torch.from_numpy(data['train'][1][:8]).to(device)
        results = []
        for config in manifest['configurations']:
            runtime.check_control(CATEGORY)
            model = build(config).to(device=device, memory_format=torch.channels_last)
            opt = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.0001)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x), labels)
            loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step()
            model.eval()
            with torch.no_grad():
                result = model(x)
            assert result.dtype == torch.float32 and torch.isfinite(result).all()
            results.append(dict(name=config['name'], training_loss=float(loss), passed=True))
            del model, opt, result, loss
        torch.cuda.synchronize()
        runtime.atomic_json(HERE / 'gpu_preflight.json', dict(passed=True, checks=results,
            device=torch.cuda.get_device_name(), at_utc=runtime.utc_now()))
        del x, labels, mean, std
        torch.cuda.empty_cache()
    event('GPU BF16/FP32 preflight passed for all 12 models')


@torch.inference_mode()
def predict(model, images, norm):
    model.eval()
    return torch.cat([model((part.permute(0,3,1,2).float()/255-norm[0])/norm[1])
                      for part in images.split(128)])


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def rng_state(batch_rng, aug_rng):
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(), batch=copy.deepcopy(batch_rng.bit_generator.state),
                augmentation=copy.deepcopy(aug_rng.bit_generator.state))


def restore_rng(state, batch_rng, aug_rng):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state_all(state['cuda'])
    batch_rng.bit_generator.state = state['batch']
    aug_rng.bit_generator.state = state['augmentation']


def verify_completed(folder, config):
    metrics = read(folder / 'metrics.json')
    assert metrics['config_sha256'] == config['config_sha256']
    assert metrics['epochs_completed'] == 80 and metrics['validation_n'] == 5000
    for name, expected in metrics['artifact_sha256'].items():
        assert sha(folder / name) == expected, f'Completed artifact changed: {folder / name}'
    return metrics


def train_run(config, data, audit):
    folder = HERE / 'runs' / config['run_id']
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'metrics.json').exists():
        verify_completed(folder, config)
        return
    if (folder / 'config.json').exists():
        assert read(folder / 'config.json') == config
    else:
        runtime.atomic_json(folder / 'config.json', config)
    device = torch.device('cuda')
    configure_gpu(config['seed'])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    lock_start = time.monotonic()
    images, y = [torch.from_numpy(v).to(device) for v in data['train'][:2]]
    val_images, val_y = [torch.from_numpy(v).to(device) for v in data['validation'][:2]]
    norm = [torch.tensor(v, dtype=torch.float32, device=device).view(1,3,1,1)
            for v in (audit['train_rgb_mean'], audit['train_rgb_std'])]
    model = build(config['architecture']).to(device=device, memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
    batch_rng = np.random.default_rng(config['seed'] + 10000)
    aug_rng = np.random.default_rng(config['seed'] + 20000)
    history, best_state, best_correct, best_epoch = [], None, -1, 0
    cumulative_lock_seconds = 0.
    previous_peak_allocated = previous_peak_reserved = 0
    run = dict(config=config, config_sha256=config['config_sha256'],
               started_at_utc=runtime.utc_now(), phase='running',
               python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
               cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               device=torch.cuda.get_device_name(), training_autocast='bfloat16',
            evaluation_autocast='off', cudnn_deterministic=True, cudnn_benchmark=False,
               tf32=False, platform=platform.platform(), pid=os.getpid())
    if (folder / 'last.pt').exists():
        checkpoint = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        assert checkpoint['config_sha256'] == config['config_sha256']
        model.load_state_dict(checkpoint['model_state'])
        opt.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        restore_rng(checkpoint['rng_state'], batch_rng, aug_rng)
        history = checkpoint['history']
        best_state, best_correct, best_epoch = (checkpoint[k] for k in ['best_state','best_correct','best_epoch'])
        cumulative_lock_seconds = checkpoint['lock_seconds']
        previous_peak_allocated = checkpoint['peak_allocated_bytes']
        previous_peak_reserved = checkpoint['peak_reserved_bytes']
        run = {**read(folder / 'run.json'), 'phase':'running', 'pid':os.getpid(), 'resumed_at_utc':runtime.utc_now()}
        atomic_csv(folder / 'history.csv', history)
        save(folder / 'best.pt', dict(model_state=best_state, config=config,
             config_sha256=config['config_sha256'], epoch=best_epoch, validation_correct=best_correct))
        del checkpoint
        event('Resumed committed epoch', run_id=config['run_id'], epoch=len(history))
    runtime.atomic_json(folder / 'run.json', run)
    axis = torch.arange(32, device=device)
    for epoch in range(len(history)+1, 81):
        runtime.check_control(CATEGORY)
        status('running', active_run=config['run_id'], epoch=epoch-1, message='Training with shared GPU lock')
        torch.cuda.synchronize()
        t0 = time.monotonic()
        lr = opt.param_groups[0]['lr']
        order_np, flips_np, rejected = epoch_plan(data['train'][1], batch_rng, aug_rng)
        plan_hash = hashlib.sha256(order_np.tobytes()+flips_np.tobytes()).hexdigest()
        order = torch.from_numpy(order_np).to(device)
        flips = torch.from_numpy(flips_np).to(device)
        model.train()
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        for start in range(0, 40000, 128):
            idx = order[start:start+128]
            columns = torch.where(flips[start:start+128, None, None], 31-axis, axis)
            raw = images[idx[:,None,None], axis[:,None], columns]
            x = (raw.permute(0,3,1,2).float()/255-norm[0])/norm[1]
            labels = y[idx]
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
            totals[0] += loss.detach() * len(idx)
            totals[1] += (logits.detach().argmax(1) == labels).sum()
        val_logits = predict(model, val_images, norm)
        val_loss = float(F.cross_entropy(val_logits, val_y))
        correct = int((val_logits.argmax(1) == val_y).sum())
        train_loss, train_acc = (totals/40000).tolist()
        assert np.isfinite([train_loss, val_loss]).all(), f'Nonfinite loss epoch {epoch}'
        torch.cuda.synchronize()
        compute_seconds = time.monotonic()-t0
        if correct > best_correct:
            best_state, best_correct, best_epoch = cpu_state(model), correct, epoch
            save(folder / 'best.pt', dict(model_state=best_state, config=config,
                 config_sha256=config['config_sha256'], epoch=best_epoch, validation_correct=best_correct))
        scheduler.step()
        row = dict(epoch=epoch, lr=lr, train_loss=train_loss, train_accuracy=train_acc,
                   validation_loss=val_loss, validation_correct=correct, validation_accuracy=correct/5000,
                   compute_seconds=compute_seconds, sampler_rejections=rejected, plan_sha256=plan_hash,
                   examples=40000, updates=313)
        history.append(row)
        checkpoint = dict(config=config, config_sha256=config['config_sha256'], epoch=epoch,
            model_state=model.state_dict(), optimizer_state=opt.state_dict(),
            scheduler_state=scheduler.state_dict(), rng_state=rng_state(batch_rng, aug_rng),
            best_state=best_state, best_correct=best_correct, best_epoch=best_epoch, history=history,
            lock_seconds=cumulative_lock_seconds+time.monotonic()-lock_start,
            peak_allocated_bytes=max(previous_peak_allocated, torch.cuda.max_memory_allocated()),
            peak_reserved_bytes=max(previous_peak_reserved, torch.cuda.max_memory_reserved()))
        save(folder / 'last.pt', checkpoint)
        del checkpoint
        atomic_csv(folder / 'history.csv', history)
        with (folder / 'training.log').open('a', encoding='utf-8') as log:
            log.write(json.dumps(dict(at_utc=runtime.utc_now(), **row))+'\n')
        status('running', active_run=config['run_id'], epoch=epoch,
               message=f'Committed epoch {epoch}/80; best validation {best_correct}/5000 at {best_epoch}')
        if epoch % 10 == 0 or epoch == 1:
            event('Epoch committed', run_id=config['run_id'], epoch=epoch,
                  correct=correct, best_correct=best_correct, compute_seconds=round(compute_seconds, 3))
        runtime.check_control(CATEGORY)
    eval_start = time.monotonic()
    model.load_state_dict(best_state)
    logits = predict(model, val_images, norm)
    pred = logits.argmax(1)
    confusion = torch.bincount(val_y*10+pred, minlength=100).reshape(10,10).cpu().numpy()
    assert int(confusion.trace()) == best_correct
    confidence = logits.softmax(1).amax(1).cpu().tolist()
    predictions = pred.cpu().tolist()
    prediction_rows = [dict(id=row['id'], label=int(row['label']), prediction=p, confidence=c)
                       for row,p,c in zip(data['validation'][2], predictions, confidence)]
    atomic_csv(folder / 'validation_predictions.csv', prediction_rows)
    tp = confusion.diagonal()
    recall = tp / np.maximum(confusion.sum(1), 1)
    f1 = 2*tp / np.maximum(confusion.sum(0)+confusion.sum(1), 1)
    shift = None
    if config['architecture']['name'] in ('baseline','blur'):
        # Evaluation-only integer translation; neither augmentation nor selection.
        padded = F.pad(val_images.permute(0,3,1,2), (1,1,1,1), mode='reflect').permute(0,2,3,1)
        shift_rows, shift_records = [], []
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            shifted = padded[:,1+dy:33+dy,1+dx:33+dx,:]
            shifted_pred = predict(model, shifted, norm).argmax(1)
            shift_records.append(dict(dy=dy, dx=dx, accuracy=float((shifted_pred==val_y).float().mean()),
                                      agreement=float((shifted_pred==pred).float().mean())))
            shift_rows += [dict(id=row['id'], label=int(row['label']), dy=dy, dx=dx, prediction=p)
                           for row,p in zip(data['validation'][2], shifted_pred.cpu().tolist())]
        atomic_csv(folder / 'shift_predictions.csv', shift_rows)
        shift = dict(per_shift=shift_records,
            mean_accuracy=float(np.mean([r['accuracy'] for r in shift_records])),
            mean_agreement=float(np.mean([r['agreement'] for r in shift_records])))
    torch.cuda.synchronize()
    evaluation_seconds = time.monotonic()-eval_start
    run.update(phase='completed', completed_at_utc=runtime.utc_now(), epochs_completed=80,
               lock_seconds=cumulative_lock_seconds+time.monotonic()-lock_start,
               gpu_compute_seconds=sum(r['compute_seconds'] for r in history),
               final_evaluation_seconds=evaluation_seconds)
    runtime.atomic_json(folder / 'run.json', run)
    artifacts = ['config.json','run.json','history.csv','best.pt','last.pt','validation_predictions.csv']
    if shift is not None:
        artifacts.append('shift_predictions.csv')
    metrics = dict(run_id=config['run_id'], configuration=config['architecture']['name'], seed=config['seed'],
        config_sha256=config['config_sha256'], epochs_completed=80, validation_n=5000,
        validation_correct=best_correct, validation_accuracy=best_correct/5000, best_epoch=best_epoch,
        validation_nll=float(F.cross_entropy(logits,val_y)), macro_f1=float(f1.mean()),
        class_recall=recall.tolist(), class_f1=f1.tolist(), worst_class_recall=float(recall.min()),
        confusion_matrix=confusion.tolist(), parameters=config['architecture']['parameters'],
        macs=config['architecture']['macs'], gpu_compute_seconds=run['gpu_compute_seconds'],
        lock_seconds=run['lock_seconds'], final_evaluation_seconds=evaluation_seconds,
        peak_allocated_bytes=max(previous_peak_allocated,torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=max(previous_peak_reserved,torch.cuda.max_memory_reserved()),
        training_examples=3200000, optimizer_updates=25040, shift_validation=shift,
        artifact_sha256={name:sha(folder/name) for name in artifacts})
    runtime.atomic_json(folder / 'metrics.json', metrics)
    event('Run completed', run_id=config['run_id'], correct=best_correct, best_epoch=best_epoch,
          gpu_compute_seconds=round(run['gpu_compute_seconds'], 2))


def run_all():
    manifest = verify_freeze()
    smoke()
    data, audit = load_data()
    for config in manifest['runs']:
        runtime.check_control(CATEGORY)
        folder = HERE/'runs'/config['run_id']
        if (folder/'metrics.json').exists():
            verify_completed(folder, config)
            continue
        status('queued', active_run=config['run_id'], epoch=0, message='Waiting for FIFO shared GPU lock')
        event('Queued run', run_id=config['run_id'])
        with runtime.gpu_lock(CATEGORY, config['run_id']):
            runtime.check_control(CATEGORY)
            try:
                train_run(config, data, audit)
            finally:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        from experiments.research_sweep.architecture.report import report
        report(final=False)
    from experiments.research_sweep.architecture.report import report
    report(final=True)
    status('completed', active_run=None, epoch=80, message='All 42 runs and report artifacts verified')


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','smoke','run','report'])
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    # A category-specific process lock prevents accidental duplicate runners.
    import msvcrt
    with (HERE/'worker.lock').open('a+b') as handle:
        handle.seek(0,2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            if args.action == 'prepare':
                prepare()
            elif args.action == 'smoke':
                smoke()
            elif args.action == 'run':
                run_all()
            else:
                from experiments.research_sweep.architecture.report import report
                report(final=counts()['completed_runs'] == 42)
        except (runtime.Paused, runtime.Cancelled) as exc:
            status('paused', message=str(exc)+'; checkpoint preserved; explicit resume required')
            event('Worker stopped by user control', reason=type(exc).__name__)
            return 20 if isinstance(exc,runtime.Paused) else 21
        except Exception as exc:
            failures = HERE/'failures'
            failures.mkdir(exist_ok=True)
            runtime.atomic_json(failures/f'{time.time_ns()}.json', dict(at_utc=runtime.utc_now(),
                action=args.action, error=repr(exc), traceback=traceback.format_exc()))
            status('failed', message=repr(exc))
            event('Worker failed; evidence preserved', error=repr(exc))
            raise
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
