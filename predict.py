"""Reproduce a released CSV from original competition PNGs on CPU or CUDA."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from PIL import Image
from experiments.research_sweep.combinations.methods import build


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load(checkpoint, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = build(saved['recipe']).to(device=device, memory_format=torch.channels_last).eval()
    model.load_state_dict(saved['model_state'], strict=True)
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == saved['parameters'] <= 5_000_000
    norm = [torch.tensor(saved[k], device=device).view(1, 3, 1, 1) for k in ['mean', 'std']]
    return model, norm, parameters


@torch.inference_mode()
def probabilities(model, norm, pixels, device, batch_size=128, policy=None):
    raw, flip = [], []
    for start in range(0, len(pixels), batch_size):
        x = torch.as_tensor(np.array(pixels[start:start+batch_size]), device=device).permute(0, 3, 1, 2).float() / 255
        x = ((x - norm[0]) / norm[1]).contiguous(memory_format=torch.channels_last)
        a = model(x).softmax(1)
        assert torch.isfinite(a).all()
        raw.append(a.cpu().numpy())
        if policy != 'raw':
            b = model(x.flip(3).contiguous(memory_format=torch.channels_last)).softmax(1)
            assert torch.isfinite(b).all()
            flip.append(((a+b)/2).cpu().numpy())
    result = {'raw': np.concatenate(raw)}
    if flip:
        result['flip'] = np.concatenate(flip)
    return result


def configure():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--policy', choices=['raw', 'flip'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    configure()
    sample = rows(args.data_dir / 'sample_submission.csv')
    test = rows(args.data_dir / 'test.csv')
    ids = [r['id'] for r in sample]
    assert len(ids) == len(set(ids)) == 2000 and list(sample[0]) == ['id', 'label']
    assert [r['id'] for r in test] == ids
    images = []
    for row in test:
        assert Path(row['filename']).name == row['filename']
        with Image.open(args.data_dir / 'test_images' / row['filename']) as image:
            assert image.mode == 'RGB' and image.size == (32, 32)
            images.append(np.array(image))
    device = torch.device(args.device)
    model, norm, parameters = load(args.checkpoint, device)
    labels = probabilities(model, norm, np.stack(images), device, policy=args.policy)[args.policy].argmax(1).tolist()
    if args.compare:
        expected = rows(args.compare)
        assert list(expected[0]) == ['id', 'label']
        assert [r['id'] for r in expected] == ids and [int(r['label']) for r in expected] == labels
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'label'])
        writer.writerows(zip(ids, labels))
    print(json.dumps({'passed': True, 'rows': len(labels), 'parameters': parameters,
                      'policy': args.policy, 'compared': bool(args.compare),
                      'csv_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
