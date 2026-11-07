"""Exact sampling, normalization and augmentation used for the second submission."""

import math
import numpy as np
import torch
from torch.nn import functional as F
from model import build


def augmentation_plan(seed, epoch, views, n=40000):
    flips = np.random.default_rng([seed, epoch, 2002]).random((views, n)) < 0.5
    angles, cuts = ([], [])
    for view in range(views):
        r = np.random.default_rng([seed, epoch, 3003, view])
        a = r.uniform(-1, 1, n).astype(np.float32)
        a *= r.random(n) < 0.5
        angles.append(a)
        r = np.random.default_rng([seed, epoch, 4004, view])
        cuts.append(
            np.stack(
                [r.random(n) < 0.5, r.integers(0, 32, n), r.integers(0, 32, n)], axis=1
            )
        )
    return (flips, np.stack(angles), np.stack(cuts))


def augment(images, ids, norm, flip, angle, cut, cfg):
    x = (images[ids].permute(0, 3, 1, 2).float() / 255 - norm[0]) / norm[1]
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    if cfg["rotation"]:
        radians = angle * (cfg["rotation"] * math.pi / 180)
        theta = torch.zeros(len(ids), 2, 3, device=x.device)
        theta[:, 0, 0] = theta[:, 1, 1] = radians.cos()
        theta[:, 0, 1] = -radians.sin()
        theta[:, 1, 0] = radians.sin()
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )
    if cfg["cutout"]:
        axis = torch.arange(32, device=x.device)
        h = cfg["cutout"] // 2
        rows = (axis[None, :] >= cut[:, 1, None] - h) & (
            axis[None, :] < cut[:, 1, None] + h
        )
        cols = (axis[None, :] >= cut[:, 2, None] - h) & (
            axis[None, :] < cut[:, 2, None] + h
        )
        mask = cut[:, 0, None, None].bool() & rows[:, :, None] & cols[:, None, :]
        x = x.masked_fill(mask[:, None], 0)
    return x.contiguous(memory_format=torch.channels_last)


@torch.inference_mode()
def predict(net, images, norm):
    net.eval()
    logits = torch.cat(
        [
            net((part.permute(0, 3, 1, 2).float() / 255 - norm[0]) / norm[1])
            for part in images.split(128)
        ]
    )
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
    return logits


def sampler(labels, seed, epoch, batch_size):
    """Uniform shuffle followed by minimum class-coverage repair; no repeats/drops."""
    rng = np.random.default_rng([seed, epoch, 1001])
    order = rng.permutation(len(labels))
    bins = np.arange(len(labels)) // batch_size
    n_batches = int(bins[-1]) + 1
    counts = np.bincount(bins * 10 + labels[order], minlength=n_batches * 10).reshape(
        -1, 10
    )
    swaps = 0
    for b, cls in np.argwhere(counts == 0):
        if counts[b, cls]:
            continue
        start, end = (int(b) * batch_size, min((int(b) + 1) * batch_size, len(order)))
        candidates = np.flatnonzero(labels[order] == cls)
        donors = candidates[
            (counts[bins[candidates], cls] > 1) & (bins[candidates] != b)
        ]
        if not len(donors):
            raise RuntimeError("Cannot repair class coverage")
        donor = int(donors[0])
        d = int(bins[donor])
        receiver = next(
            (j for j in range(start, end) if counts[b, labels[order[j]]] > 1)
        )
        displaced = int(labels[order[receiver]])
        order[receiver], order[donor] = (order[donor], order[receiver])
        counts[b, cls] += 1
        counts[d, cls] -= 1
        counts[b, displaced] -= 1
        counts[d, displaced] += 1
        swaps += 1
    assert counts.min() >= 1 and np.all(np.bincount(order, minlength=len(labels)) == 1)
    return (order, counts, swaps)


def objective(net, x, y, cfg):
    """Cross-entropy with BF16 CUDA forward computation; no external teacher."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = net(x)
        loss = F.cross_entropy(logits, y)
    return loss, logits


def make_scheduler(optimizer, cfg):
    assert cfg["scheduler"] == "cosine" and cfg["warmup"] == 0
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])
