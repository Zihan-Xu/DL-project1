"""Registered crop and Mixup additions; no modification of frozen study code."""
import numpy as np
import torch
from torch.nn import functional as F
import base_operators as original

BASE_PLAN = original.augmentation_plan
CROPS = None
CROPS_DEVICE = None
MIX_RNG = None


def augmentation_plan(seed, epoch, views, n=40000):
    global CROPS, CROPS_DEVICE, MIX_RNG
    assert views == 1
    CROPS = np.random.default_rng([seed, epoch, 5005]).integers(0, 9, (n, 2))
    CROPS_DEVICE = None
    MIX_RNG = np.random.default_rng([seed, epoch, 6006])
    return BASE_PLAN(seed, epoch, views, n)


def augment(images, ids, norm, flip, angle, cut, cfg):
    global CROPS_DEVICE
    x = (images[ids].permute(0, 3, 1, 2).float() / 255 - norm[0]) / norm[1]
    if cfg.get('crop_padding', 0):
        assert cfg['crop_padding'] == 4
        if CROPS_DEVICE is None or CROPS_DEVICE.device != ids.device:
            CROPS_DEVICE = torch.as_tensor(CROPS, device=ids.device)
        crop = CROPS_DEVICE[ids]
        padded = F.pad(x, (4, 4, 4, 4), mode='reflect')
        axis = torch.arange(32, device=x.device)
        ix = crop[:, 0, None, None] + axis[None, :, None]
        iy = crop[:, 1, None, None] + axis[None, None, :]
        # NHWC indexing avoids interpolation and keeps exactly 32x32 source pixels.
        x = padded.permute(0, 2, 3, 1)[torch.arange(len(ids), device=x.device)[:, None, None], ix, iy].permute(0, 3, 1, 2)
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    assert cfg['rotation'] == 0
    if cfg['cutout']:
        axis = torch.arange(32, device=x.device)
        h = cfg['cutout'] // 2
        rows = (axis[None, :] >= cut[:, 1, None]-h) & (axis[None, :] < cut[:, 1, None]+h)
        cols = (axis[None, :] >= cut[:, 2, None]-h) & (axis[None, :] < cut[:, 2, None]+h)
        mask = cut[:, 0, None, None].bool() & rows[:, :, None] & cols[:, None, :]
        x = x.masked_fill(mask[:, None], 0)
    return x.contiguous(memory_format=torch.channels_last)


def objective(net, x, y, cfg, teacher=None):
    assert teacher is None
    mixed = cfg.get('mixup_alpha', 0) > 0 and MIX_RNG.random() < cfg['mixup_probability']
    if mixed:
        lam = float(MIX_RNG.beta(cfg['mixup_alpha'], cfg['mixup_alpha']))
        permutation = torch.as_tensor(MIX_RNG.permutation(len(y)), device=y.device)
        x = (lam*x + (1-lam)*x[permutation]).contiguous(memory_format=torch.channels_last)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = net(x)
        loss = F.cross_entropy(logits, y)
        if mixed:
            loss = lam*loss + (1-lam)*F.cross_entropy(logits, y[permutation])
    return loss, logits

# Portable bindings to the unchanged model, sampler, schedule and inference.
build=original.build
sampler=original.sampler
make_scheduler=original.make_scheduler
predict=original.predict
