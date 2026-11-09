"""Combination operators; imports original frozen methods without editing them."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.research_sweep.architecture.models import ResidualNet, PostBlock, SE, BlurPool
from experiments.research_sweep.training.methods import sampler, make_scheduler


class ComboBlock(PostBlock):
    def forward(self, x):
        z = x
        for module in list(self.residual)[:-2]:
            z = module(z)
        z = F.dropout(z, self.dropout_probability, self.training)
        z = self.residual[-1](self.residual[-2](z))
        z = self.attention(z)
        if self.training and self.drop_probability:
            keep = 1 - self.drop_probability
            z = z * (torch.rand((len(z), 1, 1, 1), device=z.device) < keep).to(z.dtype) / keep
        return self.relu(z + self.shortcut(x))


def build(cfg):
    widths = tuple(cfg['width'] * v for v in (16, 32, 64))
    net = ResidualNet(widths, 3, 'preact' if cfg['preact'] else 'blur' if cfg['blur'] else 'plain')
    for i, block in enumerate(m for m in net if isinstance(m, PostBlock)):
        block.__class__ = ComboBlock
        block.dropout_probability = cfg['dropout']
        block.drop_probability = cfg['drop_path'] * (i + 1) / 9
        if cfg['se']:
            block.attention = SE(block.residual[-1].num_features)
            for layer in block.attention.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    assert sum(p.numel() for p in net.parameters()) <= 5_000_000
    return net


def cost(cfg):
    net = build(cfg).eval()
    totals = {'learned_macs': 0, 'fixed_blur_macs': 0}
    def hook(layer, args, out):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            totals['learned_macs'] += out.numel() * layer.weight[0].numel()
        elif isinstance(layer, BlurPool):
            totals['fixed_blur_macs'] += out.numel() * 9
    handles = [m.register_forward_hook(hook) for m in net.modules()
               if isinstance(m, (nn.Conv2d, nn.Linear, BlurPool))]
    with torch.no_grad():
        assert net(torch.zeros(1, 3, 32, 32)).shape == (1, 10)
    for h in handles:
        h.remove()
    return dict(parameters=sum(p.numel() for p in net.parameters()), macs=sum(totals.values()), **totals)


def augmentation_plan(seed, epoch, views, n=40000):
    flips = np.random.default_rng([seed, epoch, 2002]).random((views, n)) < .5
    # A per-view stream keeps view zero identical when a second view is added.
    angles, cuts = [], []
    for view in range(views):
        r = np.random.default_rng([seed, epoch, 3003, view])
        a = r.uniform(-1, 1, n).astype(np.float32)
        a *= r.random(n) < .5
        angles.append(a)
        r = np.random.default_rng([seed, epoch, 4004, view])
        cuts.append(np.stack([r.random(n) < .5, r.integers(0, 32, n), r.integers(0, 32, n)], axis=1))
    return flips, np.stack(angles), np.stack(cuts)


def augment(images, ids, norm, flip, angle, cut, cfg):
    x = (images[ids].permute(0, 3, 1, 2).float() / 255 - norm[0]) / norm[1]
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    if cfg['rotation']:
        radians = angle * (cfg['rotation'] * math.pi / 180)
        theta = torch.zeros(len(ids), 2, 3, device=x.device)
        theta[:, 0, 0] = theta[:, 1, 1] = radians.cos()
        theta[:, 0, 1] = -radians.sin()
        theta[:, 1, 0] = radians.sin()
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    if cfg['cutout']:
        axis = torch.arange(32, device=x.device)
        h = cfg['cutout'] // 2
        rows = (axis[None, :] >= cut[:, 1, None] - h) & (axis[None, :] < cut[:, 1, None] + h)
        cols = (axis[None, :] >= cut[:, 2, None] - h) & (axis[None, :] < cut[:, 2, None] + h)
        mask = cut[:, 0, None, None].bool() & rows[:, :, None] & cols[:, None, :]
        x = x.masked_fill(mask[:, None], 0)
    return x.contiguous(memory_format=torch.channels_last)


def objective(net, x, y, cfg, teacher=None):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = net(x)
        ce = F.cross_entropy(logits, y)
    if teacher is None:
        return ce, logits
    with torch.no_grad():
        targets = teacher(x).float()
    t = cfg['kd_temperature']
    kd = F.kl_div(F.log_softmax(logits.float()/t, dim=1), F.softmax(targets/t, dim=1), reduction='batchmean') * t*t
    return .5 * ce + .5 * kd, logits


@torch.inference_mode()
def predict(net, images, norm):
    net.eval()
    logits = torch.cat([net((part.permute(0, 3, 1, 2).float()/255-norm[0])/norm[1]) for part in images.split(128)])
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
    return logits
