"""CIFAR residual architecture ablations; original project-style implementation.

Papers and implementation decisions are recorded in PLAN.md. No external weights.
"""
import torch
from torch import nn
from torch.nn import functional as F


class BlurPool(nn.Module):
    """Fixed separable [1,2,1]^2 / 16, reflection boundary, even-grid sampling."""
    def __init__(self, channels):
        super().__init__()
        v = torch.tensor([1., 2., 1.])
        self.register_buffer('kernel', (v[:, None] * v[None, :] / 16)
                             .reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x):
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'),
                        self.kernel, stride=2, groups=self.channels)


class SE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.reduce = nn.Conv2d(channels, channels // 8, 1)
        self.expand = nn.Conv2d(channels // 8, channels, 1)

    def forward(self, x):
        z = x.mean((2, 3), keepdim=True)
        return x * self.expand(F.relu(self.reduce(z))).sigmoid()


class ECA(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 3, padding=1, bias=False)

    def forward(self, x):
        z = self.conv(x.mean((2, 3)).unsqueeze(1)).sigmoid()
        return x * z.squeeze(1)[:, :, None, None]


class PostBlock(nn.Module):
    def __init__(self, inputs, outputs, stride, variant):
        super().__init__()
        blur = variant == 'blur' and stride == 2
        layers = [nn.Conv2d(inputs, outputs, 3, 1 if blur else stride, 1, bias=False),
                  nn.BatchNorm2d(outputs), nn.ReLU(inplace=True)]
        # Restore the original feature-map size before the second convolution.
        if blur:
            layers.append(BlurPool(outputs))
        layers += [nn.Conv2d(outputs, outputs, 3, 1, 1,
                            groups=4 if variant == 'group4' else 1, bias=False),
                   nn.BatchNorm2d(outputs)]
        self.residual = nn.Sequential(*layers)
        if inputs == outputs and stride == 1:
            self.shortcut = nn.Identity()
        else:
            shortcuts = [nn.Conv2d(inputs, outputs, 1, 1 if blur else stride, bias=False),
                         nn.BatchNorm2d(outputs)]
            if blur:
                shortcuts.append(BlurPool(outputs))
            self.shortcut = nn.Sequential(*shortcuts)
        self.attention = SE(outputs) if variant == 'se8' else ECA() if variant == 'eca3' else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.attention(self.residual(x)) + self.shortcut(x))


class PreBlock(nn.Module):
    def __init__(self, inputs, outputs, stride, first=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inputs)
        self.conv1 = nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outputs)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, 1, 1, bias=False)
        self.shortcut = (nn.Identity() if inputs == outputs and stride == 1 else
                         nn.Conv2d(inputs, outputs, 1, stride, bias=False))
        self.first = first

    def forward(self, x):
        z = F.relu(self.bn1(x))
        # First unit activation precedes splitting; projection also consumes z.
        # Ordinary same-size shortcuts remain exact identities on x.
        shortcut = self.shortcut(z) if self.first or not isinstance(self.shortcut, nn.Identity) else x
        residual = self.conv2(F.relu(self.bn2(self.conv1(z))))
        return shortcut + residual


class ResidualNet(nn.Sequential):
    def __init__(self, widths=(48, 96, 192), blocks=3, variant='plain'):
        layers = [nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)]
        pre = variant == 'preact'
        if not pre:
            layers += [nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True)]
        inputs = widths[0]
        for stage, outputs in enumerate(widths):
            for index in range(blocks):
                stride = 2 if stage > 0 and index == 0 else 1
                layers.append(PreBlock(inputs, outputs, stride, first=stage == index == 0)
                              if pre else PostBlock(inputs, outputs, stride, variant))
                inputs = outputs
        if pre:
            layers += [nn.BatchNorm2d(widths[-1]), nn.ReLU(inplace=True)]
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(widths[-1], 10)]
        super().__init__(*layers)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')


def build(config):
    net = ResidualNet(config['widths'], config['blocks'], config['variant'])
    assert sum(p.numel() for p in net.parameters() if p.requires_grad) <= 5_000_000
    return net


def complexity(config):
    net = build(config).eval()
    totals = {'learned_macs': 0, 'fixed_blur_macs': 0}
    hooks = []
    def hook(layer, args, out):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
            totals['learned_macs'] += out.numel() * layer.weight[0].numel()
        elif isinstance(layer, nn.Linear):
            totals['learned_macs'] += out.numel() * layer.in_features
        elif isinstance(layer, BlurPool):
            totals['fixed_blur_macs'] += out.numel() * 9
    for layer in net.modules():
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear, BlurPool)):
            hooks.append(layer.register_forward_hook(hook))
    with torch.no_grad():
        shape = list(net(torch.zeros(1, 3, 32, 32)).shape)
    for handle in hooks:
        handle.remove()
    return dict(parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
                output_shape=shape, macs=sum(totals.values()), **totals)
