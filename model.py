"""The submitted 3x CIFAR ResNet-20, with anti-aliased downsampling."""

import torch
from torch import nn
from torch.nn import functional as F


class BlurPool(nn.Module):
    """Fixed separable [1,2,1]^2 / 16, reflection boundary, even-grid sampling."""

    def __init__(self, channels):
        super().__init__()
        v = torch.tensor([1.0, 2.0, 1.0])
        self.register_buffer(
            "kernel",
            (v[:, None] * v[None, :] / 16)
            .reshape(1, 1, 3, 3)
            .repeat(channels, 1, 1, 1),
        )
        self.channels = channels

    def forward(self, x):
        return F.conv2d(
            F.pad(x, (1, 1, 1, 1), mode="reflect"),
            self.kernel,
            stride=2,
            groups=self.channels,
        )


class ResidualBlock(nn.Module):
    """Two convolutions plus a shortcut, dropout and per-sample residual dropping."""

    def __init__(self, inputs, outputs, stride, dropout, drop_probability):
        super().__init__()
        layers = [
            nn.Conv2d(inputs, outputs, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        if stride == 2:
            layers.append(BlurPool(outputs))
        layers.extend(
            [nn.Conv2d(outputs, outputs, 3, 1, 1, bias=False), nn.BatchNorm2d(outputs)]
        )
        self.residual = nn.Sequential(*layers)
        if inputs == outputs and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inputs, outputs, 1, bias=False),
                nn.BatchNorm2d(outputs),
                BlurPool(outputs),
            )
        self.relu = nn.ReLU(inplace=True)
        self.dropout_probability = dropout
        self.drop_probability = drop_probability

    def forward(self, x):
        z = x
        for module in list(self.residual)[:-2]:
            z = module(z)
        z = F.dropout(z, self.dropout_probability, self.training)
        z = self.residual[-1](self.residual[-2](z))
        if self.training and self.drop_probability:
            keep = 1 - self.drop_probability
            mask = (torch.rand((len(z), 1, 1, 1), device=z.device) < keep).to(z.dtype)
            z = z * mask / keep
        return self.relu(z + self.shortcut(x))


class ResNet20(nn.Sequential):
    """48/96/192 channels; nine residual blocks; 2,436,346 trainable parameters."""

    def __init__(self, dropout=0.1, drop_path=0.1):
        widths = (48, 96, 192)
        layers = [
            nn.Conv2d(3, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        ]
        inputs = 48
        for stage, outputs in enumerate(widths):
            for index in range(3):
                stride = 2 if stage > 0 and index == 0 else 1
                probability = drop_path * (stage * 3 + index + 1) / 9
                layers.append(
                    ResidualBlock(inputs, outputs, stride, dropout, probability)
                )
                inputs = outputs
        layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(192, 10)])
        super().__init__(*layers)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="relu"
                )


def build(recipe):
    """Build only the architecture represented by this submitted checkpoint."""
    assert recipe["width"] == 3 and recipe["blur"]
    assert not recipe["preact"] and not recipe["se"]
    assert recipe["kd_temperature"] == 0 and recipe["views"] == 1
    return ResNet20(recipe["dropout"], recipe["drop_path"])
