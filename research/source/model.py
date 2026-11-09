"""CIFAR ResNet-20 with learned projection shortcuts at the two stage changes."""
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, stride):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outputs),
        )
        self.shortcut = nn.Identity() if stride == 1 else nn.Sequential(
            nn.Conv2d(inputs, outputs, 1, stride, bias=False), nn.BatchNorm2d(outputs)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.residual(x) + self.shortcut(x))


class ResNet20(nn.Sequential):
    def __init__(self, width_multiplier=1):
        if not isinstance(width_multiplier, int) or width_multiplier < 1:
            raise ValueError("width_multiplier must be a positive integer")
        widths = [base * width_multiplier for base in (16, 32, 64)]
        layers = [
            nn.Conv2d(3, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True)
        ]
        inputs = widths[0]
        for outputs in widths:
            for _ in range(3):
                layers.append(ResidualBlock(inputs, outputs, 2 if outputs != inputs else 1))
                inputs = outputs
        layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(widths[-1], 10)])
        super().__init__(*layers)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
