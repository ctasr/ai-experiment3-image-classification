"""面向32×32图像的自建CNN、残差网络及通道注意力。"""
import torch
from torch import nn


def conv_block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SmallCNN(nn.Module):
    def __init__(self, width=32, dropout=.2):
        super().__init__()
        layers = []
        cin = 3
        for cout in (width, width * 2, width * 4):
            layers.extend([conv_block(cin, cout), conv_block(cout, cout), nn.MaxPool2d(2)])
            cin = cout
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin, 10))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(4, channels // 8)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, attention=False):
        super().__init__()
        self.conv1 = conv_block(cin, cout, stride)
        self.conv2 = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.attention = ChannelAttention(cout) if attention else nn.Identity()
        self.shortcut = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.attention(self.conv2(self.conv1(x))) + self.shortcut(x))


class CIFARResNet(nn.Module):
    def __init__(self, width=16, attention=False):
        super().__init__()
        self.stem = conv_block(3, width)
        layers = []
        cin = width
        for stage, cout in enumerate((width, width * 2, width * 4)):
            for block in range(3):
                layers.append(ResidualBlock(cin, cout, 2 if stage and block == 0 else 1, attention))
                cin = cout
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(cin, 10)

    def forward(self, x):
        return self.classifier(self.pool(self.features(self.stem(x))).flatten(1))


def make_model(config):
    if config["architecture"] == "cnn":
        model = SmallCNN(config.get("width", 32), config.get("dropout", .2))
    elif config["architecture"] == "resnet20":
        model = CIFARResNet(config.get("width", 16), config.get("attention", False))
    else:
        raise ValueError(config["architecture"])
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    return model
