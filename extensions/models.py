"""第二轮模型与预处理，复用首轮CIFAR网络。"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from data import ROOT, BatchTransform
from models import make_model

WEIGHTS = ROOT / "data/pretrained/resnet18-f37072fd.pth"


class TransferResNet(nn.Module):
    def __init__(self, pretrained=True, head_only=False):
        super().__init__()
        self.backbone = resnet18(weights=None)
        if pretrained:
            self.backbone.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 10)
        self.head_only = head_only
        if head_only:
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad_(name.startswith("fc."))

    def train(self, mode=True):
        super().train(mode)
        if self.head_only and mode:
            # 冻结特征提取时也冻结BN统计，避免把更新BN混作纯分类头训练。
            self.backbone.eval()
            self.backbone.fc.train(True)
        return self

    def forward(self, x):
        return self.backbone(x)


class Transform(BatchTransform):
    def __init__(self, stats, config):
        self.image_size = config.get("image_size", 32)
        if config.get("model", "cifar") == "transfer":
            stats = {"mean": [.485, .456, .406], "std": [.229, .224, .225]}
        super().__init__(stats, config.get("augmentation", True), config.get("cutout", 0))

    def forward(self, images, training=False):
        x = super().forward(images, training)
        if self.image_size != 32:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False, antialias=True)
        return x.contiguous(memory_format=torch.channels_last)


def build(config):
    if config.get("model", "cifar") == "transfer":
        return TransferResNet(config.get("pretrained", True), config.get("head_only", False))
    return make_model(config)


def parameter_groups(model, config):
    lr = config["learning_rate"]
    if config.get("model", "cifar") == "transfer" and not config.get("head_only", False):
        head, body = [], []
        for name, parameter in model.named_parameters():
            (head if name.startswith("backbone.fc.") else body).append(parameter)
        return [{"params": body, "lr": lr * config.get("backbone_ratio", .1)}, {"params": head, "lr": lr}]
    return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
