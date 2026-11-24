"""固定划分与批量预处理。训练流程只读取官方训练批次。"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parent


def read_partition():
    manifest = json.loads((ROOT / "data/dataset_manifest.json").read_text(encoding="utf-8"))
    path = ROOT / manifest["split_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["split_sha256"]:
        raise ValueError("划分文件已改变")
    split = json.loads(path.read_text(encoding="utf-8"))
    train, validation = split["train_indices"], split["validation_indices"]
    if len(train) != 45000 or len(validation) != 5000 or set(train) & set(validation) or set(train) | set(validation) != set(range(50000)):
        raise ValueError("训练/验证划分不完整或有重叠")
    return manifest, train, validation


def training_data():
    manifest, train, validation = read_partition()
    source = CIFAR10(str(ROOT / "data/raw"), train=True, download=False)
    images = torch.from_numpy(source.data.copy()).permute(0, 3, 1, 2).contiguous()
    labels = torch.tensor(source.targets, dtype=torch.long)
    stats_path = ROOT / "data/normalization.json"
    mean = source.data[train].mean(axis=(0, 1, 2), dtype=np.float64) / 255
    std = source.data[train].std(axis=(0, 1, 2), dtype=np.float64) / 255
    stats = {"source": "fixed 45000 training images only", "split_sha256": manifest["split_sha256"], "mean": mean.tolist(), "std": std.tolist()}
    if stats_path.exists() and json.loads(stats_path.read_text(encoding="utf-8")) != stats:
        raise ValueError("归一化记录与训练划分不一致")
    if not stats_path.exists():
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return TensorDataset(images[train], labels[train]), TensorDataset(images[validation], labels[validation]), stats


def make_loader(dataset, batch_size, generator=None, training=False):
    # 无随机CPU变换，随机裁剪等在GPU批量执行；避免工作进程随机状态影响续训。
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, generator=generator, num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False)


class BatchTransform(nn.Module):
    def __init__(self, stats, augmentation=True, cutout=0):
        super().__init__()
        self.register_buffer("mean", torch.tensor(stats["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(stats["std"]).view(1, 3, 1, 1))
        self.augmentation = augmentation
        self.cutout = cutout

    def forward(self, images, training=False):
        x = images
        if training and self.augmentation:
            batch = len(x)
            padded = F.pad(x, (4, 4, 4, 4))
            rows = torch.randint(0, 9, (batch, 1, 1), device=x.device) + torch.arange(32, device=x.device).view(1, 32, 1)
            cols = torch.randint(0, 9, (batch, 1, 1), device=x.device) + torch.arange(32, device=x.device).view(1, 1, 32)
            x = padded.permute(0, 2, 3, 1)[torch.arange(batch, device=x.device).view(batch, 1, 1), rows, cols].permute(0, 3, 1, 2)
            flip = torch.rand((batch, 1, 1, 1), device=x.device) < .5
            x = torch.where(flip, x.flip(-1), x)
        x = (x.float() / 255 - self.mean) / self.std
        if training and self.cutout:
            cy = torch.randint(0, 32, (len(x), 1, 1), device=x.device)
            cx = torch.randint(0, 32, (len(x), 1, 1), device=x.device)
            yy = torch.arange(32, device=x.device).view(1, 32, 1)
            xx = torch.arange(32, device=x.device).view(1, 1, 32)
            mask = (yy >= cy - self.cutout // 2) & (yy < cy + self.cutout - self.cutout // 2) & (xx >= cx - self.cutout // 2) & (xx < cx + self.cutout - self.cutout // 2)
            x = x.masked_fill(mask.unsqueeze(1), 0)
        return x.contiguous(memory_format=torch.channels_last)
