"""Actual MobileNetV3-Small V1; legacy ResNet is delegated only for parity checks."""
import hashlib
import random
import shutil
import ssl
import urllib.parse
import urllib.request
import numpy as np
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small
from extensions.models import build as old_build, parameter_groups as old_parameter_groups, WEIGHTS as RESNET_WEIGHTS
from .common import OUT, ROOT, WEIGHTS, WEIGHT_URL, WEIGHT_PREFIX, read, write, sha


def download_weights():
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS.exists():
        temporary = WEIGHTS.with_suffix(".download")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_default_certs()
        with urllib.request.urlopen(WEIGHT_URL, timeout=60, context=context) as response:
            assert urllib.parse.urlparse(response.url).hostname == "download.pytorch.org", "Unapproved redirect"
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        assert sha(temporary).startswith(WEIGHT_PREFIX), "Official published hash prefix failed"
        temporary.replace(WEIGHTS)
    digest = sha(WEIGHTS)
    assert digest.startswith(WEIGHT_PREFIX), "Existing Mobile weights are not the fixed official V1"
    identity = {"url": WEIGHT_URL, "path": str(WEIGHTS.relative_to(ROOT)), "sha256": digest,
                "published_sha256_prefix": WEIGHT_PREFIX, "prefix_verified": True,
                "enum": "MobileNet_V3_Small_Weights.IMAGENET1K_V1", "original_classes": 1000,
                "role": "Only pretrained cases load these parameters; random cases do not"}
    if (OUT/"weights.json").exists():
        assert read(OUT/"weights.json") == identity
    else:
        write(OUT/"weights.json", identity)
    return identity


def tensor_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class MobileTransfer(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        # Construct/load the original 1000-class architecture before replacing ONLY its last head.
        self.backbone = mobilenet_v3_small(weights=None, num_classes=1000, width_mult=1.0,
                                           reduced_tail=False, dilated=False, dropout=.2)
        if pretrained:
            identity = read(OUT/"weights.json")
            assert sha(WEIGHTS) == identity["sha256"] and identity["sha256"].startswith(WEIGHT_PREFIX)
            state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state, strict=True)
        assert self.backbone.classifier[-1].in_features == 1024
        self.backbone.classifier[-1] = nn.Linear(1024, 10)
        assert sum(p.numel() for p in self.parameters()) == 1528106

    def forward(self, images):
        return self.backbone(images)


def build(cfg):
    if cfg["architecture"] == "mobilenet_v3_small":
        assert cfg["model"] == "transfer" and cfg["image_size"] == 96 and not cfg.get("head_only", False)
        return MobileTransfer(cfg["pretrained"])
    assert cfg["architecture"] == "resnet18", "Only the old ResNet18 parity reference is allowed"
    return old_build(cfg)


def parameter_groups(model, cfg):
    if cfg["architecture"] == "resnet18":
        return old_parameter_groups(model, cfg)
    assert cfg["backbone_ratio"] == 1.0
    return [{"params": [p for p in model.parameters() if p.requires_grad]}]


def weight_identity(cfg):
    if not cfg.get("pretrained", False):
        return {"loaded": False, "sha256": None, "role": "Random parameters; common ImageNet input statistics are still used"}
    if cfg["architecture"] == "resnet18":
        return {"loaded": True, "sha256": sha(RESNET_WEIGHTS), "path": str(RESNET_WEIGHTS.relative_to(ROOT)), "role": "Legacy parity reference only"}
    return {"loaded": True, **read(OUT/"weights.json")}


def metadata(model, cfg):
    head = model.backbone.classifier[-1] if cfg["architecture"] == "mobilenet_v3_small" else model.backbone.fc
    return {"class": type(model).__name__, "backbone_class": type(model.backbone).__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "head_shape": list(head.weight.shape), "head_sha256": tensor_hash(head.state_dict()),
            "initial_model_sha256": tensor_hash(model.state_dict()),
            "batchnorm_modules": sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()),
            "pretraining": weight_identity(cfg), "normalization": "ImageNet fixed mean/std", "image_size": 96}
