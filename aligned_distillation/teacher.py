"""冻结四视图教师；只接受图像，不接受训练标签。"""
import numpy as np
import torch
from torch.nn import functional as F
from data import ROOT
from extensions.models import build, Transform
from distillation.targets import read, sha, sources

OUT = ROOT / "results_aligned_distillation"


def identity():
    selected = read(ROOT / "results_round3/selected_runs.json")
    rows = {k: next(r for r in selected[k] if r["config"]["seed"] == 42)
            for k in ("cutmix", "transfer")}
    return rows, {"source_sha256": sources(),
                  "weights": {k: sha(ROOT/r["artifact"]/"best.pt") for k, r in rows.items()},
                  "policy": "equal_ensemble_flip", "calibrated": False,
                  "teacher_seed": 42}


def augment_raw(images):
    """与旧BatchTransform逐个随机调用一致，返回归一化前的共享视图。"""
    batch = len(images)
    padded = F.pad(images, (4, 4, 4, 4))
    rows = torch.randint(0, 9, (batch, 1, 1), device=images.device) + torch.arange(32, device=images.device).view(1, 32, 1)
    cols = torch.randint(0, 9, (batch, 1, 1), device=images.device) + torch.arange(32, device=images.device).view(1, 1, 32)
    view = padded.permute(0, 2, 3, 1)[torch.arange(batch, device=images.device).view(batch, 1, 1), rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand((batch, 1, 1, 1), device=images.device) < .5
    return torch.where(flip, view.flip(-1), view).contiguous(memory_format=torch.channels_last)


class Teacher:
    def __init__(self):
        rows, self.identity = identity()
        self.parts = []
        for family, row in rows.items():
            saved = torch.load(ROOT/row["artifact"]/"best.pt", map_location="cpu", weights_only=False)
            assert saved["source_sha256"] == self.identity["source_sha256"]
            model = build(saved["config"]).cuda().to(memory_format=torch.channels_last)
            model.load_state_dict(saved["model"])
            model.eval().requires_grad_(False)
            transform = Transform(saved["stats"], saved["config"]).cuda()
            before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.parts.append((model, transform, saved["config"], before))

    @torch.no_grad()
    def probabilities(self, images):
        combined = torch.zeros((len(images), 10), dtype=torch.float64, device=images.device)
        for model, transform, cfg, _ in self.parts:
            assert not model.training
            batch_size = cfg["batch_size"]
            for flip in (False, True):
                for start in range(0, len(images), batch_size):
                    x = images[start:start+batch_size]
                    if flip:
                        x = x.flip(-1)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.get("amp", True)):
                        logits = model(transform(x, training=False))
                    combined[start:start+len(x)] += logits.float().softmax(1).double() / 4
        return combined.float()

    def unchanged(self):
        for model, _, _, before in self.parts:
            assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
            assert all(torch.equal(v.detach().cpu(), before[k]) for k, v in model.state_dict().items())
        assert identity()[1] == self.identity
        return True
