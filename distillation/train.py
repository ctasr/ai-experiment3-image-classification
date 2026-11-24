"""独立蒸馏适配器，复用已验证的训练循环、优化器和完整epoch断点。"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from data import ROOT
from train import seed_everything, write_json
import extensions.train as engine
from distillation.targets import CACHE, OUT, prepare, read, sha

DIRECTORY = ROOT / "runs/distillation"
BASE_SOURCES = tuple(engine.SOURCES)
BASE_MIXED = engine.mixed_epoch
BASE_DATA = engine.training_data
SOURCES = BASE_SOURCES + ("distillation/train.py", "distillation/targets.py")
ACTIVE = None
TARGETS = None


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCES}


def training_data():
    train_set, validation_set, stats = BASE_DATA()
    if ACTIVE["distill_alpha"] > 0:
        cached = np.load(CACHE)
        assert np.array_equal(train_set.tensors[1].numpy(), cached["labels"])
        train_set = TensorDataset(*train_set.tensors, torch.arange(len(train_set)))
    return train_set, validation_set, stats


def mix_targets(targets, order, lam):
    return lam * targets + (1 - lam) * targets[order]


def cutmix_patch(x, order, lam, cx, cy):
    height, width = x.shape[-2:]
    cut_w, cut_h = int(width * math.sqrt(1 - lam)), int(height * math.sqrt(1 - lam))
    left, right = max(0, cx - cut_w // 2), min(width, cx + (cut_w + 1) // 2)
    top, bottom = max(0, cy - cut_h // 2), min(height, cy + (cut_h + 1) // 2)
    x[:, :, top:bottom, left:right] = x[order, :, top:bottom, left:right]
    return x, 1 - (right - left) * (bottom - top) / (height * width)


def loss_parts(logits, labels, order, lam, soft_targets, temperature, alpha):
    hard = lam * F.cross_entropy(logits, labels) + (1 - lam) * F.cross_entropy(logits, labels[order])
    if alpha == 0:
        return hard, hard, hard.detach() * 0
    kl = F.kl_div(F.log_softmax(logits.float() / temperature, dim=1), soft_targets.float(), reduction="batchmean") * temperature ** 2
    return (1 - alpha) * hard + alpha * kl, hard, kl


def mixed_epoch(model, loader, transform, device, optimizer, method, mix_alpha, amp):
    alpha, temperature = ACTIVE["distill_alpha"], ACTIVE["distill_temperature"]
    if alpha == 0:
        return BASE_MIXED(model, loader, transform, device, optimizer, method, mix_alpha, amp)
    assert method == "cutmix"
    model.train()
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    for x, y, indices in loader:
        x, y, indices = x.to(device, non_blocking=True), y.to(device, non_blocking=True), indices.to(device, non_blocking=True)
        x = transform(x, training=True)
        order = torch.randperm(len(y), device=device)
        lam = float(np.random.beta(mix_alpha, mix_alpha))
        height, width = x.shape[-2:]
        cx, cy = int(np.random.randint(width)), int(np.random.randint(height))
        x, lam = cutmix_patch(x, order, lam, cx, cy)
        soft = mix_targets(TARGETS[indices], order, lam)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
            loss, hard, kd = loss_parts(logits, y, order, lam, soft, temperature, alpha)
        loss.backward()
        optimizer.step()
        correct = lam * (logits.argmax(1) == y).sum() + (1 - lam) * (logits.argmax(1) == y[order]).sum()
        totals += torch.stack((loss.detach().double() * len(y), hard.detach().double() * len(y), kd.detach().double() * len(y), correct.detach().double(), torch.tensor(len(y), device=device).double()))
    loss, hard, kd, correct, count = totals.cpu().tolist()
    assert count == len(loader.dataset)
    if not all(math.isfinite(value) for value in (loss, hard, kd, correct)):
        raise FloatingPointError("蒸馏损失非有限值")
    return {"loss": loss / count, "ce": hard / count, "kd": kd / count, "accuracy": correct / count, "samples": int(count)}


def run(config, run_id, stop_after=None, reference=False):
    global ACTIVE, TARGETS
    assert 0 <= config["distill_alpha"] <= 1 and config["distill_temperature"] > 0
    torch.set_num_threads(8)
    seed_everything(config["seed"])
    metadata = prepare()
    assert metadata["cache_sha256"] == config["teacher_cache_sha256"]
    ACTIVE = config
    TARGETS = None
    if config["distill_alpha"] > 0:
        probability = torch.from_numpy(np.load(CACHE)["probabilities"]).float().clamp_min(1e-30)
        TARGETS = (probability.log() / config["distill_temperature"]).softmax(1).cuda()
    engine.DIRECTORY = DIRECTORY
    engine.SOURCES = BASE_SOURCES if reference else SOURCES
    engine.mixed_epoch = BASE_MIXED if reference else mixed_epoch
    engine.training_data = BASE_DATA if reference else training_data
    if not reference and (OUT / "source_manifest.json").exists():
        assert read(OUT / "source_manifest.json")["source_sha256"] == source_hashes()
    folder = DIRECTORY / run_id
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "objective_metadata.json", {"total_loss": "(1-alpha)*mixed hard CE + alpha*T^2*KL(teacher_target||student_T)", "train_ce": "CutMix硬标签加权CE", "train_kd": "温度缩放后的KL", "train_accuracy": "实际面积lambda加权标签命中率", "validation": "无随机增强、原始硬标签CE和accuracy", "teacher_cache_sha256": metadata["cache_sha256"], "reference_old_engine": reference})
    result = engine.run(config, run_id, stop_after)
    assert sha(CACHE) == metadata["cache_sha256"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    run(read(Path(args.config)), args.run_id, args.stop_after, args.reference)
