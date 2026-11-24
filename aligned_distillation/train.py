"""独立在线对齐版本，旧源码不改；完整epoch断点包含全部新训练源码SHA。"""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from data import ROOT
from train import seed_everything, write_json
import extensions.train as engine
from distillation.train import loss_parts, cutmix_patch, mix_targets, SOURCES as OLD_SOURCES
from distillation.targets import read, sha
from aligned_distillation.teacher import Teacher, augment_raw, identity, OUT

DIRECTORY = ROOT / "runs/aligned_distillation"
BASE_MIXED = engine.mixed_epoch
BASE_SOURCES = tuple(engine.SOURCES)
SOURCES = tuple(OLD_SOURCES) + ("aligned_distillation/teacher.py", "aligned_distillation/train.py")
ACTIVE = None
TEACHER = None


def source_hashes():
    return {name: sha(ROOT/name) for name in SOURCES}


def target_for_view(teacher, view, order, lam, temperature, alignment):
    """crop_mix混合每张当前裁剪的目标；mixed_view直接查询实际CutMix输入。"""
    probability = teacher.probabilities(view).clamp_min(1e-30)
    target = (probability.log() / temperature).softmax(1)
    if alignment == "crop_mix":
        return mix_targets(target, order, lam)
    assert alignment == "mixed_view"
    return target


def mixed_epoch(model, loader, transform, device, optimizer, method, mix_alpha, amp):
    alpha = ACTIVE["distill_alpha"]
    if alpha == 0:
        return BASE_MIXED(model, loader, transform, device, optimizer, method, mix_alpha, amp)
    assert method == "cutmix" and not ACTIVE.get("cutout", 0)
    temperature, alignment = ACTIVE["distill_temperature"], ACTIVE["alignment"]
    model.train()
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        view = augment_raw(x)
        order = torch.randperm(len(y), device=device)
        lam = float(np.random.beta(mix_alpha, mix_alpha))
        cx, cy = int(np.random.randint(32)), int(np.random.randint(32))
        if alignment == "crop_mix":
            # 教师查询当前裁剪，然后按真正的CutMix面积混合软标签。
            probability = TEACHER.probabilities(view).clamp_min(1e-30)
            raw_target = (probability.log() / temperature).softmax(1)
            view, lam = cutmix_patch(view, order, lam, cx, cy)
            soft = mix_targets(raw_target, order, lam)
        else:
            # 教师直接查询学生将要看到的同一张CutMix图像，不再近似混合软目标。
            assert alignment == "mixed_view"
            view, lam = cutmix_patch(view, order, lam, cx, cy)
            soft = target_for_view(TEACHER, view, order, lam, temperature, alignment)
        x = transform(view, training=False)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
            loss, hard, kd = loss_parts(logits, y, order, lam, soft, temperature, alpha)
        loss.backward()
        optimizer.step()
        correct = lam*(logits.argmax(1) == y).sum() + (1-lam)*(logits.argmax(1) == y[order]).sum()
        totals += torch.stack((loss.detach().double()*len(y), hard.detach().double()*len(y), kd.detach().double()*len(y), correct.detach().double(), torch.tensor(len(y), device=device).double()))
    loss, hard, kd, correct, count = totals.cpu().tolist()
    assert count == len(loader.dataset)
    if not all(math.isfinite(v) for v in (loss, hard, kd, correct)):
        raise FloatingPointError("在线对齐蒸馏出现非有限指标")
    return {"loss": loss/count, "ce": hard/count, "kd": kd/count, "accuracy": correct/count, "samples": int(count)}


def run(cfg, name, stop_after=None, reference=False):
    global ACTIVE, TEACHER
    ACTIVE = cfg
    assert cfg["alignment"] in ("crop_mix", "mixed_view")
    assert 0 <= cfg["distill_alpha"] <= 1 and cfg["distill_temperature"] > 0
    torch.set_num_threads(8)
    seed_everything(cfg["seed"])
    expected = read(OUT/"teacher_identity.json")
    assert identity()[1] == expected
    TEACHER = Teacher() if cfg["distill_alpha"] > 0 else None
    engine.DIRECTORY = DIRECTORY
    engine.SOURCES = BASE_SOURCES if reference else SOURCES
    engine.mixed_epoch = BASE_MIXED if reference else mixed_epoch
    if (OUT/"source_manifest.json").exists() and not reference:
        assert read(OUT/"source_manifest.json")["source_sha256"] == source_hashes()
    folder = DIRECTORY/name
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder/"objective_metadata.json", {"teacher_identity": expected, "alignment": cfg["alignment"], "loss": "(1-alpha)*area-weighted hard CE + alpha*T^2 KL", "train_accuracy": "actual CutMix area lambda-weighted label hit rate", "teacher_query": "shared current crop; mixed_view also shares actual CutMix patch", "external_ImageNet_knowledge": True})
    result = engine.run(cfg, name, stop_after)
    if TEACHER is not None:
        assert TEACHER.unchanged()
    write_json(folder/"teacher_check.json", {"weights_bn_gradients_unchanged": True, "teacher_loaded": TEACHER is not None})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    run(read(Path(args.config)), args.run_id, args.stop_after, args.reference)
