"""仅为45000张训练图生成教师弱视图概率，不产生验证或测试训练目标。"""
import json
from pathlib import Path
import numpy as np
import torch
from data import ROOT, training_data, read_partition, make_loader
from train import seed_everything, write_json
from extensions.models import build, Transform
from extensions.train import SOURCES as OLD_SOURCES

OUT = ROOT / "results_distillation"
CACHE = OUT / "teacher_train_probabilities.npz"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return {name: sha(ROOT / name) for name in OLD_SOURCES}


def prepare():
    OUT.mkdir(exist_ok=True)
    manifest, train_indices, validation_indices = read_partition()
    families = read(ROOT / "results_round3/selected_runs.json")
    teacher = {role: next(row for row in families[role] if row["config"]["seed"] == 42) for role in ("cutmix", "transfer")}
    identity = {"policy": "equal_ensemble_flip", "seed": 42, "source_sha256": sources(), "targets_code_sha256": sha(Path(__file__)), "split_sha256": manifest["split_sha256"], "weights": {role: sha(ROOT / row["artifact"] / "best.pt") for role, row in teacher.items()}, "views": ["original", "horizontal_flip"], "calibrated": False}
    path = OUT / "teacher_manifest.json"
    if path.exists():
        previous = read(path)
        assert previous["identity"] == identity and previous["cache_sha256"] == sha(CACHE)
        cached = np.load(CACHE)
        assert np.array_equal(cached["original_indices"], train_indices)
        assert cached["probabilities"].shape == (45000, 10)
        return previous
    torch.set_num_threads(8)
    seed_everything(42)
    train_set, validation_set, _ = training_data()
    del validation_set
    assert not set(train_indices) & set(validation_indices)
    assert len(train_set) == 45000 and np.array_equal(np.bincount(train_set.tensors[1].numpy()), np.full(10, 4500))
    combined = np.zeros((45000, 10), dtype=np.float64)
    for role, row in teacher.items():
        saved = torch.load(ROOT / row["artifact"] / "best.pt", map_location="cpu", weights_only=False)
        assert saved["source_sha256"] == identity["source_sha256"] and saved["config"] == row["config"]
        model = build(saved["config"]).cuda().to(memory_format=torch.channels_last)
        model.load_state_dict(saved["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        transform = Transform(saved["stats"], saved["config"]).cuda()
        for flip in (False, True):
            probabilities = []
            with torch.inference_mode():
                for x, labels in make_loader(train_set, saved["config"]["batch_size"]):
                    x = x.cuda()
                    if flip:
                        x = x.flip(-1)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=saved["config"].get("amp", True)):
                        logits = model(transform(x, training=False))
                    probabilities.append(logits.float().softmax(1).cpu())
            values = torch.cat(probabilities).numpy()
            assert values.shape == combined.shape and np.allclose(values.sum(1), 1, atol=1e-6)
            combined += values.astype(np.float64) / 4.
            print("教师训练目标", role, "flip=" + str(flip), len(values), flush=True)
        assert all(torch.equal(tensor.cpu(), saved["model"][name]) for name, tensor in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        del model, transform
        torch.cuda.empty_cache()
    assert np.isfinite(combined).all() and np.allclose(combined.sum(1), 1, atol=1e-6)
    np.savez_compressed(CACHE, probabilities=combined, labels=train_set.tensors[1].numpy(), original_indices=np.array(train_indices))
    result = {"identity": identity, "cache_sha256": sha(CACHE), "samples": 45000, "per_class": [4500] * 10, "training_only": True, "validation_or_test_targets_generated": False, "teacher_weights_and_bn_unchanged": True, "teacher_no_grad": True, "target_view": "固定原图/翻转弱视图，学生强增强使用同样本目标，未逐裁剪在线计算教师", "external_knowledge": "教师含ImageNet预训练模型，学生间接使用其知识"}
    assert all(sha(ROOT / row["artifact"] / "best.pt") == identity["weights"][role] for role, row in teacher.items())
    write_json(path, result)
    return result


if __name__ == "__main__":
    prepare()
