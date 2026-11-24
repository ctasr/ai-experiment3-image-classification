"""拓展训练：复用首轮训练/验证步骤，增加混合增强与迁移学习参数组。"""
import argparse
import csv
import hashlib
import json
import math
import platform
import time
import numpy as np
import torch
from torch import nn
from data import ROOT, training_data, make_loader
from train import seed_everything, rng_state, restore_rng, save_checkpoint, write_json, epoch_pass
from extensions.models import build, Transform, parameter_groups, WEIGHTS

DIRECTORY = ROOT / "runs/round2"
SOURCES = ("train.py", "data.py", "models.py", "prepare_data.py", "extensions/train.py", "extensions/models.py")


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def mixed_epoch(model, loader, transform, device, optimizer, method, alpha, amp):
    model.train()
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = transform(x, training=True)
        order = torch.randperm(len(y), device=device)
        lam = float(np.random.beta(alpha, alpha))
        if method == "mixup":
            x = lam * x + (1 - lam) * x[order]
        elif method == "cutmix":
            height, width = x.shape[-2:]
            cut_w, cut_h = int(width * math.sqrt(1 - lam)), int(height * math.sqrt(1 - lam))
            cx, cy = int(np.random.randint(width)), int(np.random.randint(height))
            left, right = max(0, cx - cut_w // 2), min(width, cx + (cut_w + 1) // 2)
            top, bottom = max(0, cy - cut_h // 2), min(height, cy + (cut_h + 1) // 2)
            x[:, :, top:bottom, left:right] = x[order, :, top:bottom, left:right]
            lam = 1 - (right - left) * (bottom - top) / (height * width)
        else:
            raise ValueError(method)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
            loss = lam * nn.functional.cross_entropy(logits, y) + (1 - lam) * nn.functional.cross_entropy(logits, y[order])
        loss.backward()
        optimizer.step()
        correct = lam * (logits.argmax(1) == y).sum() + (1 - lam) * (logits.argmax(1) == y[order]).sum()
        totals += torch.stack((loss.detach().double() * len(y), correct.detach().double(), torch.tensor(len(y), device=device).double()))
    loss, correct, count = totals.cpu().tolist()
    if not math.isfinite(loss):
        raise FloatingPointError("混合增强训练损失发散")
    assert count == len(loader.dataset)
    return {"loss": loss / count, "ce": loss / count, "accuracy": correct / count, "samples": int(count)}


def run(config, run_id, stop_after=None):
    torch.set_num_threads(8)
    seed_everything(config["seed"])
    assert torch.cuda.is_available(), "需要CUDA，不自动降级CPU"
    device = torch.device("cuda")
    folder = DIRECTORY / run_id
    folder.mkdir(parents=True, exist_ok=True)
    config_path = folder / "config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding="utf-8")) == config
    write_json(config_path, config)
    sources = source_hashes()
    result_path = folder / "result.json"
    if result_path.exists():
        assert json.loads((folder / "environment.json").read_text(encoding="utf-8"))["source_sha256"] == sources
        return json.loads(result_path.read_text(encoding="utf-8"))
    train_set, validation_set, stats = training_data()
    if config.get("pilot_samples"):
        from torch.utils.data import Subset
        train_set = Subset(train_set, range(config["pilot_samples"]))
        validation_set = Subset(validation_set, range(1000))
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = make_loader(train_set, config["batch_size"], generator, training=True)
    validation_loader = make_loader(validation_set, config["batch_size"])
    transform = Transform(stats, config).to(device)
    model = build(config).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(parameter_groups(model, config), lr=config["learning_rate"], momentum=.9, nesterov=True, weight_decay=config["weight_decay"])
    epochs, warmup = config["epochs"], min(config.get("warmup_epochs", 5), config["epochs"])
    def factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(epoch - warmup, epochs - warmup) / max(1, epochs - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0))
    start, best_accuracy, best_ce, history = 0, -1., float("inf"), []
    environment = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": __import__("torchvision").__version__, "numpy": np.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "source_sha256": sources, "normalization": "ImageNet fixed statistics" if config.get("model") == "transfer" else stats, "train_accuracy_definition": "lambda-weighted accuracy on mixed labels" if config.get("mix", "none") != "none" else "hard-label accuracy on augmented training images"}
    if config.get("model") == "transfer" and config.get("pretrained", True):
        environment["pretrained_sha256"] = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    checkpoint = folder / "last.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["config"] == config and saved["source_sha256"] == sources and saved["stats"] == stats
        assert saved["pretrained_sha256"] == environment.get("pretrained_sha256")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, history = saved["epoch"], saved["history"]
        best_accuracy, best_ce = saved["best_accuracy"], saved["best_ce"]
        restore_rng(saved["rng"], generator)
    write_json(folder / "environment.json", environment)
    try:
        for epoch in range(start, epochs):
            if (ROOT / "PAUSE_ROUND2").exists():
                write_json(folder / "status.json", {"state": "paused", "completed_epochs": epoch})
                return {"state": "paused"}
            began, learning_rate = time.monotonic(), optimizer.param_groups[-1]["lr"]
            if config.get("mix", "none") != "none":
                assert not config.get("label_smoothing", 0), "先单独控制混合增强因素"
                trained = mixed_epoch(model, train_loader, transform, device, optimizer, config["mix"], config["mix_alpha"], config.get("amp", True))
            else:
                trained = epoch_pass(model, train_loader, transform, criterion, device, optimizer, config.get("amp", True))
            validated = epoch_pass(model, validation_loader, transform, criterion, device, amp=config.get("amp", True))
            scheduler.step()
            improved = (validated["accuracy"], -validated["ce"]) > (best_accuracy, -best_ce)
            if improved:
                best_accuracy, best_ce = validated["accuracy"], validated["ce"]
            row = {"epoch": epoch + 1, "learning_rate": learning_rate, **{f"train_{key}": value for key, value in trained.items()}, **{f"validation_{key}": value for key, value in validated.items()}, "seconds": time.monotonic() - began}
            history.append(row)
            state = {"epoch": epoch + 1, "config": config, "source_sha256": sources, "stats": stats, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state(generator), "history": history, "best_accuracy": best_accuracy, "best_ce": best_ce, "pretrained_sha256": environment.get("pretrained_sha256")}
            save_checkpoint(checkpoint, state)
            if improved:
                save_checkpoint(folder / "best.pt", {key: value for key, value in state.items() if key not in ("optimizer", "scheduler", "rng", "history")})
            with (folder / "history.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            write_json(folder / "status.json", {"state": "running", "completed_epochs": epoch + 1, "total_epochs": epochs, "best_validation_accuracy": best_accuracy, "latest": row})
            print(f"{run_id} {epoch+1}/{epochs} loss={trained['loss']:.4f} train={trained['accuracy']:.4f} val={validated['accuracy']:.4f} best={best_accuracy:.4f} {row['seconds']:.1f}s", flush=True)
            if stop_after and epoch + 1 == stop_after and stop_after < epochs:
                return {"state": "paused"}
    except FloatingPointError as exc:
        result = {"state": "diverged", "run_id": run_id, "config": config, "best_validation_accuracy": -1., "best_validation_ce": 1e30, "epochs_completed": len(history), "reason": str(exc)}
        write_json(result_path, result)
        write_json(folder / "status.json", result)
        return result
    chosen = max(history, key=lambda row: (row["validation_accuracy"], -row["validation_ce"]))
    result = {"state": "completed", "run_id": run_id, "config": config, "best_validation_accuracy": best_accuracy, "best_validation_ce": best_ce, "best_epoch": chosen["epoch"], "epochs_completed": len(history), "parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "training_seconds": sum(row["seconds"] for row in history)}
    write_json(result_path, result)
    write_json(folder / "status.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(json.loads(open(args.config, encoding="utf-8").read()), args.run_id)
