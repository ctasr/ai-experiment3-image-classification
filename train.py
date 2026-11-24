"""训练与验证，支持在完整epoch边界恢复；不加载测试集。"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
from torch import nn
from data import ROOT, BatchTransform, make_loader, training_data
from models import make_model


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def epoch_pass(model, loader, transform, criterion, device, optimizer=None, amp=True):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = transform(images, training=training)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
                ce = nn.functional.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            count = len(labels)
            totals += torch.stack((loss.detach().double() * count, ce.detach().double() * count, (logits.argmax(1) == labels).sum().double(), torch.tensor(count, device=device, dtype=torch.float64)))
    loss_sum, ce_sum, correct, count = totals.cpu().tolist()
    if not math.isfinite(loss_sum):
        raise FloatingPointError("训练损失非有限，记录该配置发散")
    if count != len(loader.dataset):
        raise RuntimeError("未完整遍历数据集")
    return {"loss": loss_sum / count, "ce": ce_sum / count, "accuracy": correct / count, "samples": int(count)}


def run(config, run_id, resume=True, stop_after=None):
    torch.set_num_threads(8)
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not config.get("allow_cpu", False):
        raise RuntimeError("正式实验需要CUDA，避免误用CPU造成错误时间预估")
    folder = ROOT / "runs" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    config_path = folder / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("同名实验的配置已改变，不能覆盖或混用")
    write_json(config_path, config)
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ("models.py", "data.py", "train.py", "prepare_data.py")}
    environment = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": __import__("torchvision").__version__, "numpy": np.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu", "platform": platform.platform(), "deterministic_algorithms": True, "source_sha256": sources}
    train_set, validation_set, stats = training_data()
    if config.get("pilot_samples"):
        # 仅供运行检查，目录单独标注pilot，不纳入正式模型比较。
        from torch.utils.data import Subset
        train_set = Subset(train_set, range(config["pilot_samples"]))
        validation_set = Subset(validation_set, range(min(1000, len(validation_set))))
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = make_loader(train_set, config["batch_size"], generator, training=True)
    validation_loader = make_loader(validation_set, config["batch_size"])
    transform = BatchTransform(stats, config.get("augmentation", True), config.get("cutout", 0)).to(device)
    model = make_model(config).to(device, memory_format=torch.channels_last)
    if config.get("optimizer", "sgd") == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=.9, weight_decay=config["weight_decay"], nesterov=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    epochs = config["epochs"]
    warmup = min(config.get("warmup_epochs", 5), epochs)
    def factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(epoch - warmup, epochs - warmup) / max(1, epochs - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0))
    start_epoch, best_accuracy, best_ce, bad_epochs = 0, -1., float("inf"), 0
    history = []
    checkpoint_path = folder / "last.pt"
    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["config"] != config or saved["source_sha256"] != sources or saved["stats"] != stats:
            raise ValueError("断点的配置、源码或数据统计量不同，禁止混合恢复")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = saved["epoch"]
        history = saved["history"]
        best_accuracy, best_ce, bad_epochs = saved["best_accuracy"], saved["best_ce"], saved["bad_epochs"]
        restore_rng(saved["rng"], generator)
        print(f"恢复 {run_id}：已完成{start_epoch}轮", flush=True)
    write_json(folder / "environment.json", environment)
    parameters = sum(p.numel() for p in model.parameters())
    began = time.monotonic()
    try:
        for epoch in range(start_epoch, epochs):
            if (ROOT / "PAUSE").exists():
                write_json(folder / "status.json", {"state": "paused", "completed_epochs": epoch})
                return {"state": "paused"}
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_started = time.monotonic()
            trained = epoch_pass(model, train_loader, transform, criterion, device, optimizer, config.get("amp", True))
            validated = epoch_pass(model, validation_loader, transform, criterion, device, amp=config.get("amp", True))
            scheduler.step()
            improved = (validated["accuracy"], -validated["ce"]) > (best_accuracy, -best_ce)
            if improved:
                best_accuracy, best_ce, bad_epochs = validated["accuracy"], validated["ce"], 0
            else:
                bad_epochs += 1
            row = {"epoch": epoch + 1, "learning_rate": current_lr, **{f"train_{k}": v for k, v in trained.items()}, **{f"validation_{k}": v for k, v in validated.items()}, "seconds": time.monotonic() - epoch_started}
            history.append(row)
            state = {"epoch": epoch + 1, "config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state(generator), "history": history, "best_accuracy": best_accuracy, "best_ce": best_ce, "bad_epochs": bad_epochs, "source_sha256": sources, "stats": stats}
            save_checkpoint(checkpoint_path, state)
            if improved:
                save_checkpoint(folder / "best.pt", {"model": model.state_dict(), "config": config, "epoch": epoch + 1, "validation_accuracy": best_accuracy, "validation_ce": best_ce, "stats": stats, "source_sha256": sources})
            with (folder / "history.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            write_json(folder / "status.json", {"state": "running", "completed_epochs": epoch + 1, "total_epochs": epochs, "best_validation_accuracy": best_accuracy, "latest": row})
            print(f"{run_id} {epoch+1}/{epochs} loss={trained['loss']:.4f} train={trained['accuracy']:.4f} val={validated['accuracy']:.4f} best={best_accuracy:.4f} {row['seconds']:.1f}s", flush=True)
            if stop_after and epoch + 1 >= stop_after and epoch + 1 < epochs:
                write_json(folder / "status.json", {"state": "paused", "completed_epochs": epoch + 1})
                return {"state": "paused"}
            patience = config.get("patience", 0)
            if patience and epoch + 1 >= config.get("minimum_epochs", epochs // 2) and bad_epochs >= patience:
                break
    except FloatingPointError as exc:
        result = {"state": "diverged", "run_id": run_id, "config": config, "epochs_completed": len(history), "best_validation_accuracy": -1., "best_validation_ce": 1e30, "parameters": parameters, "reason": str(exc)}
        write_json(folder / "result.json", result)
        write_json(folder / "status.json", result)
        return result
    except KeyboardInterrupt:
        write_json(folder / "status.json", {"state": "interrupted", "completed_epochs": len(history), "resume_from": "last fully completed epoch"})
        raise
    result = {"state": "completed", "run_id": run_id, "config": config, "epochs_completed": len(history), "best_epoch": max(history, key=lambda row: (row["validation_accuracy"], -row["validation_ce"]))["epoch"], "best_validation_accuracy": best_accuracy, "best_validation_ce": best_ce, "parameters": parameters, "training_seconds": sum(row["seconds"] for row in history), "current_process_seconds": time.monotonic() - began}
    write_json(folder / "result.json", result)
    write_json(folder / "status.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text(encoding="utf-8")), args.run_id)
