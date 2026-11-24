"""Independent deterministic transfer engine; checkpoints never use the old namespace."""
import csv
import math
import platform
import time
import numpy as np
import torch
from torch import nn
from data import training_data, make_loader
from train import seed_everything, rng_state, restore_rng, save_checkpoint, epoch_pass
from extensions.models import Transform
from . import models
from .common import ROOT, OUT, RUNS, read, write, sha, source_hashes, assert_inputs


def check_best_checkpoint(folder, last):
    """Best-first commit may be one epoch ahead after a crash; replay from last, never its counters."""
    best = torch.load(folder/"best.pt", map_location="cpu", weights_only=False)
    for key in ("config", "source_sha256", "stats", "pretrained_sha256"):
        assert best[key] == last[key], f"best/last provenance differs: {key}"
    chosen = max(last["history"], key=lambda row: (row["validation_accuracy"], -row["validation_ce"]))
    assert chosen["validation_accuracy"] == last["best_accuracy"] and chosen["validation_ce"] == last["best_ce"]
    if best["epoch"] == chosen["epoch"]:
        assert best["best_accuracy"] == last["best_accuracy"] and best["best_ce"] == last["best_ce"]
        return {"state": "consistent", "best_epoch": best["epoch"], "last_epoch": last["epoch"]}
    assert best["epoch"] == last["epoch"]+1 and last["epoch"] < last["config"]["epochs"], "Missing/stale best is not a recoverable best-first transaction"
    assert (best["best_accuracy"], -best["best_ce"]) > (last["best_accuracy"], -last["best_ce"])
    recovery = {"state": "best_one_epoch_ahead_replay_from_last", "best_epoch": best["epoch"], "last_epoch": last["epoch"],
                "ahead_best_sha256": sha(folder/"best.pt"), "rule": "Restore model/optimizer/RNG/history exclusively from last; recompute the next epoch, never import ahead-best counters."}
    write(folder/"resume_transaction_recovery.json", recovery)
    return recovery


def run(cfg, name, stop_after=None):
    assert_inputs()
    assert cfg["epochs"] > 0 and cfg["batch_size"] == 128 and cfg["image_size"] == 96
    assert cfg.get("mix", "none") == "none" and not cfg.get("label_smoothing", 0) and not cfg.get("patience", 0)
    assert not any(k.startswith("distill") for k in cfg)
    torch.set_num_threads(8)
    seed_everything(cfg["seed"])
    assert torch.cuda.is_available(), "Formal Mobile study requires CUDA; no silent CPU fallback"
    folder = RUNS/name; folder.mkdir(parents=True, exist_ok=True)
    if (folder/"config.json").exists():
        assert read(folder/"config.json") == cfg, "Different configuration cannot reuse a checkpoint"
    write(folder/"config.json", cfg)
    sources = source_hashes(); identity = models.weight_identity(cfg)
    if (folder/"result.json").exists():
        result = read(folder/"result.json")
        assert read(folder/"environment.json")["source_sha256"] == sources
        saved = torch.load(folder/"last.pt", map_location="cpu", weights_only=False)
        assert saved["source_sha256"] == sources and saved["config"] == cfg and saved["pretrained_sha256"] == identity["sha256"]
        check_best_checkpoint(folder, saved)
        assert result["state"] == "completed" and result["epochs_completed"] == cfg["epochs"]
        return result
    train_set, val_set, stats = training_data()
    if cfg.get("pilot_samples"):
        from torch.utils.data import Subset
        train_set = Subset(train_set, range(cfg["pilot_samples"])); val_set = Subset(val_set, range(1000))
    assert len(train_set) == cfg.get("pilot_samples", 45000) and len(val_set) == (1000 if cfg.get("pilot_samples") else 5000)
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = make_loader(train_set, 128, generator, training=True)
    validation_loader = make_loader(val_set, 128)
    transform = Transform(stats, cfg).cuda()
    model = models.build(cfg).cuda().to(memory_format=torch.channels_last)
    initial = models.metadata(model, cfg)
    before_bn = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "running_mean" in k}
    optimizer = torch.optim.SGD(models.parameter_groups(model, cfg), lr=cfg["learning_rate"], momentum=.9,
                                nesterov=True, weight_decay=cfg["weight_decay"])
    epochs, warm = cfg["epochs"], min(cfg.get("warmup_epochs", 5), cfg["epochs"])
    def factor(epoch):
        return (epoch+1)/warm if epoch < warm else .5*(1+math.cos(math.pi*min(epoch-warm, epochs-warm)/max(1, epochs-warm)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.)
    start, best_accuracy, best_ce, history = 0, -1., float("inf"), []
    env = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
           "numpy": np.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
           "source_sha256": sources, "normalization": "ImageNet fixed statistics", "image_size": 96,
           "train_accuracy_definition": "hard-label accuracy on augmented training images",
           "pretraining_identity": identity, "pretrained_sha256": identity["sha256"],
           "split_sha256": sha(ROOT/"data/splits/train_validation_seed42.json"),
           "normalization_sha256": sha(ROOT/"data/normalization.json"), "initialization": initial}
    checkpoint = folder/"last.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["config"] == cfg and saved["source_sha256"] == sources and saved["stats"] == stats
        assert saved["pretrained_sha256"] == identity["sha256"]
        check_best_checkpoint(folder, saved)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        start, history = saved["epoch"], saved["history"]
        best_accuracy, best_ce = saved["best_accuracy"], saved["best_ce"]
        restore_rng(saved["rng"], generator)
    if (folder/"environment.json").exists():
        old = read(folder/"environment.json")
        assert old == env
    else:
        write(folder/"environment.json", env)
    write(folder/"initialization.json", initial)
    if stop_after and start >= stop_after and start < epochs:
        return {"state": "paused", "completed_epochs": start}
    torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(start, epochs):
            if (ROOT/"PAUSE_MOBILE").exists():
                write(folder/"status.json", {"state": "paused", "completed_epochs": epoch, "reason": "PAUSE_MOBILE"})
                return {"state": "paused"}
            began, lr = time.monotonic(), optimizer.param_groups[-1]["lr"]
            trained = epoch_pass(model, train_loader, transform, criterion, torch.device("cuda"), optimizer, cfg.get("amp", True))
            assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters() if p.requires_grad), "Missing or nonfinite gradient"
            validated = epoch_pass(model, validation_loader, transform, criterion, torch.device("cuda"), amp=cfg.get("amp", True))
            scheduler.step()
            improved = (validated["accuracy"], -validated["ce"]) > (best_accuracy, -best_ce)
            if improved:
                best_accuracy, best_ce = validated["accuracy"], validated["ce"]
            row = {"epoch": epoch+1, "learning_rate": lr, **{f"train_{k}": v for k, v in trained.items()},
                   **{f"validation_{k}": v for k, v in validated.items()}, "seconds": time.monotonic()-began}
            history.append(row)
            state = {"epoch": epoch+1, "config": cfg, "source_sha256": sources, "stats": stats, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state(generator),
                     "history": history, "best_accuracy": best_accuracy, "best_ce": best_ce,
                     "pretrained_sha256": identity["sha256"]}
            # Commit improved best first. A crash may leave best ahead, but never last claiming a missing best.
            if improved:
                save_checkpoint(folder/"best.pt", {k: v for k, v in state.items() if k not in ("optimizer", "scheduler", "rng", "history")})
            save_checkpoint(checkpoint, state)
            with (folder/"history.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
            write(folder/"runtime_integrity.json", {"actual_model": type(model.backbone).__name__, "gradients_finite_and_present": True,
                  "batchnorm_updated": any(not torch.equal(v.detach().cpu(), before_bn[k]) for k, v in model.state_dict().items() if k in before_bn),
                  "all_parameters_trainable": all(p.requires_grad for p in model.parameters()),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "train_samples": trained["samples"], "validation_samples": validated["samples"],
                  "last_train_batch": len(train_set)%128 or 128, "last_validation_batch": len(val_set)%128 or 128,
                  "teacher_loaded": False, "pretrained_parameters_loaded": bool(cfg["pretrained"]),
                  "timing_scope": "epoch wall includes gradient-finiteness synchronization, pauses/system load; not pure GPU time or FLOPs"})
            write(folder/"status.json", {"state": "running", "completed_epochs": epoch+1, "total_epochs": epochs,
                  "best_validation_accuracy": best_accuracy, "latest": row})
            print(f"{name} {epoch+1}/{epochs} train={trained['accuracy']:.5f} val={validated['accuracy']:.5f} ce={validated['ce']:.6f} {row['seconds']:.2f}s", flush=True)
            if stop_after and epoch+1 >= stop_after and epoch+1 < epochs:
                write(folder/"status.json", {"state": "paused", "completed_epochs": epoch+1, "total_epochs": epochs, "reason": "preflight_stop_after"})
                return {"state": "paused"}
    except Exception as exc:
        write(folder/"failure.json", {"error": repr(exc), "completed_epochs": len(history), "resume": "last complete epoch only; preserve evidence"})
        raise
    chosen = max(history, key=lambda v: (v["validation_accuracy"], -v["validation_ce"]))
    check_best_checkpoint(folder, torch.load(checkpoint, map_location="cpu", weights_only=False))
    result = {"state": "completed", "run_id": name, "config": cfg, "best_validation_accuracy": best_accuracy,
              "best_validation_ce": best_ce, "best_epoch": chosen["epoch"], "epochs_completed": len(history),
              "parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "training_seconds": sum(v["seconds"] for v in history)}
    write(folder/"result.json", result); write(folder/"status.json", result)
    assert_inputs()
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--run-id", required=True); parser.add_argument("--stop-after", type=int)
    args = parser.parse_args(); run(read(args.config), args.run_id, args.stop_after)
