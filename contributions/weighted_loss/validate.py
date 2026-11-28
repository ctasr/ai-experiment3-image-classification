"""Raw-image validation only, with explicit endpoint/source and input locks."""
from pathlib import Path
import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from . import adapter


def evaluate(checkpoint, data_dir, output_dir, mode, endpoint="last", device="cuda"):
    checkpoint, output_dir = Path(checkpoint).resolve(), Path(output_dir).resolve()
    before = adapter.sha(checkpoint)
    data_before = adapter.data_identity(data_dir)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    adapter.configuration(cfg, mode, cfg["seed"], {"validation_recipe_check": True})
    if cfg.get("source_set_mode") != mode or cfg.get("pilot_samples"):
        raise ValueError("Wrong mode or a preflight checkpoint")
    if endpoint == "last" and saved["epoch"] != 240:
        raise ValueError("LAST validation requires the complete 240-epoch endpoint")
    if endpoint == "last" and not all(key in saved for key in ("history", "optimizer", "scheduler", "rng")):
        raise ValueError("LAST checkpoint is missing full training state")
    if endpoint == "last" and (len(saved["history"]) != 240 or
            any(row["epoch"] != index + 1 or row["train_samples"] != 45000 or row["validation_samples"] != 5000
                for index, row in enumerate(saved["history"]))):
        raise ValueError("LAST history is not the complete formal 45k/5k run")
    if endpoint == "best" and any(key in saved for key in ("history", "optimizer", "scheduler", "rng")):
        raise ValueError("BEST endpoint requires the selected BEST checkpoint, not LAST")
    current_sources = adapter.source_hashes(mode)
    legacy = adapter.provenance()["legacy_source_dictionaries"]
    if saved["source_sha256"] != current_sources and saved["source_sha256"] not in legacy.values():
        raise ValueError("Unknown checkpoint source dictionary")
    if saved["source_sha256"] != current_sources and mode != "weighted_single":
        raise ValueError("Legacy sources cover weighted_single only")
    if cfg.get("portable_protocol"):
        if cfg["portable_protocol"] != adapter.PROTOCOL or cfg["portable_inputs"]["data"] != data_before:
            raise ValueError("Checkpoint dataset identity differs")
    _, validation, stats = adapter.training_data(data_dir)
    if saved["stats"] != stats:
        raise ValueError("Checkpoint normalization/split differs")
    with adapter.runtime():
        torch.set_num_threads(8)
        adapter.CORE.seed_everything(cfg["seed"])
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("This frozen BF16 evaluation protocol requires CUDA")
        model = adapter.ENGINE.build(cfg).to(target, memory_format=torch.channels_last)
        model.load_state_dict(saved["model"]); model.eval().requires_grad_(False)
        transform = adapter.ENGINE.Transform(stats, cfg).to(target)
        loader = adapter.ENGINE.make_loader(validation, cfg["batch_size"])
        probabilities, labels, losses = [], [], []
        native_totals = torch.zeros(2, dtype=torch.float64, device=target)
        with torch.inference_mode():
            for images, truth in loader:
                images, truth = images.to(target), truth.to(target)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    logits = model(transform(images, training=False))
                    loss = F.cross_entropy(logits, truth, reduction="none")
                    mean_loss = F.cross_entropy(logits, truth)
                native_totals += torch.stack((mean_loss.detach().double() * len(truth), torch.tensor(len(truth), device=target).double()))
                probabilities.append(logits.float().softmax(1).cpu().numpy())
                losses.append(loss.float().cpu().numpy()); labels.append(truth.cpu().numpy())
        probabilities, labels, losses = np.concatenate(probabilities), np.concatenate(labels), np.concatenate(losses)
    predictions = probabilities.argmax(1)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    if len(labels) != 5000 or not np.all(confusion.sum(1) == 500):
        raise ValueError("Validation sample/class counts differ")
    if adapter.sha(checkpoint) != before or adapter.data_identity(data_dir) != data_before or adapter.source_hashes(mode) != current_sources:
        raise ValueError("Inputs changed during evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"state": "validation_complete", "checkpoint_sha256": before, "checkpoint_epoch": saved["epoch"],
              "checkpoint_source_sha256": saved["source_sha256"], "evaluation_source_sha256": {**current_sources, "contributions/weighted_loss/validate.py": adapter.sha(__file__)},
              "mode": mode, "endpoint": endpoint, "precision": "native BF16 raw image, one forward, uncalibrated",
              "validation_samples": len(labels), "accuracy": float((predictions == labels).mean()),
              "ce": float((native_totals[0] / native_totals[1]).cpu()),
              "per_image_native_loss_mean_FP64": float(losses.astype(np.float64).mean()), "confusion": confusion.tolist(),
              "per_class_accuracy": (confusion.diagonal() / confusion.sum(1)).tolist(),
              "data": data_before, "official_test_read": False, "test_selection": False}
    np.savez_compressed(output_dir / "validation_predictions.npz", probabilities=probabilities, labels=labels, losses=losses)
    adapter.write(output_dir / "results.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=adapter.MODES, required=True)
    parser.add_argument("--endpoint", choices=("last", "best"), default="last")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(args.checkpoint, args.data_dir, args.output_dir, args.mode, args.endpoint, args.device), ensure_ascii=False))


if __name__ == "__main__":
    main()
