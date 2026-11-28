"""Read-only BEST/raw-BF16 validation of existing A/B runs; never loads test."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import adapter as a
from data import make_loader
from extensions.models import build, Transform
from train import epoch_pass, seed_everything


def nll64(logits, labels):
    z = np.asarray(logits, dtype=np.float64)
    maximum = z.max(1)
    return maximum + np.log(np.exp(z - maximum[:, None]).sum(1)) - z[np.arange(len(labels)), labels]


def one(folder, seed, role, dataset, stats, data_dir, output):
    folder = Path(folder).resolve()
    cfg = a.config(seed, role)
    if a.read(folder / "config.json") != cfg:
        raise ValueError("Expected fixed width32/160 configuration")
    history = a.history(folder)
    chosen = max(history, key=lambda r: (r["validation_accuracy"], -r["validation_ce"]))
    checkpoint = folder / "best.pt"
    identity = {"checkpoint_sha256": a.sha(checkpoint), "history_sha256": a.sha(folder / "history.csv"), "config": cfg, "evaluation_sources": {f"contributions/alignment/{name}": a.sha(a.PACKAGE / name) for name in ("adapter.py", "validate.py")}, "dataset": a.data_identity(data_dir)}
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source10 = a.read(a.PACKAGE / "evidence/source10.json")["source_sha256"]
    if saved["source_sha256"] not in (source10, a.sources()) or saved["config"] != cfg or saved["stats"] != stats or saved["epoch"] != chosen["epoch"]:
        raise ValueError("Checkpoint provenance or selected epoch differs")
    if saved["source_sha256"] == source10:
        original = a.read(a.PACKAGE / "evidence/original_validation_analysis.json")["pairs"][str(seed)][role]
        if identity["checkpoint_sha256"] != original["weight_sha256"]:
            raise ValueError("Legacy validation accepts only the six indexed original A/B checkpoints")
    else:
        provenance = a.read(folder / "portable_provenance.json")["identity"]
        expected_teacher = a.read(a.PACKAGE / "evidence/source10.json")["teacher_identity"]
        if provenance["source_sha256"] != saved["source_sha256"] or provenance["teacher_identity"] != expected_teacher or provenance["data"] != a.data_identity(data_dir):
            raise ValueError("Portable checkpoint input provenance differs")
    cache = output / f"seed{seed}_{role}.npz"
    meta = cache.with_suffix(".json")
    if cache.exists() or meta.exists():
        if not cache.exists() or not meta.exists() or a.read(meta)["identity"] != identity or a.read(meta)["cache_sha256"] != a.sha(cache):
            raise ValueError("Cache belongs to different inputs; use a new validation output")
        values = np.load(cache)
        replay = a.read(meta)["native_validation"]
        logits, labels = values["logits"], values["labels"]
    else:
        with a.preserve_runtime():
            seed_everything(seed)
            torch.set_num_threads(8)
            if not torch.cuda.is_available():
                raise RuntimeError("Historical raw BF16 validation needs CUDA; no CPU/FP32 substitute is labelled exact")
            model = build(cfg).cuda().to(memory_format=torch.channels_last)
            model.load_state_dict(saved["model"])
            model.eval().requires_grad_(False)
            before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            transform = Transform(stats, cfg).cuda()
            captured = []
            hook = model.register_forward_hook(lambda _m, _args, z: captured.append((z.detach().float().cpu().numpy(), z.detach().float().softmax(1).cpu().numpy())))
            try:
                replay = epoch_pass(model, make_loader(dataset, 256, training=False), transform, nn.CrossEntropyLoss(), torch.device("cuda"), amp=True)
            finally:
                hook.remove()
            logits = np.concatenate([item[0] for item in captured])
            probabilities = np.concatenate([item[1] for item in captured])
            labels = dataset.tensors[1].numpy().copy()
            if len(captured) != 20 or len(captured[-1][0]) != 136 or any(not torch.equal(v.detach().cpu(), before[k]) for k, v in model.state_dict().items()):
                raise ValueError("Validation batching or model state changed")
        np.savez_compressed(cache, logits=logits, probabilities=probabilities, labels=labels, per_sample_nll=nll64(logits, labels))
        a.write(meta, {"identity": identity, "native_validation": replay, "cache_sha256": a.sha(cache), "official_test_images": 0})
    predicted = logits.argmax(1)
    if replay["accuracy"] != chosen["validation_accuracy"] or abs(replay["ce"] - chosen["validation_ce"]) > 1e-6 or len(labels) != 5000 or logits.shape != (5000, 10) or not np.isfinite(logits).all() or not np.array_equal(labels, dataset.tensors[1].numpy()) or float((predicted == labels).mean()) != replay["accuracy"]:
        raise ValueError("Raw BF16 replay does not match the recorded BEST validation")
    cm = np.zeros((10, 10), dtype=np.int64)
    np.add.at(cm, (labels, predicted), 1)
    if not np.array_equal(cm.sum(1), np.full(10, 500)):
        raise ValueError("Validation class counts differ")
    row = {"seed": seed, "role": role, "best_epoch": chosen["epoch"], "accuracy": chosen["validation_accuracy"], "native_ce": chosen["validation_ce"], "replay_ce": replay["ce"], "per_sample_nll_mean_float64": float(nll64(logits, labels).mean()), "confusion_matrix": cm.tolist(), "per_class_accuracy": (cm.diagonal() / 500).tolist(), "checkpoint_sha256": identity["checkpoint_sha256"], "training_sources": saved["source_sha256"], "cache_sha256": a.sha(cache), "samples": 5000, "batches": 20, "last_batch": 136, "median_epoch_seconds": float(np.median([r["seconds"] for r in history])), "all_raw_seconds": [r["seconds"] for r in history]}
    del saved
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="same data directory used for training, including raw/ and splits/")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs-dir", help="portable output/runs containing A_seed42, B_seed42, etc")
    source.add_argument("--run-map", help="JSON mapping seed to A/B existing run directory paths; relative paths resolve against the JSON file")
    parser.add_argument("--seeds", nargs="+", type=int, choices=a.SEEDS, default=list(a.SEEDS))
    parser.add_argument("--output-dir", required=True, help="isolated analysis output; input run files are never changed")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate seeds are not allowed")
    data_dir = Path(args.data_dir).resolve()
    output = Path(args.output_dir).resolve()
    marker = output / "validation_identity.json"
    a.sources()
    if args.run_map:
        run_map_path = Path(args.run_map).resolve()
        data = a.read(run_map_path)
        mapping = {seed: {role: (run_map_path.parent / data[str(seed)][role]).resolve() for role in a.ROLES} for seed in args.seeds}
    else:
        runs = Path(args.runs_dir).resolve()
        mapping = {seed: {role: runs / f"{role}_seed{seed}" for role in a.ROLES} for seed in args.seeds}
    if output == data_dir or output.is_relative_to(data_dir) or any(output == p or p.is_relative_to(output) or output.is_relative_to(p) for row in mapping.values() for p in row.values()):
        raise ValueError("Validation output must be separate from the data and all input runs")
    identity = {"kind": "portable_alignment_validation_v1", "data": a.data_identity(data_dir), "runs": {str(seed): {role: {"best_sha256": a.sha(folder / "best.pt"), "history_sha256": a.sha(folder / "history.csv")} for role, folder in row.items()} for seed, row in mapping.items()}, "evaluation_sources": {name: a.sha(a.PACKAGE / name) for name in ("adapter.py", "validate.py")}}
    if marker.exists():
        if a.read(marker) != identity:
            raise ValueError("Validation output has another identity")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Refusing unrelated analysis output")
    else:
        a.write(marker, identity)
    _, validation, stats = a.load_data(data_dir)
    pairs = {}
    for seed in args.seeds:
        pairs[str(seed)] = {role: one(mapping[seed][role], seed, role, validation, stats, data_dir, output) for role in a.ROLES}
        pairs[str(seed)]["delta"] = {"accuracy_B_minus_A": pairs[str(seed)]["B"]["accuracy"] - pairs[str(seed)]["A"]["accuracy"], "native_ce_B_minus_A": pairs[str(seed)]["B"]["native_ce"] - pairs[str(seed)]["A"]["native_ce"]}
    complete = set(args.seeds) == set(a.SEEDS)
    acc = [p["delta"]["accuracy_B_minus_A"] for p in pairs.values()]
    ce = [p["delta"]["native_ce_B_minus_A"] for p in pairs.values()]
    result = {"pairs": pairs, "mean_accuracy_B_minus_A": float(np.mean(acc)), "mean_native_ce_B_minus_A": float(np.mean(ce)), "three_fixed_seeds_complete": complete, "original_conclusion_screen_supported": bool(np.mean(acc) >= .001 and sum(v >= 0 for v in acc) >= 2 and np.mean(ce) <= 0) if complete else None, "official_test_images": 0, "new_training": 0, "temperature_fitted": False, "selection": "BEST from complete160 history; maximum accuracy, then minimum raw CE; first tie", "limitations": "same fixed5000 validation images and Teacher42; no independent test or selection/retraining uncertainty; no automatic promotion", "manual_visual_review_complete": False}
    a.write(output / "validation_analysis.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
