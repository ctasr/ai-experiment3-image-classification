"""Locked-checkpoint, raw-BF16 validation for the fixed A/B seed controls.

No model, CUDA operation, or dataset is created at import time. The only data
path used by ``analyze`` is the five original CIFAR-10 training batches and the
frozen 45k/5k partition. This module has no official-test entry point.
"""
from pathlib import Path
import csv
import math
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from alignment_confirmation import common as c
from data import read_partition, make_loader
from extensions.models import build, Transform
from train import seed_everything


BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = (20261125, 7317, 160)
REPLAY_CE_ATOL = 1e-6


def _local(value):
    root = Path(c.ROOT).resolve()
    path = (root / str(value).replace("\\", "/")).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Validation input escapes the project")
    return path


def _history(folder):
    with (folder / "history.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["epoch"]) for row in rows] == list(range(1, 161))
    for row in rows:
        assert int(row["train_samples"]) == 45000
        assert int(row["validation_samples"]) == 5000
        assert all(math.isfinite(float(row[key])) for key in
                   ("validation_accuracy", "validation_ce", "seconds"))
        assert float(row["seconds"]) > 0
    return rows


def _training_validation():
    """Read training batch pickles directly; do not instantiate CIFAR10(test)."""
    manifest, train_indices, validation_indices = read_partition()
    assert len(train_indices) == 45000 and len(validation_indices) == 5000
    images, labels = [], []
    batches = manifest["official_train_batches"]
    assert len(batches) == 5
    assert [Path(item["file"]).name for item in batches] == [f"data_batch_{i}" for i in range(1, 6)]
    for item in batches:
        path = _local(item["file"])
        assert c.sha(path) == item["sha256"]
        with path.open("rb") as stream:
            saved = pickle.load(stream, encoding="bytes")
        raw = saved[b"data"]
        assert raw.dtype == np.uint8 and raw.shape == (10000, 3072)
        images.append(raw.reshape(-1, 3, 32, 32))
        labels.extend(saved[b"labels"])
    raw = np.concatenate(images)
    labels = np.asarray(labels, dtype=np.int64)
    indices = np.asarray(validation_indices, dtype=np.int64)
    truth = labels[indices]
    assert np.array_equal(np.bincount(truth, minlength=10), np.full(10, 500))
    dataset = TensorDataset(torch.from_numpy(raw[indices].copy()), torch.from_numpy(truth.copy()))
    return dataset, truth, indices, manifest


def _expected_config(seed, role):
    config = dict(c.a_config(int(seed)))
    if role == "B":
        config["alignment"] = "mixed_view"
    assert role in ("A", "B") and config["seed"] == int(seed)
    assert config["batch_size"] == 256 and config["epochs"] == 160 and config["amp"] is True
    assert config["model"] == "cifar" and config["width"] == 32
    return config


def _run_folder(seed, role):
    seed_rows = c.OLD_RUNS.get(int(seed), c.OLD_RUNS.get(str(seed), {}))
    if role in seed_rows:
        return _local(seed_rows[role])
    assert role == "A" and int(seed) in (17, 73)
    return Path(c.RUNS) / c.new_run_id(int(seed))


def _input_identity(folder, checkpoint, config, manifest):
    names = ("data.py", "train.py", "extensions/models.py",
             "alignment_confirmation/common.py", "alignment_confirmation/evaluate.py")
    return {
        "weight_sha256": c.sha(checkpoint),
        "training_sources": c.training_source_hashes(),
        "evaluation_sources": {name: c.sha(Path(c.ROOT) / name) for name in names},
        "config": config,
        "history_sha256": c.sha(folder / "history.csv"),
        "result_sha256": c.sha(folder / "result.json"),
        "normalization_sha256": c.sha(Path(c.ROOT) / "data/normalization.json"),
        "dataset_manifest_sha256": c.sha(Path(c.ROOT) / "data/dataset_manifest.json"),
        "split_sha256": manifest["split_sha256"],
        "teacher_identity_sha256": c.sha(Path(c.OUT) / "teacher_identity.json"),
        "protocol": "original validation5000/B256/eval/no augmentation/no flip/BF16 one forward",
        "ce_protocol": "float32 batch-mean cross_entropy under CUDA autocast, float64 sample-weighted totals",
        "criterion": "CrossEntropyLoss(label_smoothing=0)", "temperature_fitted": False,
    }


def _numpy_nll(logits, labels):
    scores = np.asarray(logits, dtype=np.float64)
    maximum = scores.max(1)
    return maximum + np.log(np.exp(scores - maximum[:, None]).sum(1)) - scores[np.arange(len(labels)), labels]


def _check_cache(cache, truth, indices):
    p, z, y, ids, nll = (cache[name] for name in
                         ("probabilities", "logits", "labels", "indices", "per_sample_nll"))
    assert p.shape == z.shape == (5000, 10) and p.dtype == z.dtype == np.float32
    assert y.shape == ids.shape == nll.shape == (5000,)
    assert np.array_equal(y, truth) and np.array_equal(ids, indices)
    assert np.isfinite(p).all() and np.isfinite(z).all() and np.isfinite(nll).all()
    assert np.allclose(p.sum(1), 1, atol=2e-6, rtol=0)
    assert np.allclose(nll, _numpy_nll(z, y), atol=1e-12, rtol=0)
    assert np.array_equal(p.argmax(1), z.argmax(1))
    return p, z, y, ids, nll


def _predict(saved, config, dataset, truth, indices):
    """Exactly one raw BF16 forward per batch, matching the old epoch CE."""
    with c.preserve_runtime():
        torch.set_num_threads(8)
        seed_everything(config["seed"])
        assert torch.cuda.is_available(), "This historical BF16 replay requires CUDA"
        device = torch.device("cuda")
        model = build(config).to(device).to(memory_format=torch.channels_last)
        model.load_state_dict(saved["model"])
        model.eval().requires_grad_(False)
        transform = Transform(saved["stats"], config).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0)
        before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        totals = torch.zeros(4, device=device, dtype=torch.float64)
        logits_rows, probability_rows, label_rows, batch_sizes = [], [], [], []
        with torch.no_grad():
            for x, y in make_loader(dataset, 256, training=False):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    logits = model(transform(x, training=False))
                    loss = criterion(logits, y)
                    ce = nn.functional.cross_entropy(logits, y)
                count = len(y)
                totals += torch.stack((loss.detach().double() * count,
                                       ce.detach().double() * count,
                                       (logits.argmax(1) == y).sum().double(),
                                       torch.tensor(count, device=device, dtype=torch.float64)))
                logits_rows.append(logits.float().cpu().numpy())
                probability_rows.append(logits.float().softmax(1).cpu().numpy())
                label_rows.append(y.cpu().numpy())
                batch_sizes.append(count)
        assert batch_sizes == [256] * 19 + [136]
        loss_sum, ce_sum, correct, count = totals.cpu().tolist()
        assert count == 5000 and loss_sum == ce_sum
        assert all(torch.equal(value.detach().cpu(), before[key])
                   for key, value in model.state_dict().items())
        z, p, y = np.concatenate(logits_rows), np.concatenate(probability_rows), np.concatenate(label_rows)
        cache = {"probabilities": p, "logits": z, "labels": y, "indices": indices.copy(),
                 "per_sample_nll": _numpy_nll(z, y)}
        _check_cache(cache, truth, indices)
        replay = {"accuracy": correct / count, "ce": ce_sum / count,
                  "samples": int(count), "batches": 20, "batch_sizes": batch_sizes,
                  "model_state_unchanged": True, "forwards": 20,
                  "per_sample_nll_mean_float64": float(cache["per_sample_nll"].mean())}
        del model, transform, before, logits_rows, probability_rows, label_rows
        torch.cuda.empty_cache()
    return cache, replay


def _one(seed, role, dataset, truth, indices, manifest):
    folder = _run_folder(seed, role)
    config = _expected_config(seed, role)
    history = _history(folder)
    result = c.read(folder / "result.json")
    assert result["state"] == "completed" and result["epochs_completed"] == 160
    assert result["config"] == config == c.read(folder / "config.json")
    selected = max(history, key=lambda row: (float(row["validation_accuracy"]), -float(row["validation_ce"])))
    best_epoch, best_accuracy, best_ce = (int(selected["epoch"]), float(selected["validation_accuracy"]),
                                        float(selected["validation_ce"]))
    assert result["best_epoch"] == best_epoch
    assert result["best_validation_accuracy"] == best_accuracy and result["best_validation_ce"] == best_ce
    checkpoint = folder / "best.pt"
    weight_sha = c.sha(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert c.sha(checkpoint) == weight_sha, "Checkpoint changed while being loaded"
    assert saved["source_sha256"] == c.training_source_hashes()
    assert saved["config"] == config and saved["epoch"] == best_epoch
    assert saved["best_accuracy"] == best_accuracy and saved["best_ce"] == best_ce
    assert saved["stats"] == c.read(Path(c.ROOT) / "data/normalization.json")
    identity = _input_identity(folder, checkpoint, config, manifest)
    assert identity["weight_sha256"] == weight_sha
    cache_dir = Path(c.OUT) / "validation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"seed{seed}_{role}.npz"
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() or metadata_path.exists():
        assert cache_path.exists() and metadata_path.exists(), "Incomplete cache must be inspected, not overwritten"
        metadata = c.read(metadata_path)
        assert metadata["identity"] == identity and metadata["cache_sha256"] == c.sha(cache_path)
        with np.load(cache_path, allow_pickle=False) as stored:
            cache = {key: stored[key].copy() for key in stored.files}
        replay = metadata["replay"]
    else:
        cache, replay = _predict(saved, config, dataset, truth, indices)
        assert replay["accuracy"] == best_accuracy
        assert abs(replay["ce"] - best_ce) <= REPLAY_CE_ATOL
        temporary = cache_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **cache)
        temporary.replace(cache_path)
        metadata = {"identity": identity, "cache_sha256": c.sha(cache_path), "replay": replay}
        c.write(metadata_path, metadata)
    p, z, labels, ids, nll = _check_cache(cache, truth, indices)
    predicted = z.argmax(1)
    assert float((predicted == labels).mean()) == replay["accuracy"] == best_accuracy
    assert abs(replay["ce"] - best_ce) <= REPLAY_CE_ATOL
    assert replay["batch_sizes"] == [256] * 19 + [136] and replay["model_state_unchanged"]
    cm = np.bincount(labels * 10 + predicted, minlength=100).reshape(10, 10)
    assert np.array_equal(cm.sum(1), np.full(10, 500))
    wrong = np.flatnonzero(predicted != labels)
    confidence = p.max(1)
    wrong = wrong[np.lexsort((ids[wrong], -confidence[wrong]))]
    examples = [{"validation_row": int(i), "official_train_index": int(ids[i]),
                 "true": int(labels[i]), "predicted": int(predicted[i]),
                 "confidence": float(confidence[i])} for i in wrong[:12]]
    seconds = np.asarray([float(row["seconds"]) for row in history], dtype=np.float64)
    raw_seconds, median = float(seconds.sum()), float(np.median(seconds))
    assert math.isclose(result["training_seconds"], raw_seconds, rel_tol=0, abs_tol=1e-6)
    pause_threshold = max(60., 3 * median)
    pauses = [{"epoch": int(i + 1), "seconds": float(value),
               "classification": "long observed epoch; cause not inferred"}
              for i, value in enumerate(seconds) if value > pause_threshold]
    row = {"run_id": result["run_id"], "artifact": c.rel(folder),
           "history_csv": c.rel(folder / "history.csv"), "config": config,
           "selection": {"best_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
                         "best_validation_ce": best_ce, "rule": "maximum accuracy, then minimum CE; first exact tie"},
           "validation": {"accuracy": best_accuracy, "ce": best_ce,
                          "confusion_matrix": cm.tolist(), "per_class_accuracy": (cm.diagonal() / 500).tolist(),
                          "replay_ce": replay["ce"], "ce_replay_absolute_difference": abs(replay["ce"] - best_ce),
                          "per_sample_nll_mean_float64": float(nll.mean()), "samples": 5000},
           "cost": {"median_epoch_seconds": median, "raw_seconds": raw_seconds,
                    "pauses": pauses, "long_epoch_threshold_seconds": pause_threshold,
                    "all_raw_epoch_times_retained": True},
           "error_examples": examples, "cache": c.rel(cache_path),
           "cache_sha256": c.sha(cache_path), "cache_metadata": c.rel(metadata_path),
           "weight_sha256": identity["weight_sha256"], "training_sources": saved["source_sha256"]}
    del saved
    return row, {"correct": (predicted == labels).astype(np.float64), "nll": nll, "labels": labels}


def _bootstrap(seed_values, caches):
    """Class-stratified shared image draws; seeds are never independent images."""
    labels = caches[seed_values[0]]["A"]["labels"]
    for seed in seed_values:
        for role in ("A", "B"):
            assert np.array_equal(caches[seed][role]["labels"], labels)
    correct_delta = np.stack([caches[seed]["B"]["correct"] - caches[seed]["A"]["correct"] for seed in seed_values])
    nll_delta = np.stack([caches[seed]["B"]["nll"] - caches[seed]["A"]["nll"] for seed in seed_values])
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(BOOTSTRAP_SEED)))
    distributions = np.empty((BOOTSTRAP_REPLICATES, len(seed_values), 2), dtype=np.float64)
    class_rows = [np.flatnonzero(labels == category) for category in range(10)]
    assert all(len(rows) == 500 for rows in class_rows)
    for start in range(0, BOOTSTRAP_REPLICATES, 50):
        count = min(50, BOOTSTRAP_REPLICATES - start)
        draw = np.concatenate([rows[rng.integers(0, 500, size=(count, 500))] for rows in class_rows], axis=1)
        distributions[start:start + count, :, 0] = correct_delta[:, draw].mean(2).T
        distributions[start:start + count, :, 1] = nll_delta[:, draw].mean(2).T
    pair_intervals = {}
    for index, seed in enumerate(seed_values):
        pair_intervals[str(seed)] = {
            "accuracy_CI95": np.quantile(distributions[:, index, 0], [.025, .975]).tolist(),
            "ce_CI95": np.quantile(distributions[:, index, 1], [.025, .975]).tolist(),
            "ce_interval_definition": "float64 per-image NLL from the same raw BF16 logits",
        }
    joint = distributions.mean(1)
    return pair_intervals, {"accuracy_CI95": np.quantile(joint[:, 0], [.025, .975]).tolist(),
                            "ce_CI95": np.quantile(joint[:, 1], [.025, .975]).tolist()}


def analyze(seeds=(42, 17, 73)):
    """Replay only completed fixed seeds and write validation-only evidence."""
    seed_values = tuple(int(seed) for seed in seeds)
    assert seed_values and len(set(seed_values)) == len(seed_values)
    assert all(seed in (42, 17, 73) for seed in seed_values)
    c.verify_protected()
    training_audits_sha = c.sha(Path(c.OUT) / "training_audits.json")
    dataset, truth, indices, manifest = _training_validation()
    pairs, caches = {}, {}
    for seed in seed_values:
        caches[seed] = {}
        rows = {}
        for role in ("A", "B"):
            rows[role], caches[seed][role] = _one(seed, role, dataset, truth, indices, manifest)
        assert {key: value for key, value in rows["A"]["config"].items() if key != "alignment"} == {
            key: value for key, value in rows["B"]["config"].items() if key != "alignment"}
        pairs[str(seed)] = {**rows,
                           "delta": {metric: rows["B"]["validation"][metric] - rows["A"]["validation"][metric]
                                     for metric in ("accuracy", "ce")}}
    intervals, joint_interval = _bootstrap(seed_values, caches)
    for seed in seed_values:
        pairs[str(seed)]["paired"] = intervals[str(seed)]
    acc = [pairs[str(seed)]["delta"]["accuracy"] for seed in seed_values]
    ce = [pairs[str(seed)]["delta"]["ce"] for seed in seed_values]
    complete = set(seed_values) == {42, 17, 73}
    mean_accuracy, mean_ce = float(np.mean(acc)), float(np.mean(ce))
    nonnegative = sum(value >= 0 for value in acc)
    supported = bool(mean_accuracy >= .001 and nonnegative >= 2 and mean_ce <= 0) if complete else None
    analysis = {
        "pairs": pairs,
        "training_audits_sha256": training_audits_sha,
        "gate": {"complete": complete, "supported": supported,
                 "mean_accuracy_delta": mean_accuracy, "mean_ce_delta": mean_ce,
                 "nonnegative_accuracy_seeds": nonnegative,
                 "criterion": "three fixed seeds: mean(B-A)accuracy>=.001, at least2 nonnegative accuracy, mean(B-A)CE<=0",
                 "purpose": "conclusion only; no automatic official test, reproduction, or model promotion"},
        "paired_mean_interval": joint_interval,
        "protocol": {"dataset": "fixed official-training-derived validation5000 only", "batch_size": 256,
                     "batches": 20, "last_batch": 136, "amp": "CUDA BF16", "views": "original single forward",
                     "selection": "original best.pt by full160 validation(max accuracy,-CE)",
                     "main_ce": "historical BF16 batch-weighted training validation CE, replay matched within1e-6",
                     "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "private_rng": "PCG64",
                                   "seed_sequence": list(BOOTSTRAP_SEED), "stratified": True,
                                   "shared_images_across_seeds": True, "images_per_draw": 5000,
                                   "scope": "fixed-checkpoint conditional image uncertainty, not retraining or model-selection uncertainty"},
                     "official_test_images": 0, "external_images": 0, "temperature_fitting": False},
        "manual_visual_review_pending": True,
    }
    c.verify_protected()
    assert c.sha(Path(c.OUT) / "training_audits.json") == training_audits_sha
    c.write(Path(c.OUT) / "validation_analysis.json", analysis)
    return analysis
