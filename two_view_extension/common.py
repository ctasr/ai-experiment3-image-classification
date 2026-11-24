from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import pickle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_two_view_extension"
FIG = ROOT / "figures_two_view_extension"
MODELS = {
    "kd": "runs/horizon/horizon240_seed42/best.pt",
    "no_kd": "runs/horizon_control/horizon240_no_kd_seed42/best.pt",
}
CASES = ["clean", "brightness_plus8", "brightness_minus8", "translate_right1"]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(name, obj):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def progress(stage, **fields):
    write("progress.json", {"stage": stage, "updated_at_utc": now(), **fields})


def verify_seal():
    seal = read(OUT / "source_manifest.json")
    for group in ("compute_sources", "protected_inputs"):
        for rel, expected in seal[group].items():
            if sha(ROOT / rel) != expected:
                raise RuntimeError(f"Frozen {group} changed: {rel}")
    return seal


def load_validation():
    """Load only the five official training pickle batches. Never CIFAR10/test."""
    manifest = read(ROOT / "data/dataset_manifest.json")
    split = read(ROOT / manifest["split_file"])
    assert sha(ROOT / manifest["split_file"]) == manifest["split_sha256"]
    train, validation = split["train_indices"], split["validation_indices"]
    assert len(train) == 45000 and len(validation) == 5000
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(range(50000))
    arrays, labels = [], []
    for row in manifest["official_train_batches"]:
        path = ROOT / row["file"]
        assert path.name.startswith("data_batch_") and sha(path) == row["sha256"]
        with path.open("rb") as f:
            batch = pickle.load(f, encoding="bytes")
        arrays.append(batch[b"data"].reshape(-1, 3, 32, 32))
        labels.extend(batch[b"labels"])
    data, labels = np.concatenate(arrays), np.asarray(labels, dtype=np.int64)
    indices = np.asarray(validation, dtype=np.int64)
    y = labels[indices]
    assert np.array_equal(np.bincount(y, minlength=10), np.full(10, 500))
    old = read(ROOT / "results_final_interpretability_v1/results.json")
    subset_indices = np.asarray([v["official_train_index"] for v in old["records"]])
    by_index = {int(v): i for i, v in enumerate(indices)}
    subset_rows = np.asarray([by_index[int(v)] for v in subset_indices])
    assert len(subset_rows) == len(set(subset_rows)) == 200
    assert np.array_equal(np.bincount(y[subset_rows], minlength=10), np.full(10, 20))
    assert np.array_equal(y[subset_rows], [v["true"] for v in old["records"]])
    return torch.from_numpy(data[indices].copy()), y, indices, subset_rows, old


def load_model(name, device, seal):
    from models import make_model
    saved = torch.load(ROOT / MODELS[name], map_location="cpu", weights_only=False)
    assert saved["source_sha256"] == seal["model_training_sources"][name]
    cfg = saved["config"]
    assert cfg["seed"] == 42 and cfg["width"] == 48 and cfg["epochs"] == 240
    assert cfg["distill_alpha"] == (0.25 if name == "kd" else 0.0)
    model = make_model(cfg).eval().requires_grad_(False)
    model.load_state_dict(saved["model"])
    assert sum(p.numel() for p in model.parameters()) == 2436346
    assert saved["stats"]["mean"] == read(ROOT / "data/normalization.json")["mean"]
    assert saved["stats"]["std"] == read(ROOT / "data/normalization.json")["std"]
    return model.to(device), saved["stats"], cfg


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def assert_frozen(model, before):
    assert not model.training and state_hash(model) == before
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def normalise(raw, stats, case="clean"):
    if case in ("brightness_plus8", "brightness_minus8"):
        shift = 8 if case == "brightness_plus8" else -8
        raw = (raw.to(torch.int16) + shift).clamp(0, 255).to(torch.uint8)
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=raw.device)[None, :, None, None]
    std = torch.as_tensor(stats["std"], dtype=torch.float32, device=raw.device)[None, :, None, None]
    x = (raw.float() / 255 - mean) / std
    if case == "translate_right1":
        moved = torch.zeros_like(x)  # zero in normalised space = channel training mean
        moved[:, :, :, 1:] = x[:, :, :, :-1]
        x = moved
    assert case in CASES
    return x.contiguous(memory_format=torch.channels_last)


def js_distance(p, q):
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    m = (p + q) / 2
    return .5 * np.sum(p * (np.log(np.clip(p, 1e-300, 1)) - np.log(np.clip(m, 1e-300, 1))) +
                         q * (np.log(np.clip(q, 1e-300, 1)) - np.log(np.clip(m, 1e-300, 1))), axis=1)


def per_image(p, labels):
    p = np.asarray(p, dtype=np.float64)
    assert p.shape == (len(labels), 10) and np.isfinite(p).all()
    assert np.allclose(p.sum(1), 1, atol=2e-6)
    pred = p.argmax(1)
    return pred, (pred == labels).astype(np.float64), -np.log(np.clip(p[np.arange(len(p)), labels], 1e-300, 1))


def transition(clean, perturbed, labels):
    a, b = clean == labels, perturbed == labels
    return {"correct_to_correct": int((a & b).sum()), "correct_to_wrong": int((a & ~b).sum()),
            "wrong_to_correct": int((~a & b).sum()), "wrong_to_wrong": int((~a & ~b).sum()),
            "wrong_changed_label": int((~a & ~b & (clean != perturbed)).sum())}


def bootstrap(values, labels=None, seed=42):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if labels is None:
        means = values[rng.integers(0, len(values), size=(1000, len(values)))].mean(1)
    else:
        means = np.zeros(1000)
        for c in range(10):
            group = values[np.asarray(labels) == c]
            means += group[rng.integers(0, len(group), size=(1000, len(group)))].sum(1) / len(values)
    return np.quantile(means, [.025, .975]).tolist()
