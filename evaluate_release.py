"""Evaluate one released student without importing historical experiment drivers.

All third-party and project imports are delayed until after argument parsing, so
``python evaluate_release.py --help`` needs neither dependencies nor artifacts.
The default CUDA path follows the frozen report predictor: BF16 network calls,
FP32 softmax, then NumPy float64 conversion and equal original/flip averaging.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 17, 73), default=42,
                        help="released training seed (default: 42)")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="default: 128 for test, 256 for validation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("auto", "bfloat16", "float32"),
                        default="auto", help="auto: CUDA BF16, CPU FP32")
    parser.add_argument("--output", type=Path, default=None,
                        help="optional JSON file; existing files are never overwritten")
    return parser


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(root, relative):
    """Resolve manifest paths inside this checkout, independent of the cwd."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes the checkout: {relative}")
    return path


def check_file(root, relative, expected):
    path = repository_path(root, relative)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {relative}; prepare the release assets/data as described in README.md."
        )
    if sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {relative}")
    return path


def validate_release_manifest(manifest):
    """Schema 2 adds delivery metadata; original model/source fields stay strict."""
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] not in (1, 2):
        raise ValueError("Unsupported release_checkpoints.json schema_version")
    required = {"repository", "tag", "training_source_sha256", "checkpoints",
                "split_file", "split_sha256", "normalization_file", "normalization_sha256"}
    if not required.issubset(manifest):
        raise ValueError("Release manifest lacks original required fields")
    def digest(value):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Invalid SHA256 in release manifest")
    def relative(value):
        if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("Invalid relative path in release manifest")
    sources = manifest["training_source_sha256"]
    if not isinstance(sources, dict) or len(sources) != 12:
        raise ValueError("Expected exactly 12 frozen training sources")
    for name, value in sources.items():
        relative(name); digest(value)
    for name in ("split", "normalization"):
        relative(manifest[name + "_file"]); digest(manifest[name + "_sha256"])
    entries = manifest["checkpoints"]
    if not isinstance(entries, dict) or set(entries) != {"42", "17", "73"}:
        raise ValueError("Expected the three released student seeds")
    for seed, entry in entries.items():
        if not isinstance(entry, dict) or not {"path", "sha256", "config"}.issubset(entry):
            raise ValueError("Released checkpoint lacks path/SHA256/config")
        relative(entry["path"]); digest(entry["sha256"])
        if not isinstance(entry["config"], dict) or entry["config"].get("seed") != int(seed):
            raise ValueError("Released checkpoint config has a different seed")
        if "bytes" in entry and (type(entry["bytes"]) is not int or entry["bytes"] <= 0):
            raise ValueError("Invalid checkpoint byte count")
    return sources


def validation_training_arrays(root, dataset_manifest):
    """Read only five SHA-locked official train pickles in torchvision order.

    CIFAR10(train=True) integrity-checks test_batch as well. This equivalent
    train-only decoding avoids that unrelated file read; no transforms change.
    """
    import pickle
    import numpy as np

    rows = dataset_manifest.get("official_train_batches")
    expected = [f"data/raw/cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)]
    if not isinstance(rows, list) or len(rows) != 5 or [row.get("file") for row in rows] != expected:
        raise ValueError("Expected the five original training files in fixed order")
    images, labels = [], []
    for row in rows:
        if row.get("count") != 10000:
            raise ValueError("Unexpected official training batch size")
        source = check_file(root, row["file"], row["sha256"])
        with source.open("rb") as stream:
            batch = pickle.load(stream, encoding="latin1")
        data = np.asarray(batch["data"])
        target = np.asarray(batch["labels"])
        if data.shape != (10000, 3072) or data.dtype != np.uint8:
            raise ValueError("Unexpected official training image shape/dtype")
        if target.shape != (10000,) or not np.issubdtype(target.dtype, np.integer) or np.any((target < 0) | (target >= 10)):
            raise ValueError("Unexpected official training labels")
        images.append(data); labels.extend(batch["labels"])
    # Exact installed torchvision train-branch concatenation and HWC layout.
    data = np.vstack(images).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
    return data, np.asarray(labels, dtype=np.int64)


def check_saved_identity(saved, entry, source_hashes, seed, expected_stats):
    """Bind a weights-only student to its exact released config and 12 sources."""
    required = {"model", "config", "stats", "source_sha256"}
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError("Student checkpoint lacks model/config/stats/source_sha256")
    if not isinstance(source_hashes, dict) or len(source_hashes) != 12:
        raise ValueError("The release must identify exactly 12 frozen training sources")
    if saved["source_sha256"] != source_hashes:
        raise ValueError("Checkpoint training-source identities differ from the release")
    if saved["config"] != entry["config"] or saved["config"].get("seed") != seed:
        raise ValueError("Checkpoint config or training seed differs from the release")
    cfg = saved["config"]
    if (cfg.get("architecture"), cfg.get("model"), cfg.get("width"),
            cfg.get("epochs"), cfg.get("distill_alpha")) != ("resnet20", "cifar", 48, 240, .25):
        raise ValueError("Expected the frozen width-48, 240-epoch KD student")
    if saved["stats"] != expected_stats:
        raise ValueError("Checkpoint normalization differs from the released training statistics")


def probability_mean(raw, flipped):
    """Convert each FP32 branch to float64 before averaging, as in the report."""
    import numpy as np

    raw = np.asarray(raw).astype(np.float64)
    flipped = np.asarray(flipped).astype(np.float64)
    if raw.shape != flipped.shape:
        raise ValueError("Original and flip probability arrays have different shapes")
    return raw, (raw + flipped) / 2


def classification_metrics(probabilities, labels):
    """Hard-label metrics; NLL uses the original predictor's 1e-30 floor."""
    import numpy as np

    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels)
    if (y.ndim != 1 or len(y) == 0 or p.shape != (len(y), 10)
            or not np.issubdtype(y.dtype, np.integer)
            or np.any((y < 0) | (y >= 10))):
        raise ValueError("Expected nonempty integer labels and an N x 10 probability array")
    if (not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            # Unaveraged FP32 softmax rows can exceed 2e-7 rounding error.
            or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-6)):
        raise ValueError("Invalid class probabilities")
    predicted = p.argmax(1)
    matrix = np.bincount(y * 10 + predicted, minlength=100).reshape(10, 10)
    counts = matrix.sum(1)
    per_class = [float(matrix[i, i] / counts[i]) if counts[i] else None
                 for i in range(10)]
    return {
        "samples": int(len(y)),
        "correct": int((predicted == y).sum()),
        "accuracy": float((predicted == y).mean()),
        "nll": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-30)).mean()),
        "confusion_matrix": matrix.tolist(),
        "per_class_accuracy": per_class,
        "per_class_samples": counts.tolist(),
    }


def evaluate(args):
    # Validate identities before importing the preserved model/data modules.
    root = REPOSITORY_ROOT
    manifest_path = root / "release_checkpoints.json"
    manifest = read_json(manifest_path)
    source_hashes = validate_release_manifest(manifest)
    for relative, digest in source_hashes.items():
        check_file(root, relative, digest)
    entry = manifest["checkpoints"][str(args.seed)]
    checkpoint = check_file(root, entry["path"], entry["sha256"])
    if "bytes" in entry and checkpoint.stat().st_size != entry["bytes"]:
        raise ValueError("Checkpoint byte count differs from the release")
    split_path = check_file(root, manifest["split_file"], manifest["split_sha256"])
    stats_path = check_file(root, manifest["normalization_file"],
                            manifest["normalization_sha256"])
    expected_stats = read_json(stats_path)
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset
    from data import ROOT, read_partition, make_loader
    from extensions.models import build, Transform
    from train import seed_everything

    if ROOT.resolve() != root.resolve():
        raise ValueError("Imported data.py does not belong to this checkout")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu --precision float32")
    device = torch.device(device_name)
    precision = args.precision
    if precision == "auto":
        precision = "bfloat16" if device.type == "cuda" else "float32"
    if device.type == "cuda" and precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device lacks BF16 support; select --precision float32")
    torch.set_num_threads(8)
    seed_everything(42)
    batch_size = args.batch_size or (128 if args.split == "test" else 256)

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    check_saved_identity(saved, entry, source_hashes, args.seed, expected_stats)
    model = build(saved["config"]).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(saved["model"], strict=True)
    model.eval().requires_grad_(False)
    transform = Transform(saved["stats"], saved["config"]).to(device)

    # Read the locked partition even for test, and hash only the raw data needed
    # for the requested split. This entry never downloads or creates dataset files.
    dataset_manifest, _, validation_indices = read_partition()
    if (dataset_manifest["split_file"] != manifest["split_file"]
            or dataset_manifest["split_sha256"] != manifest["split_sha256"]):
        raise ValueError("Dataset manifest refers to a different validation split")
    if args.split == "validation":
        raw_files = dataset_manifest["official_train_batches"]
    else:
        raw_files = [dataset_manifest["test"]]
    data_hashes = {item["file"]: item["sha256"] for item in raw_files}
    if args.split == "validation":
        source_images, source_labels = validation_training_arrays(root, dataset_manifest)
    else:
        for relative, digest in data_hashes.items():
            check_file(root, relative, digest)
        from torchvision.datasets import CIFAR10
        source = CIFAR10(str(root / "data/raw"), train=False, download=False)
        source_images, source_labels = source.data, np.asarray(source.targets, dtype=np.int64)
    indices = validation_indices if args.split == "validation" else list(range(10000))
    labels = source_labels[indices]
    expected_per_class = 500 if args.split == "validation" else 1000
    if not np.array_equal(np.bincount(labels, minlength=10), np.full(10, expected_per_class)):
        raise ValueError("Unexpected class counts in the requested official split")
    dataset = TensorDataset(
        torch.from_numpy(source_images[indices].copy()).permute(0, 3, 1, 2).contiguous(),
        torch.from_numpy(labels),
    )
    raw_parts, flip_parts, truth = [], [], []
    with torch.inference_mode():
        for images, target in make_loader(dataset, batch_size, training=False):
            images = images.to(device, non_blocking=True)
            context = (torch.autocast(device.type, dtype=torch.bfloat16)
                       if precision == "bfloat16" else nullcontext())
            with context:
                raw_logits = model(transform(images, training=False))
                flip_logits = model(transform(images.flip(-1), training=False))
            # Deliberately outside autocast, matching the archived predictor.
            raw_parts.append(raw_logits.float().softmax(1).cpu().numpy())
            flip_parts.append(flip_logits.float().softmax(1).cpu().numpy())
            truth.append(target.numpy())
    raw, two_view = probability_mean(np.concatenate(raw_parts), np.concatenate(flip_parts))
    labels = np.concatenate(truth)
    # Recheck the input identity; inference does not rewrite any scientific artifact.
    if sha256(checkpoint) != entry["sha256"]:
        raise ValueError("Checkpoint changed during evaluation")
    reference_path = device.type == "cuda" and precision == "bfloat16"
    return {
        "raw": classification_metrics(raw, labels),
        "two_view": classification_metrics(two_view, labels),
        "metadata": {
            "repository": manifest.get("repository"), "release_tag": manifest.get("tag"),
            "training_seed": args.seed, "evaluation_seed": 42, "split": args.split,
            "class_names": CLASS_NAMES, "batch_size": batch_size,
            "device": str(device), "network_precision": precision,
            "softmax_precision": "float32", "probability_averaging_precision": "float64",
            "policies": ["original", "equal_original_horizontal_flip"],
            "temperature_applied": False, "teacher_loaded": False,
            "deterministic_algorithms": True, "threads": 8, "tf32": False,
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "checkpoint": entry["path"], "checkpoint_sha256": entry["sha256"],
            "training_source_sha256": source_hashes,
            "split_file": str(split_path.relative_to(root)),
            "split_sha256": manifest["split_sha256"],
            "normalization_sha256": manifest["normalization_sha256"],
            "raw_data_sha256": data_hashes, "config": saved["config"],
            "release_manifest_sha256": sha256(manifest_path),
            "evaluation_script_sha256": sha256(Path(__file__)),
            "evaluation_entry_version": "portable-release-evaluation-v2",
            "validation_loader": "five_SHA_locked_official_training_pickles_no_test_file_read" if args.split == "validation" else None,
            "precision_scope": (
                "Uses the original CUDA/BF16 prediction arithmetic; exact matching also "
                "depends on hardware, software and batch size."
                if reference_path else
                "CPU or FP32 evaluation is not guaranteed to match the original GPU/BF16 results."
            ),
            "nll_probability_floor": 1e-30,
            "scope": "One fixed released student; no calibration, training or model selection.",
        },
    }


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    try:
        result = evaluate(args)
        content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(content)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also closes the race after the early existence check.
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(content)
            print(f"Saved evaluation JSON: {args.output}", file=sys.stderr)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
