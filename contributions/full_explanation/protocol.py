"""Explicit input identities and the frozen, pre-inference 5000-image design."""
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import pickle
import random

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SEEDS = (42, 17, 73)
PROTOCOL = "full-validation-explanation-v1"
CASES = ["clean", "brightness_plus8", "brightness_minus8", "translate_right1"]
SOURCE_FILES = ("__init__.py", "protocol.py", "engine.py", "legacy_math.py", "run.py", "check.py",
                "analysis.py", "figures.py", "config.json", "provenance.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def provenance():
    value = read(HERE / "provenance.json")
    if sha(HERE / "legacy_math.py") != value["legacy_math_sha256"]:
        raise ValueError("Copied original analytic/CPU-autograd mathematics changed")
    return value


def sources():
    legacy = provenance()["model_training_sources"]
    if len(legacy) != 12 or any(sha(REPO / name) != digest for name, digest in legacy.items()):
        raise ValueError("Frozen model-training Source12 differs")
    return {**legacy, **{"contributions/full_explanation/" + name: sha(HERE / name) for name in SOURCE_FILES}}


def data_path(data_dir, relative):
    parts = PurePosixPath(str(relative).replace("\\", "/")).parts
    if parts[:1] == ("data",):
        parts = parts[1:]
    target = (Path(data_dir) / Path(*parts)).resolve()
    if not parts or ".." in parts or not target.is_relative_to(Path(data_dir).resolve()):
        raise ValueError("Dataset path escaped explicit data directory")
    return target


def data_identity(data_dir):
    data_dir = Path(data_dir).resolve()
    expected = provenance()["data_metadata_sha256"]
    actual = {name: sha(data_dir / name) for name in expected}
    if actual != expected:
        raise ValueError("Original split/normalization/manifest bytes differ")
    manifest = read(data_dir / "dataset_manifest.json")
    batches = {}
    for item in manifest["official_train_batches"]:
        path = data_path(data_dir, item["file"])
        if not path.name.startswith("data_batch_") or sha(path) != item["sha256"]:
            raise ValueError("Original official-training batch differs")
        batches[path.name] = item["sha256"]
    return {"metadata_sha256": actual, "training_batch_sha256": batches,
            "official_test_or_external_file_opened": False}


def load_validation(data_dir):
    """Original train-pickle reconstruction; never constructs CIFAR10/test."""
    manifest = read(Path(data_dir) / "dataset_manifest.json")
    split = read(data_path(data_dir, manifest["split_file"]))
    train, validation = split["train_indices"], split["validation_indices"]
    if (len(train) != 45000 or len(validation) != 5000 or len(set(train)) != 45000
            or len(set(validation)) != 5000 or set(train) & set(validation)
            or set(train) | set(validation) != set(range(50000))):
        raise ValueError("Not the complete disjoint original45k/5k partition")
    arrays, labels = [], []
    for item in manifest["official_train_batches"]:
        path = data_path(data_dir, item["file"])
        if sha(path) != item["sha256"]:
            raise ValueError("Training batch SHA changed")
        with path.open("rb") as stream:
            batch = pickle.load(stream, encoding="bytes")
        arrays.append(batch[b"data"].reshape(-1, 3, 32, 32)); labels.extend(batch[b"labels"])
    data, labels = np.concatenate(arrays), np.asarray(labels, dtype=np.int64)
    indices = np.asarray(validation, dtype=np.int64); y = labels[indices]
    if not np.array_equal(np.bincount(y, minlength=10), np.full(10, 500)):
        raise ValueError("Original validation class counts differ")
    return torch.from_numpy(data[indices].copy()), y, indices


def references(reference_dir):
    reference_dir = Path(reference_dir).resolve()
    expected = provenance()["reference_files_sha256"]
    if {name: sha(reference_dir / name) for name in expected} != expected:
        raise ValueError("Original reference cache bytes differ")
    result = {}
    for seed in SEEDS:
        with np.load(reference_dir / f"legacy_{seed}.npz", allow_pickle=False) as source:
            result[seed] = {name: source[name] for name in source.files}
        if "loc" not in result[seed]:
            result[seed]["loc"] = result[seed]["positions"][:, 0, 0] * 25 + result[seed]["positions"][:, 0, 1]
        if "flat" not in result[seed]:
            result[seed]["flat"] = np.ptp(result[seed]["heat"], axis=(1, 2)) == 0
        if "target" not in result[seed]:
            result[seed]["target"] = result[seed]["q"].argmax(1)
    with np.load(reference_dir / "seed42_clean5000.npz", allow_pickle=False) as source:
        clean = {name: source[name] for name in source.files}
    return result, clean


def design(data_dir, reference_dir):
    """Lock IDs/targets/windows before any new model inference or prediction."""
    legacy, clean = references(reference_dir)
    manifest = read(Path(data_dir) / "dataset_manifest.json")
    original = np.asarray(read(data_path(data_dir, manifest["split_file"]))["validation_indices"], dtype=np.int64)
    if not np.array_equal(original, clean["official_train_indices"]):
        raise ValueError("Old5000 cache order differs from original validation partition")
    old_indices = legacy[42]["indices"]
    lookup = {int(value): row for row, value in enumerate(original)}
    old_rows = np.asarray([lookup[int(value)] for value in old_indices], dtype=np.int64)
    if len(old_rows) != 200 or len(set(old_rows.tolist())) != 200:
        raise ValueError("Original200 cache indices differ")
    # Target comes from original full5000 q, never from new batch32 predictions.
    frozen_target = clean["clean_q"].argmax(1).astype(np.int64)
    if not np.array_equal(frozen_target[old_rows], legacy[42]["target"]):
        raise ValueError("Old200 and original5000 target classes disagree")
    old_set = set(old_rows.tolist())
    processing_rows = np.asarray([*old_rows.tolist(), *[row for row in range(5000) if row not in old_set]], dtype=np.int64)
    rng = np.random.default_rng(42)
    random_positions = rng.integers(0, 25, size=(5000, 10, 2))
    for seed in SEEDS:
        ref = legacy[seed]
        if not (np.array_equal(ref["indices"], old_indices) and np.array_equal(ref["labels"], clean["labels"][old_rows])
                and np.array_equal(ref["target"], frozen_target[old_rows])
                and np.array_equal(ref["positions"][:, 1:], random_positions[:200])):
            raise ValueError("Legacy seed targets/indices/labels/random windows differ")
    return {"validation_rows": processing_rows, "indices": original[processing_rows],
            "labels": clean["labels"][processing_rows], "target": frozen_target[processing_rows],
            "random_positions": random_positions, "legacy200": np.arange(5000) < 200}, legacy


def add_inputs(parser):
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--weights-42", type=Path, required=True)
    parser.add_argument("--weights-17", type=Path, required=True)
    parser.add_argument("--weights-73", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)


def input_identity(args):
    weights = {str(seed): sha(getattr(args, f"weights_{seed}")) for seed in SEEDS}
    if weights != provenance()["model_weights_sha256"]:
        raise ValueError("Original three BEST student weight bytes differ")
    reference_hashes = {name: sha(Path(args.reference_dir) / name) for name in provenance()["reference_files_sha256"]}
    if reference_hashes != provenance()["reference_files_sha256"]:
        raise ValueError("Original200/full-target reference bytes differ")
    return {"protocol": PROTOCOL, "source_sha256": sources(), "weights_sha256": weights,
            "data": data_identity(args.data_dir), "references_sha256": reference_hashes,
            "config": read(HERE / "config.json"), "new_training": 0, "new_official_test": 0,
            "new_temperature_or_threshold": 0}


def prepare(args):
    output = Path(args.output_dir).resolve()
    for path in (REPO, args.data_dir, args.reference_dir, args.weights_42, args.weights_17, args.weights_73):
        source = Path(path).resolve()
        if output == source or source.is_relative_to(output):
            raise ValueError("Output must not contain input files or checkout")
    if output.is_relative_to(Path(args.data_dir).resolve()) or output.is_relative_to(Path(args.reference_dir).resolve()):
        raise ValueError("Output must not be inside immutable data/references")
    identity = input_identity(args)
    marker = output / "identity.json"
    if marker.exists():
        if read(marker) != identity:
            raise ValueError("Output belongs to a different source or input identity")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Nonempty output lacks this stage's identity")
    else:
        output.mkdir(parents=True, exist_ok=True); write(marker, identity)
    arrays, _ = design(args.data_dir, args.reference_dir)
    path = output / "frozen_design.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays) or any(not np.array_equal(existing[name], value) for name, value in arrays.items()):
                raise ValueError("Previously frozen inference target/order/window design differs")
    else:
        save_npz(path, **arrays)
    write(output / "design_manifest.json", {"identity_sha256": sha(marker), "design_sha256": sha(path),
          "samples": 5000, "unique_images": 5000, "per_class": [500] * 10,
          "segments": [200, 4800], "target_locked_before_inference": True,
          "target_source": "original seed42 clean5000 q cache argmax", "random_generator": "private PCG64 default_rng(42)",
          "random_prefix200_exact": True, "global_RNG_not_used_for_design": True})
    return identity, arrays


@contextmanager
def preserve_runtime():
    initialized = torch.cuda.is_initialized()
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
             torch.cuda.get_rng_state_all() if initialized else None, torch.get_num_threads(),
             torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.utils.deterministic.fill_uninitialized_memory)
    try:
        yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])
        torch.set_num_threads(state[4]); torch.use_deterministic_algorithms(state[5], warn_only=state[6])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = state[7:9]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = state[9:11]
        torch.utils.deterministic.fill_uninitialized_memory = state[11]


def fp32_runtime(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Expected one of the three frozen original model seeds")
    deterministic = torch.are_deterministic_algorithms_enabled()
    return {"seed": seed, "global_deterministic_algorithms": deterministic,
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "bilinear_interpolation_dispatch": "torch._decomp.decompositions._upsample_linear_vec" if deterministic else "torch._C._nn.upsample_bilinear2d",
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "CPU_threads": torch.get_num_threads(), "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
            "policy": "original seed42 native bilinear route; original17/73 deterministic decomposition route"}


def configure_FP32(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Expected one of the three frozen original model seeds")
    if not torch.utils.deterministic.fill_uninitialized_memory:
        raise RuntimeError("Original deterministic fill default must remain enabled")
    torch.set_num_threads(8)
    # Original seed42 two_view_extension used the fresh-process defaultFalse;
    # seed17/73 seed_explanation_study called train.seed_everything (True).
    # In this PyTorch version that flag selects a different bilinear forward.
    torch.use_deterministic_algorithms(seed != 42)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    return fp32_runtime(seed)


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


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def assert_frozen(model, before):
    if model.training or state_hash(model) != before or any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Model weights, BN state or gradients changed")
