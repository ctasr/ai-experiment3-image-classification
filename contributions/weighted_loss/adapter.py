"""Path-only and loss-only routing over the frozen main-12 training loop.

This is a new source version. It deliberately refuses to resume a legacy
Source15/Source16 checkpoint, even when the loss arithmetic is identical.
"""
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
import traceback

import numpy as np
import torch
from torch.utils.data import TensorDataset

import aligned_distillation.train as CORE
import extensions.models as MODEL_MODULE
from .loss import MODES, combined, original_loss_parts

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENGINE = CORE.engine
TEACHER_MODULE = sys.modules[CORE.Teacher.__module__]
ORIGINAL_TEACHER = CORE.Teacher
ORIGINAL_AUGMENT = CORE.augment_raw
PROTOCOL = "portable-weighted-v1"
TRAINING_FILES = ("__init__.py", "legacy_loss.py", "loss.py", "adapter.py", "train.py", "config.json", "provenance.json")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def provenance():
    value = read(HERE / "provenance.json")
    assert sha(HERE / "legacy_loss.py") == value["original_loss_sha256"]
    return value


def source_hashes(mode="weighted_single"):
    if mode not in MODES:
        raise ValueError("Unknown objective source identity")
    original = provenance()["frozen_main12"]
    if len(original) != 12:
        raise ValueError("Expected the frozen main-12 source dictionary")
    for name, expected in original.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    additions = {"contributions/weighted_loss/" + name: sha(HERE / name) for name in TRAINING_FILES}
    additions[f"contributions/weighted_loss/configs/{mode}.json"] = sha(HERE / "configs" / (mode + ".json"))
    return {**original, **additions}


def data_path(data_dir, relative):
    # The historical manifest uses repository-relative data/... names.
    parts = PurePosixPath(str(relative).replace("\\", "/")).parts
    if parts and parts[0] == "data":
        parts = parts[1:]
    if not parts or any(part in ("..", "") for part in parts):
        raise ValueError("Unsafe dataset manifest path")
    result = (Path(data_dir) / Path(*parts)).resolve()
    if not result.is_relative_to(Path(data_dir).resolve()):
        raise ValueError("Dataset path escaped --data-dir")
    return result


def data_identity(data_dir):
    data_dir = Path(data_dir).resolve()
    manifest = read(data_dir / "dataset_manifest.json")
    split_path = data_path(data_dir, manifest["split_file"])
    if sha(split_path) != manifest["split_sha256"]:
        raise ValueError("Split SHA mismatch")
    split = read(split_path)
    training, validation = split["train_indices"], split["validation_indices"]
    if (len(training) != 45000 or len(validation) != 5000 or len(set(training)) != 45000
            or len(set(validation)) != 5000 or set(training) & set(validation)
            or set(training) | set(validation) != set(range(50000))):
        raise ValueError("Expected the complete disjoint 45k/5k official-training split")
    raw = {}
    for item in manifest["official_train_batches"]:
        path = data_path(data_dir, item["file"])
        actual = sha(path)
        if actual != item["sha256"]:
            raise ValueError(f"Training batch SHA mismatch: {path.name}")
        raw[path.name] = actual
    metadata = {"manifest_sha256": sha(data_dir / "dataset_manifest.json"),
                "split_sha256": sha(split_path), "normalization_sha256": sha(data_dir / "normalization.json")}
    if metadata != provenance()["frozen_data_metadata"]:
        raise ValueError("Expected the frozen split, manifest and normalization records")
    return {**metadata,
            "training_batches_sha256": raw, "training_samples": 45000, "validation_samples": 5000,
            "official_test_read": False}


def training_data(data_dir):
    """Same tensor order and FP64 train-only statistics as frozen data.py."""
    manifest = read(Path(data_dir) / "dataset_manifest.json")
    split = read(data_path(data_dir, manifest["split_file"]))
    train, validation = split["train_indices"], split["validation_indices"]
    # Match CIFAR10(train=True)'s tensor construction while opening only the
    # five frozen training batches. Its generic integrity check also hashes
    # test_batch; that extra file access is unnecessary for this entry point.
    arrays, targets = [], []
    for item in manifest["official_train_batches"]:
        with data_path(data_dir, item["file"]).open("rb") as stream:
            batch = pickle.load(stream, encoding="latin1")
        arrays.append(batch["data"]); targets.extend(batch["labels"])
    source_data = np.vstack(arrays).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
    images = torch.from_numpy(source_data.copy()).permute(0, 3, 1, 2).contiguous()
    labels = torch.tensor(targets, dtype=torch.long)
    mean = source_data[train].mean(axis=(0, 1, 2), dtype=np.float64) / 255
    std = source_data[train].std(axis=(0, 1, 2), dtype=np.float64) / 255
    stats = {"source": "fixed 45000 training images only", "split_sha256": manifest["split_sha256"],
             "mean": mean.tolist(), "std": std.tolist()}
    if read(Path(data_dir) / "normalization.json") != stats:
        raise ValueError("Normalization differs from the fixed training-only statistics")
    if not np.all(np.bincount(np.asarray(targets)[train], minlength=10) == 4500):
        raise ValueError("Unexpected per-class training counts")
    if not np.all(np.bincount(np.asarray(targets)[validation], minlength=10) == 500):
        raise ValueError("Unexpected per-class validation counts")
    return TensorDataset(images[train], labels[train]), TensorDataset(images[validation], labels[validation]), stats


def teacher_inputs(cutmix_path, transfer_path):
    paths = {"cutmix": Path(cutmix_path).resolve(), "transfer": Path(transfer_path).resolve()}
    saved = {key: torch.load(path, map_location="cpu", weights_only=False) for key, path in paths.items()}
    expected = {name: value for name, value in provenance()["frozen_main12"].items()
                if name in ("train.py", "data.py", "models.py", "prepare_data.py", "extensions/train.py", "extensions/models.py")}
    for key, checkpoint in saved.items():
        if checkpoint["source_sha256"] != expected or checkpoint["config"]["seed"] != 42:
            raise ValueError(f"Unexpected {key} Teacher source/seed")
        if sha(paths[key]) != provenance()["frozen_teacher_weights"][key]:
            raise ValueError(f"Expected the original frozen {key} Teacher weight bytes")
    if saved["cutmix"]["config"].get("architecture") != "resnet20" or saved["cutmix"]["config"].get("width") != 32:
        raise ValueError("CutMix Teacher must be the original width-32 residual model")
    if saved["transfer"]["config"].get("model") != "transfer":
        raise ValueError("Transfer Teacher must be the original ImageNet transfer model")
    def identity():
        return {}, {"source_sha256": expected, "weights": {key: sha(path) for key, path in paths.items()},
                    "policy": "equal_ensemble_flip", "calibrated": False, "teacher_seed": 42}
    return paths, saved, identity


def configuration(cfg, mode, seed, inputs, pilot_samples=None):
    cfg = dict(cfg)
    if cfg.get("source_set_mode", mode) != mode:
        raise ValueError("Config mode and --mode disagree")
    for field in ("source_set_mode", "portable_protocol", "portable_inputs"):
        cfg.pop(field, None)
    fixed = read(HERE / "config.json")
    # Only the named seed and diagnostic sample subset may vary.
    cfg.pop("seed", None)
    cfg.pop("pilot_samples", None)
    expected = dict(fixed)
    expected.pop("seed")
    if cfg != expected or mode not in MODES or seed not in (42, 17, 73):
        raise ValueError("Configuration differs from fixed width48/KD240 recipe")
    if pilot_samples not in (None, 456):
        raise ValueError("Only the explicit 456-image GPU preflight subset is supported")
    cfg.update(seed=seed, source_set_mode=mode, portable_protocol=PROTOCOL, portable_inputs=inputs)
    if pilot_samples is not None:
        cfg["pilot_samples"] = pilot_samples
    return cfg


def prepare_output(output_dir, data_dir, teacher_paths, imagenet_weights, identity):
    """Reject unrelated directories and output locations that contain inputs."""
    output_dir, data_dir = Path(output_dir).resolve(), Path(data_dir).resolve()
    for source in (data_dir, ROOT, Path(imagenet_weights).resolve(), *teacher_paths.values()):
        source = Path(source).resolve()
        if output_dir == source or source.is_relative_to(output_dir):
            raise ValueError("Output must not contain the checkout or input files")
    if output_dir.is_relative_to(data_dir):
        raise ValueError("Output must not be inside the immutable dataset directory")
    marker = output_dir / "portable_weighted_identity.json"
    if marker.exists():
        if read(marker) != identity:
            raise ValueError("Output belongs to a different objective, source package or inputs")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Nonempty output has no matching portable identity; use a fresh directory")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        write(marker, identity)


def check_trace(path, completed_epochs, cfg):
    """Do not append duplicate batches after an uncommitted partial epoch."""
    samples = cfg.get("pilot_samples", 45000)
    batches = math.ceil(samples / cfg["batch_size"])
    expected_count, count = completed_epochs * batches, 0
    path = Path(path)
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for count, line in enumerate(stream, 1):
                row = json.loads(line)
                epoch, batch = (count - 1) // batches + 1, (count - 1) % batches + 1
                size = cfg["batch_size"] if batch < batches else samples - cfg["batch_size"] * (batches - 1)
                if (row["epoch"], row["batch_in_epoch"], row["global_batch"], row["batch_size"]) != (epoch, batch, count, size):
                    raise ValueError("Trace batch identities differ from the committed epoch sequence")
    if count != expected_count:
        raise ValueError("Trace and epoch checkpoint disagree. Preserve the partial trace for explicit recovery; do not silently append duplicate batches.")
    return {"committed_epochs": completed_epochs, "records": count, "batches_per_epoch": batches,
            "final_batch_samples": samples - cfg["batch_size"] * (batches - 1)}


def bindings():
    return ([(CORE, key) for key in ("OUT", "DIRECTORY", "SOURCES", "Teacher", "identity", "ACTIVE", "TEACHER", "loss_parts", "augment_raw")]
            + [(ENGINE, key) for key in ("ROOT", "DIRECTORY", "SOURCES", "mixed_epoch", "training_data", "source_hashes")]
            + [(TEACHER_MODULE, key) for key in ("OUT", "identity")]
            + [(MODEL_MODULE, "WEIGHTS")])


@contextmanager
def runtime():
    initialized = torch.cuda.is_initialized()
    saved = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
             torch.cuda.get_rng_state_all() if initialized else None, torch.get_num_threads(),
             torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.utils.deterministic.fill_uninitialized_memory)
    receipt = {"CUDA_initialized_before": initialized}
    try:
        yield receipt
    finally:
        random.setstate(saved[0]); np.random.set_state(saved[1]); torch.set_rng_state(saved[2])
        if saved[3] is not None:
            torch.cuda.set_rng_state_all(saved[3])
        torch.set_num_threads(saved[4]); torch.use_deterministic_algorithms(saved[5], warn_only=saved[6])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = saved[7:9]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[9:11]
        torch.utils.deterministic.fill_uninitialized_memory = saved[11]
        receipt.update(CPU_RNG_threads_flags_restored=True,
                       prior_CUDA_RNG_restored=initialized,
                       first_CUDA_initialization_cannot_be_reversed=not initialized and torch.cuda.is_initialized())


@contextmanager
def scope(cfg, output_dir, data_dir, saved_teachers, identity, start_epoch, trace_path, imagenet_weights):
    if (ENGINE.mixed_epoch is not CORE.BASE_MIXED or tuple(ENGINE.SOURCES) != tuple(CORE.BASE_SOURCES)
            or CORE.loss_parts is not original_loss_parts or CORE.Teacher is not ORIGINAL_TEACHER):
        raise RuntimeError("Use a fresh process; another training adapter has changed the frozen engine")
    if not torch.utils.deterministic.fill_uninitialized_memory:
        raise RuntimeError("The frozen deterministic fill default must remain enabled")
    output_dir = Path(output_dir).resolve()
    missing = object()
    previous = [(module, key, getattr(module, key, missing)) for module, key in bindings()]
    current, counter = {}, 0
    audit = {"protocol": PROTOCOL, "mode": cfg["source_set_mode"], "frozen_main_loop_reused": True}
    trace = None
    try:
        with runtime() as state:
            try:
                class PortableTeacher(ORIGINAL_TEACHER):
                    def __init__(self):
                        self.identity = identity()[1]
                        self.parts = []
                        # Same family order, build calls, transforms and RNG consumption.
                        for family in ("cutmix", "transfer"):
                            saved = saved_teachers[family]
                            model = TEACHER_MODULE.build(saved["config"]).cuda().to(memory_format=torch.channels_last)
                            model.load_state_dict(saved["model"])
                            model.eval().requires_grad_(False)
                            transform = TEACHER_MODULE.Transform(saved["stats"], saved["config"]).cuda()
                            before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                            self.parts.append((model, transform, saved["config"], before))

                    def probabilities(self, images):
                        result = super().probabilities(images)
                        current["teacher_input_sha256"] = tensor_sha(images)
                        current["teacher_raw_q_sha256"] = tensor_sha(result)
                        return result

                def augmented(images):
                    view = ORIGINAL_AUGMENT(images)
                    current["view_sha256"] = tensor_sha(view)
                    return view

                def routed_loss(logits, labels, order, lam, q, temperature, alpha):
                    nonlocal counter
                    result, extra, details, original = combined(logits, labels, order, lam, q, temperature, alpha, cfg["source_set_mode"])
                    counter += 1
                    batches = math.ceil(cfg.get("pilot_samples", 45000) / cfg["batch_size"])
                    epoch, batch = start_epoch + (counter - 1) // batches + 1, (counter - 1) % batches + 1
                    row = {"epoch": epoch, "batch_in_epoch": batch, "global_batch": (epoch - 1) * batches + batch,
                           "batch_size": len(labels), "actual_lambda": float(lam), "labels_sha256": tensor_sha(labels),
                           "order_sha256": tensor_sha(order), "teacher_t1_q_sha256": tensor_sha(q),
                           "original_KL": float(original[2].detach()), "new_extra": float(extra.detach()),
                           "modified_KD": float(result[2].detach()), "hard_CE": float(original[1].detach()),
                           "returned_total": float(result[0].detach()), "mode": cfg["source_set_mode"], **current}
                    if not all(math.isfinite(row[key]) for key in ("original_KL", "new_extra", "modified_KD", "hard_CE", "returned_total")):
                        raise FloatingPointError("Non-finite routed loss")
                    if details is not None and not bool(details["all_finite"]):
                        raise FloatingPointError("Non-finite frozen loss diagnostics")
                    trace.write(json.dumps(row, allow_nan=False) + "\n"); trace.flush(); current.clear()
                    return result

                sources = source_hashes(cfg["source_set_mode"])
                output_dir.mkdir(parents=True, exist_ok=True)
                for filename, value in (("teacher_identity.json", identity()[1]), ("source_manifest.json", {"protocol": PROTOCOL, "mode": cfg["source_set_mode"], "source_sha256": sources})):
                    path = output_dir / filename
                    if path.exists() and read(path) != value:
                        raise ValueError(f"Existing output identity differs: {filename}")
                    write(path, value)
                trace_path = Path(trace_path)
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace = trace_path.open("a", encoding="utf-8")
                CORE.OUT, CORE.DIRECTORY, CORE.SOURCES = output_dir, output_dir / "runs", tuple(sources)
                CORE.identity, CORE.Teacher = identity, PortableTeacher
                CORE.augment_raw, CORE.loss_parts = augmented, routed_loss
                TEACHER_MODULE.OUT, TEACHER_MODULE.identity = output_dir, identity
                ENGINE.training_data = lambda: training_data(data_dir)
                ENGINE.ROOT, ENGINE.source_hashes = output_dir, lambda: source_hashes(cfg["source_set_mode"])
                MODEL_MODULE.WEIGHTS = Path(imagenet_weights).resolve()
                yield audit
            except BaseException as exc:
                audit["exception"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
                raise
            finally:
                if trace is not None:
                    trace.close()
                current.clear()
                for module, key, value in reversed(previous):
                    if value is missing:
                        if hasattr(module, key):
                            delattr(module, key)
                    else:
                        setattr(module, key, value)
                audit["all_bindings_restored"] = all((not hasattr(module, key)) if value is missing else getattr(module, key) is value for module, key, value in previous)
                audit["batches_seen"] = counter
                audit["runtime"] = state
    finally:
        pass


def run(config_path, mode, seed, data_dir, teacher_cutmix, teacher_transfer, output_dir,
        run_id=None, stop_after=240, pilot_samples=None, imagenet_weights=None):
    if type(stop_after) is not int or not 1 <= stop_after <= 240:
        raise ValueError("--stop-after must be an epoch in [1,240]")
    run_id = run_id or f"{mode}_seed{seed}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or not run_id.startswith(mode + "_"):
        raise ValueError("Run ID must start with its mode and contain no path components")
    output_dir = Path(output_dir).resolve()
    folder = output_dir / "runs" / run_id
    sources = source_hashes(mode)
    data = data_identity(data_dir)
    paths, saved_teachers, identity = teacher_inputs(teacher_cutmix, teacher_transfer)
    imagenet_weights = Path(imagenet_weights) if imagenet_weights is not None else Path(data_dir) / "pretrained/resnet18-f37072fd.pth"
    initialization_sha = sha(imagenet_weights)
    if initialization_sha != saved_teachers["transfer"]["pretrained_sha256"]:
        raise ValueError("ImageNet initialization differs from the frozen transfer Teacher")
    inputs = {"data": data, "teacher": identity()[1], "imagenet_initialization_sha256": initialization_sha}
    cfg = configuration(read(config_path), mode, seed, inputs, pilot_samples)
    stats = read(Path(data_dir) / "normalization.json")
    if any(saved["stats"] != stats for saved in saved_teachers.values()):
        raise ValueError("Teacher normalization/split differs from the requested data")
    common_config = {key: value for key, value in cfg.items() if key != "seed"}
    prepare_output(output_dir, data_dir, paths, imagenet_weights,
                   {"protocol": PROTOCOL, "mode": mode, "source_sha256": sources,
                    "inputs": inputs, "common_config": common_config})
    start = 0
    if (folder / "config.json").exists() and read(folder / "config.json") != cfg:
        raise ValueError("Cannot resume a different config, input identity or mode")
    if (folder / "last.pt").exists():
        saved = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
        if saved["config"] != cfg or saved["source_sha256"] != sources:
            raise ValueError("Cannot mix legacy or changed sources with this checkpoint")
        start = int(saved["epoch"])
        if start != len(saved["history"]):
            raise ValueError("Checkpoint epoch/history mismatch")
        if start >= stop_after:
            return {"state": "already_at_target", "completed_epochs": start, "core_called": False}
    check_trace(folder / "trace.jsonl", start, cfg)
    folder.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".training.lock"
    lock = lock_path.open("x", encoding="utf-8")
    lock.write(json.dumps({"pid": os.getpid(), "run_id": run_id, "mode": mode})); lock.flush()
    audit, failure = {}, None
    try:
        with scope(cfg, output_dir, data_dir, saved_teachers, identity, start, folder / "trace.jsonl", imagenet_weights) as audit:
            result = CORE.run(cfg, run_id, stop_after=stop_after)
            if (folder / "last.pt").exists():
                committed = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
                if committed["config"] != cfg or committed["source_sha256"] != sources:
                    raise ValueError("Final checkpoint source/config differs")
                if committed["epoch"] != len(committed["history"]):
                    raise ValueError("Final epoch/history differs")
                audit["trace"] = check_trace(folder / "trace.jsonl", committed["epoch"], cfg)
            elif result.get("state") == "paused":
                audit["trace"] = check_trace(folder / "trace.jsonl", 0, cfg)
            else:
                raise ValueError("No epoch checkpoint was produced")
            # The frozen core's generic metadata names full KL. Correct that
            # metadata in this new owned run so its KD metric is not mislabeled.
            objective = read(folder / "objective_metadata.json")
            formulas = {"original": ".75 CE + .25 full_KL",
                        "weighted_single": ".75 CE + .25 [full_KL + lambda*q_yi*D_yi + (1-lambda)*q_yj*D_yj]",
                        "confidence_mass": ".75 CE + .25 mean([1+lambda*q_yi+(1-lambda)*q_yj]*per_image_full_KL)"}
            objective.update(protocol=PROTOCOL, mode=mode, endpoint="fixed LAST240",
                             config=cfg, source_sha256=sources, loss=formulas[mode],
                             confidence_mass_is_gradient_matched=False,
                             train_kd="full KL plus weighted conditional extras" if mode == "weighted_single" else
                                      "per-image coefficient-weighted full KL" if mode == "confidence_mass" else "full KL")
            write(folder / "objective_metadata.json", objective)
        if data_identity(data_dir) != data or identity()[1] != inputs["teacher"] or source_hashes(mode) != sources or sha(imagenet_weights) != initialization_sha:
            raise ValueError("Input/source files changed during training")
        return result
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        try:
            write(folder / ("portable_failure.json" if failure else "portable_audit.json"),
                  {**audit, "start_epoch": start, "stop_after": stop_after, "failure": failure,
                   "source_sha256": sources, "inputs": inputs})
        finally:
            lock.close()
            lock_path.unlink()
