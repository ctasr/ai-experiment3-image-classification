"""Path routing around the frozen Source10 trainer; training maths stay there.

The adapter is a NEW source identity. It never labels a newly produced checkpoint
as an original Source10 artifact. Use a fresh Python process for each invocation.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset

import aligned_distillation.train as core
import aligned_distillation.teacher as teacher_module
import extensions.models as model_module

REPO = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
SEEDS = (42, 17, 73)
ROLES = {"A": "crop_mix", "B": "mixed_view"}
engine = core.engine


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sources():
    expected = read(PACKAGE / "evidence/source10.json")["source_sha256"]
    actual = {name: sha(REPO / name) for name in expected}
    if actual != expected:
        raise ValueError("Frozen Source10 differs from the recorded experiment")
    adapter_names = ("__init__.py", "adapter.py", "train.py", "config.json")
    extra = {f"contributions/alignment/{name}": sha(PACKAGE / name) for name in adapter_names}
    return {**actual, **extra}


def config(seed, role):
    if seed not in SEEDS or role not in ROLES:
        raise ValueError("Only fixed seeds 42/17/73 and roles A/B are defined")
    result = read(PACKAGE / "config.json")
    result.update(seed=seed, alignment=ROLES[role])
    return result


@dataclass(frozen=True)
class Paths:
    data: Path
    cutmix: Path
    transfer: Path
    imagenet: Path
    output: Path

    @classmethod
    def from_args(cls, args):
        data = Path(args.data_dir).resolve()
        imagenet = Path(args.imagenet_weights).resolve() if args.imagenet_weights else data / "pretrained/resnet18-f37072fd.pth"
        return cls(data, Path(args.teacher_cutmix).resolve(), Path(args.teacher_transfer).resolve(), imagenet, Path(args.output_dir).resolve())


def add_paths(parser):
    parser.add_argument("--data-dir", required=True, help="directory containing raw/, splits/, dataset_manifest.json and normalization.json")
    parser.add_argument("--teacher-cutmix", required=True, help="frozen seed42 CutMix teacher checkpoint")
    parser.add_argument("--teacher-transfer", required=True, help="frozen seed42 ResNet18 teacher checkpoint")
    parser.add_argument("--imagenet-weights", help="original ResNet18 initialization file; default DATA/pretrained/resnet18-f37072fd.pth")
    parser.add_argument("--output-dir", required=True, help="new isolated output root; existing unrelated files are rejected")


def data_identity(data):
    reference = read(PACKAGE / "evidence/source10.json")
    manifest = read(data / "dataset_manifest.json")
    split = data / "splits/train_validation_seed42.json"
    if sha(data / "dataset_manifest.json") != reference["dataset_manifest_sha256"]:
        raise ValueError("The recorded official dataset manifest differs")
    if sha(split) != reference["split_sha256"] or sha(data / "normalization.json") != reference["normalization_sha256"]:
        raise ValueError("The fixed split or normalization differs")
    if manifest["split_sha256"] != reference["split_sha256"]:
        raise ValueError("Dataset manifest has a different split")
    return {"dataset_manifest_sha256": sha(data / "dataset_manifest.json"), "split_sha256": sha(split), "normalization_sha256": sha(data / "normalization.json")}


def load_data(data):
    """Same uint8 tensors and float64 statistics as data.training_data, new path."""
    data_identity(data)
    manifest = read(data / "dataset_manifest.json")
    split = read(data / "splits/train_validation_seed42.json")
    train, validation = split["train_indices"], split["validation_indices"]
    if len(train) != 45000 or len(validation) != 5000 or len(set(train)) != 45000 or len(set(validation)) != 5000 or set(train) & set(validation) or set(train) | set(validation) != set(range(50000)):
        raise ValueError("Invalid 45k/5k split")
    arrays, targets = [], []
    for item in manifest["official_train_batches"]:
        path = data / "raw/cifar-10-batches-py" / Path(item["file"]).name
        if sha(path) != item["sha256"]:
            raise ValueError(f"Official training batch changed: {path.name}")
        # Match CIFAR10(train=True)'s exact vstack/reshape/transpose order,
        # without its generic integrity check opening the official test file.
        with path.open("rb") as stream:
            saved = pickle.load(stream, encoding="latin1")
        arrays.append(saved["data"])
        targets.extend(saved["labels"])
    source_data = np.vstack(arrays).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
    images = torch.from_numpy(source_data.copy()).permute(0, 3, 1, 2).contiguous()
    labels = torch.tensor(targets, dtype=torch.long)
    if not np.array_equal(np.bincount(np.array(targets)[train]), np.full(10, 4500)) or not np.array_equal(np.bincount(np.array(targets)[validation]), np.full(10, 500)):
        raise ValueError("Class counts differ from the fixed split")
    mean = source_data[train].mean(axis=(0, 1, 2), dtype=np.float64) / 255
    std = source_data[train].std(axis=(0, 1, 2), dtype=np.float64) / 255
    stats = {"source": "fixed 45000 training images only", "split_sha256": manifest["split_sha256"], "mean": mean.tolist(), "std": std.tolist()}
    if stats != read(data / "normalization.json"):
        raise ValueError("45k training statistics do not match normalization.json")
    return TensorDataset(images[train], labels[train]), TensorDataset(images[validation], labels[validation]), stats


def input_identity(paths):
    reference = read(PACKAGE / "evidence/source10.json")
    weights = {"cutmix": sha(paths.cutmix), "transfer": sha(paths.transfer)}
    if weights != reference["teacher_identity"]["weights"]:
        raise ValueError("Expected the original fixed Teacher42 weights")
    if sha(paths.imagenet) != reference["imagenet_initialization_sha256"]:
        raise ValueError("The ResNet18 initialization file differs from the original Teacher construction")
    combined = sources()
    return {"kind": "portable_alignment_v1", "source_sha256": combined, "legacy_core_source_sha256": reference["source_sha256"], "adapter_source_sha256": {n: h for n, h in combined.items() if n not in reference["source_sha256"]}, "data": data_identity(paths.data), "teacher_identity": reference["teacher_identity"], "imagenet_initialization_sha256": sha(paths.imagenet)}


def prepare_output(paths):
    """Only a fresh directory or exactly matching owned output may be used."""
    for source in (paths.data, paths.cutmix, paths.transfer, paths.imagenet, REPO):
        if paths.output == source or source.is_relative_to(paths.output):
            raise ValueError("Output must not contain input files or the checkout")
    if paths.output.is_relative_to(paths.data):
        raise ValueError("Output must not be placed inside the immutable data directory")
    identity = input_identity(paths)
    marker = paths.output / "portable_alignment_identity.json"
    if marker.exists():
        if read(marker) != identity:
            raise ValueError("Output belongs to different sources/inputs; use a new directory")
    elif paths.output.exists() and any(paths.output.iterdir()):
        raise ValueError("Refusing to overwrite an unrelated output directory")
    else:
        paths.output.mkdir(parents=True, exist_ok=True)
        write(marker, identity)
        write(paths.output / "input_paths.json", {k: str(v) for k, v in vars(paths).items()})
        write(paths.output / "source_manifest.json", {"source_sha256": identity["source_sha256"], "legacy_core_source_sha256": identity["legacy_core_source_sha256"], "adapter_source_sha256": identity["adapter_source_sha256"], "identity_note": "new adapter sources are additional, not original Source10"})
        write(paths.output / "teacher_identity.json", identity["teacher_identity"])
    return identity


@contextmanager
def process_lock(output):
    path = output / "active_driver.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"An existing lock prevents duplicate training: {path}. Check its PID before removing a stale lock.") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


@contextmanager
def preserve_runtime():
    initialized = torch.cuda.is_initialized()
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all() if initialized else None)
    flags = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(), torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.utils.deterministic.fill_uninitialized_memory)
    try:
        yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])
        torch.set_num_threads(flags[0]); torch.use_deterministic_algorithms(flags[1], warn_only=flags[2])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = flags[3:5]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags[5:7]
        torch.utils.deterministic.fill_uninitialized_memory = flags[7]


@contextmanager
def training_scope(paths):
    if engine.mixed_epoch is not core.BASE_MIXED or tuple(engine.SOURCES) != tuple(core.BASE_SOURCES):
        raise RuntimeError("A fresh process is required; another training adapter has changed the engine")
    if not torch.utils.deterministic.fill_uninitialized_memory:
        raise RuntimeError("The original deterministic fill default must remain enabled")
    teacher_identity = read(PACKAGE / "evidence/source10.json")["teacher_identity"]
    teacher_paths = {"cutmix": paths.cutmix, "transfer": paths.transfer}

    def routed_identity():
        if {name: sha(path) for name, path in teacher_paths.items()} != teacher_identity["weights"]:
            raise ValueError("Frozen Teacher files changed")
        return {}, teacher_identity

    class RoutedTeacher(teacher_module.Teacher):
        def __init__(self):
            # This is the original constructor order, with only checkpoint paths routed.
            self.identity = routed_identity()[1]
            self.parts = []
            for family in ("cutmix", "transfer"):
                saved = torch.load(teacher_paths[family], map_location="cpu", weights_only=False)
                if saved["source_sha256"] != teacher_identity["source_sha256"]:
                    raise ValueError("Teacher checkpoint provenance differs")
                model = teacher_module.build(saved["config"]).cuda().to(memory_format=torch.channels_last)
                model.load_state_dict(saved["model"])
                model.eval().requires_grad_(False)
                transform = teacher_module.Transform(saved["stats"], saved["config"]).cuda()
                before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                self.parts.append((model, transform, saved["config"], before))

    bindings = [(core, n) for n in ("DIRECTORY", "OUT", "ACTIVE", "TEACHER", "Teacher", "identity", "SOURCES")]
    bindings += [(teacher_module, "identity"), (model_module, "WEIGHTS")]
    bindings += [(engine, n) for n in ("ROOT", "DIRECTORY", "SOURCES", "mixed_epoch", "training_data", "source_hashes")]
    saved_bindings = [(module, name, getattr(module, name)) for module, name in bindings]
    with preserve_runtime():
        try:
            core.DIRECTORY, core.OUT = paths.output / "runs", paths.output
            core.SOURCES = tuple(sources())
            core.identity = teacher_module.identity = routed_identity
            core.Teacher = RoutedTeacher
            model_module.WEIGHTS = paths.imagenet
            engine.ROOT = paths.output  # isolated PAUSE_ROUND2 sentinel only
            engine.training_data = lambda: load_data(paths.data)
            engine.source_hashes = sources
            yield
        finally:
            for module, name, value in reversed(saved_bindings):
                setattr(module, name, value)


def history(folder, require_complete=True):
    with (Path(folder) / "history.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [{k: int(v) if k in ("epoch", "train_samples", "validation_samples") else float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    if not rows or [r["epoch"] for r in rows] != list(range(1, len(rows) + 1)) or (require_complete and len(rows) != 160):
        raise ValueError("Incomplete or invalid history")
    for index, row in enumerate(rows):
        factor = (index + 1) / 5 if index < 5 else .5 * (1 + math.cos(math.pi * min(index - 5, 155) / 155))
        if row["train_samples"] != 45000 or row["validation_samples"] != 5000 or abs(row["learning_rate"] - .1 * factor) > 1e-14 or not all(math.isfinite(v) for v in row.values()):
            raise ValueError("Sample accounting, learning rate, or history differs")
    return rows
