"""Fixed fresh-partition confirmation CLI. No execution on import.

Uses new Teacher LAST160/LAST60 and six predeclared student LAST240 cases.
No old CIFAR Teacher/student checkpoint, official test or retired dataset is read.
Old source files remain unchanged; data/source/Teacher bindings are process-local.
Review and real CPU/GPU preflights are required before formal training.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import random
import struct
import sys
import time

HERE = Path(__file__).resolve().parent
OWNED_ROOT = None
SPLIT_SEED = 20261127
LEGACY_LOSS_SHA = "272ebb7e19cc34023ea7d060972b1d57d2f2a65c39179cd528960e97582a206d"
SEEDS = (42, 17, 73)
TEACHERS = {
    "cutmix": ("configs_round3/cutmix_lr0.1_wd0.0005_seed42.json", 160),
    "transfer": ("configs_round3/transfer_lr0.003_wd0.002_seed42.json", 60),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def owned(path):
    path = Path(path).resolve()
    if OWNED_ROOT is None or not path.is_relative_to(OWNED_ROOT):
        raise ValueError(f"Output escaped the marked crosssplit workspace: {path}")
    return path


def resolve_workspace(project, argument, creating):
    raw = Path(argument).expanduser()
    work = (raw if raw.is_absolute() else project / raw).resolve()
    if work == project or project.is_relative_to(work):
        raise ValueError("Workspace cannot be the project or its ancestor")
    if not work.name.startswith("crosssplit"):
        raise ValueError("Use a dedicated directory named crosssplit...")
    if work.is_relative_to(project) and not work.is_relative_to(project / "outputs"):
        raise ValueError("In-project workspace must be under outputs/, never data/runs/results/source directories")
    if creating:
        if work.exists() and any(work.iterdir()):
            raise ValueError("Prepare requires an empty new workspace; do not overlay or reseal old records")
    else:
        marker = read(work / "workspace_marker.json")
        if marker != {"protocol": "crosssplit-fixed-v1", "project": str(project)}:
            raise ValueError("Wrong workspace marker or source project")
    return work


def package_source_hashes():
    return {"contributions/crosssplit/" + name: sha(HERE / name)
            for name in ("__init__.py", "run.py")}


def write(path, value):
    path = owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".new")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


@contextmanager
def training_lock(work):
    # One byte lock per workspace. Root's umbrella driver also serializes GPU
    # stages across workspaces; never kill a process to acquire this lock.
    import msvcrt
    file = owned(work / "gpu_case.lock")
    with file.open("a+b") as stream:
        if file.stat().st_size == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def raw_train(project, entries):
    import numpy as np
    images, labels = [], []
    assert len(entries) == 5
    assert [Path(item["file"]).name for item in entries] == [f"data_batch_{i}" for i in range(1, 6)]
    for entry in entries:
        path = project / entry["file"]
        assert sha(path) == entry["sha256"]
        with path.open("rb") as stream:
            batch = pickle.load(stream, encoding="latin1")
        data = batch["data"]
        assert data.shape == (10000, 3072) and data.dtype == np.uint8
        images.append(data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        labels.extend(int(v) for v in batch["labels"])
    return np.concatenate(images), np.asarray(labels, dtype=np.int64)


def make_partition(labels):
    """Pure label/index split, private RNG, no image or metric consultation."""
    import numpy as np
    labels = np.asarray(labels, dtype=np.int64)
    assert len(labels) == 50000 and np.bincount(labels).tolist() == [5000] * 10
    rng, train, val = random.Random(SPLIT_SEED), [], []
    for label in range(10):
        indices = np.flatnonzero(labels == label).tolist()
        rng.shuffle(indices)
        val.extend(indices[:500]); train.extend(indices[500:])
    train.sort(); val.sort()
    assert len(train) == 45000 and len(val) == 5000
    assert not set(train) & set(val) and set(train) | set(val) == set(range(50000))
    assert np.bincount(labels[train]).tolist() == [4500] * 10
    assert np.bincount(labels[val]).tolist() == [500] * 10
    return {"seed": SPLIT_SEED, "source": "official train batches only",
            "algorithm": "private Python random.Random; perclass shuffle; valfirst500; sorted combined indices",
            "train_indices": train, "validation_indices": val}


def prepare(project, work, loss):
    """CPU only. Seal labels-based partition before ANY new performance."""
    import numpy as np
    write(work / "workspace_marker.json", {"protocol": "crosssplit-fixed-v1", "project": str(project)})
    assert sha(loss) == LEGACY_LOSS_SHA, "Use the byte-identical published weighted_loss/legacy_loss.py math"
    assert not (work / "plan.json").exists(), "Use existing sealed workspace, never reseal it in place"
    original_sources = read(project / "results_horizon/source_manifest.json")["source_sha256"]
    assert len(original_sources) == 12
    assert all(sha(project / name) == digest for name, digest in original_sources.items())
    entries = read(project / "data/dataset_manifest.json")["official_train_batches"]
    images, labels = raw_train(project, entries)
    split = make_partition(labels)
    train, val = split["train_indices"], split["validation_indices"]
    write(work / "partition.json", split)
    # Exact integer channel moments avoid allocating a 1GB float64 temporary.
    # This new train-only normalization algorithm is frozen for BOTH arms.
    sums = np.zeros(3, dtype=np.uint64); squares = np.zeros(3, dtype=np.uint64)
    for start in range(0, 45000, 512):
        x = images[train[start:start + 512]].astype(np.uint64)
        sums += x.sum(axis=(0, 1, 2), dtype=np.uint64)
        squares += (x * x).sum(axis=(0, 1, 2), dtype=np.uint64)
    count = 45000 * 32 * 32
    mean = sums.astype(np.float64) / count
    variance = squares.astype(np.float64) / count - mean * mean
    assert np.all(variance > 0)
    stats = {"source": "new fixed45000 training images only", "split_sha256": sha(work / "partition.json"),
             "mean": (mean / 255).tolist(), "std": (np.sqrt(variance) / 255).tolist(),
             "normalization_algorithm": "uint64 sums/squares; population sqrt(E[x2]-E[x]^2)/255",
             "channel_sums_uint64": [int(v) for v in sums],
             "channel_squares_uint64": [int(v) for v in squares]}
    write(work / "normalization.json", stats)
    configs, original_configs = {}, {}
    for role, (name, epochs) in TEACHERS.items():
        cfg = read(project / name)
        assert cfg["seed"] == 42 and cfg["epochs"] == epochs
        cfg.update(crosssplit_seed=SPLIT_SEED, partition_sha256=sha(work / "partition.json"),
                   declared_checkpoint_selector="LAST")
        key = "teacher_" + role
        write(work / "configs" / (key + ".json"), cfg)
        configs[key] = {"path": "configs/" + key + ".json", "sha256": sha(work / "configs" / (key + ".json"))}
        original_configs[key] = {"path": name, "sha256": sha(project / name)}
    for seed in SEEDS:
        name = "configs_horizon/" + ("horizon240_seed42.json" if seed == 42 else f"repeat_horizon240_seed{seed}.json")
        base = read(project / name)
        assert base["seed"] == seed and base["epochs"] == 240 and base["width"] == 48
        for mode in ("original", "weighted_single"):
            cfg = dict(base, crosssplit_seed=SPLIT_SEED, partition_sha256=sha(work / "partition.json"),
                       declared_checkpoint_selector="LAST", source_set_mode=mode)
            key = f"student_{mode}_{seed}"
            write(work / "configs" / (key + ".json"), cfg)
            configs[key] = {"path": "configs/" + key + ".json", "sha256": sha(work / "configs" / (key + ".json"))}
            original_configs[key] = {"path": name, "sha256": sha(project / name)}
    pretrained = project / "data/pretrained/resnet18-f37072fd.pth"
    plan = {"status": "sealed_not_started", "project": str(project), "split_seed": SPLIT_SEED,
            "partition_sha256": sha(work / "partition.json"), "normalization_sha256": sha(work / "normalization.json"),
            "raw_official_train_batches": entries, "original_source12": original_sources,
            "candidate_sha256": sha(__file__), "package_source_sha256": package_source_hashes(),
            "package_root": str(HERE), "loss_file": str(loss), "loss_sha256": sha(loss),
            "imagenet_initialization": {"path": str(pretrained), "sha256": sha(pretrained)},
            "configs": configs, "original_configs": original_configs,
            "student_seed_order": [42, 17, 73], "student_modes": ["original", "weighted_single"],
            "formal_case_tag": "main",
            "all_six_student_cases_required": True, "condition_on_validation_results": False,
            "Teacher_selectors": {"cutmix": "LAST160", "transfer": "LAST60"},
            "student_selector": "LAST240", "test_access": "none",
            "scope": "new internal partition confirmation; historical dataset/test exposure remains"}
    write(work / "plan.json", plan)
    write(work / "CPU_seal_receipt.json", {"passed": True, "plan_sha256": sha(work / "plan.json"),
          "train45000_val5000_disjoint": True, "perclass4500_500": True,
          "new_validation_not_seen_in_any_new_training": "enforced by new data callback",
          "performance_computed": False, "GPU_used": False})
    print("SEALED: one new partition, new norm, two Teacher configs, all six student configs. No model execution.")


def verify(project, work):
    plan = read(work / "plan.json")
    assert Path(plan["project"]).resolve() == project
    assert plan["split_seed"] == SPLIT_SEED and plan["all_six_student_cases_required"] is True
    assert plan["candidate_sha256"] == sha(__file__)
    assert plan["package_source_sha256"] == package_source_hashes()
    assert Path(plan["package_root"]).resolve() == HERE
    assert plan["partition_sha256"] == sha(work / "partition.json")
    assert plan["normalization_sha256"] == sha(work / "normalization.json")
    for name, digest in plan["original_source12"].items():
        assert sha(project / name) == digest
    for entry in plan["configs"].values():
        assert sha(work / entry["path"]) == entry["sha256"]
    for entry in plan["original_configs"].values():
        assert sha(project / entry["path"]) == entry["sha256"]
    assert sha(plan["loss_file"]) == plan["loss_sha256"]
    assert plan["loss_sha256"] == LEGACY_LOSS_SHA
    assert sha(plan["imagenet_initialization"]["path"]) == plan["imagenet_initialization"]["sha256"]
    return plan


def source_set(plan, kind, mode="original"):
    names = list(plan["original_source12"])
    if kind == "teacher":
        names = names[:6]
    result = {name: plan["original_source12"][name] for name in names}
    result.update(plan["package_source_sha256"])
    if mode == "weighted_single":
        result["contributions/weighted_loss/legacy_loss.py"] = plan["loss_sha256"]
    return result


def runtime(project, work, plan):
    sys.path.insert(0, str(project)); sys.dont_write_bytecode = True
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset
    import train as base
    import extensions.train as engine
    import extensions.models as mm
    import aligned_distillation.train as core
    from aligned_distillation.teacher import Teacher as OldTeacherMath
    assert Path(base.__file__).resolve() == project / "train.py"
    split, stats = read(work / "partition.json"), read(work / "normalization.json")

    def new_training_data():
        images, labels = raw_train(project, plan["raw_official_train_batches"])
        train_ids, val_ids = split["train_indices"], split["validation_indices"]
        assert len(train_ids) == 45000 and len(val_ids) == 5000 and not set(train_ids) & set(val_ids)
        assert np.bincount(labels[train_ids]).tolist() == [4500] * 10
        assert np.bincount(labels[val_ids]).tolist() == [500] * 10
        x = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).contiguous()
        y = torch.from_numpy(labels.copy())
        return TensorDataset(x[train_ids], y[train_ids]), TensorDataset(x[val_ids], y[val_ids]), stats

    engine.training_data = new_training_data
    return np, torch, base, engine, mm, core, OldTeacherMath


def case(work, kind, role, seed, tag, purpose):
    assert purpose in ("preflight", "formal")
    name = "teacher_" + role if kind == "teacher" else f"student_{role}_{seed}"
    return owned(work / "runs" / purpose / (name + "__" + tag))


def teacher_identity(project, work, plan):
    lock = read(work / "teacher_lock.json")
    assert lock["plan_sha256"] == sha(work / "plan.json")
    assert lock["partition_sha256"] == plan["partition_sha256"]
    assert set(lock["teachers"]) == set(TEACHERS)
    for role, entry in lock["teachers"].items():
        path = owned(work / entry["checkpoint"])
        assert path.is_relative_to(work / "runs/formal")
        assert sha(path) == entry["sha256"]
        assert entry["epoch"] == TEACHERS[role][1]
    return lock


def tensor_sha(value, torch):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def require_gate(work, plan, kind):
    """A bare passed flag cannot authorize a formal run without its evidence."""
    file = "teachers.json" if kind == "teacher" else "students.json"
    receipt = read(work / "gates" / file)
    assert receipt["passed"] is True and receipt["plan_sha256"] == sha(work / "plan.json")
    assert receipt["kind"] == kind and receipt["real45k5k"] is True
    assert receipt["continuous_vs_resumed_full_state_exact"] is True
    roles = set(TEACHERS) if kind == "teacher" else {"original", "weighted_single"}
    assert {row["role"] for row in receipt["roles"]} == roles
    expected_epoch = 2 if kind == "teacher" else 8
    for row in receipt["roles"]:
        assert row["epochs"] == expected_epoch and len(row["checkpoint_files"]) == 2
        for entry in row["checkpoint_files"]:
            checkpoint = owned(entry["path"])
            assert checkpoint.is_relative_to(work / "runs/preflight")
            assert sha(checkpoint) == entry["sha256"]
    if kind == "student":
        assert receipt["shared_input_teacher_q_exact"] is True
        teacher_identity(Path(plan["project"]), work, plan)
    return receipt


def semantic_digest(saved, np, torch):
    """Compare exact semantic state; timing fields alone are excluded."""
    digest = hashlib.sha256()
    saved = dict(saved)
    saved["history"] = [{k: v for k, v in row.items() if k != "seconds"} for row in saved["history"]]

    def add(value):
        if isinstance(value, torch.Tensor):
            x = value.detach().cpu().contiguous()
            digest.update(repr(("tensor", str(x.dtype), tuple(x.shape))).encode())
            digest.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, np.ndarray):
            digest.update(repr(("ndarray", str(value.dtype), value.shape)).encode()); digest.update(value.tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict{")
            for key in sorted(value, key=lambda x: (type(x).__name__, repr(x))):
                add(key); add(value[key])
            digest.update(b"}")
        elif isinstance(value, (tuple, list)):
            digest.update(type(value).__name__.encode() + b"[")
            for x in value:
                add(x)
            digest.update(b"]")
        elif isinstance(value, float):
            digest.update(b"float" + struct.pack("!d", value))
        else:
            digest.update(repr((type(value).__name__, value)).encode())
    add(saved)
    return digest.hexdigest()


def effective_objective(base_metadata, cfg, mode, plan_sha256):
    """Correct metadata only; never computes a loss or consumes any RNG."""
    assert mode in ("original", "weighted_single")
    result = dict(base_metadata)
    alpha, temperature = float(cfg["distill_alpha"]), float(cfg["distill_temperature"])
    assert alpha == .25 and temperature == 1.
    if mode == "original":
        formula = "(1-alpha)*CE_CutMix + alpha*K_T"
        kd_definition = "K_T = T^2 * full KL(q_T || student_T)"
    else:
        formula = "(1-alpha)*CE_CutMix + alpha*(K_T + E_T)"
        kd_definition = "K_T + E_T; E_T is the unchanged area-weighted singleton conditional-KL extra, including T^2"
    result.update(
        metadata_protocol="crosssplit-effective-objective-v2",
        loss=formula, source_set_mode=mode, alpha=alpha, temperature=temperature,
        train_loss_definition=formula,
        train_ce_definition="actual CutMix-area-weighted hard-label CE on the augmented mixed image",
        train_kd_definition=kd_definition,
        K_T_definition="T^2 * full KL(q_T || student_T)",
        E_T_definition="unchanged weighted_loss/legacy_loss.extras(..., mode='weighted_single') output, already multiplied by T^2",
        actual_total_construction=("original loss_parts total unchanged" if mode == "original"
                                   else "original loss_parts total + alpha*E_T; returned train_kd = original_KL + E_T"),
        teacher_query="shared actual crop/flip/CutMix tensor; fixed original/flip internal Teacher views",
        teacher_checkpoint_selection="new Teacher LAST160 and LAST60, never selected by new validation scores",
        student_checkpoint_selection="fixed LAST240; BEST records are diagnostics and are not the confirmation endpoint",
        validation_metric_scope="clean-image hard-label CE/accuracy, distinct from the mixed-image training objective",
        public_plan_sha256=plan_sha256,
        base_core_metadata_copy="objective_metadata_base_core.json")
    return result


def train_case(args, project, work, plan):
    kind, role = args.kind, args.role
    if kind == "teacher":
        assert role in TEACHERS and args.seed == 42
        key = "teacher_" + role
    else:
        assert role in ("original", "weighted_single") and args.seed in SEEDS
        key = f"student_{role}_{args.seed}"
    cfg = read(work / plan["configs"][key]["path"])
    folder = case(work, kind, role, args.seed, args.tag, args.purpose)
    assert args.tag and Path(args.tag).name == args.tag and args.tag not in (".", "..")
    if args.purpose == "formal":
        assert args.tag == plan["formal_case_tag"], "Formal cases have one sealed name, no trial tags"
        assert args.stop_after is None
        require_gate(work, plan, kind)
    else:
        assert args.tag != plan["formal_case_tag"], "Reserved formal tag"
        assert args.stop_after in ((1, 2) if kind == "teacher" else (4, 8))
    if args.purpose == "formal" and (folder / "crosssplit_receipt.json").exists():
        old_receipt = read(folder / "crosssplit_receipt.json")
        if old_receipt["epochs"] == cfg["epochs"]:
            assert old_receipt["plan_sha256"] == sha(work / "plan.json")
            assert old_receipt["source_sha256"] == source_set(plan, kind, role)
            assert sha(folder / "last.pt") == old_receipt["checkpoint_sha256"]
            assert read(folder / "config.json") == cfg
            if kind == "student":
                teacher_identity(project, work, plan)
            print("Already completed formal case; verified without importing model runtime.")
            return
    np, torch, base, engine, mm, core, old_teacher = runtime(project, work, plan)
    sources = source_set(plan, kind, role)
    engine.DIRECTORY, engine.SOURCES = folder.parent, tuple(sources)
    engine.source_hashes = lambda: dict(sources)
    original_loss = core.loss_parts
    lock = None
    source_dir = owned(work / "case_manifests" / folder.name)
    source_dir.mkdir(parents=True, exist_ok=True)
    trace_stream = None
    original_core_writer = core.write_json
    saved_epoch = 0
    if (folder / "last.pt").exists():
        saved = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
        assert saved["config"] == cfg and saved["source_sha256"] == sources and saved["stats"] == read(work / "normalization.json")
        saved_epoch = int(saved["epoch"])
        if args.purpose == "preflight" and saved_epoch >= args.stop_after:
            assert saved_epoch == args.stop_after, "Existing run exceeded bounded preflight target"
            assert (folder / "crosssplit_receipt.json").exists(), "At target without final receipt; inspect or choose a new preflight tag, never run past target"
            receipt = read(folder / "crosssplit_receipt.json")
            assert receipt["epochs"] == saved_epoch and receipt["checkpoint_sha256"] == sha(folder / "last.pt")
            assert receipt["plan_sha256"] == sha(work / "plan.json")
            print("Already at requested preflight epoch; verified without another core call.")
            return
    try:
        if kind == "student":
            lock = teacher_identity(project, work, plan)
            # Never invoke the old Teacher constructor or its old identity().
            class FreshTeacher(old_teacher):
                def __init__(self):
                    self.parts = []
                    self.lock = lock
                    for teacher_role in ("cutmix", "transfer"):
                        entry = lock["teachers"][teacher_role]
                        checkpoint = owned(work / entry["checkpoint"])
                        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                        expected_cfg = read(work / plan["configs"]["teacher_" + teacher_role]["path"])
                        assert saved["config"] == expected_cfg and saved["stats"] == read(work / "normalization.json")
                        assert saved["source_sha256"] == source_set(plan, "teacher")
                        assert saved["epoch"] == TEACHERS[teacher_role][1]
                        model = mm.build(saved["config"]).cuda().to(memory_format=torch.channels_last)
                        model.load_state_dict(saved["model"], strict=True)
                        model.eval().requires_grad_(False)
                        transform = mm.Transform(saved["stats"], saved["config"]).cuda()
                        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        self.parts.append((model, transform, saved["config"], before))

                def probabilities(self, images):
                    values = super().probabilities(images)
                    current["actual_mixed_image_sha256"] = tensor_sha(images, torch)
                    current["raw_teacher_q_sha256"] = tensor_sha(values, torch)
                    return values

                def unchanged(self):
                    for model, _, _, before in self.parts:
                        assert not model.training
                        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
                        assert all(torch.equal(v.detach().cpu(), before[k]) for k, v in model.state_dict().items())
                    assert teacher_identity(project, work, plan) == self.lock
                    return True

            current, batch_counter = {}, saved_epoch * math.ceil(45000 / 256)
            trace_path = folder / "input_loss_trace.jsonl"
            folder.mkdir(parents=True, exist_ok=True)
            # Preserve rolled-back partial traces, retaining a committed prefix.
            if trace_path.exists():
                previous = trace_path.with_name("trace_before_resume_" + str(time.time_ns()) + ".jsonl")
                assert previous.resolve().is_relative_to(work)
                trace_path.rename(previous)
                with previous.open(encoding="utf-8") as src, trace_path.open("x", encoding="utf-8") as dst:
                    for line in src:
                        row = json.loads(line)
                        if row["global_batch"] <= batch_counter:
                            dst.write(line)
            trace_stream = trace_path.open("a", encoding="utf-8")
            loss_module = None
            if role == "weighted_single":
                spec = importlib.util.spec_from_file_location("crosssplit_frozen_weighted_math", plan["loss_file"])
                loss_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(loss_module)

            def loss_parts(logits, labels, order, lam, q, temperature, alpha):
                nonlocal batch_counter
                original = original_loss(logits, labels, order, lam, q, temperature, alpha)
                extra = original[2].detach() * 0
                result = original
                if role == "weighted_single":
                    extra, detail = loss_module.extras(logits, labels, order, lam, q, temperature, mode="weighted_single")
                    assert bool(detail["all_finite"])
                    result = (original[0] + alpha * extra, original[1], original[2] + extra)
                batch_counter += 1
                n_batch = math.ceil(45000 / 256)
                row = {"global_batch": batch_counter, "epoch": (batch_counter - 1) // n_batch + 1,
                       "batch_in_epoch": (batch_counter - 1) % n_batch + 1, "samples": len(labels),
                       "lambda": float(lam), "labels_sha256": tensor_sha(labels, torch),
                       "order_sha256": tensor_sha(order, torch), "target_q_sha256": tensor_sha(q, torch),
                       "original_KL": float(original[2].detach()), "extra": float(extra.detach()),
                       "hard_CE": float(original[1].detach()), "total": float(result[0].detach()), **current}
                assert set(("actual_mixed_image_sha256", "raw_teacher_q_sha256")).issubset(row)
                assert all(math.isfinite(row[key]) for key in ("original_KL", "extra", "hard_CE", "total"))
                trace_stream.write(json.dumps(row, allow_nan=False) + "\n")
                trace_stream.flush(); current.clear()
                return result

            core.OUT, core.DIRECTORY, core.SOURCES = source_dir, folder.parent, tuple(sources)
            core.source_hashes = lambda: dict(sources)
            core.Teacher = FreshTeacher
            core.identity = lambda: ({}, lock)
            core.loss_parts = loss_parts
            def objective_writer(path, value):
                path = Path(path)
                if path.name == "objective_metadata.json":
                    # Keep the exact generic base declaration as provenance.
                    # This occurs before engine.run seeds/builds the student;
                    # only plain dictionaries/JSON are touched, not RNG/loss.
                    raw = path.with_name("objective_metadata_base_core.json")
                    if raw.exists():
                        assert read(raw) == value
                    else:
                        original_core_writer(raw, value)
                    value = effective_objective(value, cfg, role, sha(work / "plan.json"))
                original_core_writer(path, value)
            core.write_json = objective_writer
            write(source_dir / "teacher_identity.json", lock)
            write(source_dir / "source_manifest.json", {"source_sha256": sources})
        else:
            write(source_dir / "source_manifest.json", {"source_sha256": sources})
        write(work / "worker.json", {"pid": os.getpid(), "stage": "training", "kind": kind,
              "role": role, "seed": args.seed, "purpose": args.purpose, "folder": str(folder),
              "selector": "LAST", "started_unix": time.time(), "plan_sha256": sha(work / "plan.json")})
        began = time.monotonic()
        result = (engine.run(cfg, folder.name, args.stop_after) if kind == "teacher"
                  else core.run(cfg, folder.name, args.stop_after))
        last = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
        target = cfg["epochs"] if args.purpose == "formal" else args.stop_after
        assert last["epoch"] == target and len(last["history"]) == target
        assert all(row["train_samples"] == 45000 and row["validation_samples"] == 5000 for row in last["history"])
        assert last["config"] == cfg and last["source_sha256"] == sources
        assert verify(project, work) == plan
        if lock is not None:
            assert teacher_identity(project, work, plan) == lock
        write(folder / "crosssplit_receipt.json", {"state": "completed_target", "purpose": args.purpose,
              "kind": kind, "role": role, "seed": args.seed, "epochs": target,
              "partition_sha256": plan["partition_sha256"], "source_sha256": sources,
              "plan_sha256": sha(work / "plan.json"),
              "checkpoint": "last.pt", "checkpoint_sha256": sha(folder / "last.pt"),
              "selected_BEST": False, "train_val_disjoint": True, "old_CIFAR_Teacher_loaded": False,
              "wall_seconds_this_process": time.monotonic() - began, "result": result})
        write(work / "worker.json", {"pid": os.getpid(), "stage": "case_target_completed",
              "kind": kind, "role": role, "seed": args.seed, "purpose": args.purpose,
              "epochs": target, "folder": str(folder), "scientific_closure": False,
              "next_required": "remaining fixed cases, analysis, real figures and independent checks"})
    finally:
        if trace_stream is not None:
            trace_stream.close()
        core.write_json = original_core_writer


def lock_teachers(project, work, plan, tag):
    import torch
    assert tag == plan["formal_case_tag"]
    entries = {}
    for role, (_, epochs) in TEACHERS.items():
        folder = case(work, "teacher", role, 42, tag, "formal")
        saved = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
        assert saved["epoch"] == epochs and saved["source_sha256"] == source_set(plan, "teacher")
        assert len(saved["history"]) == epochs
        assert all(row["train_samples"] == 45000 and row["validation_samples"] == 5000 for row in saved["history"])
        assert read(folder / "crosssplit_receipt.json")["purpose"] == "formal"
        assert saved["config"] == read(work / plan["configs"]["teacher_" + role]["path"])
        assert saved["stats"] == read(work / "normalization.json")
        entries[role] = {"checkpoint": (folder / "last.pt").relative_to(work).as_posix(),
                         "sha256": sha(folder / "last.pt"), "epoch": epochs,
                         "selector": "LAST", "training_seed": 42,
                         "source_sha256": saved["source_sha256"]}
    lock = {"plan_sha256": sha(work / "plan.json"), "partition_sha256": plan["partition_sha256"],
            "teachers": entries, "policy": "equal_two_teacher_original_flip", "calibrated": False,
            "training_index_intersection_validation": 0}
    if (work / "teacher_lock.json").exists():
        assert read(work / "teacher_lock.json") == lock
    write(work / "teacher_lock.json", lock)
    print("LOCKED: new LAST160+LAST60 only; neither is an old CIFAR Teacher.")


def gate(args, project, work, plan):
    """CPU compare completed preflights; no forward and no CUDA initialization."""
    import numpy as np
    import torch
    assert not torch.cuda.is_initialized()
    roles = list(TEACHERS) if args.kind == "teacher" else ["original", "weighted_single"]
    receipts, traces = [], []
    for role in roles:
        folders = [case(work, args.kind, role, 42, tag, "preflight") for tag in (args.continuous_tag, args.resumed_tag)]
        saved = [torch.load(folder / "last.pt", map_location="cpu", weights_only=False) for folder in folders]
        expected_epoch = 2 if args.kind == "teacher" else 8
        assert all(item["epoch"] == expected_epoch and len(item["history"]) == expected_epoch for item in saved)
        expected_key = "teacher_" + role if args.kind == "teacher" else f"student_{role}_42"
        assert all(item["config"] == read(work / plan["configs"][expected_key]["path"]) for item in saved)
        assert all(item["stats"] == read(work / "normalization.json") for item in saved)
        assert all(item["source_sha256"] == source_set(plan, args.kind, role) for item in saved)
        digests = [semantic_digest(item, np, torch) for item in saved]
        assert digests[0] == digests[1], ("continuous/resume divergence", role)
        assert all(row["train_samples"] == 45000 and row["validation_samples"] == 5000 for item in saved for row in item["history"])
        if args.kind == "student":
            assert all(read(folder / "teacher_check.json")["weights_bn_gradients_unchanged"] is True for folder in folders)
            trace_pair = [[json.loads(line) for line in (folder / "input_loss_trace.jsonl").read_text(encoding="utf-8").splitlines()] for folder in folders]
            assert trace_pair[0] == trace_pair[1], ("resumed input/loss trace diverged", role)
            assert len(trace_pair[0]) == 8 * 176
            assert all(row["samples"] == (200 if row["batch_in_epoch"] == 176 else 256) for row in trace_pair[0])
            traces.append(trace_pair[0])
        receipts.append({"role": role, "semantic_digest": digests[0], "epochs": expected_epoch,
                         "checkpoint_files": [{"path": str(folder / "last.pt"), "sha256": sha(folder / "last.pt")} for folder in folders]})
    if args.kind == "student":
        common = ("global_batch", "epoch", "batch_in_epoch", "samples", "lambda", "labels_sha256",
                  "order_sha256", "target_q_sha256", "actual_mixed_image_sha256", "raw_teacher_q_sha256")
        assert [{k: row[k] for k in common} for row in traces[0]] == [{k: row[k] for k in common} for row in traces[1]], "Arms do not share actual inputs/Teacher targets"
        teacher_identity(project, work, plan)
    assert not torch.cuda.is_initialized()
    file = "teachers.json" if args.kind == "teacher" else "students.json"
    write(work / "gates" / file, {"passed": True, "plan_sha256": sha(work / "plan.json"),
          "kind": args.kind, "continuous_vs_resumed_full_state_exact": True,
          "real45k5k": True, "source_guard": True, "roles": receipts,
          "shared_input_teacher_q_exact": args.kind == "student", "CUDA_initialized_by_comparator": False,
          "scope": "Teacher first2 or student first8 with original full schedule; not a completed formal run"})
    print("PASS:", file, "from actual checkpoints and traces")


def evaluate(args, project, work, plan):
    """Fixed LAST only; new45k/new5k, no official-test path or Teacher load."""
    mode, seed = args.mode, args.seed
    folder = case(work, "student", mode, seed, plan["formal_case_tag"], "formal")
    receipt = read(folder / "crosssplit_receipt.json")
    assert receipt["epochs"] == 240 and receipt["purpose"] == "formal"
    checkpoint = folder / "last.pt"
    assert receipt["checkpoint_sha256"] == sha(checkpoint)
    assert receipt["plan_sha256"] == sha(work / "plan.json")
    output = owned(work / "analysis" / f"{mode}_{seed}")
    if (output / "metrics.json").exists():
        old = read(output / "metrics.json")
        assert old["checkpoint_sha256"] == sha(checkpoint) and old["plan_sha256"] == sha(work / "plan.json")
        assert old["validation_cache_sha256"] == sha(output / "validation_cache.npz")
        print("Existing fixed-LAST analysis verified; no repeat forward.")
        return
    np, torch, base, engine, mm, _, _ = runtime(project, work, plan)
    torch.set_num_threads(8); base.seed_everything(42)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = read(work / plan["configs"][f"student_{mode}_{seed}"]["path"])
    assert saved["config"] == cfg and saved["source_sha256"] == source_set(plan, "student", mode)
    assert saved["epoch"] == 240
    train_set, val_set, stats = engine.training_data()
    from data import make_loader
    train_loader = make_loader(train_set, 256, generator=torch.Generator().manual_seed(29517))
    val_loader = make_loader(val_set, 256, generator=torch.Generator().manual_seed(29518))
    model = mm.build(cfg).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(saved["model"], strict=True); model.eval().requires_grad_(False)
    transform = mm.Transform(stats, cfg).cuda().eval()
    before = {name: tensor_sha(value, torch) for name, value in model.state_dict().items()}
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0)
    clean_train = base.epoch_pass(model, train_loader, transform, criterion, torch.device("cuda"), amp=True)
    labels = val_set.tensors[1]
    logit_parts, ce_parts = [], []

    class Collect(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.student = model; self.position = 0

        def forward(self, images):
            logits = self.student(images)
            target = labels[self.position:self.position + len(images)].to(logits.device)
            ce = torch.nn.functional.cross_entropy(logits.float(), target, reduction="none")
            logit_parts.append(logits.detach().float().cpu().numpy())
            ce_parts.append(ce.detach().float().cpu().numpy())
            self.position += len(images)
            return logits

    collected = Collect()
    clean_val = base.epoch_pass(collected, val_loader, transform, criterion, torch.device("cuda"), amp=True)
    for name in ("loss", "ce", "accuracy", "samples"):
        assert clean_val[name] == saved["history"][-1]["validation_" + name], ("native validation bridge failed", name)
    assert collected.position == 5000
    assert {name: tensor_sha(value, torch) for name, value in model.state_dict().items()} == before
    assert sha(checkpoint) == receipt["checkpoint_sha256"]
    logits, per_image_ce = np.concatenate(logit_parts), np.concatenate(ce_parts)
    truth = labels.numpy(); predicted = logits.argmax(1)
    cm = np.bincount(truth * 10 + predicted, minlength=100).reshape(10, 10)
    assert cm.sum(1).tolist() == [500] * 10
    assert float((predicted == truth).mean()) == clean_val["accuracy"]
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "validation_cache.npz"
    assert not cache.exists(), "Uncommitted existing cache requires inspection; do not overwrite"
    with cache.open("xb") as stream:
        np.savez_compressed(stream, source_ids=np.asarray(read(work / "partition.json")["validation_indices"], dtype=np.int64),
                            labels=truth, logits_from_BF16_cast_float32=logits,
                            hard_CE_per_image_float32=per_image_ce, prediction=predicted)
    metrics = {"checkpoint_sha256": sha(checkpoint), "plan_sha256": sha(work / "plan.json"),
               "selector": "LAST240", "mode": mode, "training_seed": seed,
               "precision": "CUDA BF16 model, FP32 hard CE; no TTA/temperature", "batch_size": 256,
               "clean_train45k": clean_train, "clean_validation5k": clean_val,
               "per_image_CE_float64_mean": float(per_image_ce.astype(np.float64).mean()),
               "CE_rounding_scope": "Native batch-reduced CE is reported separately; per-image FP32 CE supports paired descriptive analysis and can differ in final rounding digits.",
               "validation_confusion_matrix": cm.tolist(), "per_class_accuracy": (cm.diagonal() / 500).tolist(),
               "validation_cache_sha256": sha(cache), "source_sha256": saved["source_sha256"],
               "model_BN_and_checkpoint_unchanged": True, "native_last_validation_exact": True,
               "Teacher_loaded_for_analysis": False, "official_test_read": False,
               "manual_plot_review": "pending; this CLI does not mark scientific closure"}
    assert verify(project, work) == plan
    teacher_identity(project, work, plan)
    write(output / "metrics.json", metrics)
    print("SAVED: fixed LAST45k/5k, CM/perclass and paired validation cache; figures/manual review pending.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(HERE.parents[1]), help="Checkout/core root, default inferred from package")
    parser.add_argument("--workspace", default="outputs/crosssplit_20261127", help="Dedicated new marked output directory")
    parser.add_argument("--loss-file", help="Optional frozen pure legacy_loss.py path; default is sibling weighted_loss/legacy_loss.py")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    t = sub.add_parser("train")
    t.add_argument("--kind", choices=("teacher", "student"), required=True)
    t.add_argument("--role", required=True)
    t.add_argument("--seed", type=int, choices=SEEDS, default=42)
    t.add_argument("--purpose", choices=("preflight", "formal"), required=True)
    t.add_argument("--tag", required=True)
    t.add_argument("--stop-after", type=int)
    l = sub.add_parser("lock-teachers"); l.add_argument("--tag", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--kind", choices=("teacher", "student"), required=True)
    g.add_argument("--continuous-tag", required=True); g.add_argument("--resumed-tag", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--mode", choices=("original", "weighted_single"), required=True)
    e.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    global OWNED_ROOT
    project = Path(args.project).resolve()
    work = resolve_workspace(project, args.workspace, creating=args.command == "prepare")
    OWNED_ROOT = work
    work.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        loss = Path(args.loss_file).resolve() if args.loss_file else HERE.parent / "weighted_loss/legacy_loss.py"
        prepare(project, work, loss)
        return
    plan = verify(project, work)
    if args.command == "train":
        with training_lock(work):
            train_case(args, project, work, plan)
    elif args.command == "lock-teachers":
        lock_teachers(project, work, plan, args.tag)
    elif args.command == "gate":
        gate(args, project, work, plan)
    elif args.command == "evaluate":
        evaluate(args, project, work, plan)


if __name__ == "__main__":
    main()
