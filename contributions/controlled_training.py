"""Fixed augmentation repetitions and a matched-recipe architecture comparison.

This adapter uses the recorded six-source training loop without changing it.
It writes a new source identity and accepts only the fixed published cases.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import hashlib
import json
import math
import csv
import shutil

import numpy as np

import torch

import extensions.train as engine
from contributions.alignment import adapter as shared


ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_suffix(".json")
CORE_SOURCES = ("train.py", "data.py", "models.py", "prepare_data.py",
                "extensions/train.py", "extensions/models.py")
BASE_BUILD = engine.build
BASE_LOADER = engine.make_loader
BASE_TRANSFORM = engine.Transform
BASE_WRITE = engine.write_json
BASE_MIXED = engine.mixed_epoch
AUGMENTATION_METHODS = ("se", "cutout", "smoothing", "cutmix", "combined")
PROOF_KIND = "controlled_training_actual_GPU_prefix8_resume4_v1"


def fixed_cases():
    document = shared.read(CONFIG)
    cases = document["cases"]
    expected_names = {f"augmentation_{method}_seed{seed}" for method in AUGMENTATION_METHODS for seed in (17, 73)}
    expected_names |= {f"preflight_augmentation_{method}_seed42" for method in AUGMENTATION_METHODS}
    expected_names |= {f"architecture_{arch}_seed{seed}" for arch in ("cnn", "resnet20") for seed in (42, 17, 73)}
    if document["protocol"] != "fixed_controlled_training_v1" or set(cases) != expected_names:
        raise ValueError("The fixed case registry differs")
    for name, case in cases.items():
        config = case["config"]
        if case["official_test"] is not False or config["epochs"] != 160 or config["batch_size"] != 256 or "pilot_samples" in config:
            raise ValueError("Only full45k/5k 160-epoch fixed cases are permitted")
        if case["study"] == "augmentation_confirmation":
            original = ROOT / case["original_method_config"]
            if shared.sha(original) != case["original_method_config_sha256"]:
                raise ValueError("The indexed original augmentation config changed")
            expected = shared.read(original)
            expected["seed"] = config["seed"]
            if config != expected or bool(case.get("preflight_only")) != name.startswith("preflight_"):
                raise ValueError("Augmentation cases may change only the original training seed")
        elif case["study"] == "matched_architecture":
            architecture = "cnn" if "_cnn_" in name else "resnet20"
            expected = {"architecture": architecture, "width": 32 if architecture == "cnn" else 16,
                        "attention": False, "dropout": 0., "augmentation": True, "cutout": 0,
                        "label_smoothing": 0., "optimizer": "sgd", "learning_rate": .1,
                        "weight_decay": .0005, "batch_size": 256, "epochs": 160,
                        "warmup_epochs": 5, "seed": int(name.rsplit("seed", 1)[1]),
                        "amp": True, "patience": 0, "model": "cifar", "image_size": 32, "mix": "none"}
            if config != expected or case.get("preflight_only"):
                raise ValueError("Matched architecture recipe differs")
        else:
            raise ValueError("Unregistered study")
        if config["seed"] not in (42, 17, 73):
            raise ValueError("Unregistered training seed")
    if sum(not row.get("preflight_only", False) for row in cases.values()) != document["new_formal_cases"]:
        raise ValueError("Formal/preflight case count differs")
    return cases


def source_identity():
    reference = shared.read(shared.PACKAGE / "evidence/source10.json")["source_sha256"]
    actual = {name: shared.sha(ROOT / name) for name in CORE_SOURCES}
    if actual != {name: reference[name] for name in CORE_SOURCES}:
        raise ValueError("The recorded six-source core changed")
    # Include all shared adapter sources, rather than hiding the loader dependency.
    return {**shared.sources(),
            "contributions/controlled_training.py": shared.sha(Path(__file__)),
            "contributions/controlled_training.json": shared.sha(CONFIG)}


def prepare_output(output, data, case, trace_epochs, preflight=False):
    if output == ROOT or ROOT.is_relative_to(output) or output.is_relative_to(data) or data.is_relative_to(output):
        raise ValueError("Output must be isolated from the checkout and original data")
    identity = {"kind": "fixed_controlled_training_v1", "case": case,
                "source_sha256": source_identity(), "data": shared.data_identity(data),
                "trace_epochs": trace_epochs,
                "invocation_class": "preflight" if preflight else "formal",
                "official_test_read": False,
                "model_initialization_rng_isolated": case["study"] == "matched_architecture"}
    marker = output / "identity.json"
    if marker.exists():
        if shared.read(marker) != identity:
            raise ValueError("Output belongs to different inputs or sources")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Refusing an unmarked nonempty output directory")
    else:
        shared.write(marker, identity)
        shared.write(output / "input_paths.json", {"data_dir": str(data), "checkout": str(ROOT)})
    return identity


def digest_tensor(value):
    cpu = value.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(cpu.shape), str(cpu.dtype))).encode() + cpu.numpy().tobytes()).hexdigest()


class Trace:
    def __init__(self, output, limit, starting_epoch):
        self.path = output / "input_trace.json"
        self.limit = limit
        self.current_epoch = starting_epoch + 1
        self.rows = shared.read(self.path)["epochs"] if self.path.exists() else []
        self.hooks = []
        expected = min(starting_epoch, limit)
        if [row["epoch"] for row in self.rows] != list(range(1, expected + 1)):
            raise ValueError("Trace/checkpoint boundary differs; retain this run and diagnose it")
        if any("actual_model_input_sha256" not in row or row["samples"] != 45000 for row in self.rows):
            raise ValueError("Old or incomplete trace format cannot be mixed with actual-input tracing")
        self.reset()

    def reset(self):
        self.raw = hashlib.sha256()
        self.augmented = hashlib.sha256()
        self.labels = hashlib.sha256()
        self.actual = hashlib.sha256()
        self.batch_sizes = []
        self.actual_batch_sizes = []
        self.transformed_batches = 0

    @property
    def enabled(self):
        return self.current_epoch <= self.limit

    def raw_batch(self, x, y):
        if self.enabled:
            self.raw.update(bytes.fromhex(digest_tensor(x)))
            self.labels.update(bytes.fromhex(digest_tensor(y)))
            self.batch_sizes.append(len(y))

    def augmented_batch(self, x):
        if self.enabled:
            self.augmented.update(bytes.fromhex(digest_tensor(x)))
            self.transformed_batches += 1

    def actual_batch(self, x):
        if self.enabled:
            self.actual.update(bytes.fromhex(digest_tensor(x)))
            self.actual_batch_sizes.append(len(x))

    def close(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def commit(self, status):
        epoch = status["completed_epochs"]
        if epoch != self.current_epoch:
            raise ValueError("Unexpected epoch at input-trace commit")
        if self.enabled:
            expected_batches = [256] * 175 + [200]
            if self.batch_sizes != expected_batches or self.actual_batch_sizes != expected_batches or self.transformed_batches != 176:
                raise ValueError("Full45k actual/before-mix training batch accounting differs")
            if status["latest"]["train_samples"] != 45000 or status["latest"]["validation_samples"] != 5000:
                raise ValueError("Status is not a full45k/5k epoch")
            self.rows.append({"epoch": epoch, "samples": sum(self.batch_sizes),
                              "batch_sizes": self.batch_sizes,
                              "raw_sha256": self.raw.hexdigest(),
                              "labels_sha256": self.labels.hexdigest(),
                              "transformed_before_mix_sha256": self.augmented.hexdigest(),
                              "actual_model_input_sha256": self.actual.hexdigest()})
            shared.write(self.path, {"kind": "observed_training_inputs_v2",
                                    "semantics": "raw loader -> transformed before mixing -> actual model input after mixing; no validation batches",
                                    "rng_calls_added": 0, "epochs": self.rows})
        self.current_epoch += 1
        self.reset()


class TracedLoader:
    def __init__(self, loader, trace):
        self.loader, self.trace, self.dataset = loader, trace, loader.dataset

    def __iter__(self):
        for x, y in self.loader:
            self.trace.raw_batch(x, y)
            yield x, y

    def __len__(self):
        return len(self.loader)


@contextmanager
def training_scope(output, data, case, trace):
    if (engine.build is not BASE_BUILD or engine.Transform is not BASE_TRANSFORM
            or engine.make_loader is not BASE_LOADER or engine.mixed_epoch is not BASE_MIXED
            or tuple(engine.SOURCES) != CORE_SOURCES or engine.ROOT != ROOT
            or engine.DIRECTORY != ROOT / "runs/round2"
            or BASE_BUILD.__module__ != "extensions.models" or BASE_MIXED.__module__ != "extensions.train"):
        raise RuntimeError("Use a fresh Python process for this adapter")
    bindings = [(name, getattr(engine, name)) for name in
                ("ROOT", "DIRECTORY", "SOURCES", "training_data", "source_hashes", "build", "make_loader", "Transform", "write_json")]

    def build(config):
        if case["study"] == "matched_architecture":
            # Different-sized CPU initializations do not move training RNGs.
            with shared.preserve_runtime():
                model = BASE_BUILD(config)
        else:
            model = BASE_BUILD(config)
        if trace.limit:
            def observe(module, inputs):
                if module.training:
                    trace.actual_batch(inputs[0])
            trace.hooks.append(model.register_forward_pre_hook(observe))
        return model

    def loader(dataset, batch_size, generator=None, training=False):
        result = BASE_LOADER(dataset, batch_size, generator, training)
        return TracedLoader(result, trace) if training and trace.limit else result

    class Transform(BASE_TRANSFORM):
        def forward(self, images, training=False):
            result = super().forward(images, training)
            if training:
                trace.augmented_batch(result)
            return result

    def write(path, value):
        BASE_WRITE(path, value)
        if Path(path).name == "status.json" and value.get("state") == "running":
            trace.commit(value)

    with shared.preserve_runtime():
        try:
            engine.ROOT = output  # isolated pause sentinel; original project is read-only
            engine.DIRECTORY = output / "runs"
            engine.SOURCES = tuple(source_identity())
            engine.training_data = lambda: shared.load_data(data)
            engine.source_hashes = source_identity
            engine.build, engine.make_loader = build, loader
            engine.Transform, engine.write_json = Transform, write
            yield
        finally:
            trace.close()
            for name, original in reversed(bindings):
                setattr(engine, name, original)


def saved_boundary(folder, config, source, stats):
    path = folder / "last.pt"
    if not path.exists():
        if any((folder / name).exists() for name in ("best.pt", "history.csv", "result.json")):
            raise ValueError("Saved run artifacts lack their LAST checkpoint; retain and diagnose them")
        return 0
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["config"] != config or saved["source_sha256"] != source or saved["stats"] != stats:
        raise ValueError("Checkpoint sources or configuration differ")
    rows = saved["history"]
    epoch = saved["epoch"]
    if [row["epoch"] for row in rows] != list(range(1, epoch + 1)):
        raise ValueError("Checkpoint history is incomplete")
    if any(row["train_samples"] != 45000 or row["validation_samples"] != 5000 for row in rows):
        raise ValueError("This protocol requires the complete 45k/5k partition")
    for index, row in enumerate(rows):
        factor = (index + 1) / 5 if index < 5 else .5 * (1 + math.cos(math.pi * min(index - 5, 155) / 155))
        if not all(math.isfinite(value) for value in row.values()) or abs(row["learning_rate"] - .1 * factor) > 1e-14:
            raise ValueError("Saved learning rates or numeric history differ")
    if not 0 < epoch <= config["epochs"]:
        raise ValueError("Checkpoint has an invalid epoch")
    result_path = folder / "result.json"
    if result_path.exists():
        result = shared.read(result_path)
        if result.get("state") == "diverged":
            raise FloatingPointError("This frozen case has a recorded divergence; do not silently restart it")
        if epoch != config["epochs"] or result.get("state") != "completed" or result.get("epochs_completed") != epoch or result.get("config") != config:
            raise ValueError("Cached completion disagrees with the actual checkpoint")
    return epoch


def required_preflight_cases(study):
    if study == "augmentation_confirmation":
        return [f"preflight_augmentation_{method}_seed42" for method in AUGMENTATION_METHODS]
    if study == "matched_architecture":
        return [f"architecture_{model}_seed42" for model in ("cnn", "resnet20")]
    raise ValueError("Unknown preflight study")


def validate_proof(proof, study, data):
    required = required_preflight_cases(study)
    if (proof.get("kind") != PROOF_KIND or proof.get("all_passed") is not True
            or proof.get("source_sha256") != source_identity() or proof.get("study") != study
            or proof.get("data") != shared.data_identity(data)
            or proof.get("covered_cases") != required
            or proof.get("continuous_epochs") != 8 or proof.get("resume_epoch") != 4
            or proof.get("actual_GPU_artifacts_checked") is not True):
        raise ValueError("Preflight is incomplete or belongs to another study, data or source version")
    if study == "augmentation_confirmation" and proof.get("all_five_original_seed42_history_bridges_exact") is not True:
        raise ValueError("All five original method prefix histories must be bridged")
    if study == "matched_architecture" and proof.get("matched_actual_training_inputs_exact") is not True:
        raise ValueError("Both architectures must observe the same actual training inputs")


def _equal(left, right, location="root"):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError(f"Tensor mismatch at {location}")
    elif isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or left.dtype != right.dtype or not np.array_equal(left, right):
            raise ValueError(f"Array mismatch at {location}")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError(f"Mapping mismatch at {location}")
        for key in left:
            _equal(left[key], right[key], f"{location}.{key}")
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError(f"Sequence mismatch at {location}")
        for index, (x, y) in enumerate(zip(left, right)):
            _equal(x, y, f"{location}[{index}]")
    elif left != right:
        raise ValueError(f"Value mismatch at {location}")


def _nonseconds(rows):
    return [{k: v for k, v in row.items() if k != "seconds"} for row in rows]


def _history_file(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{k: int(v) if k in ("epoch", "train_samples", "validation_samples") else float(v)
                 for k, v in row.items()} for row in csv.DictReader(stream)]


def verify_gpu_preflight(manifest_path, proof_output):
    """Read completed artifacts only; never constructs a model or initializes CUDA.

    Manifest cases map names to continuous_output/resumed_output and, for the
    augmentation study, an explicit original_reference_run directory. Relative
    paths resolve against the manifest file. Both outputs must be preflight8
    identities; resumed_output/preflight_boundary_epoch4.pt proves the boundary.
    """
    manifest_path, proof_output = Path(manifest_path).resolve(), Path(proof_output).resolve()
    if proof_output.exists():
        raise ValueError("Proof receipts are immutable; choose a new output")
    manifest = shared.read(manifest_path)
    study = manifest["study"]
    cases = fixed_cases()
    required = required_preflight_cases(study)
    if set(manifest["cases"]) != set(required):
        raise ValueError("The GPU preflight manifest does not cover this entire fixed study")
    data = (manifest_path.parent / manifest["data_dir"]).resolve()
    data_id, sources = shared.data_identity(data), source_identity()
    stats = shared.read(data / "normalization.json")
    checks, traces = {}, {}
    for name in required:
        case = cases[name]
        roots = {role: (manifest_path.parent / manifest["cases"][name][role]).resolve()
                 for role in ("continuous_output", "resumed_output")}
        snapshots = {}
        for role, output in roots.items():
            marker = shared.read(output / "identity.json")
            if (marker.get("invocation_class") != "preflight" or marker.get("case") != case
                    or marker.get("source_sha256") != sources or marker.get("data") != data_id
                    or marker.get("trace_epochs") != 8):
                raise ValueError("Not a matching preflight8 output identity")
            folder = output / "runs" / name
            if saved_boundary(folder, case["config"], sources, stats) != 8:
                raise ValueError("The preflight did not finish exactly eight full-data epochs")
            last = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
            if not last.get("rng", {}).get("cuda") or shared.read(folder / "environment.json").get("gpu") in (None, "cpu"):
                raise ValueError("CPU or missing hardware artifacts cannot certify GPU parity")
            trace = Trace(output, 8, 8)
            if any(row["batch_sizes"] != [256] * 175 + [200] for row in trace.rows):
                raise ValueError("Trace does not cover full45k with its last200 batch")
            snapshots[role] = last
            traces[(name, role)] = trace.rows
        left, right = snapshots.values()
        _equal({k: v for k, v in left.items() if k != "history"}, {k: v for k, v in right.items() if k != "history"}, name)
        _equal(_nonseconds(left["history"]), _nonseconds(right["history"]), name + ".history")
        for filename in ("best.pt",):
            x = torch.load(roots["continuous_output"] / "runs" / name / filename, map_location="cpu", weights_only=False)
            y = torch.load(roots["resumed_output"] / "runs" / name / filename, map_location="cpu", weights_only=False)
            _equal(x, y, name + ".BEST")
        _equal(traces[(name, "continuous_output")], traces[(name, "resumed_output")], name + ".trace")
        boundary_path = roots["resumed_output"] / "preflight_boundary_epoch4.pt"
        boundary = torch.load(boundary_path, map_location="cpu", weights_only=False)
        if boundary["epoch"] != 4 or boundary["source_sha256"] != sources or boundary["config"] != case["config"] or boundary["stats"] != stats:
            raise ValueError("The preserved resume4 checkpoint differs")
        _equal(_nonseconds(boundary["history"]), _nonseconds(right["history"][:4]), name + ".resume4_history")
        result = {"continuous_resume_numeric_states_exact": True, "actual_input_trace_exact": True,
                  "resume4_checkpoint_sha256": shared.sha(boundary_path), "last_checkpoint_sha256": {role: shared.sha(root / "runs" / name / "last.pt") for role, root in roots.items()}}
        if study == "augmentation_confirmation":
            reference = (manifest_path.parent / manifest["cases"][name]["original_reference_run"]).resolve()
            if shared.read(reference / "config.json") != case["config"]:
                raise ValueError("The original seed42 reference configuration differs")
            expected6 = {n: shared.read(shared.PACKAGE / "evidence/source10.json")["source_sha256"][n] for n in CORE_SOURCES}
            if shared.read(reference / "environment.json")["source_sha256"] != expected6:
                raise ValueError("Original reference is not the frozen Source6 experiment")
            original_history = _history_file(reference / "history.csv")
            if len(original_history) != 160:
                raise ValueError("The original reference must be a completed160 history")
            _equal(_nonseconds(original_history[:8]), _nonseconds(left["history"]), name + ".original_prefix8")
            result["original_seed42_prefix8_history_exact"] = True
            result["original_history_sha256"] = shared.sha(reference / "history.csv")
        checks[name] = result
    matched = False
    if study == "matched_architecture":
        _equal(traces[(required[0], "continuous_output")], traces[(required[1], "continuous_output")], "matched_architecture_actual_inputs")
        matched = True
    proof = {"kind": PROOF_KIND, "all_passed": True, "study": study, "source_sha256": sources,
             "data": data_id, "covered_cases": required, "continuous_epochs": 8, "resume_epoch": 4,
             "actual_GPU_artifacts_checked": True, "checks": checks,
             "all_five_original_seed42_history_bridges_exact": study == "augmentation_confirmation",
             "matched_actual_training_inputs_exact": matched,
             "scope": "actual full45k/5k prefix8, preserved4->8 restore, fixed-source same-device/software artifacts; not full160 or all-device proof"}
    validate_proof(proof, study, data)
    shared.write(proof_output, proof)
    return proof


def evaluate_last(run_folder, data_dir, analysis_output, preflight=False):
    """Read-only raw BF16 LAST160 evaluation on clean45k/5k, never test.

    The native hard CE is distinct from FP64 per-image NLL and, for smoothing
    cases, from the historical label-smoothed validation_loss.
    """
    folder, data, output = Path(run_folder).resolve(), Path(data_dir).resolve(), Path(analysis_output).resolve()
    cases = fixed_cases()
    if folder.name not in cases or (cases[folder.name].get("preflight_only") and not preflight):
        raise ValueError("Evaluation needs a registered formal case")
    case, source = cases[folder.name], source_identity()
    training_root = folder.parent.parent
    marker = shared.read(training_root / "identity.json")
    data_id = shared.data_identity(data)
    if marker.get("invocation_class") != ("preflight" if preflight else "formal") or marker.get("case") != case or marker.get("source_sha256") != source or marker.get("data") != data_id:
        raise ValueError("The run is not this frozen formal protocol")
    if (training_root / "active_driver.lock").exists():
        raise RuntimeError("Do not evaluate a run while its training driver owns the lock")
    if output == training_root or output.is_relative_to(training_root) or training_root.is_relative_to(output) or output.is_relative_to(data) or data.is_relative_to(output):
        raise ValueError("Evaluation output must be separate from inputs")
    stats = shared.read(data / "normalization.json")
    target = 8 if preflight else 160
    endpoint = "PREFLIGHT_LAST8" if preflight else "LAST160"
    if saved_boundary(folder, case["config"], source, stats) != target:
        raise ValueError(f"Only complete {endpoint} is evaluated here")
    if shared.read(folder / "config.json") != case["config"] or shared.read(folder / "environment.json")["source_sha256"] != source:
        raise ValueError("Run config/environment provenance differs")
    checkpoint = folder / "last.pt"
    locks = {name: shared.sha(folder / name) for name in ("last.pt", "config.json", "history.csv", "environment.json")}
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not set(("model", "optimizer", "scheduler", "rng")).issubset(saved):
        raise ValueError("A native complete LAST state is required before any CUDA initialization")
    _equal(_history_file(folder / "history.csv"), saved["history"], "saved_LAST_history_csv")
    identity = {"kind": "controlled_" + endpoint + "_clean_evaluation_v1", "case": case, "source_sha256": source,
                "data": data_id, "input_file_sha256": locks, "checkpoint": endpoint, "preflight_only": preflight,
                "precision": "native CUDA BF16 raw single-view; no calibration", "official_test_images": 0}
    marker_path = output / "evaluation_identity.json"
    if marker_path.exists():
        if shared.read(marker_path) != identity:
            raise ValueError("Evaluation output belongs to different inputs")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Refusing unmarked nonempty evaluation output")
    else:
        shared.write(marker_path, identity)
    if (output / "evaluation.json").exists():
        result = shared.read(output / "evaluation.json")
        if all(shared.sha(output / row["cache"]) == row["cache_sha256"] for row in result["partitions"].values()):
            return result
        raise ValueError("A completed evaluation cache changed")
    train_set, val_set, actual_stats = shared.load_data(data)
    if actual_stats != stats or len(train_set) != 45000 or len(val_set) != 5000:
        raise ValueError("Clean evaluation requires all fixed45k/5k samples")
    split = shared.read(data / "splits/train_validation_seed42.json")
    partitions = {}
    with shared.preserve_runtime():
        # No training/teacher is built or called. Native epoch_pass switches the
        # student to eval, disables gradients and applies no random transform.
        if not torch.cuda.is_available():
            raise RuntimeError("Historical BF16 replay requires CUDA")
        torch.set_num_threads(8)
        engine.seed_everything(case["config"]["seed"])
        model = BASE_BUILD(case["config"]).cuda().to(memory_format=torch.channels_last)
        model.load_state_dict(saved["model"])
        model.eval().requires_grad_(False)
        before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        transform = BASE_TRANSFORM(stats, case["config"]).cuda()
        for name, dataset, index_key, per_class_count in (("train", train_set, "train_indices", 4500), ("validation", val_set, "validation_indices", 500)):
            cache, meta = output / f"{name}_raw.npz", output / f"{name}_raw.json"
            labels, indices = dataset.tensors[1].numpy(), np.asarray(split[index_key], dtype=np.int64)
            if cache.exists() or meta.exists():
                if not cache.exists() or not meta.exists():
                    raise ValueError("Interrupted cache write retained; use a new analysis output")
                existing = shared.read(meta)
                if existing["identity"] != identity or existing["cache_sha256"] != shared.sha(cache):
                    raise ValueError("Partition cache provenance differs")
                row = existing["metrics"]
                with np.load(cache) as arrays:
                    if not np.array_equal(arrays["labels"], labels) or not np.array_equal(arrays["indices"], indices):
                        raise ValueError("Partition cache sample ordering differs")
                partitions[name] = row
                continue
            captured = []
            hook = model.register_forward_hook(lambda _model, _inputs, logits: captured.append((logits.detach().float().cpu().numpy(), logits.detach().float().softmax(1).cpu().numpy())))
            try:
                native = engine.epoch_pass(model, BASE_LOADER(dataset, 256, training=False), transform, torch.nn.CrossEntropyLoss(label_smoothing=0), torch.device("cuda"), amp=True)
            finally:
                hook.remove()
            logits = np.concatenate([item[0] for item in captured])
            probabilities = np.concatenate([item[1] for item in captured])
            expected_batches = 176 if name == "train" else 20
            expected_last = 200 if name == "train" else 136
            if len(captured) != expected_batches or len(captured[-1][0]) != expected_last or logits.shape != (len(dataset), 10) or not np.isfinite(logits).all():
                raise ValueError("Unexpected full-partition batching/logits")
            predicted = logits.argmax(1)
            if float((predicted == labels).mean()) != native["accuracy"] or native["samples"] != len(dataset):
                raise ValueError("Native and cached sample accounting differs")
            z = logits.astype(np.float64)
            maximum = z.max(1)
            nll = maximum + np.log(np.exp(z - maximum[:, None]).sum(1)) - z[np.arange(len(labels)), labels]
            matrix = np.zeros((10, 10), dtype=np.int64)
            np.add.at(matrix, (labels, predicted), 1)
            if not np.array_equal(matrix.sum(1), np.full(10, per_class_count)):
                raise ValueError("Unexpected per-class sample counts")
            np.savez_compressed(cache, logits=logits, probabilities=probabilities, labels=labels, indices=indices, per_sample_nll_float64=nll)
            wrong = np.flatnonzero(predicted != labels)
            confidence = probabilities.max(1)
            wrong = wrong[np.argsort(-confidence[wrong], kind="stable")[:20]]
            row = {"samples": len(dataset), "batches": expected_batches, "last_batch": expected_last,
                   "accuracy": native["accuracy"], "native_BF16_batch_weighted_hard_CE": native["ce"],
                   "FP64_per_image_NLL_mean": float(nll.mean()), "confusion_matrix": matrix.tolist(),
                   "per_class_accuracy": (matrix.diagonal() / per_class_count).tolist(),
                   "high_confidence_errors": [{"original_train_index": int(indices[i]), "true": int(labels[i]), "prediction": int(predicted[i]), "confidence_uncalibrated": float(confidence[i])} for i in wrong],
                   "cache": cache.name, "cache_sha256": shared.sha(cache)}
            shared.write(meta, {"identity": identity, "cache_sha256": row["cache_sha256"], "metrics": row})
            partitions[name] = row
        if any(not torch.equal(value.detach().cpu(), before[key]) for key, value in model.state_dict().items()) or any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError("Evaluation changed student parameters, BN or gradients")
    last = saved["history"][-1]
    accuracy_exact = partitions["validation"]["accuracy"] == last["validation_accuracy"]
    ce_difference = abs(partitions["validation"]["native_BF16_batch_weighted_hard_CE"] - last["validation_ce"])
    if not accuracy_exact or ce_difference > 1e-6 or last["validation_samples"] != 5000:
        raise ValueError("Validation accuracy/hard CE do not replay LAST history")
    if source_identity() != source or shared.data_identity(data) != data_id or locks != {name: shared.sha(folder / name) for name in locks}:
        raise ValueError("Source/data/input weights changed during read-only evaluation")
    result = {"state": "evaluated", "case": folder.name, "checkpoint": endpoint, "preflight_only": preflight, "partitions": partitions,
              "validation_accuracy_exact": accuracy_exact, "validation_native_hard_CE_difference": ce_difference,
              "validation_hard_CE_tolerance": 1e-6, "historical_validation_loss_not_used_as_hard_CE": True,
              "native_CE_and_FP64_NLL_are_distinct_reductions": True, "train_eval_no_random_augmentation": True,
              "student_eval_teacher_not_loaded": True, "weights_BN_gradients_and_input_files_unchanged": True,
              "raw_epoch_seconds": [row["seconds"] for row in saved["history"]],
              "official_test_images": 0, "new_training": 0, "manual_visual_review_complete": False}
    shared.write(output / "evaluation.json", result)
    return result


def train(args):
    cases = fixed_cases()
    if args.case not in cases:
        raise ValueError("Only fixed registered cases are accepted")
    case = cases[args.case]
    config = case["config"]
    if case.get("preflight_only") and not args.preflight:
        raise ValueError("This case is a preflight only, never a formal candidate")
    data, output = Path(args.data_dir).resolve(), Path(args.output_dir).resolve()
    if not 0 <= args.trace_epochs <= 8:
        raise ValueError("Trace is restricted to the fixed first-eight-epoch preflight")
    if args.preflight:
        if args.stop_after is None or not 1 <= args.stop_after <= 8:
            raise ValueError("Preflight must stop within the first eight full-data epochs")
        if args.trace_epochs != 8:
            raise ValueError("GPU preflight needs the fixed eight-epoch actual-input trace")
    else:
        if not args.preflight_proof:
            raise ValueError("Formal training requires an actual matching GPU preflight proof")
        proof = shared.read(args.preflight_proof)
        validate_proof(proof, case["study"], data)
    if args.stop_after is not None and not 1 <= args.stop_after <= config["epochs"]:
        raise ValueError("Invalid stopping boundary")
    identity = prepare_output(output, data, case, args.trace_epochs, args.preflight)
    with shared.process_lock(output):
        folder = output / "runs" / args.case
        start = saved_boundary(folder, config, identity["source_sha256"], shared.read(data / "normalization.json"))
        target = args.stop_after or config["epochs"]
        if start > target:
            raise ValueError("Checkpoint already exceeds the requested stopping epoch")
        # Frozen legacy loop only checks stop_after after an epoch. Guard a
        # repeated command at an already reached boundary before entering it.
        if start == target:
            return {"state": "already_at_requested_boundary", "epoch": start}
        trace = Trace(output, args.trace_epochs, start)
        with training_scope(output, data, case, trace):
            result = engine.run(config, args.case, stop_after=args.stop_after)
        if source_identity() != identity["source_sha256"]:
            raise ValueError("Sources changed during execution")
        if shared.data_identity(data) != identity["data"]:
            raise ValueError("Data metadata changed during execution")
        shared.write(output / "invocation_result.json", result)
        if result.get("state") == "diverged":
            raise FloatingPointError("Fixed run diverged; retain the recorded negative result")
        if args.preflight and target == 4 and saved_boundary(folder, config, identity["source_sha256"], shared.read(data / "normalization.json")) == 4:
            boundary = output / "preflight_boundary_epoch4.pt"
            if boundary.exists():
                if shared.sha(boundary) != shared.sha(folder / "last.pt"):
                    raise ValueError("A different preserved prefix4 checkpoint already exists")
            else:
                temporary = boundary.with_suffix(".tmp")
                shutil.copyfile(folder / "last.pt", temporary)
                temporary.replace(boundary)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--trace-epochs", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-proof")
    parser.add_argument("--verify-preflight-manifest", help="read-only artifact check; never starts training")
    parser.add_argument("--proof-output", help="new immutable proof JSON destination")
    parser.add_argument("--evaluate-run", help="read-only complete formal LAST160 clean45k/5k evaluation")
    parser.add_argument("--evaluate-preflight-run", help="explicit preflight-only LAST8 evaluation, never formal")
    args = parser.parse_args()
    if args.verify_preflight_manifest:
        if not args.proof_output:
            parser.error("--verify-preflight-manifest requires --proof-output")
        if args.case or args.data_dir or args.output_dir or args.stop_after or args.preflight or args.preflight_proof or args.evaluate_run or args.evaluate_preflight_run:
            parser.error("Read-only proof verification cannot be combined with training arguments")
        result = verify_gpu_preflight(args.verify_preflight_manifest, args.proof_output)
    elif args.evaluate_run or args.evaluate_preflight_run:
        if args.evaluate_run and args.evaluate_preflight_run:
            parser.error("Choose formal LAST160 or preflight LAST8 evaluation")
        if not args.data_dir or not args.output_dir:
            parser.error("--evaluate-run requires --data-dir and --output-dir")
        if args.case or args.stop_after or args.preflight or args.preflight_proof or args.proof_output:
            parser.error("Read-only LAST evaluation cannot be combined with training/proof arguments")
        result = evaluate_last(args.evaluate_run or args.evaluate_preflight_run, args.data_dir, args.output_dir, preflight=bool(args.evaluate_preflight_run))
    else:
        if args.proof_output:
            parser.error("--proof-output requires --verify-preflight-manifest")
        if not args.case or not args.data_dir or not args.output_dir:
            parser.error("training requires --case, --data-dir and --output-dir")
        result = train(args)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
