"""CPU checks and read-only comparisons; never starts a training process."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from . import adapter as a


def equal(left, right, location="root"):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError(f"Tensor mismatch: {location}")
    elif isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or left.dtype != right.dtype or not np.array_equal(left, right):
            raise ValueError(f"Array mismatch: {location}")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError(f"Mapping mismatch: {location}")
        for key in left:
            equal(left[key], right[key], f"{location}.{key}")
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError(f"Sequence mismatch: {location}")
        for index, (x, y) in enumerate(zip(left, right)):
            equal(x, y, f"{location}[{index}]")
    elif left != right:
        raise ValueError(f"Value mismatch: {location}")


def compare_runs(actual, reference, epochs):
    actual, reference = Path(actual).resolve(), Path(reference).resolve()
    ah = a.history(actual, require_complete=epochs == 160)
    rh = a.history(reference, require_complete=epochs == 160)
    if len(ah) != epochs or len(rh) != epochs:
        raise ValueError("Unexpected checkpoint boundary")
    equal([{k: v for k, v in row.items() if k != "seconds"} for row in ah], [{k: v for k, v in row.items() if k != "seconds"} for row in rh], "nonseconds_history")
    checked = {}
    for filename in ("last.pt", "best.pt"):
        x = torch.load(actual / filename, map_location="cpu", weights_only=False)
        y = torch.load(reference / filename, map_location="cpu", weights_only=False)
        if filename == "last.pt" and (x["epoch"] != epochs or y["epoch"] != epochs):
            raise ValueError("Wrong LAST epoch")
        # SHA metadata are intentionally different when comparing a new adapter to
        # the original experiment. All numerical/state fields must still match.
        ignore = {"source_sha256", "history"}
        equal({k: v for k, v in x.items() if k not in ignore}, {k: v for k, v in y.items() if k not in ignore}, filename)
        if "history" in x and "history" in y:
            equal([{k: v for k, v in row.items() if k != "seconds"} for row in x["history"]], [{k: v for k, v in row.items() if k != "seconds"} for row in y["history"]], filename + ".history")
        checked[filename] = {"actual_sha256": a.sha(actual / filename), "reference_sha256": a.sha(reference / filename), "numeric_fields_exact": True, "source_identity_equal": x["source_sha256"] == y["source_sha256"]}
        del x, y
    return {"passed": True, "epochs": epochs, "nonseconds_history_exact": True, "checkpoints": checked, "no_full_checkpoint_byte_equality_claim": True}


def cpu(data_dir=None):
    if torch.cuda.is_initialized():
        raise RuntimeError("Run CPU checks in a fresh process")
    source_hashes = a.sources()
    for seed in a.SEEDS:
        ca, cb = a.config(seed, "A"), a.config(seed, "B")
        equal({k: v for k, v in ca.items() if k != "alignment"}, {k: v for k, v in cb.items() if k != "alignment"}, "matched_config")
        if (ca["width"], ca["epochs"], ca["batch_size"], ca["distill_alpha"], ca["distill_temperature"]) != (32, 160, 256, .25, 1.):
            raise ValueError("Fixed configuration changed")
    original = a.read(a.PACKAGE / "evidence/index.json")
    for row in original["entries"]:
        if a.sha(a.PACKAGE / "evidence" / row["file"]) != row["sha256"]:
            raise ValueError("Original raw numerical evidence changed")
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    with a.preserve_runtime():
        random.random(); np.random.random(); torch.rand(3)
        torch.set_num_threads(1)
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    equal(before, after, "runtime_restoration")
    result = {"kind": "cpu_checks_only", "passed": True, "source_sha256": source_hashes, "fixed_AB_configs_match_except_alignment": True, "original_numeric_evidence_bytes_exact": True, "runtime_RNG_and_threads_restored": True, "train_batches": 176, "train_last_batch": 200, "validation_batches": 20, "validation_last_batch": 136, "GPU_preflight_passed": False, "CUDA_initialized": torch.cuda.is_initialized()}
    if data_dir:
        train, val, stats = a.load_data(Path(data_dir).resolve())
        result["data"] = {"train_samples": len(train), "validation_samples": len(val), "stats": stats, "official_test_used_for_metrics": False}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cp = sub.add_parser("cpu")
    cp.add_argument("--data-dir")
    cp.add_argument("--output")
    ep = sub.add_parser("evidence")
    ep.add_argument("--output")
    gp = sub.add_parser("gpu-proof", help="compare already-run A/B seed42 continuous3 and resumed1->3")
    gp.add_argument("--continuous-root", required=True)
    gp.add_argument("--resumed-root", required=True)
    gp.add_argument("--output", required=True)
    compare = sub.add_parser("compare", help="compare a completed portable run with its original numeric artifacts")
    compare.add_argument("--actual-run", required=True)
    compare.add_argument("--reference-run", required=True)
    compare.add_argument("--epochs", type=int, default=160)
    compare.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "cpu":
        result = cpu(args.data_dir)
    elif args.command == "evidence":
        cpu()
        data = a.read(a.PACKAGE / "evidence/original_validation_analysis.json")
        result = {"kind": "archived_original_validation_evidence_not_new_inference", "pairs": {seed: {role: {"accuracy": pair[role]["validation"]["accuracy"], "ce": pair[role]["validation"]["ce"], "best_epoch": pair[role]["selection"]["best_epoch"]} for role in a.ROLES} for seed, pair in data["pairs"].items()}, "gate": data["gate"]}
    elif args.command == "compare":
        result = compare_runs(args.actual_run, args.reference_run, args.epochs)
        result["kind"] = "read_only_original_numeric_comparison"
    else:
        continuous, resumed = Path(args.continuous_root).resolve(), Path(args.resumed_root).resolve()
        identity = a.read(continuous / "portable_alignment_identity.json")
        equal(identity, a.read(resumed / "portable_alignment_identity.json"), "preflight_inputs")
        equal(identity["source_sha256"], a.sources(), "current_sources")
        checks = {}
        for role in a.ROLES:
            left, right = continuous / "runs" / f"{role}_seed42", resumed / "runs" / f"{role}_seed42"
            checks[role] = compare_runs(left, right, 3)
            for folder in (left, right):
                teacher = a.read(folder / "teacher_check.json")
                if not teacher["teacher_loaded"] or not teacher["weights_bn_gradients_unchanged"]:
                    raise ValueError("Teacher freeze proof missing")
                saved = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
                if not saved["rng"]["cuda"]:
                    raise ValueError("CPU-only run cannot certify GPU parity")
                del saved
        result = {"kind": "gpu_continuous_resume_AB3", "passed": True, "identity": identity, "checks": checks, "checker_sha256": a.sha(Path(__file__)), "scope": "same GPU/software, actual45k/5k A/B seed42 first3; continuous vs epoch1->3 restore; not full160 reproduction or all-device proof", "training_core_delegated_unchanged": True}
    if args.output:
        destination = Path(args.output).resolve()
        if destination.exists():
            raise ValueError("Comparison receipts are immutable; choose a new --output")
        a.write(destination, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
