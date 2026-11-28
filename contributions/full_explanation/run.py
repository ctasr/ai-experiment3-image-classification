"""Frozen prepare -> actualGPU200 bridge -> remaining4800 -> analysis workflow."""
from pathlib import Path
import argparse
import json
import os
import time
import traceback

import numpy as np
import torch

from . import protocol as p, engine, check


def load_npz(path):
    with np.load(path, allow_pickle=False) as value:
        return {name: value[name] for name in value.files}


def cached(path, identity_sha):
    path = Path(path); receipt = path.with_suffix(".json")
    if path.exists() != receipt.exists():
        raise RuntimeError("Partial phase output exists without matching receipt; preserve it for explicit recovery")
    if not path.exists():
        return None
    value = p.read(receipt)
    if value["identity_sha256"] != identity_sha or value["data_sha256"] != p.sha(path) or value["state"] != "computed":
        raise RuntimeError("Cached phase checksum/source identity differs")
    return load_npz(path), value


def compute(args, stage):
    identity, design = p.prepare(args)
    output = Path(args.output_dir).resolve(); identity_sha = p.sha(output / "identity.json")
    cpu = p.read(output / "CPU_preflight.json")
    if (not cpu.get("passed") or cpu.get("kind") != "CPU_synthetic_and_original_cache_protocol" or cpu["identity_sha256"] != identity_sha
            or cpu["source_sha256"] != identity["source_sha256"] or cpu["checker_sha256"] != p.sha(Path(check.__file__))):
        raise RuntimeError("Current frozen sources/inputs need actual CPU preflight")
    if cpu["design_sha256"] != p.sha(output / "frozen_design.npz"):
        raise RuntimeError("CPU proof does not cover these frozen design bytes")
    if stage == "full":
        proof = p.read(output / "GPU_bridge.json")
        if (not proof.get("passed") or proof.get("kind") != "actual_GPU_original200_three_seed_bridge" or proof["identity_sha256"] != identity_sha
                or proof["source_sha256"] != identity["source_sha256"] or proof["checker_sha256"] != p.sha(Path(check.__file__))):
            raise RuntimeError("All three actualGPU200 cache/semantic bridges must pass before5000")
        if proof["design_sha256"] != p.sha(output / "frozen_design.npz"):
            raise RuntimeError("GPU proof does not cover these frozen target/order/window design bytes")
        passed_bridges = {value["seed"]: value for value in proof["seeds"]}
        if set(passed_bridges) != set(p.SEEDS):
            raise RuntimeError("GPU proof does not retain every fixed model seed")
    raw, labels, indices = p.load_validation(args.data_dir)
    if not np.array_equal(indices[design["validation_rows"]], design["indices"]) or not np.array_equal(labels[design["validation_rows"]], design["labels"]):
        raise ValueError("Actual training-batch validation rows/labels differ from frozen design")
    raw = raw[design["validation_rows"]]
    _, legacy = p.design(args.data_dir, args.reference_dir)
    stats = p.read(Path(args.data_dir) / "normalization.json")
    rows = check.bridge_rows(legacy)
    if stage == "bridge":
        if not np.array_equal(raw[:200].permute(0, 2, 3, 1).numpy(), legacy[42]["images"]):
            raise ValueError("Actual original200 image pixels differ from frozen cache")
    started = time.monotonic(); results = []
    with p.preserve_runtime():
        for seed in p.SEEDS:
            runtime = p.configure_FP32(seed)
            path = output / ("bridge" if stage == "bridge" else "full") / f"seed{seed}.npz"
            existing = cached(path, identity_sha)
            if existing:
                if existing[1].get("runtime") != runtime:
                    raise RuntimeError("Cached seed-specific interpolation runtime differs")
                results.append(existing[1]); continue
            model = engine.load_model(getattr(args, f"weights_{seed}"), seed, stats, torch.device("cuda"))
            before = p.state_hash(model)
            trace_path = output / ("bridge" if stage == "bridge" else "full") / f"seed{seed}_events.jsonl"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            if trace_path.exists():
                raise RuntimeError("Uncommitted phase trace exists; preserve it for explicit recovery")
            with trace_path.open("x", encoding="utf-8") as trace:
                def event(value):
                    value.update(seed=seed, stage=stage)
                    trace.write(json.dumps(value, allow_nan=False) + "\n"); trace.flush()
                    p.write(output / "progress.json", {"state": stage, "seed": seed, **value})
                if stage == "bridge":
                    data, features, compute_proof = engine.compute_segment(model, stats, raw[:200], design["target"][:200],
                            design["random_positions"][:200], design["indices"][:200], event, rows)
                    # Preserve actual computed arrays even if the bridge fails.
                    # Unverified data can never satisfy cached() or the5000 gate.
                    data.update(indices=design["indices"][:200], labels=design["labels"][:200], target=design["target"][:200])
                    p.save_npz(path, **data)
                    p.write(path.with_suffix(".json"), {"state": "unverified_bridge", "seed": seed,
                            "identity_sha256": identity_sha, "source_sha256": identity["source_sha256"],
                            "data_sha256": p.sha(path), "compute": compute_proof, "runtime": runtime,
                            "actual_same_layout_and_semantic_gates_pending": True})
                    exact_proof = check.original200_exact(data, legacy[seed], design["indices"][:200], design["labels"][:200], design["target"][:200])
                    semantic_proof = check.gpu_semantics(model, stats, raw[:200], design["target"][:200], data, features, rows)
                    extra = {"same_layout_original200_exact": exact_proof, "single_image_semantics": semantic_proof}
                    prefix = slice(0, 200)
                else:
                    bridge = cached(output / "bridge" / f"seed{seed}.npz", identity_sha)
                    if bridge is None:
                        raise RuntimeError("Missing passed original200 bridge cache")
                    if bridge[1]["data_sha256"] != passed_bridges[seed]["data_sha256"]:
                        raise RuntimeError("Original200 bridge cache no longer matches the passed GPU proof")
                    if bridge[1].get("runtime") != runtime:
                        raise RuntimeError("Original200 and remaining4800 seed runtimes differ")
                    remaining, _, compute_proof = engine.compute_segment(model, stats, raw[200:], design["target"][200:],
                            design["random_positions"][200:], design["indices"][200:], event)
                    data = {name: np.concatenate((bridge[0][name], remaining[name])) for name in engine.FIELDS}
                    extra = {"original200_reused_from_this_run_GPU_bridge": True,
                             "bridge_data_sha256": bridge[1]["data_sha256"], "new_inference_samples": 4800,
                             "prefix_and_new_samples": [200, 4800]}
                    prefix = slice(None)
                p.assert_frozen(model, before)
                if p.fp32_runtime(seed) != runtime:
                    raise RuntimeError("Seed-specific FP32/interpolation runtime changed during inference")
                data.update(indices=design["indices"][prefix], labels=design["labels"][prefix], target=design["target"][prefix])
                if stage != "bridge":
                    p.save_npz(path, **data)
                value = {"state": "computed", "seed": seed, "stage": stage, "samples": len(data["indices"]),
                         "identity_sha256": identity_sha, "data_sha256": p.sha(path), "source_sha256": identity["source_sha256"],
                         "compute": compute_proof, "runtime": runtime, "weights_BN_gradients_unchanged": True, **extra}
                p.write(path.with_suffix(".json"), value); results.append(value)
            del model; torch.cuda.empty_cache()
            if p.input_identity(args) != identity:
                raise RuntimeError("Frozen input/source bytes changed during inference")
    if stage == "bridge":
        result = {"passed": True, "kind": "actual_GPU_original200_three_seed_bridge",
                  "identity_sha256": identity_sha, "design_sha256": p.sha(output / "frozen_design.npz"),
                  "source_sha256": identity["source_sha256"], "checker_sha256": p.sha(Path(check.__file__)),
                  "all_seeds": list(p.SEEDS), "seeds": results, "GPU_autograd": 0,
                  "wall_seconds_including_pauses": time.monotonic() - started,
                  "all5000_compute_permitted_only_for_this_identity": True}
        p.write(output / "GPU_bridge.json", result)
    else:
        result = {"state": "full5000_compute_complete_analysis_pending", "identity_sha256": identity_sha,
                  "source_sha256": identity["source_sha256"], "seeds": results, "unique_images": 5000,
                  "not15000_independent": True, "all_three_seeds_retained": True, "new_training": 0,
                  "new_official_or_external_test": 0, "new_temperature_fit": 0,
                  "wall_seconds_including_pauses": time.monotonic() - started,
                  "analysis_and_actual_visual_review_pending": True}
        p.write(output / "compute_complete.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "bridge", "full")); p.add_inputs(parser)
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        identity, _ = p.prepare(args)
        print(json.dumps({"state": "prepared", "GPU_started": False, "source_sha256": identity["source_sha256"]})); return
    output = Path(args.output_dir).resolve()
    if not (output / "identity.json").exists():
        raise RuntimeError("Run explicit CPU preparation/preflight before GPU dispatch")
    lock_path = output / "active_inference.lock"
    lock = lock_path.open("x", encoding="utf-8"); lock.write(str(os.getpid())); lock.flush()
    try:
        p.write(output / "worker.json", {"state": args.stage, "pid": os.getpid(), "new_training": 0})
        result = compute(args, args.stage)
        p.write(output / "worker.json", {"state": "bridge_passed" if args.stage == "bridge" else result["state"], "pid": os.getpid()})
        print(json.dumps({"state": "bridge_passed" if args.stage == "bridge" else result["state"]}), flush=True)
    except BaseException as exc:
        p.write(output / "worker.json", {"state": "needs_root_review", "pid": os.getpid(),
                "reason": str(exc), "traceback": traceback.format_exc(), "full5000_approved": False})
        raise
    finally:
        lock.close(); lock_path.unlink()


if __name__ == "__main__":
    main()
