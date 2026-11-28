"""Explicit CPU/tiny-GPU/formal240 boundaries for the fixed weighted KD plan."""
from pathlib import Path
from contextlib import contextmanager
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import traceback
from itertools import zip_longest

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] / "实验三-图像分类"
OLD = HERE.parent / "source_set_kd_pilot"
SPATIAL = HERE.parent / "balanced_spatial_dropout_pilot"
PLAN_SHA = "e006815ffbf8c71f08bf17d7701be8628a0e266e4bab6c28c6a3976c56da3a61"
IMPLEMENTATION_SHA = "d22d81174a152d33f37241cbcd6c121819827d3ee45a333b9f91c4d498a18128"
FORMAL_NAME = "weighted240_seed42"
INPUT_KEYS = ("epoch", "batch_in_epoch", "global_batch", "batch_size", "actual_lambda", "order_sha256",
              "labels_sha256", "view_sha256", "teacher_input_sha256", "teacher_raw_q_sha256", "teacher_t1_q_sha256")
LOSS_KEYS = ("original_KL", "new_extra", "modified_KD", "hard_CE", "returned_total")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(name, value):
    path = (HERE / name).resolve()
    assert path.is_relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_adapter():
    old_path = list(sys.path)
    old_bytecode = sys.dont_write_bytecode
    try:
        sys.path.insert(0, str(PROJECT))
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location("private_weighted_confirmation_adapter", HERE / "adapter.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode


def implementation():
    assert sha(HERE / "plan.json") == PLAN_SHA
    assert sha(HERE / "IMPLEMENTATION_READY.json") == IMPLEMENTATION_SHA
    approval = read(HERE / "IMPLEMENTATION_READY.json")
    assert approval["approved"] is True and approval["plan_sha256"] == PLAN_SHA
    decision = HERE / "decision.json"
    if not decision.exists():
        decision = HERE / "prospective_decision.json"
    assert sha(decision) == approval["decision_sha256"]
    return approval, decision


def verify(adapter=None):
    implementation()
    manifest = read(HERE / "source_manifest.json")
    assert manifest["plan_sha256"] == PLAN_SHA and manifest["IMPLEMENTATION_READY_sha256"] == IMPLEMENTATION_SHA
    for name, expected in manifest["private_sources"].items():
        assert sha(HERE / name) == expected
    for path, expected in manifest["protected_inputs"].items():
        assert sha(Path(path)) == expected, "Protected file changed: " + path
    adapter = adapter or load_adapter()
    assert adapter.source_hashes() == manifest["training_sources"] and len(manifest["training_sources"]) == 15
    assert sha(Path(read(HERE / "plan.json")["fixed_algorithm"]["loss_module"])) == read(HERE / "plan.json")["fixed_algorithm"]["loss_sha256"]
    return {"passed": True, "protected_count": len(manifest["protected_inputs"])}


def ready(name, adapter):
    approval = read(HERE / name)
    assert approval["approved"] is True
    for key, filename in (("plan_sha256", "plan.json"), ("source_manifest_sha256", "source_manifest.json"),
                          ("prepare_CPU_sha256", "prepare_CPU.json")):
        assert approval[key] == sha(HERE / filename)
    if name == "FORMAL_READY.json":
        assert approval["preflight_GPU_sha256"] == sha(HERE / "preflight_GPU.json")
        assert read(HERE / "preflight_GPU.json")["state"] == "passed"
    verify(adapter)
    return approval


def cfg(adapter, mode, tiny=False):
    config = read(PROJECT / "runs/horizon/horizon240_seed42/config.json")
    config["source_set_mode"] = mode
    if tiny:
        config["pilot_samples"] = 456
    return adapter.configuration(config, mode)


@contextmanager
def awake():
    assert os.name == "nt"
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel.SetThreadExecutionState
    function.argtypes = [ctypes.c_uint]; function.restype = ctypes.c_uint
    previous = int(function(0x80000003))
    if not previous:
        raise ctypes.WinError(ctypes.get_last_error())
    record = {"previous_request": previous, "requested": 0x80000003, "restored": False, "powercfg_changed": False}
    try:
        yield record
    finally:
        if not function(previous):
            raise ctypes.WinError(ctypes.get_last_error())
        record["restored"] = True


@contextmanager
def exclusive():
    assert os.name == "nt"
    import msvcrt
    stream = (HERE / "driver.lock").open("a+b")
    locked = False
    try:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        locked = True
        yield
    finally:
        if locked:
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        stream.close()


def checkpoint(adapter, folder, kind="last.pt"):
    return adapter.torch.load(Path(folder) / kind, map_location="cpu", weights_only=False)


def equal(adapter, a, b):
    if isinstance(a, adapter.torch.Tensor):
        return isinstance(b, adapter.torch.Tensor) and adapter.torch.equal(a, b)
    if isinstance(a, adapter.CORE.np.ndarray):
        return isinstance(b, adapter.CORE.np.ndarray) and adapter.CORE.np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(adapter, a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(adapter, x, y) for x, y in zip(a, b))
    return a == b


def compare(adapter, left_folder, right_folder):
    left, right = checkpoint(adapter, left_folder), checkpoint(adapter, right_folder)
    for key in ("epoch", "model", "optimizer", "scheduler", "rng", "best_accuracy", "best_ce", "stats"):
        assert equal(adapter, left[key], right[key]), "Checkpoint identity: " + key
    histories = [[{key: value for key, value in row.items() if key != "seconds"} for row in saved["history"]]
                 for saved in (left, right)]
    assert histories[0] == histories[1]
    first, second = checkpoint(adapter, left_folder, "best.pt"), checkpoint(adapter, right_folder, "best.pt")
    for key in ("epoch", "model", "best_accuracy", "best_ce", "stats"):
        assert equal(adapter, first[key], second[key]), "Best checkpoint identity: " + key


def traces(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit_trace(path, epochs, bpe):
    count = 0
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            epoch, batch = count // bpe + 1, count % bpe + 1
            size = 200 if batch == bpe else 256
            assert (row["epoch"], row["batch_in_epoch"], row["global_batch"], row["batch_size"]) == (epoch, batch, count + 1, size)
            assert all(key in row for key in INPUT_KEYS + LOSS_KEYS)
            assert all(math.isfinite(row[key]) for key in LOSS_KEYS)
            for key in INPUT_KEYS:
                if key.endswith("sha256"):
                    assert isinstance(row[key], str) and len(row[key]) == 64
                    assert all(character in "0123456789abcdef" for character in row[key].lower())
            count += 1
    assert count == epochs * bpe
    return {"records": count, "sample_counts_indices_losses_finite": True}


def shared_trace_equal(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert all(a[key] == b[key] for key in INPUT_KEYS + LOSS_KEYS)
        shared = set(a) & set(b)
        assert all(a[key] == b[key] for key in shared)


def prepare():
    implementation()
    assert not (HERE / "source_manifest.json").exists()
    adapter = load_adapter()
    assert not adapter.torch.cuda.is_initialized()
    with adapter.runtime():
        adapter.torch.set_num_threads(1)
        loss_proof = adapter.cpu_loss_check()
        assert loss_proof.get("state") == "passed" and loss_proof.get("case_count", len(loss_proof.get("cases", []))) >= 8
        route_proof = adapter.cpu_route_check(cfg(adapter, "weighted_single", True))
        assert route_proof.get("passed", route_proof.get("state") == "passed")
    approval, decision = implementation()
    protected = dict(read(SPATIAL / "source_manifest.json")["protected_inputs"])
    plan = read(HERE / "plan.json")
    protected.update(plan["protected_inputs"])
    extra = [HERE / "plan.json", HERE / "IMPLEMENTATION_READY.json", decision,
             SPATIAL / "source_manifest.json", OLD / "source_manifest.json", OLD / "pilot_results.json"]
    for name in ("last.pt", "best.pt", "history.csv", "config.json"):
        extra.append(OLD / "pilot_runs/pilot_weighted_single_seed42" / name)
    extra.append(OLD / "pilot_traces/pilot_weighted_single_seed42.jsonl")
    lock_documents = list(HERE.glob("ROOT*LOCK*.json"))
    previous_stage_locks = SPATIAL / "ROOT_PREVIOUS_STAGE_LOCKS.json"
    if previous_stage_locks.exists():
        lock_documents.append(previous_stage_locks)
    for path in lock_documents:
        extra.append(path)
        record = read(path)
        groups = (record, record.get("protected_inputs", {}), record.get("locks", {}))
        for group in groups:
            if isinstance(group, dict):
                for name, digest in group.items():
                    if isinstance(digest, str) and len(digest) == 64 and Path(name).is_absolute():
                        assert name not in protected or protected[name] == digest
                        protected[name] = digest
    # The previous spatial stage is scientifically/manual/power closed. Lock
    # its six real checkpoints, raw telemetry, evaluation and human evidence;
    # never include transient recovery folders or mutable worker/progress files.
    for directory_name in ("pilot_runs", "pilot_traces", "evaluation", "pilot_figures", "figures"):
        directory = SPATIAL / directory_name
        if directory.exists():
            extra.extend(path for path in directory.rglob("*") if path.is_file()
                         and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".lock", ".tmp"))
    extra.extend(path for path in SPATIAL.iterdir() if path.is_file()
                 and path.name not in ("driver.lock", "worker.json", "progress.json")
                 and path.suffix not in (".pyc", ".lock", ".tmp"))
    protected.update({str(path.resolve()): sha(path) for path in extra})
    assert all(sha(Path(path)) == expected for path, expected in protected.items())
    assert adapter.source_hashes() == {**plan["original_training_source_dictionary"], **{
        key: value for key, value in adapter.source_hashes().items() if key not in plan["original_training_source_dictionary"]}}
    assert len(adapter.source_hashes()) == 15
    write("source_manifest.json", {"plan_sha256": PLAN_SHA, "IMPLEMENTATION_READY_sha256": IMPLEMENTATION_SHA,
                                  "private_sources": {name: sha(HERE / name) for name in ("adapter.py", "run.py")},
                                  "training_sources": adapter.source_hashes(), "old_reference_sources": adapter.source_hashes(True),
                                  "protected_inputs": protected})
    proof = {"state": "passed", "loss": loss_proof, "route": route_proof,
             "protected_after": verify(adapter), "GPU_started": False, "formal_started": False}
    write("prepare_CPU.json", proof)
    return {"state": "CPU_prepared_waits_GPU_READY", "source_manifest_sha256": sha(HERE / "source_manifest.json"),
            "prepare_CPU_sha256": sha(HERE / "prepare_CPU.json"), "GPU_started": False}


def gpu_preflight(adapter):
    ready("GPU_READY.json", adapter)
    assert not (HERE / "preflight_runs").exists() and not (HERE / "preflight_GPU.json").exists()
    runs, out, trace = HERE / "preflight_runs", HERE / "preflight_out", HERE / "preflight_traces"
    calls = (("ref1", "original", 1, True), ("original1", "original", 1, False),
             ("weighted_cont6", "weighted_single", 6, False),
             ("weighted_resume", "weighted_single", 2, False), ("weighted_resume", "weighted_single", 6, False))
    executed = []
    with awake() as power:
        for name, mode, stop, reference in calls:
            start = checkpoint(adapter, runs / name)["epoch"] if (runs / name / "last.pt").exists() else 0
            write("worker.json", {"pid": os.getpid(), "state": "tiny_GPU_preflight", "current_run": name,
                                  "start_epoch": start, "stop_after": stop, "formal_started": False})
            adapter.run(cfg(adapter, mode, True), name, mode, stop_after=stop,
                        out_dir=out / ("source12" if reference else "source15"), run_dir=runs,
                        trace_file=trace / (name + ".jsonl"), reference=reference)
            saved = checkpoint(adapter, runs / name)
            assert saved["epoch"] == stop and saved["source_sha256"] == adapter.source_hashes(reference)
            added = saved["history"][start:]
            assert len(added) == stop - start and all(row["train_samples"] == 456 and row["validation_samples"] == 1000 for row in added)
            for row in added:
                e = row["epoch"] - 1
                factor = (e + 1) / 5 if e < 5 else .5 * (1 + math.cos(math.pi * (e - 5) / 235))
                assert abs(row["learning_rate"] - .1 * factor) < 1e-12
            executed.append({"name": name, "start_epoch": start, "stop_epoch": stop, "new_epochs": stop - start})
            verify(adapter)
        compare(adapter, runs / "ref1", runs / "original1")
        ref_trace, disabled_trace = traces(trace / "ref1.jsonl"), traces(trace / "original1.jsonl")
        shared_trace_equal(ref_trace, disabled_trace)
        compare(adapter, runs / "weighted_cont6", runs / "weighted_resume")
        continuous, resumed = traces(trace / "weighted_cont6.jsonl"), traces(trace / "weighted_resume.jsonl")
        assert continuous == resumed
        trace_proof = audit_trace(trace / "weighted_cont6.jsonl", 6, 2)
        assert sum(row["new_epochs"] for row in executed) == 14
        assert sum(len(traces(path)) for path in trace.glob("*.jsonl")) == 28
        last_sha = sha(runs / "weighted_resume/last.pt")
        boundary = adapter.run(cfg(adapter, "weighted_single", True), "weighted_resume", "weighted_single", stop_after=6,
                               out_dir=out / "source15", run_dir=runs, trace_file=trace / "weighted_resume.jsonl")
        assert boundary["core_called"] is False and sha(runs / "weighted_resume/last.pt") == last_sha
    result = {"state": "passed", "tiny_epochs": 14, "SGD_steps": 28, "train_exposures": 14 * 456,
              "validation_exposures": 14000, "executed_calls": executed, "trace_proof": trace_proof,
              "disabled_Source12_exact": True, "weighted6_vs2plus4_exact": True, "boundary_reentry_no_core": True,
              "power_thread_request": power, "protected_after": verify(adapter), "formal_started": False,
              "output_locks": {str(path.relative_to(HERE)): sha(path) for path in runs.rglob("*") if path.is_file()}}
    write("preflight_GPU.json", result)
    write("worker.json", {"pid": os.getpid(), "state": "preflight_passed_waits_FORMAL_READY", "formal_started": False})
    return {"state": result["state"], "preflight_GPU_sha256": sha(HERE / "preflight_GPU.json"), "formal_started": False}


def _first12_identity(adapter, folder, trace_path):
    old_folder = OLD / "pilot_runs/pilot_weighted_single_seed42"
    own = checkpoint(adapter, folder)
    previous = checkpoint(adapter, old_folder)
    assert own["epoch"] == 12 and len(own["history"]) == 12
    assert own["config"] == previous["config"] == cfg(adapter, "weighted_single")
    assert own["source_sha256"] == adapter.source_hashes()
    assert previous["source_sha256"] == read(OLD / "source_manifest.json")["training_sources"]
    del own, previous
    compare(adapter, folder, old_folder)
    def rows(path):
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    count = 0
    old_trace = OLD / "pilot_traces/pilot_weighted_single_seed42.jsonl"
    for current, previous in zip_longest(rows(trace_path), rows(old_trace)):
        assert current is not None and previous is not None
        shared_trace_equal((current,), (previous,))
        count += 1
    assert count == 2112
    result = {"state": "passed", "own_Source15_checkpoint_sha256": sha(folder / "last.pt"),
              "reference_SourceSet_checkpoint_sha256": sha(old_folder / "last.pt"),
              "own_first12_trace_sha256": sha(trace_path), "reference_trace_sha256": sha(OLD / "pilot_traces/pilot_weighted_single_seed42.jsonl"),
              "Source15": adapter.source_hashes(), "full_model_optimizer_scheduler_RNG_non_timehistory_exact": True,
              "config_exact_and_both_source_dictionaries_checked": True,
              "all2112_input_q_loss_records_exact": True, "own_from_scratch_case_continues": True,
              "scope": "12-prefix identity only, not a full240 exact reproduction"}
    write("first12_identity.json", result)
    return result


def formal(adapter):
    ready("FORMAL_READY.json", adapter)
    assert not (HERE / "formal_results.json").exists()
    runs, out, trace_path = HERE / "formal_runs", HERE / "formal_out", HERE / "formal_traces" / (FORMAL_NAME + ".jsonl")
    folder = runs / FORMAL_NAME
    existing = checkpoint(adapter, folder)["epoch"] if (folder / "last.pt").exists() else 0
    if existing > 12:
        assert read(HERE / "first12_identity.json")["state"] == "passed"
        assert read(HERE / "first12_identity.json")["Source15"] == adapter.source_hashes()
    began = time.monotonic()
    with awake() as power:
        if existing <= 12:
            write("worker.json", {"pid": os.getpid(), "state": "formal240_first12_boundary", "current_run": FORMAL_NAME,
                                  "start_epoch": existing, "stop_after": 12, "formal240_completed_added": 0})
            adapter.run(cfg(adapter, "weighted_single"), FORMAL_NAME, "weighted_single", stop_after=12,
                        out_dir=out, run_dir=runs, trace_file=trace_path)
            _first12_identity(adapter, folder, trace_path)
            verify(adapter)
        write("worker.json", {"pid": os.getpid(), "state": "formal240_training", "current_run": FORMAL_NAME,
                              "stop_after": 240, "first12_identity_passed": True, "formal240_completed_added": 0})
        adapter.run(cfg(adapter, "weighted_single"), FORMAL_NAME, "weighted_single", stop_after=240,
                    out_dir=out, run_dir=runs, trace_file=trace_path)
        saved = checkpoint(adapter, folder)
        assert saved["epoch"] == len(saved["history"]) == 240 and saved["source_sha256"] == adapter.source_hashes()
        assert saved["config"] == cfg(adapter, "weighted_single")
        for index, row in enumerate(saved["history"]):
            assert row["epoch"] == index + 1 and row["train_samples"] == 45000 and row["validation_samples"] == 5000
            factor = (index + 1) / 5 if index < 5 else .5 * (1 + math.cos(math.pi * (index - 5) / 235))
            assert abs(row["learning_rate"] - .1 * factor) < 1e-12
        trace_proof = audit_trace(trace_path, 240, 176)
    plan = read(HERE / "plan.json"); baseline = plan["baseline_last"][0]; last = saved["history"][-1]
    correct = round(last["validation_accuracy"] * 5000)
    median = statistics.median(row["seconds"] for row in saved["history"])
    quality = {"additional_correct5": correct - baseline["correct"] >= 5,
               "accuracy_delta001": last["validation_accuracy"] - baseline["accuracy"] >= .001 - 1e-12,
               "CE_delta_minus005": last["validation_ce"] - baseline["hard_validation_CE"] <= -.005}
    ratio = median / baseline["median_raw_epoch_seconds"]
    result = {"state": "program_complete_manual_pending", "run_id": FORMAL_NAME, "seed": 42,
              "last240": last, "last_correct": correct, "baseline_last": baseline,
              "quality_gate": quality, "quality_supported": all(quality.values()),
              "resource_gate": {"median_raw_wall_ratio_le1_25": ratio <= 1.25}, "median_wall_ratio": ratio,
              "raw_epoch_seconds": [row["seconds"] for row in saved["history"]], "median_epoch_seconds": median,
              "first12_identity_sha256": sha(HERE / "first12_identity.json"), "trace_proof": trace_proof,
              "last_sha256": sha(folder / "last.pt"), "best_sha256": sha(folder / "best.pt"),
              "history_sha256": sha(folder / "history.csv"), "trace_sha256": sha(trace_path),
              "Source15": adapter.source_hashes(), "power_thread_request": power,
              "train_kd_meaning": "original full KL plus area-weighted singleton-DKD conditional extras; not the original full KL alone",
              "elapsed_monotonic_seconds_including_pauses": time.monotonic() - began,
              "protected_after": verify(adapter), "formal240_completed_added": 1,
              "offline_scientific_visual_power_closure_pending": True, "automatic_seed_test_reproduction": False,
              "overall_goal_achieved": False}
    write("formal_results.json", result)
    write("worker.json", {"pid": os.getpid(), "state": result["state"], "run_id": FORMAL_NAME,
                          "formal240_completed_added": 1, "overall_goal_achieved": False})
    return {"state": result["state"], "formal_results_sha256": sha(HERE / "formal_results.json"), "automatic_expansion": False}


def main(argv=None):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--gpu-preflight", action="store_true")
    group.add_argument("--formal", action="store_true")
    args = parser.parse_args(argv)
    try:
        with exclusive():
            if args.prepare:
                result = prepare()
            else:
                adapter = load_adapter()
                with adapter.runtime():
                    adapter.torch.set_num_threads(1)
                    result = gpu_preflight(adapter) if args.gpu_preflight else formal(adapter)
    except BaseException as exc:
        phase = "prepare" if args.prepare else "preflight" if args.gpu_preflight else "formal"
        write("failure_" + phase + ".json", {"pid": os.getpid(), "state": "technical_failure_no_auto_retry",
                                             "reason": repr(exc), "traceback": traceback.format_exc(), "overall_goal_achieved": False})
        raise
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
