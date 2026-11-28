"""Isolated output routing around the unchanged original Source10 trainer.

No training mathematics, scheduler, teacher, or random stream is reimplemented.
This package must be entered in a fresh process, without later training packages.
"""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import random
import sys

import numpy as np
import torch

from data import ROOT
import aligned_distillation.train as CORE
import aligned_distillation.teacher as TEACHER_MODULE

ENGINE = CORE.engine
OUT = ROOT / "results_alignment_confirmation"
RUNS = ROOT / "runs/alignment_confirmation"
FIG = ROOT / "figures_alignment_confirmation"
CONFIGS = ROOT / "configs_alignment_confirmation"
DECISION = ROOT.parent / "tmp/experiment3_innovation_screening_20261125/ab_confirmation_root_decision.json"
DECISION_SHA = "8bc88a517aaa2abb1ed59241d3b611f42e36d1a68f94b553255f86e355f97d92"
SEEDS = (17, 73)
OLD_RUNS = {
    42: {"A": "runs/aligned_distillation/aligned_crop_mix_seed42", "B": "runs/aligned_distillation/aligned_mixed_view_seed42"},
    17: {"B": "runs/aligned_distillation/repeat_mixed_view_seed17"},
    73: {"B": "runs/aligned_distillation/repeat_mixed_view_seed73"},
}
ANALYSIS_FILES = ("alignment_confirmation/common.py", "alignment_confirmation/evaluate.py", "alignment_confirmation/plot.py", "alignment_confirmation/run.py")
FORBIDDEN_PREFIXES = ("capacity_study", "horizon_study", "relative_progress", "relative_progress_fast", "relative_progress_runtime", "relative_progress_seed17", "noise_augmentation", "noise_augmentation_study", "online_ema", "clean_anchor_consistency")


def now():
    return datetime.now().astimezone().isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    # New metadata can only be written inside this stage's named directories.
    allowed = (OUT.resolve(), RUNS.resolve(), FIG.resolve(), CONFIGS.resolve())
    if not any(path.resolve().is_relative_to(base) for base in allowed):
        raise ValueError(f"Output outside isolated stage: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def training_source_hashes():
    expected = read(ROOT / "results_aligned_distillation/source_manifest.json")["source_sha256"]
    assert len(expected) == 10 and tuple(expected) == tuple(CORE.SOURCES)
    actual = {name: sha(ROOT / name) for name in expected}
    assert actual == expected, "Original Source10 changed"
    return actual


def analysis_source_hashes():
    return {**training_source_hashes(), **{name: sha(ROOT / name) for name in ANALYSIS_FILES}}


def new_run_id(seed):
    assert seed in SEEDS
    return f"confirm_crop_mix_seed{seed}"


def a_config(seed):
    assert seed in (42, 17, 73)
    cfg = read(ROOT / "configs_aligned_distillation/aligned_crop_mix_seed42.json")
    cfg["seed"] = seed
    return cfg


def assert_original_context():
    polluted = [name for name in sys.modules if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES)]
    assert not polluted, f"Fresh isolated process required; polluted modules: {polluted}"
    assert tuple(CORE.BASE_SOURCES) == ("train.py", "data.py", "models.py", "prepare_data.py", "extensions/train.py", "extensions/models.py")
    assert CORE.BASE_MIXED.__module__ == "extensions.train"
    assert ENGINE.mixed_epoch is CORE.BASE_MIXED
    assert tuple(ENGINE.SOURCES) == tuple(CORE.BASE_SOURCES)
    assert ENGINE.DIRECTORY == ROOT / "runs/round2"
    assert CORE.DIRECTORY == ROOT / "runs/aligned_distillation"
    assert CORE.OUT == TEACHER_MODULE.OUT == ROOT / "results_aligned_distillation"
    assert CORE.Teacher is TEACHER_MODULE.Teacher
    assert torch.utils.deterministic.fill_uninitialized_memory is True, "Use original default, never inherit runtime fill=False"


@contextmanager
def preserve_runtime():
    """Restore RNG/flags around analysis and isolated calls, including errors.

    CUDA may initialize during an explicit evaluation; it cannot be uninitialized.
    If it was already initialized, all device RNG states are restored exactly.
    The next original training call seeds again in its unchanged original order.
    """
    initialized = torch.cuda.is_initialized()
    saved = {
        "python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if initialized else None,
        "threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "fill": torch.utils.deterministic.fill_uninitialized_memory,
    }
    try:
        yield
    finally:
        random.setstate(saved["python"])
        np.random.set_state(saved["numpy"])
        torch.set_rng_state(saved["cpu"])
        if saved["cuda"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda"])
        torch.set_num_threads(saved["threads"])
        torch.use_deterministic_algorithms(saved["deterministic"], warn_only=saved["warn_only"])
        torch.backends.cudnn.benchmark = saved["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = saved["cudnn_deterministic"]
        torch.backends.cuda.matmul.allow_tf32 = saved["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = saved["cudnn_tf32"]
        torch.utils.deterministic.fill_uninitialized_memory = saved["fill"]


@contextmanager
def training_scope():
    assert_original_context()
    bindings = [(CORE, name) for name in ("DIRECTORY", "OUT", "ACTIVE", "TEACHER")]
    bindings += [(TEACHER_MODULE, "OUT")]
    bindings += [(ENGINE, name) for name in ("DIRECTORY", "SOURCES", "mixed_epoch")]
    saved = [(module, name, getattr(module, name)) for module, name in bindings]
    with preserve_runtime():
        try:
            CORE.DIRECTORY, CORE.OUT = RUNS, OUT
            TEACHER_MODULE.OUT = OUT
            yield
        finally:
            # Release a newly allocated Teacher reference, never mutate old weights.
            for module, name, value in reversed(saved):
                setattr(module, name, value)


def run_legacy(seed):
    cfg = a_config(seed)
    with training_scope():
        # No random operation is inserted between original seed/Teacher/engine calls.
        return CORE.run(cfg, new_run_id(seed))


def protected_paths():
    old = read(ROOT.parent / "tmp/experiment3_innovation_screening_20261125/finite_step_acceptance_pilot_v2/plan.json")
    protected = dict(old["protected_project_inputs"])
    additional = ["results_aligned_distillation/source_manifest.json", "results_aligned_distillation/teacher_identity.json", "results_aligned_distillation/preflight.json", "results_aligned_distillation/selection_before_repeats.json", "results_aligned_distillation/validation_confirmation.json", "results_relative_progress_seed17/reference_vs_zero.json", "results_relative_progress_seed17/stage_closed.json", "results_relative_progress_seed17/results.json", "results_relative_progress_seed17/source_manifest.json"]
    for cases in OLD_RUNS.values():
        for folder in cases.values():
            additional += [f"{folder}/{name}" for name in ("best.pt", "last.pt", "config.json", "history.csv", "result.json", "environment.json", "objective_metadata.json", "teacher_check.json")]
    additional += [f"configs_aligned_distillation/{name}.json" for name in ("aligned_crop_mix_seed42", "aligned_mixed_view_seed42", "repeat_mixed_view_seed17", "repeat_mixed_view_seed73")]
    for name in additional:
        protected[name] = sha(ROOT / name)
    # Check inherited locks before recording anything new.
    for name, expected in old["protected_project_inputs"].items():
        assert sha(ROOT / name) == expected, f"Inherited protected input changed: {name}"
    return protected


def verify_protected():
    manifest = read(OUT / "source_manifest.json")
    for name, expected in manifest["protected_inputs"].items():
        assert sha(ROOT / name) == expected, f"Protected input changed: {name}"
    for name, expected in manifest["protected_private_inputs"].items():
        assert sha(Path(name)) == expected, f"Protected private decision changed: {name}"
    assert sha(OUT / "plan.json") == manifest["plan_sha256"]
    assert analysis_source_hashes() == manifest["evaluation_sources"]
    assert training_source_hashes() == manifest["source_sha256"]
    assert sha(ROOT / "alignment_confirmation/check.py") == manifest["separate_CPU_checker_sha256"]
    assert sha(ROOT / "alignment_confirmation/__init__.py") == manifest["package_entry_sha256"]
    assert read(OUT / "teacher_identity.json") == manifest["teacher_identity"]
    assert CORE.identity()[1] == manifest["teacher_identity"]
    for seed in SEEDS:
        assert read(CONFIGS / f"{new_run_id(seed)}.json") == a_config(seed)
    new_locks = verify_new_run_locks()
    return {"passed": True, "protected_count": len(manifest["protected_inputs"]), "private_count": len(manifest["protected_private_inputs"]), "completed_new_controls_locked": new_locks, "checked_at": now()}


def verify_new_run_locks():
    path = OUT / "training_audits.json"
    if not path.exists():
        return []
    audits = read(path)["new_runs"]
    checked = []
    for seed, audit in audits.items():
        assert int(seed) in SEEDS
        folder = RUNS / new_run_id(int(seed))
        assert audit["passed"] and audit["config"] == a_config(int(seed))
        for name, expected in audit["weight_locks"].items():
            assert name in ("best.pt", "last.pt") and sha(folder / name) == expected, f"New control weight changed: {seed}/{name}"
        assert set(audit["weight_locks"]) == {"best.pt", "last.pt"}
        assert sha(folder / "history.csv") == audit["history_sha256"]
        checked.append(int(seed))
    return sorted(checked)


def audit_run(folder, expected_cfg=None):
    """CPU-only audit of actual saved160 epochs and the original best selection."""
    folder = Path(folder)
    cfg = read(folder / "config.json")
    if expected_cfg is not None:
        assert cfg == expected_cfg
    with (folder / "history.csv").open(encoding="utf-8-sig", newline="") as stream:
        history = [{key: (int(value) if key in ("epoch", "train_samples", "validation_samples") else float(value)) for key, value in row.items()} for row in csv.DictReader(stream)]
    assert len(history) == 160 and [row["epoch"] for row in history] == list(range(1, 161))
    assert all(row["train_samples"] == 45000 and row["validation_samples"] == 5000 for row in history)
    assert all(math.isfinite(value) for row in history for value in row.values())
    assert all(row["seconds"] > 0 for row in history)
    # Exact LambdaLR factors used by original extensions.train.run.
    for index, row in enumerate(history):
        factor = (index + 1) / 5 if index < 5 else .5 * (1 + math.cos(math.pi * min(index - 5, 155) / 155))
        assert abs(row["learning_rate"] - cfg["learning_rate"] * factor) < 1e-14
    result = read(folder / "result.json")
    assert result["state"] == "completed" and result["epochs_completed"] == 160 and result["config"] == cfg
    chosen = max(history, key=lambda row: (row["validation_accuracy"], -row["validation_ce"]))
    assert result["best_epoch"] == chosen["epoch"]
    assert result["best_validation_accuracy"] == chosen["validation_accuracy"]
    assert result["best_validation_ce"] == chosen["validation_ce"]
    sources = training_source_hashes()
    env = read(folder / "environment.json")
    assert env["source_sha256"] == sources
    assert read(folder / "teacher_check.json")["weights_bn_gradients_unchanged"] is True
    objective = read(folder / "objective_metadata.json")
    assert objective["teacher_identity"] == read(OUT / "teacher_identity.json")
    assert objective["alignment"] == cfg["alignment"]
    stats = read(ROOT / "data/normalization.json")
    locks = {}
    for name in ("best.pt", "last.pt"):
        checkpoint = torch.load(folder / name, map_location="cpu", weights_only=False)
        assert checkpoint["config"] == cfg and checkpoint["source_sha256"] == sources and checkpoint["stats"] == stats
        assert checkpoint["best_accuracy"] == chosen["validation_accuracy"] and checkpoint["best_ce"] == chosen["validation_ce"]
        if name == "best.pt":
            assert checkpoint["epoch"] == chosen["epoch"]
        else:
            assert checkpoint["epoch"] == 160 and checkpoint["history"] == history
            assert set(("optimizer", "scheduler", "rng")).issubset(checkpoint)
        locks[name] = sha(folder / name)
        del checkpoint
    return {"passed": True, "run_id": folder.name, "config": cfg, "source_count": 10, "selection": {"best_epoch": chosen["epoch"], "best_validation_accuracy": chosen["validation_accuracy"], "best_validation_ce": chosen["validation_ce"]}, "weight_locks": locks, "history_sha256": sha(folder / "history.csv"), "history_epochs": 160, "train_batches_each": 176, "train_last_batch": 200, "validation_batches_each": 20, "validation_last_batch": 136, "all_learning_rates_match_original": True, "training_accuracy": "CutMix actual-area weighted hits; not raw fit", "teacher_per_image_views": 4, "teacher_forward_calls_not_assumed_four_per_batch": True}


def prepare():
    assert_original_context()
    assert sha(DECISION) == DECISION_SHA
    decision = read(DECISION)
    assert decision["new_training_seeds"] == [17, 73] and decision["both_planned_runs_unconditional_on_first_result"]
    if (OUT / "source_manifest.json").exists():
        return verify_protected()
    OUT.mkdir(parents=True, exist_ok=True)
    sources = training_source_hashes()
    old_preflight = read(ROOT / "results_aligned_distillation/preflight.json")
    assert old_preflight["state"] == "passed" and old_preflight["source_sha256"] == sources
    teacher = read(ROOT / "results_aligned_distillation/teacher_identity.json")
    assert teacher == CORE.identity()[1] and teacher["calibrated"] is False
    write(OUT / "teacher_identity.json", teacher)
    for seed in SEEDS:
        cfg = a_config(seed)
        b = read(ROOT / OLD_RUNS[seed]["B"] / "config.json")
        assert {k: v for k, v in cfg.items() if k != "alignment"} == {k: v for k, v in b.items() if k != "alignment"}
        assert cfg["alignment"] == "crop_mix" and b["alignment"] == "mixed_view"
        write(CONFIGS / f"{new_run_id(seed)}.json", cfg)
    plan = {
        "created_at": now(), "stage": "alignment_confirmation", "state": "prepared_no_READY_no_GPU",
        "root_decision_sha256": DECISION_SHA,
        "purpose": "Complete two missing crop_mix controls for the original160-epoch A/B teacher-target comparison; existing design confirmation, not a new algorithm or accuracy candidate",
        "fixed_order": list(SEEDS), "epochs_each": 160, "both_runs_unconditional": True,
        "run_ids": [new_run_id(seed) for seed in SEEDS], "new_formal_training_count": 2,
        "configuration": "Original A42 config with seed alone replaced by17/73; original Source10 and defaults unchanged",
        "training_core": "aligned_distillation.train.run, byte-for-byte original; new output routing only",
        "training_sources": sources, "teacher_identity": teacher,
        "legacy_preflight_reused": {"file": "results_aligned_distillation/preflight.json", "sha256": sha(ROOT / "results_aligned_distillation/preflight.json"), "reason": "Unchanged mathematics/source; new CPU check covers output context and restoration, no new GPU pilot"},
        "comparison": {"A": "Current-crop teacher probabilities mixed by actual CutMix area", "B": "Teacher query on actual student CutMix input", "existing_runs": OLD_RUNS, "only_factor": "alignment", "teacher": "same fixed seed42 uncalibrated four-view ensemble; indirect ImageNet knowledge retained"},
        "evaluation": {"split": "original5000 validation images only", "batch_size": 256, "batch_count": 20, "last_batch": 136, "precision": "original BF16 autocast", "view": "single unaugmented raw view", "selection": "Each original160history maximum accuracy, then lower hard-label CE; first tied row", "replay_accuracy_tolerance": 0, "replay_ce_tolerance": 1e-6, "no_new_model_selection": True},
        "final_conclusion_screen": {"mean_B_minus_A_accuracy_min": .001, "nonnegative_accuracy_seed_count_min": 2, "mean_B_minus_A_ce_max": 0, "seed_count": 3, "conclusion_only": True, "cannot_cancel_seed73": True, "cannot_authorize_test_reproduction_or_promotion": True},
        "intervals": {"kind": "descriptive paired stratified bootstrap conditional on locked models", "resamples": 2000, "rng": "private PCG64 fixed seed", "shared_image_indices_across_seeds": True, "unit": "5000 common validation images, never15000 independent", "no_retraining_or_selection_uncertainty": True},
        "retention": "All seeds/signs/outcomes/raw timing/pauses retained; failed earlier gates unchanged",
        "cost": {"effective_training_estimate_minutes": 85.1, "whole_flow_estimate_hours": [1.5, 2], "sleep_extra": True, "basis": "Original A42 median15.9535 sec/epoch times320", "not_a_new_global_invention": True, "positive_result_guaranteed": False},
        "new_official_test_images": 0, "new_retired_external_images": 0, "new_B_training": 0, "full_reproduction": 0,
        "report_GitHub_frozen": True, "notification_complete_requires_manual_and_power_closure": True,
    }
    write(OUT / "plan.json", plan)
    protected = protected_paths()
    private = {str(DECISION): DECISION_SHA}
    base = ROOT.parent / "tmp/experiment3_innovation_screening_20261125"
    for name in ("finite_step_probe/results.json", "finite_step_acceptance_pilot_v2/results.json", "finite_step_acceptance_pilot_v2/stage_closed.json", "next_action_after_finite_failure.json"):
        path = base / name
        assert path.exists()
        private[str(path)] = sha(path)
    manifest = {"at": now(), "source_sha256": sources, "evaluation_sources": analysis_source_hashes(), "teacher_identity": teacher, "plan_sha256": sha(OUT / "plan.json"), "protected_inputs": protected, "protected_private_inputs": private, "analysis_source_count": 14, "training_source_count": 10, "separate_CPU_checker_sha256": sha(ROOT / "alignment_confirmation/check.py"), "package_entry_sha256": sha(ROOT / "alignment_confirmation/__init__.py"), "state": "prepared_requires_independent_review_and_READY"}
    write(OUT / "source_manifest.json", manifest)
    return verify_protected()
