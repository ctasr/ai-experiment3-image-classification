"""Execute both prespecified missing controls; only root may create READY."""
import argparse
import os
import time
import traceback

from alignment_confirmation import common as c


def state(value, **details):
    row = {"at": c.now(), "pid": os.getpid(), "state": value, **details}
    c.write(c.OUT / "worker.json", row)
    c.write(c.OUT / "progress.json", row)
    return row


def require_ready():
    ready = c.read(c.OUT / "READY.json")
    assert ready["approved"] is True
    assert ready["plan_sha256"] == c.sha(c.OUT / "plan.json")
    assert ready["source_manifest_sha256"] == c.sha(c.OUT / "source_manifest.json")
    assert ready["preflight_cpu_sha256"] == c.sha(c.OUT / "preflight_cpu.json")
    preflight = c.read(c.OUT / "preflight_cpu.json")
    assert preflight["state"] == "passed" and preflight["official_test_loaded"] is False
    assert c.sha(c.ROOT / "alignment_confirmation/check.py") == c.read(c.OUT / "source_manifest.json")["separate_CPU_checker_sha256"]
    assert not (c.ROOT / "PAUSE_ROUND2").exists(), "Original pause marker exists; do not silently delete it"
    c.verify_protected()
    c.assert_original_context()


def execute():
    require_ready()
    import msvcrt
    handle = (c.OUT / "driver.lock").open("a+b")
    handle.seek(0)
    if not handle.read(1):
        handle.write(b"1")
        handle.flush()
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    started = time.monotonic()
    try:
        state("running", current_run=None, fixed_seed_order=list(c.SEEDS), total_new_formal_runs=2)
        audits = {}
        # The17 outcome is never consulted before executing the fixed73 control.
        for index, seed in enumerate(c.SEEDS):
            run_id = c.new_run_id(seed)
            state("training", current_run=run_id, seed=seed, completed_new_formal_runs=index, total_epochs=160)
            result = c.run_legacy(seed)
            if result["state"] != "completed":
                state("attention_required", current_run=run_id, reason="Original engine did not complete; preserve checkpoint and do not advance", result=result)
                return
            c.assert_original_context()
            audit = c.audit_run(c.RUNS / run_id, c.a_config(seed))
            audits[str(seed)] = audit
            c.write(c.OUT / "training_audits.json", {"at": c.now(), "new_runs": audits})
            c.verify_protected()
        state("validation_analysis", current_run=None, completed_new_formal_runs=2, validation_only=True)
        from alignment_confirmation.evaluate import analyze
        from alignment_confirmation.plot import generate
        analysis = analyze(seeds=(42, 17, 73))
        state("rendering_figures", current_run=None, completed_new_formal_runs=2, official_test_images=0)
        generate(analysis)
        proof = c.verify_protected()
        result = {
            "at": c.now(), "state": "program_complete_manual_pending", "completed_new_formal_runs": 2,
            "seeds": list(c.SEEDS), "epochs_each": 160, "both_controls_executed_independent_of_first_sign": True,
            "training_audits_sha256": c.sha(c.OUT / "training_audits.json"),
            "analysis_sha256": c.sha(c.OUT / "validation_analysis.json"),
            "figure_manifest_sha256": c.sha(c.OUT / "figure_manifest.json"),
            "final_conclusion_screen": analysis["gate"], "protected_inputs_after": proof,
            "elapsed_awake_monotonic_seconds_including_long_pauses": time.monotonic() - started,
            "new_official_test_images": 0, "new_B_training": 0, "new_full_reproduction": 0,
            "manual_visual_review_pending": True, "independent_numeric_review_pending": True,
            "power_restore_pending_independent_finisher": True,
            "innovation_or_positive_value_confirmed_by_program_alone": False,
            "report_GitHub_unchanged": True,
            "notification_complete_written": False,
        }
        c.write(c.OUT / "program_complete.json", result)
        state("program_complete_manual_pending", current_run=None, completed_new_formal_runs=2, final_conclusion_screen=analysis["gate"], manual_visual_review_pending=True)
    except BaseException as exc:
        state("failed", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Write new stage metadata andCPU checks only")
    parser.add_argument("--start", action="store_true", help="Requires separate root-reviewed READY")
    args = parser.parse_args()
    if args.prepare == args.start:
        parser.error("Choose exactly one of --prepare or --start")
    if args.prepare:
        print(c.prepare(), flush=True)
        from alignment_confirmation.check import check
        print(check(), flush=True)
    else:
        execute()


if __name__ == "__main__":
    main()
