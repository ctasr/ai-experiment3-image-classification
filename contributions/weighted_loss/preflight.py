"""Synthetic CPU-only objective/gradient checks; no models, dataset or CUDA."""
from pathlib import Path
import argparse
import importlib.util
import json
import random
import tempfile

import numpy as np

import torch
from torch.nn import functional as F

from . import adapter, legacy_loss
from .loss import MODES, combined, original_loss_parts


def scope_check(scratch_dir=None):
    """Exercise restoration and loss routing without constructing a model."""
    missing = object()
    bindings = [(module, key, getattr(module, key, missing)) for module, key in adapter.bindings()]
    cpu_rng = torch.get_rng_state().clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cases = []
    parent = Path(scratch_dir).resolve() if scratch_dir is not None else Path(tempfile.gettempdir()).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="weighted_scope_CPU_", dir=parent) as directory:
        base = Path(directory).resolve()
        assert base.parent == parent
        cfg = adapter.read(adapter.HERE / "config.json")
        for mode in MODES:
            cfg_mode = adapter.configuration(cfg, mode, 42, {"CPU_diagnostic": True})
            for fail in (False, True):
                target = base / (mode + ("_failure" if fail else "_success"))
                active = None
                try:
                    identity = lambda: ({}, {"CPU_diagnostic": True})
                    with adapter.scope(cfg_mode, target, base / "data", {}, identity, 0,
                                       target / "trace.jsonl", base / "unused-initialization.pth") as active:
                        assert adapter.CORE.Teacher is not adapter.ORIGINAL_TEACHER
                        assert adapter.ENGINE.ROOT == target
                        assert adapter.ENGINE.source_hashes() == adapter.source_hashes(mode)
                        z = torch.zeros(3, 10, requires_grad=True)
                        y, order = torch.tensor([0, 1, 2]), torch.tensor([2, 0, 1])
                        actual = adapter.CORE.loss_parts(z, y, order, .4, torch.full((3, 10), .1), 1., .25)
                        expected = combined(z, y, order, .4, torch.full((3, 10), .1), 1., .25, mode)[0]
                        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
                        random.random(); np.random.random(); torch.rand(1)
                        if fail:
                            raise RuntimeError("intentional CPU routing failure")
                except RuntimeError as exc:
                    assert fail and str(exc) == "intentional CPU routing failure"
                assert active["all_bindings_restored"] and active["runtime"]["CPU_RNG_threads_flags_restored"]
                for module, key, value in bindings:
                    assert (not hasattr(module, key)) if value is missing else getattr(module, key) is value
                cases.append({"mode": mode, "intentional_exception": fail, "bindings_and_RNG_restored": True})
        owned = base / "owned"
        identity = {"CPU_diagnostic": True, "mode": "confidence_mass"}
        paths = {"cutmix": base / "cutmix.pt", "transfer": base / "transfer.pt"}
        adapter.prepare_output(owned, base / "data", paths, base / "initialization.pth", identity)
        adapter.prepare_output(owned, base / "data", paths, base / "initialization.pth", identity)
        unrelated = base / "unrelated"
        unrelated.mkdir(); (unrelated / "unrelated.txt").write_text("retain", encoding="utf-8")
        try:
            adapter.prepare_output(unrelated, base / "data", paths, base / "initialization.pth", identity)
        except ValueError:
            pass
        else:
            raise AssertionError("Unrelated output directory was accepted")
        trace = base / "committed_trace.jsonl"
        fixture = [{"epoch": 1, "batch_in_epoch": index, "global_batch": index, "batch_size": size}
                   for index, size in ((1, 256), (2, 200))]
        trace.write_text("".join(json.dumps(row) + "\n" for row in fixture), encoding="utf-8")
        assert adapter.check_trace(trace, 1, {"pilot_samples": 456, "batch_size": 256})["records"] == 2
        with trace.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": 2, "batch_in_epoch": 1, "global_batch": 3, "batch_size": 256}) + "\n")
        try:
            adapter.check_trace(trace, 1, {"pilot_samples": 456, "batch_size": 256})
        except ValueError:
            pass
        else:
            raise AssertionError("Uncommitted partial trace was accepted")
    assert torch.equal(cpu_rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    assert random.getstate() == python_rng
    restored_numpy = np.random.get_state()
    assert numpy_rng[0] == restored_numpy[0] and np.array_equal(numpy_rng[1], restored_numpy[1]) and numpy_rng[2:] == restored_numpy[2:]
    return {"state": "passed", "cases": cases, "model_forwards": 0, "CUDA_initialized": False,
            "externally_verified_python_numpy_torch_RNG": True,
            "unrelated_output_rejected": True, "partial_uncommitted_trace_rejected": True,
            "final_batch_200_accounting_checked": True}


def manual_extra(logits, labels, order, area, q, temperature):
    """Independent double-precision expression, not the copied implementation."""
    z = logits.double() / temperature
    q = q.double()
    result = z.sum(1) * 0
    for weight, classes in ((area, labels), (1 - area, labels[order])):
        if not weight:
            continue
        selected = torch.ones_like(z, dtype=torch.bool).scatter_(1, classes[:, None], False)
        nq = q.masked_fill(~selected, 0)
        cq = nq / nq.sum(1, keepdim=True)
        logp = z - torch.logsumexp(z.masked_fill(~selected, -torch.inf), dim=1, keepdim=True)
        terms = torch.special.xlogy(cq, cq) - cq * logp
        result = result + weight * q.gather(1, classes[:, None]).squeeze(1) * terms.sum(1)
    return result.mean() * temperature ** 2


def check(legacy_loss_path=None, scratch_dir=None):
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preflight must start without an initialized CUDA context")
    before_rng = torch.get_rng_state().clone()
    generator = torch.Generator(device="cpu").manual_seed(2026112701)
    frozen = legacy_loss
    legacy_external_checked = False
    if legacy_loss_path is not None:
        legacy_loss_path = Path(legacy_loss_path)
        if adapter.sha(legacy_loss_path) != adapter.provenance()["original_loss_sha256"]:
            raise ValueError("Explicit legacy loss file does not have the frozen SHA")
        spec = importlib.util.spec_from_file_location("explicit_legacy_loss_for_CPU_audit", legacy_loss_path)
        frozen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(frozen)
        legacy_external_checked = True
    cases = []
    for name, area, temperature in (("two_sources", .35, 1.), ("same_label", .35, 1.),
                                   ("self_order", .7, 1.), ("zero_i", 0., 1.), ("zero_j", 1., 1.),
                                   ("extreme", .42, 1.), ("temperature_diagnostic", .35, 2.)):
        raw = torch.randn((4, 10), generator=generator, device="cpu")
        q = torch.randn((4, 10), generator=generator, device="cpu").softmax(1)
        labels, order = torch.tensor([0, 1, 2, 3]), torch.tensor([3, 2, 1, 0])
        if name == "same_label":
            labels.fill_(2)
        if name == "self_order":
            order = torch.arange(4)
        if name == "extreme":
            raw = raw * 500
            q = torch.tensor([[1e-30] * 9 + [1.]] * 4, dtype=torch.float32)
        q_before, labels_before, order_before = q.clone(), labels.clone(), order.clone()
        row = {"name": name}
        for mode in MODES:
            a, b = raw.clone().requires_grad_(), raw.clone().requires_grad_()
            actual, extra, details, original = combined(a, labels, order, area, q, temperature, .25, mode)
            baseline = original_loss_parts(b, labels, order, area, q, temperature, .25)
            if mode == "weighted_single":
                expected_extra, _ = frozen.extras(b, labels, order, area, q, temperature, "weighted_single")
                expected = (baseline[0] + .25 * expected_extra, baseline[1], baseline[2] + expected_extra)
            elif mode == "confidence_mass":
                coefficient = 1 + (area * q[torch.arange(4), labels] + (1 - area) * q[torch.arange(4), labels[order]])
                per_row = F.kl_div(F.log_softmax(b.float() / temperature, 1), q, reduction="none").sum(1) * temperature ** 2
                kd = (coefficient * per_row).mean()
                expected = (.75 * baseline[1] + .25 * kd, baseline[1], kd)
                assert bool((coefficient >= 1).all()) and bool((coefficient <= 2 + 1e-6).all())
                assert torch.equal(details["coefficient"], coefficient)
            else:
                expected = baseline
                assert actual is original
            assert all(torch.equal(x, y) for x, y in zip(actual, expected)), (name, mode)
            grad_a = torch.autograd.grad(actual[0], a)[0]
            grad_b = torch.autograd.grad(expected[0], b)[0]
            assert torch.equal(grad_a, grad_b) and bool(torch.isfinite(grad_a).all()), (name, mode)
            assert all(bool(torch.isfinite(value).all()) for value in actual)
            row[mode] = {"loss_exact": True, "gradient_exact": True}
        a = raw.clone().requires_grad_()
        frozen_extra, _ = legacy_loss.extras(a, labels, order, area, q, temperature, "weighted_single")
        reference = manual_extra(a, labels, order, area, q, temperature)
        torch.testing.assert_close(frozen_extra.double(), reference, rtol=2e-5, atol=2e-5)
        g1 = torch.autograd.grad(frozen_extra, a)[0]
        g2 = torch.autograd.grad(reference, a)[0]
        torch.testing.assert_close(g1, g2, rtol=2e-5, atol=2e-6)
        assert torch.equal(q, q_before) and torch.equal(labels, labels_before) and torch.equal(order, order_before)
        row["independent_FP64_conditional_expression_close"] = True
        row["target_labels_order_unmodified"] = True
        cases.append(row)
    # A nonconstant confidence coefficient must be applied before batch mean.
    z = torch.tensor([[3., 0., -1.], [-2., 1., 4.]], requires_grad=True)
    q = torch.tensor([[.8, .1, .1], [.15, .7, .15]])
    result, _, details, original = combined(z, torch.tensor([0, 1]), torch.tensor([1, 0]), .3, q, 1., .25, "confidence_mass")
    wrong = details["coefficient"].mean() * original[2]
    assert not torch.isclose(result[2], wrong, rtol=1e-6, atol=1e-6)
    # Exact zero complement mass is not silently clamped/rescued. Preserve the
    # legacy boundary behavior; the training adapter rejects nonfinite output.
    q_zero = torch.tensor([[1.] + [0.] * 9])
    edge, _ = legacy_loss.extras(torch.zeros(1, 10), torch.tensor([0]), torch.tensor([0]), 1., q_zero, 1., "weighted_single")
    assert not bool(torch.isfinite(edge))
    for area in (-.01, 1.01):
        try:
            combined(torch.zeros(1, 10), torch.tensor([0]), torch.tensor([0]), area, torch.full((1, 10), .1), 1., .25, "confidence_mass")
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid lambda was accepted")
    assert torch.equal(before_rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    sources = {mode: adapter.source_hashes(mode) for mode in MODES}
    assert sources["weighted_single"] != sources["confidence_mass"]
    route = scope_check(scratch_dir)
    return {"state": "passed", "protocol": adapter.PROTOCOL, "CPU_only": True, "CUDA_initialized": False,
            "model_forwards": 0, "dataset_reads": 0, "case_count": len(cases), "cases": cases,
            "legacy_loss_sha256": adapter.sha(Path(legacy_loss.__file__)),
            "explicit_external_legacy_file_compared": legacy_external_checked,
            "per_image_weight_before_mean_checked": True,
            "zero_complement_legacy_nonfinite_behavior_preserved": True,
            "global_CPU_RNG_unchanged": True, "source_sha256_by_mode": sources,
            "CPU_scope_routing_restore": route,
            "confidence_mass_scope": "Written scalar coefficient-mass control only; gradient norms are not matched",
            "GPU_loop_resume_teacher_identity_preflight_pending": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-loss", type=Path, help="Optional explicit old loss file for CPU equality audit only")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scratch-dir", type=Path)
    args = parser.parse_args(argv)
    result = check(args.legacy_loss, args.scratch_dir or (args.output.parent if args.output else None))
    if args.output:
        adapter.write(args.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
