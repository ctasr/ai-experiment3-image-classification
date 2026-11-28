"""CPU synthetic/cache checks and actual same-layout GPU bridge gates."""
from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import protocol as p, engine
from .legacy_math import analytic_maps, cpu_autograd


def array_exact(actual, reference, name):
    if actual.dtype != reference.dtype or actual.shape != reference.shape or np.ascontiguousarray(actual).tobytes() != np.ascontiguousarray(reference).tobytes():
        difference = float(np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64)))) if actual.shape == reference.shape else None
        raise AssertionError(f"Original same-layout cache differs: {name}; max_abs={difference}")


def bridge_rows(legacy):
    rows = set(range(0, 200, 20)) | set(np.argsort(legacy[42]["indices"])[:8].tolist()) | set(range(4))
    for seed in p.SEEDS:
        rows.update(np.flatnonzero(legacy[seed]["flat"]).tolist())
    return sorted(rows)


def cpu(args):
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU checks require an uninitialized CUDA context")
    torch_rng, numpy_rng, python_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    identity, arrays = p.prepare(args)
    legacy, clean = p.references(args.reference_dir)
    for seed in p.SEEDS:
        ref = legacy[seed]
        drops = engine.fixed_target_drops(ref["q"], ref["maskedq"], ref["target"])
        array_exact(drops[:, 0] - drops[:, 1:].mean(1), ref["dq"], "original200_dq_seed" + str(seed))
    generator = torch.Generator(device="cpu").manual_seed(47211)
    proofs = []
    with p.preserve_runtime():
        torch.set_num_threads(1)
        for seed in p.SEEDS:
            saved = torch.load(getattr(args, f"weights_{seed}"), map_location="cpu", weights_only=False)
            if saved["source_sha256"] != p.provenance()["model_training_sources"] or saved["config"]["seed"] != seed:
                raise ValueError("Frozen head checkpoint provenance differs")
            weight, bias = saved["model"]["classifier.weight"], saved["model"]["classifier.bias"]
            f0 = torch.randn((4, 192, 3, 4), generator=generator) * .2
            f1 = torch.randn((4, 192, 3, 4), generator=generator) * .2
            target = torch.tensor([0, 2, 5, 9])
            proof = cpu_autograd(f0, f1, weight, bias, target)
            z0, z1 = F.linear(f0.mean((2, 3)), weight, bias), F.linear(f1.mean((2, 3)), weight, bias)
            first = analytic_maps(f0, f1, weight, z0.softmax(1), z1.softmax(1), target)
            swapped = analytic_maps(f1, f0, weight, z1.softmax(1), z0.softmax(1), target)
            torch.testing.assert_close(first[4].flip(-1), swapped[4], atol=2e-8, rtol=2e-5)
            zero = analytic_maps(torch.zeros_like(f0), torch.zeros_like(f1), weight, z0.softmax(1), z1.softmax(1), target)
            if not bool(zero[6].all()) or not bool((zero[7] == 0).all()):
                raise AssertionError("Flat-map first row-major tie behavior changed")
            proofs.append({"seed": seed, "synthetic_feature_CPU_autograd": proof,
                           "signed_flip_swap_checked": True, "flat_first_window_checked": True})
        stats = p.read(Path(args.data_dir) / "normalization.json")
        raw = torch.arange(3 * 32 * 32).remainder(256).to(torch.uint8).reshape(1, 3, 32, 32)
        image = p.normalise(raw, stats)[0]
        locations = [[0, 0], [24, 24], [12, 3], *[[1, 2]] * 8]
        variants = engine.variants_for_image(image, locations)
        for variant, (yy, xx) in zip(variants, locations):
            reference = image.clone(); reference[:, yy:yy+8, xx:xx+8] = 0
            if not torch.equal(variant, reference) or int((variant == 0).sum()) != 192:
                raise AssertionError("Normalized mean-fill mask footprint differs")
            if not torch.equal(variant.flip(-1).flip(-1), variant):
                raise AssertionError("Mask must be applied before generating the flip branch")
        # Image-major fixed target extraction agrees with an explicit loop.
        fixture = np.arange(4 * 11 * 10, dtype=np.float32).reshape(4, 11, 10)
        target = np.array([9, 1, 7, 3])
        vector = np.take_along_axis(fixture, np.broadcast_to(target[:, None, None], (4, 11, 1)), axis=2)[:, :, 0]
        array_exact(vector, np.stack([fixture[row, :, target[row]] for row in range(4)]), "target_extraction")
    if not torch.equal(torch_rng, torch.get_rng_state()) or random.getstate() != python_rng:
        raise AssertionError("CPU preparation/check changed caller global RNG")
    current_numpy = np.random.get_state()
    if numpy_rng[0] != current_numpy[0] or not np.array_equal(numpy_rng[1], current_numpy[1]) or numpy_rng[2:] != current_numpy[2:]:
        raise AssertionError("Private windows changed global NumPy RNG")
    result = {"kind": "CPU_synthetic_and_original_cache_protocol", "passed": True,
              "identity_sha256": p.sha(Path(args.output_dir) / "identity.json"),
              "design_sha256": p.sha(Path(args.output_dir) / "frozen_design.npz"),
              "source_sha256": identity["source_sha256"], "checker_sha256": p.sha(__file__),
              "head_proofs": proofs, "actual_CNN_forward_images": 0, "CUDA_initialized": False,
              "global_python_numpy_torch_RNG_unchanged": True, "mask_mean_fill_and_order_checked": True,
              "original200_random_window_prefix_exact": True, "all5000_targets_locked_from_original_cache": True,
              "actual_portable_metric_function_original200_dq_exact_all_seeds": True,
              "old200_target_vs_full5000_disagreements": 0,
              "old200_vs_old5000_probability_max_abs_descriptive": float(np.abs(legacy[42]["q"] - clean["clean_q"][[int(np.flatnonzero(clean["official_train_indices"] == value)[0]) for value in legacy[42]["indices"]]]).max()),
              "different_FP32_batch_layouts_not_claimed_bitwise_equal": True,
              "GPU_semantic_rows_in_original200": bridge_rows(legacy),
              "actual_GPU_original200_cache_and_semantic_bridge_pending": True}
    p.write(Path(args.output_dir) / "CPU_preflight.json", result)
    return result


def original200_exact(data, reference, indices, labels, target):
    for name in ("p0", "p1", "q", "map0", "map1", "heat", "loc", "flat", "positions", "maskedq", "dq"):
        array_exact(data[name], reference[name], name)
    for name, actual in (("indices", indices), ("labels", labels), ("target", target)):
        array_exact(actual, reference[name], name)
    return {"exact_fields": ["p0", "p1", "q", "map0", "map1", "heat", "loc", "flat", "positions", "maskedq", "dq", "indices", "labels", "target"],
            "comparison": "dtype/shape and canonical array bytes, original minibatch geometry retained"}


def gpu_semantics(model, stats, raw, target, data, features, rows):
    """Same per-image semantics with declared FP32 tolerance, no GPU autograd."""
    before = p.state_hash(model); device = next(model.parameters()).device
    started = time.monotonic()
    maximum, forwards = {}, 0
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        for row in rows:
            x = p.normalise(raw[row:row+1].to(device), stats)
            a, b, q, f0, f1, *_ = engine.branches(model, x, True); forwards += 2
            c = torch.tensor([int(target[row])], dtype=torch.long, device=device)
            maps = analytic_maps(f0, f1, model.classifier.weight, a, b, c)
            for name, value in (("p0", a), ("p1", b), ("q", q), ("map0", maps[2]), ("map1", maps[3]), ("heat", maps[5])):
                actual = value.cpu().numpy()[0]; reference = data[name][row]
                maximum[name] = max(maximum.get(name, 0.), float(np.abs(actual - reference).max()))
                if not np.allclose(actual, reference, atol=2e-5, rtol=2e-5):
                    raise AssertionError("Single-image FP32 semantic comparison failed: " + name)
            if int(maps[7][0]) != int(data["loc"][row]) or bool(maps[6][0]) != bool(data["flat"][row]):
                raise AssertionError("Single-image highest-window/flat behavior differs; do not run5000")
            variants = torch.stack(engine.variants_for_image(x[0], data["positions"][row]))
            _, _, masked, *_ = engine.branches(model, variants); forwards += 22
            reference = data["maskedq"][row]
            if not np.allclose(masked.cpu().numpy(), reference, atol=2e-5, rtol=2e-5):
                raise AssertionError("Single-image occlusion batching semantics differ")
    chosen = rows[:4]
    f0 = torch.tensor(np.stack([features[row]["feature0"] for row in chosen]))
    f1 = torch.tensor(np.stack([features[row]["feature1"] for row in chosen]))
    cpu_proof = cpu_autograd(f0, f1, model.classifier.weight.detach().cpu().clone(),
                             model.classifier.bias.detach().cpu().clone(), torch.as_tensor(target[chosen], dtype=torch.long))
    p.assert_frozen(model, before)
    return {"passed": True, "rows": rows, "single_image_FP32_atol": 2e-5, "single_image_FP32_rtol": 2e-5,
            "max_abs_differences": maximum, "highest_windows_and_flat_flags_exact": True,
            "different_batch_probabilities_bitwise_claimed": False,
            "actual_GPU_feature_CPU_autograd": cpu_proof,
            "GPU_autograd": 0, "additional_CNN_forward_images": forwards,
            "wall_seconds_including_CPU_autograd_and_pauses": time.monotonic() - started,
            "weights_BN_gradients_unchanged": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); p.add_inputs(parser)
    args = parser.parse_args(argv)
    result = cpu(args)
    print(json.dumps({"passed": True, "kind": result["kind"], "CUDA_initialized": False}), flush=True)


if __name__ == "__main__":
    main()
