"""Paired, class-stratified fixed-image analysis after all actual inference."""
from pathlib import Path
import argparse
import json

import numpy as np

from . import protocol as p


def stratified_summary(values, labels):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    labels = np.asarray(labels)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Empty/nonfinite sensitivity cohort")
    rng = np.random.default_rng(42)
    draws = np.zeros((1000, values.shape[1]), dtype=np.float64)
    for label in range(10):
        rows = np.flatnonzero(labels == label)
        if len(rows):
            # Identical sampled image IDs are used for every seed/metric column.
            selected = rows[rng.integers(0, len(rows), size=(1000, len(rows)))]
            draws += values[selected].sum(1) / len(values)
    return {"n": len(values), "mean": values.mean(0).tolist(),
            "conditional_image_ci95": np.quantile(draws, [.025, .975], axis=0).T.tolist(),
            "bootstrap_replicates": 1000, "private_PCG64_seed": 42,
            "resampling": "paired image identities, within true class, shared across all seed columns"}


def read_full(output):
    output = Path(output)
    identity = p.read(output / "identity.json")
    complete = p.read(output / "compute_complete.json")
    if complete["identity_sha256"] != p.sha(output / "identity.json") or complete["source_sha256"] != p.sources():
        raise ValueError("All5000 computation was not completed for the current frozen identity")
    data, receipts = {}, {}
    for seed in p.SEEDS:
        path = output / "full" / f"seed{seed}.npz"
        receipt = p.read(path.with_suffix(".json"))
        if receipt["data_sha256"] != p.sha(path) or receipt["identity_sha256"] != complete["identity_sha256"] or receipt["samples"] != 5000:
            raise ValueError("A full seed cache is incomplete or changed")
        with np.load(path, allow_pickle=False) as archive:
            data[seed] = {name: archive[name] for name in archive.files}
        receipts[seed] = receipt
    for seed in (17, 73):
        for name in ("indices", "labels", "target"):
            if not np.array_equal(data[seed][name], data[42][name]):
                raise ValueError("The three models do not share identical images/labels/target classes")
    if len(np.unique(data[42]["indices"])) != 5000 or not np.array_equal(np.bincount(data[42]["labels"], minlength=10), np.full(10, 500)):
        raise ValueError("Full original5000 image/class accounting differs")
    return identity, data, receipts


def analyze(output):
    output = Path(output)
    identity, data, receipts = read_full(output)
    labels, indices, target = (data[42][name] for name in ("labels", "indices", "target"))
    difference = np.stack([data[seed]["dq"] for seed in p.SEEDS], axis=1).astype(np.float64)
    aggregate = difference.mean(1)
    matrix = np.column_stack((difference, aggregate))
    summary = stratified_summary(matrix, labels)
    correct = target == labels
    cohorts = {}
    for name, rows in (("fixed_target_correct", correct), ("fixed_target_wrong", ~correct),
                       ("original200", np.arange(5000) < 200), ("remaining4800", np.arange(5000) >= 200)):
        cohorts[name] = stratified_summary(matrix[rows], labels[rows]) if rows.any() else None
    classes = [dict(class_id=label, **stratified_summary(matrix[labels == label], labels[labels == label])) for label in range(10)]
    per_seed, tv = [], []
    for column, seed in enumerate(p.SEEDS):
        row = data[seed]; prediction = row["q"].argmax(1)
        per_seed.append({"seed": seed, "n": 5000, "mean_high_minus_random": float(difference[:, column].mean()),
                         "high_drop_negative_count": int((row["high_drop"] < 0).sum()),
                         "high_minus_random_negative_count": int((row["dq"] < 0).sum()),
                         "flat_count": int(row["flat"].sum()), "flat_official_indices": indices[row["flat"]].tolist(),
                         "own_prediction_accuracy": float((prediction == labels).mean()),
                         "agreement_with_fixed_seed42_target": int((prediction == target).sum()),
                         "mean_high_drop": float(row["high_drop"].astype(np.float64).mean()),
                         "mean_random_drop": float(row["random_mean_drop"].astype(np.float64).mean())})
        a, b = data[42]["heat"].astype(np.float64), row["heat"].astype(np.float64)
        sa, sb = a.sum((1, 2)), b.sum((1, 2)); valid = (sa > 0) & (sb > 0)
        distance = np.full(5000, np.nan)
        distance[valid] = .5 * np.abs(a[valid] / sa[valid, None, None] - b[valid] / sb[valid, None, None]).sum((1, 2))
        tv.append(distance)
        per_seed[-1].update(heatmap_TV_vs42_valid_pairs=int(valid.sum()),
                            heatmap_TV_vs42_mean=float(distance[valid].mean()) if valid.any() else None)
    # Rules are fixed before inference. All negative/flat observations remain in
    # the raw arrays and aggregate; examples only illustrate the recorded cases.
    candidates = (("fixed_target_correct", correct), ("fixed_target_wrong", ~correct),
                  ("negative_high_drop", np.any(np.stack([data[s]["high_drop"] < 0 for s in p.SEEDS]), axis=0)),
                  ("flat_map", np.any(np.stack([data[s]["flat"] for s in p.SEEDS]), axis=0)))
    examples = []
    for reason, mask in candidates:
        available = np.flatnonzero(mask)
        if len(available):
            row = int(available[np.argmin(indices[available])])
            examples.append({"reason": reason, "row": row, "official_train_index": int(indices[row])})
    result = {"state": "analysis_complete_actual_figures_pending", "protocol": p.PROTOCOL,
              "identity_sha256": p.sha(output / "identity.json"), "source_sha256": p.sources(),
              "unique_validation_images": 5000, "model_seeds": list(p.SEEDS), "not15000_independent": True,
              "target": "original seed42 clean5000 cached prediction, locked before inference for all models",
              "columns": ["seed42", "seed17", "seed73", "per_image_three_seed_average"],
              "overall": summary, "cohorts": cohorts, "classes": classes, "seeds": per_seed,
              "seed_mean_sample_SD_descriptive": float(np.std(difference.mean(0), ddof=1)),
              "example_selection": examples, "all_negative_and_flat_results_retained": True,
              "original200_not_claimed_representative_of5000": True,
              "uncertainty_scope": "Conditional fixed-model paired-image intervals, stratified by true class; no training/Teacher/selection uncertainty or multiple-comparison correction; no causal or object-boundary proof",
              "no_temperature_threshold_or_model_selection": True, "new_official_or_external_test": 0,
              "actual_visual_review_pending": True}
    p.save_npz(output / "analysis_arrays.npz", indices=indices, labels=labels, target=target,
               per_seed_dq=difference, per_image_three_seed_average=aggregate, heatmap_TV_vs42=np.stack(tv))
    result["analysis_arrays_sha256"] = p.sha(output / "analysis_arrays.npz")
    p.write(output / "analysis.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.output_dir)
    print(json.dumps({"state": result["state"], "unique_images": 5000, "actual_visual_review_pending": True}))


if __name__ == "__main__":
    main()
