"""Synthetic probability/identity checks only; never load a model or CIFAR data."""
import copy
import unittest

import numpy as np

from evaluate_release import (
    check_saved_identity, classification_metrics, make_parser, probability_mean,
)


class EvaluationArithmeticTests(unittest.TestCase):
    def test_branch_average_converts_before_adding(self):
        raw = np.array([[.1, .9]], dtype=np.float32)
        flipped = np.array([[.3, .7]], dtype=np.float32)
        original, mean = probability_mean(raw, flipped)
        self.assertEqual(mean.dtype, np.float64)
        np.testing.assert_array_equal(original, raw.astype(np.float64))
        np.testing.assert_array_equal(mean, (raw.astype(np.float64) + flipped.astype(np.float64)) / 2)

    def test_confusion_orientation_and_hard_label_nll(self):
        labels = np.array([0, 1, 1], dtype=np.int64)
        probabilities = np.zeros((3, 10), dtype=np.float64)
        probabilities[0, :2] = [.8, .2]
        probabilities[1, :2] = [.6, .4]
        probabilities[2, :2] = [.1, .9]
        metrics = classification_metrics(probabilities, labels)
        self.assertEqual(metrics["correct"], 2)
        self.assertEqual(metrics["confusion_matrix"][1][0], 1)
        self.assertEqual(metrics["per_class_accuracy"][:2], [1, .5])
        self.assertIsNone(metrics["per_class_accuracy"][2])
        self.assertAlmostEqual(metrics["nll"], -np.log([.8, .4, .9]).mean())

    def test_reject_invalid_probabilities_and_labels(self):
        with self.assertRaises(ValueError):
            classification_metrics(np.zeros((1, 10)), np.array([0]))
        with self.assertRaises(ValueError):
            classification_metrics(np.full((1, 10), .1), np.array([-1]))

    def test_accept_fp32_softmax_rounding_without_renormalizing(self):
        probabilities = np.full((1, 10), .10000003, dtype=np.float32)
        metrics = classification_metrics(probabilities, np.array([0]))
        self.assertEqual(metrics['correct'], 1)
        self.assertAlmostEqual(metrics['nll'], -np.log(float(probabilities[0, 0])))

    def test_checkpoint_identity_rejects_seed_config_and_source_changes(self):
        config = dict(architecture="resnet20", model="cifar", width=48,
                      epochs=240, distill_alpha=.25, seed=42)
        sources = {f"source{i}.py": str(i) for i in range(12)}
        stats = {"mean": [.1, .2, .3], "std": [.2, .3, .4]}
        # Real best.pt retains these four metadata fields alongside the core keys.
        saved = dict(model={}, config=config, stats=stats, source_sha256=sources,
                     epoch=234, best_accuracy=.972, best_ce=.1604152711868286,
                     pretrained_sha256=None)
        check_saved_identity(saved, {"config": config}, sources, 42, stats)
        changed = copy.deepcopy(saved)
        changed["config"]["seed"] = 17
        with self.assertRaises(ValueError):
            check_saved_identity(changed, {"config": config}, sources, 42, stats)
        changed = copy.deepcopy(saved)
        changed["source_sha256"]["source0.py"] = "changed"
        with self.assertRaises(ValueError):
            check_saved_identity(changed, {"config": config}, sources, 42, stats)

    def test_documented_cli(self):
        args = make_parser().parse_args(["--seed", "17", "--split", "validation"])
        self.assertEqual(args.seed, 17)
        self.assertEqual(args.split, "validation")
        self.assertIsNone(args.batch_size)


if __name__ == "__main__":
    unittest.main()
