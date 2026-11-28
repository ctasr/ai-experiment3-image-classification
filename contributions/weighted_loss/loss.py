"""Loss routing; the original weighted-single mathematics is kept byte-for-byte.

``confidence_mass`` is a new explicit coefficient-mass control. Its coefficient
matches the sum of the weighted recipe's written KL coefficients; it does not
match gradient norms or remove all differences between the two objectives.
"""
import math

import torch
from torch.nn import functional as F

from distillation.train import loss_parts as original_loss_parts
from . import legacy_loss

MODES = ("original", "weighted_single", "confidence_mass")


def combined(logits, labels, order, lam, q, temperature, alpha, mode):
    """Return the original (total, hard CE, KD) interface plus audit values.

    The Teacher target is the supplied FP32 target for the actual CutMix input.
    No extra normalization, clamp, temperature fitting or target RNG is added.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown loss mode: {mode}")
    original = original_loss_parts(logits, labels, order, lam, q, temperature, alpha)
    if mode == "original":
        return original, original[2].detach() * 0, None, original
    if mode == "weighted_single":
        extra, details = legacy_loss.extras(
            logits, labels, order, lam, q, temperature, mode="weighted_single"
        )
        # Preserve the frozen adapter's arithmetic order exactly.
        return (original[0] + alpha * extra, original[1], original[2] + extra), extra, details, original
    area, temperature = float(lam), float(temperature)
    if not (math.isfinite(area) and 0 <= area <= 1 and math.isfinite(temperature) and temperature > 0):
        raise ValueError("Expected actual lambda in [0,1] and positive temperature")
    if q.dtype != torch.float32 or q.shape != logits.shape or q.device != logits.device:
        raise ValueError("Expected unchanged FP32 Teacher target matching logits")
    if labels.shape != (len(logits),) or order.shape != labels.shape:
        raise ValueError("Mismatched labels/order")
    partner = labels[order]
    source_mass = area * q.gather(1, labels[:, None]).squeeze(1)
    source_mass = source_mass + (1 - area) * q.gather(1, partner[:, None]).squeeze(1)
    coefficient = 1 + source_mass
    per_row_kl = F.kl_div(
        F.log_softmax(logits.float() / temperature, dim=1), q.float(), reduction="none"
    ).sum(1) * temperature ** 2
    # Per-image weighting must precede averaging. Multiplying a batch-average
    # KL by the average coefficient would be a different control.
    kd = (coefficient * per_row_kl).mean()
    total = (1 - alpha) * original[1] + alpha * kd
    details = {"coefficient": coefficient, "per_row_full_KL": per_row_kl,
               "explicit_source_mass": source_mass,
               "all_finite": torch.isfinite(kd) & torch.isfinite(total)}
    return (total, original[1], kd), kd - original[2], details, original
