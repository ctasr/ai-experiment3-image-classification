"""Conditional non-source KL extras only; original CE/full KL stay delegated.

Teacher q is used as supplied: no additional clamp, temperature transform,
normalization, detached statistic, or1-minus-source mass calculation.
"""
import math
import torch


def _conditional(logits, q, non_source):
    selected_q = torch.where(non_source, q, torch.zeros_like(q))
    mass = selected_q.sum(dim=1)
    log_student = logits - torch.logsumexp(logits.masked_fill(~non_source, -torch.inf), dim=1, keepdim=True)
    conditional_q = selected_q / mass[:, None]
    # xlogy(0,0)=0 implements the entropy limit on masked-out coordinates;
    # it does not modify q or rescue a zero non-source mass.
    terms = (torch.special.xlogy(selected_q, selected_q) / mass[:, None]
             - conditional_q * (mass.log()[:, None] + log_student))
    return terms.sum(dim=1), mass


def extras(logits, labels, order, lam, q, T, mode):
    """Return (mean extra*T², tensor diagnostics) for one actual CutMix batch."""
    if mode not in ("source_set", "weighted_single", "original"):
        raise ValueError(mode)
    temperature, area = float(T), float(lam)
    if not (math.isfinite(temperature) and temperature > 0 and math.isfinite(area) and 0 <= area <= 1):
        raise ValueError("Expected positive T and actual lambda in[0,1]")
    z = logits.float() / temperature
    if z.ndim != 2 or q.shape != z.shape or labels.shape != order.shape or labels.shape != (len(z),):
        raise ValueError("Mismatched logits/teacher/label/order shapes")
    if q.dtype != torch.float32 or q.device != z.device or labels.device != z.device or order.device != z.device:
        raise ValueError("Expected the unchanged FP32 Teacher target and same-device labels/order")
    if z.shape[1] < 3:
        raise ValueError("The source-set protocol requires a nonempty complement")
    if mode == "original":
        per_row = z.sum(1) * 0
        finite = torch.isfinite(per_row)
        return per_row.mean(), {"per_row_extra": per_row, "finite": finite, "all_finite": finite.all()}
    partner = labels[order]
    source = torch.zeros_like(z, dtype=torch.bool)
    if area > 0:
        source.scatter_(1, labels[:, None], True)
    if area < 1:
        source.scatter_(1, partner[:, None], True)
    qS = torch.where(source, q, torch.zeros_like(q)).sum(1)
    qN = torch.where(~source, q, torch.zeros_like(q)).sum(1)
    if mode == "source_set":
        conditional, direct_mass = _conditional(z, q, ~source)
        per_row = qS * conditional * temperature ** 2
    else:
        # Exact zero-area branches are never evaluated, avoiding0*bad values.
        per_row = z.sum(1) * 0
        if area > 0:
            complement_i = torch.ones_like(z, dtype=torch.bool).scatter_(1, labels[:, None], False)
            conditional_i, _ = _conditional(z, q, complement_i)
            per_row = per_row + area * q.gather(1, labels[:, None]).squeeze(1) * conditional_i
        if area < 1:
            complement_j = torch.ones_like(z, dtype=torch.bool).scatter_(1, partner[:, None], False)
            conditional_j, _ = _conditional(z, q, complement_j)
            per_row = per_row + (1 - area) * q.gather(1, partner[:, None]).squeeze(1) * conditional_j
        per_row = per_row * temperature ** 2
        # Report the actual set conditional for diagnostics only. Its complement
        # is nonempty in this fixed C10 protocol, including equal-label rows.
        conditional, direct_mass = _conditional(z, q, ~source)
    finite = torch.isfinite(per_row) & torch.isfinite(qS) & torch.isfinite(qN) & torch.isfinite(conditional)
    diagnostic = {"source_cardinality": source.sum(1), "qS": qS, "qN": qN,
                  "per_row_extra": per_row, "conditional_KL_non_source": conditional,
                  "direct_non_source_mass": direct_mass, "finite": finite, "all_finite": finite.all()}
    return per_row.mean(), diagnostic
