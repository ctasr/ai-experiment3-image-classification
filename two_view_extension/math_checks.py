import numpy as np
import torch
from torch.nn import functional as F


def analytic_maps(feature0, feature1, weight, p0, p1, target):
    """Actual q=0.5*(softmax(z0)+softmax(z1)) target, no extra /Z."""
    z = feature0.shape[-1] * feature0.shape[-2]
    selected = weight[target]
    a0 = .5 * p0.gather(1, target[:, None]) * (selected - p0 @ weight) / z
    a1 = .5 * p1.gather(1, target[:, None]) * (selected - p1 @ weight) / z
    signed0 = (feature0 * a0[:, :, None, None]).sum(1)
    signed1 = (feature1 * a1[:, :, None, None]).sum(1)
    # Interpolate in each input's coordinates before undoing the input flip.
    map0 = F.interpolate(signed0[:, None], (32, 32), mode="bilinear", align_corners=False)[:, 0]
    map1 = F.interpolate(signed1[:, None], (32, 32), mode="bilinear", align_corners=False)[:, 0].flip(-1)
    signed_sum = map0 + map1  # 0.5 already appears in each q derivative
    positive = signed_sum.relu()
    flat = (positive.amax((1, 2)) - positive.amin((1, 2))) == 0
    pooled = F.avg_pool2d(positive[:, None], 8, stride=1).flatten(1)
    locations = pooled.argmax(1)  # row-major first index, also for degenerate maps
    return a0, a1, map0, map1, signed_sum, positive, flat, locations


def cpu_autograd(feature0, feature1, weight, bias, target):
    """Actual frozen features checked on CPU; model parameters never need grad."""
    assert feature0.device.type == feature1.device.type == weight.device.type == "cpu"
    f0 = feature0.detach().clone().requires_grad_(True)
    f1 = feature1.detach().clone().requires_grad_(True)
    weight, bias = weight.detach(), bias.detach()
    z0, z1 = F.linear(f0.mean((2, 3)), weight, bias), F.linear(f1.mean((2, 3)), weight, bias)
    p0, p1 = z0.softmax(1), z1.softmax(1)
    q = (p0 + p1) / 2
    g0, g1 = torch.autograd.grad(q.gather(1, target[:, None]).sum(), (f0, f1))
    a0, a1, *_ = analytic_maps(f0, f1, weight, p0, p1, target)
    e0, e1 = a0[:, :, None, None].expand_as(g0), a1[:, :, None, None].expand_as(g1)
    assert torch.allclose(g0, e0, atol=2e-8, rtol=2e-5)
    assert torch.allclose(g1, e1, atol=2e-8, rtol=2e-5)
    assert torch.allclose(g0.mean((2, 3)), a0, atol=2e-8, rtol=2e-5)
    return {"passed": True, "samples": len(f0), "max_absolute_error": max(float((g0-e0).abs().max()), float((g1-e1).abs().max())),
            "autograd_target": "sum of fixed-class actual q probabilities; not batch mean", "classifier_bias_included": True,
            "model_weights_require_grad": False, "cpu_only_autograd": True,
            "per_branch_gradient_l2": [g0.detach().flatten(1).norm(dim=1).tolist(), g1.detach().flatten(1).norm(dim=1).tolist()]}


def signed_map_distance(m0, m1):
    a, b = np.maximum(m0, 0), np.maximum(m1, 0)
    sa, sb = a.sum((1, 2)), b.sum((1, 2))
    valid = (sa > 0) & (sb > 0)
    values = np.full(len(a), np.nan)
    values[valid] = .5 * np.abs(a[valid]/sa[valid, None, None] - b[valid]/sb[valid, None, None]).sum((1, 2))
    return values, valid
