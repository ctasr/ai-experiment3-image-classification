"""The original two-view inference and analytic Grad-CAM protocol, extended."""
import hashlib
import time

import numpy as np
import torch
from torch.nn import functional as F

from models import make_model
from . import protocol as p
from .legacy_math import analytic_maps

FIELDS = ("p0", "p1", "q", "map0", "map1", "signed", "heat", "flat", "loc",
          "gradient_norm0", "gradient_norm1", "masked0", "masked1", "maskedq",
          "positions", "high_drop", "random_mean_drop", "dq")


def branches(model, x, capture=False):
    features = {}
    handle = model.features.register_forward_hook(lambda m, i, o: features.update(value=o.detach())) if capture else None
    try:
        z0 = model(x)
        f0 = features["value"].clone() if capture else None
        z1 = model(x.flip(-1).contiguous(memory_format=torch.channels_last))
        f1 = features["value"].clone() if capture else None
    finally:
        if handle is not None:
            handle.remove()
    p0, p1 = z0.softmax(1), z1.softmax(1)
    return p0, p1, (p0+p1)/2, f0, f1, z0, z1


def load_model(path, seed, stats, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved["source_sha256"] != p.provenance()["model_training_sources"]
            or saved["config"]["seed"] != seed or saved["config"]["width"] != 48
            or saved["config"]["epochs"] != 240 or saved["config"]["distill_alpha"] != .25
            or saved["config"]["architecture"] != "resnet20" or saved["stats"] != stats):
        raise ValueError("Wrong frozen original student checkpoint/config/normalization")
    model = make_model(saved["config"]).eval().requires_grad_(False)
    model.load_state_dict(saved["model"])
    # Preserve original inference construction: do not convert model weights
    # to channels_last here. The original normalized input is channels_last.
    model = model.to(device)
    if sum(value.numel() for value in model.parameters()) != 2436346 or not isinstance(model.classifier, torch.nn.Linear):
        raise ValueError("Analytic probability gradient requires the frozen GAP+linear head")
    return model


def tensor_identity(value):
    array = value.detach().cpu().contiguous().numpy()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "stride": list(value.stride()),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def variants_for_image(image, locations):
    """Image-major, then high window and ten original random windows."""
    result = []
    for yy, xx in locations:
        value = image.clone()
        value[:, int(yy):int(yy)+8, int(xx):int(xx)+8] = 0
        result.append(value)
    return result


def fixed_target_drops(probabilities, masked_probabilities, target):
    """Image-major fixed-class scores; no new argmax after masking."""
    n = len(probabilities)
    if probabilities.shape != (n, 10) or masked_probabilities.shape != (n, 11, 10) or target.shape != (n,):
        raise ValueError("Wrong fixed-target probability shape")
    scores = np.take_along_axis(masked_probabilities, np.broadcast_to(target[:, None, None], (n, 11, 1)), axis=2)[:, :, 0]
    return probabilities[np.arange(n), target, None] - scores


def compute_segment(model, stats, raw, target, random_positions, indices, event=None, capture_rows=()):
    """One isolated segment; never combine the old200 tail with new4800."""
    if len(raw) != len(target) or random_positions.shape != (len(raw), 10, 2):
        raise ValueError("Segment target/window shape differs")
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("Actual protocol inference must run on CUDA; CPU synthetic checks are separate")
    before = p.state_hash(model)
    values = {name: [] for name in FIELDS[:11]}
    features, capture_rows = {}, set(int(row) for row in capture_rows)
    maximum_identity, forwards = 0., 0
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        for start in range(0, len(raw), 32):
            began = time.monotonic()
            x = p.normalise(raw[start:start+32].to(device), stats)
            p0, p1, q, f0, f1, z0, z1 = branches(model, x, True)
            forwards += 2 * len(x)
            c = torch.as_tensor(target[start:start+len(x)], dtype=torch.long, device=device)
            a0, a1, m0, m1, signed, heat, flat, loc = analytic_maps(f0, f1, model.classifier.weight, p0, p1, c)
            for feature, logits in ((f0, z0), (f1, z1)):
                ordinary = (feature * model.classifier.weight[c, :, None, None]).sum(1)
                identity = ordinary.mean((1, 2)) + model.classifier.bias[c]
                selected = logits.gather(1, c[:, None]).squeeze(1)
                maximum_identity = max(maximum_identity, float((identity - selected).abs().max()))
                if not torch.allclose(identity, selected, atol=2e-5, rtol=2e-5):
                    raise RuntimeError("Original GAP/linear CAM-logit identity failed")
            current = dict(p0=p0, p1=p1, q=q, map0=m0, map1=m1, signed=signed, heat=heat,
                           flat=flat, loc=loc, gradient_norm0=a0.flatten(1).norm(dim=1), gradient_norm1=a1.flatten(1).norm(dim=1))
            if any(not bool(torch.isfinite(value).all()) for value in current.values()):
                raise FloatingPointError("Nonfinite original-protocol map/probability")
            for name, value in current.items():
                values[name].append(value.cpu().numpy())
            for local in range(len(x)):
                row = start + local
                if row in capture_rows:
                    features[row] = {"feature0": f0[local].cpu().numpy(), "feature1": f1[local].cpu().numpy()}
            if event:
                event({"phase": "clean", "completed": start + len(x), "total": len(raw),
                       "official_indices": indices[start:start+len(x)].tolist(), "forward_images": 2 * len(x),
                       "input": tensor_identity(x), "seconds": time.monotonic() - began})
    result = {name: np.concatenate(parts) for name, parts in values.items()}
    masked0, masked1, positions = [], [], []
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        for start in range(0, len(raw), 16):
            began = time.monotonic()
            x = p.normalise(raw[start:start+16].to(device), stats)
            variants, boxes = [], []
            for local in range(len(x)):
                row = start + local; location = int(result["loc"][row])
                chosen = [[location // 25, location % 25], *random_positions[row].tolist()]
                boxes.append(chosen); variants.extend(variants_for_image(x[local], chosen))
            b0, b1 = [], []
            # Keep the original torch.stack layout. Do not convert the first
            # masked branch to channels_last; branches() only converts flip.
            stacked = torch.stack(variants)
            for batch in stacked.split(64):
                a, b, *_ = branches(model, batch)
                forwards += 2 * len(batch)
                b0.append(a.cpu().numpy()); b1.append(b.cpu().numpy())
            masked0.append(np.concatenate(b0).reshape(len(x), 11, 10))
            masked1.append(np.concatenate(b1).reshape(len(x), 11, 10)); positions.extend(boxes)
            if event:
                event({"phase": "occlusion", "completed": start + len(x), "total": len(raw),
                       "official_indices": indices[start:start+len(x)].tolist(), "forward_images": 22 * len(x),
                       "input": tensor_identity(stacked), "seconds": time.monotonic() - began})
    result["masked0"], result["masked1"] = np.concatenate(masked0), np.concatenate(masked1)
    result["maskedq"] = (result["masked0"] + result["masked1"]) / 2
    result["positions"] = np.asarray(positions, dtype=np.int64)
    drops = fixed_target_drops(result["q"], result["maskedq"], target)
    result["high_drop"], result["random_mean_drop"] = drops[:, 0], drops[:, 1:].mean(1)
    result["dq"] = result["high_drop"] - result["random_mean_drop"]
    drops0 = fixed_target_drops(result["p0"], result["masked0"], target)
    drops1 = fixed_target_drops(result["p1"], result["masked1"], target)
    identity_error = float(np.max(np.abs(drops - (drops0 + drops1) / 2)))
    if identity_error >= 2e-6:
        raise RuntimeError("Original fixed-target two-branch probability-drop identity failed")
    p.assert_frozen(model, before)
    proof = {"forward_images": forwards, "samples": len(raw), "wall_seconds_including_pauses": time.monotonic() - started,
             "ordinary_CAM_logit_identity_max_abs": maximum_identity, "q_branch_drop_identity_max_abs": identity_error,
             "weights_BN_gradients_unchanged": True, "GPU_autograd": 0, "actual_CNN_forward_images": forwards}
    return result, features, proof
