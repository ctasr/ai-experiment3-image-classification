"""Immutable identities, FP64 scoring semantics and explicit lifecycle gates."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import random

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SEEDS = (42, 17, 73)
GROUPS = ("alignment_A_BEST", "alignment_B_BEST", "main_KD_BEST", "main_KD_LAST",
          "weighted_single_LAST", "confidence_mass_LAST", "noKD240_BEST")
RAW = "raw_FP32_logits_prob64"
TWO = "raw_and_flip_FP32_logits_prob64_mean"
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
FILES = ("__init__.py", "common.py", "manifest.py", "dataset.py", "inference.py", "analysis.py", "run.py", "protocol.json", "source_metadata.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def sources():
    return {**{f"contributions/external_confirmation/{name}": sha(HERE / name) for name in FILES},
            "models.py": sha(REPO / "models.py")}


def policies(group):
    if group not in GROUPS:
        raise ValueError("Unregistered model group")
    return [RAW, TWO] if group == "main_KD_BEST" else [RAW]


def model_id(group, seed):
    if group not in GROUPS or seed not in SEEDS:
        raise ValueError("Only the fixed seven groups and three seeds are allowed")
    return f"{group}_seed{seed}"


def log_probs64(logits):
    z = np.asarray(logits)
    if z.dtype != np.float32 or z.ndim != 2 or z.shape[1] != 10 or not np.isfinite(z).all():
        raise ValueError("Expected finite saved FP32 logits of shape(N,10)")
    z = z.astype(np.float64)
    maximum = z.max(1, keepdims=True)
    return z - maximum - np.log(np.exp(z - maximum).sum(1, keepdims=True))


def probability_policy(logits0, logits1=None):
    lp0 = log_probs64(logits0)
    p0 = np.exp(lp0)
    if logits1 is None:
        return p0, lp0
    if np.shape(logits1) != np.shape(logits0):
        raise ValueError("Raw/flip logits shape differs")
    lp1 = log_probs64(logits1)
    # The decision rule uses this explicit FP64 arithmetic mean. Stable NLL is
    # computed separately from logaddexp, avoiding probability underflow.
    return (p0 + np.exp(lp1)) / 2, np.logaddexp(lp0, lp1) - np.log(2.)


def state_hash(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(name.encode()); result.update(str(array.shape).encode()); result.update(str(array.dtype).encode()); result.update(array.tobytes())
    return result.hexdigest()


@contextmanager
def preserve_runtime():
    initialized = torch.cuda.is_initialized()
    saved = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
             torch.cuda.get_rng_state_all() if initialized else None, torch.get_num_threads(),
             torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.utils.deterministic.fill_uninitialized_memory)
    try:
        yield
    finally:
        random.setstate(saved[0]); np.random.set_state(saved[1]); torch.set_rng_state(saved[2])
        if saved[3] is not None:
            torch.cuda.set_rng_state_all(saved[3])
        torch.set_num_threads(saved[4]); torch.use_deterministic_algorithms(saved[5], warn_only=saved[6])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = saved[7:9]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[9:11]
        torch.utils.deterministic.fill_uninitialized_memory = saved[11]


def configure_fp32():
    if not torch.cuda.is_available():
        raise RuntimeError("Actual evaluator preflight/inference requires CUDA")
    if not torch.utils.deterministic.fill_uninitialized_memory:
        raise RuntimeError("The fixed deterministic fill default must remain enabled")
    torch.set_num_threads(8); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False


@contextmanager
def output_lock(output):
    path = Path(output) / "active_external_stage.lock"
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError("An existing stage lock must be investigated, never silently replaced") from exc
    stream.write(str(os.getpid())); stream.flush()
    try:
        yield
    finally:
        stream.close(); path.unlink(missing_ok=True)


def check_sources(snapshot):
    if sources() != snapshot:
        raise ValueError("Frozen evaluator/model implementation sources changed")


def require_no_access(output):
    if (Path(output) / "access_started.json").exists():
        raise RuntimeError("External access has started: model/source/policy bindings may no longer change")
