"""Frozen experiment protocol, provenance and resumable orchestration utilities."""
from pathlib import Path
import contextlib
import hashlib
import json
import os
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_mobile_transfer"
RUNS = ROOT / "runs/mobile_transfer"
FIG = ROOT / "figures_mobile_transfer"
CONFIGS = ROOT / "configs_mobile_transfer"
WEIGHTS = ROOT / "data/pretrained/mobile_transfer/mobilenet_v3_small-047dcff4.pth"
WEIGHT_URL = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
WEIGHT_PREFIX = "047dcff4"
OLD_SOURCES = ("train.py", "data.py", "models.py", "prepare_data.py", "extensions/train.py", "extensions/models.py")
SOURCES = OLD_SOURCES + ("mobile_transfer_study/common.py", "mobile_transfer_study/models.py", "mobile_transfer_study/train.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes():
    return {name: sha(ROOT/name) for name in SOURCES}


def run_id(pretrained, seed):
    return f"mobile_{'pretrained' if pretrained else 'random'}_seed{seed}"


def baseline(seed):
    return next(r for r in read(ROOT/"results_round3/selected_runs.json")["transfer"] if r["config"]["seed"] == seed)


def config(pretrained, seed):
    cfg = dict(baseline(seed)["config"])
    cfg.update(architecture="mobilenet_v3_small", pretrained=bool(pretrained), head_only=False,
               backbone_ratio=1.0, image_size=96, batch_size=128, epochs=60, warmup_epochs=5,
               learning_rate=.003, weight_decay=.002, seed=int(seed), mix="none", label_smoothing=0.,
               augmentation=True, cutout=0, patience=0, amp=True, model="transfer")
    assert not any(k.startswith("distill") for k in cfg)
    return cfg


def assert_inputs():
    snapshot = read(OUT/"input_snapshot.json")
    assert all((ROOT/name).is_file() and sha(ROOT/name) == digest for name, digest in snapshot.items()), "Frozen input changed"
    manifest = read(OUT/"source_manifest.json")
    assert manifest["source_sha256"] == source_hashes(), "Training source changed; isolate a new version"
    assert manifest["plan_sha256"] == sha(OUT/"plan.json")
    identity = read(OUT/"weights.json")
    assert WEIGHTS.is_file() and sha(WEIGHTS) == identity["sha256"] and identity["sha256"].startswith(WEIGHT_PREFIX)


def gate(rows):
    """All thresholds are the authorized frozen practical rules, not significance tests."""
    pairs = []
    for seed in sorted({r["config"]["seed"] for r in rows}):
        p = next(r for r in rows if r["config"]["seed"] == seed and r["config"]["pretrained"])
        q = next(r for r in rows if r["config"]["seed"] == seed and not r["config"]["pretrained"])
        b = baseline(seed)
        pairs.append({"seed": seed, "pretrained_minus_random_accuracy": p["best_validation_accuracy"]-q["best_validation_accuracy"],
                      "pretrained_minus_random_ce": p["best_validation_ce"]-q["best_validation_ce"],
                      "pretrained_minus_resnet18_accuracy": p["best_validation_accuracy"]-b["best_validation_accuracy"]})
    values = {key: statistics.mean(p[key] for p in pairs) for key in pairs[0] if key != "seed"}
    supported = (values["pretrained_minus_random_accuracy"] >= .01-1e-12 and
                 values["pretrained_minus_random_ce"] <= -.01+1e-12 and
                 values["pretrained_minus_resnet18_accuracy"] >= -.02-1e-12 and
                 (len(pairs) == 1 or sum(p["pretrained_minus_random_accuracy"] >= 0 for p in pairs) >= 2))
    return {"supported": supported, "seeds": [p["seed"] for p in pairs], "pairs": pairs, "mean_changes": values,
            "rule": read(OUT/"plan.json")["gates"], "practical_screening_not_statistical_confirmation": True}


@contextlib.contextmanager
def process_lock():
    """Windows file locking prevents duplicate live drivers even after stale PID records."""
    import msvcrt
    OUT.mkdir(exist_ok=True)
    handle = (OUT/"driver.lock").open("a+b")
    handle.seek(0); handle.write(b"1"); handle.flush(); handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close(); raise RuntimeError("A Mobile study driver is already alive")
    try:
        yield
    finally:
        handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1); handle.close()


def progress(state, current_run=None, **extra):
    write(OUT/"progress.json", {"state": state, "current_run": current_run,
          "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra})
