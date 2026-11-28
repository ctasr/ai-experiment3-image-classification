"""FP32 original-validation preflight and sealed external logit caching."""
from pathlib import Path
import hashlib
import os
import pickle
import numpy as np
import torch

from . import common as C
from .dataset import raw_training, data_path, load_external
from .manifest import current, sealed, utc

LOGIT_ATOL = 2e-4
LOGIT_RTOL = 2e-4
PROB_ATOL = 2e-5
PROB_RTOL = 2e-5


def initialize_gpu():
    # Set before CUDA/cuBLAS initialization; the process uses no training RNG.
    configured=os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if torch.cuda.is_initialized() and configured != ":4096:8":
        raise RuntimeError("An already initialized CUDA runtime lacks deterministic cuBLAS configuration")
    if configured is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    elif configured != ":4096:8":
        raise RuntimeError("Unsupported deterministic cuBLAS workspace setting")
    torch.cuda.init();torch.cuda.get_rng_state_all()


def environment():
    import platform
    return {"python":platform.python_version(),"torch":torch.__version__,"cuda_build":torch.version.cuda,
            "cudnn":torch.backends.cudnn.version(),"GPU":torch.cuda.get_device_name(),
            "capability":list(torch.cuda.get_device_capability()),"CUBLAS_WORKSPACE_CONFIG":os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def runtime_hash():
    value = [__import__("random").getstate(), np.random.get_state(), torch.get_rng_state().numpy().tobytes(),
             [x.cpu().numpy().tobytes() for x in torch.cuda.get_rng_state_all()], torch.get_num_threads(),
             torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.utils.deterministic.fill_uninitialized_memory]
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def load_model(row):
    import models as model_module
    if Path(model_module.__file__).resolve()!=(C.REPO/"models.py").resolve():
        raise ValueError("The imported inference architecture is not the bound checkout source")
    if C.sha(row["checkpoint"]) != row["checkpoint_sha256"]:
        raise ValueError("Bound checkpoint changed before model loading")
    ckpt = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
    if (ckpt["config"] != row["config"] or ckpt["stats"] != row["stats"]
            or ckpt["source_sha256"] != row["expected_training_sources"] or ckpt["epoch"] != row["epoch"]):
        raise ValueError("Bound checkpoint metadata changed")
    model = model_module.make_model(row["config"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().requires_grad_(False)
    if C.state_hash(model) != row["model_state_sha256"]:
        raise ValueError("Checkpoint model state differs from CPU preparation")
    # Preserve the original FP32 analysis model's contiguous weight layout;
    # only normalized image tensors use channels_last.
    model = model.to("cuda")
    del ckpt
    return model


@torch.inference_mode()
def logits(model, images, stats, flip=False, batch_size=128):
    outputs = []
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device="cuda").view(1,3,1,1)
    std = torch.tensor(stats["std"], dtype=torch.float32, device="cuda").view(1,3,1,1)
    for first in range(0, len(images), batch_size):
        raw = np.ascontiguousarray(images[first:first+batch_size].transpose(0,3,1,2))
        x = torch.from_numpy(raw).to("cuda", dtype=torch.float32).div_(255.)
        x = ((x-mean)/std).contiguous(memory_format=torch.channels_last)
        if flip:
            x = x.flip(-1)
        with torch.autocast(device_type="cuda", enabled=False):
            z = model(x)
        if z.dtype != torch.float32 or z.shape != (len(raw),10) or not torch.isfinite(z).all().item():
            raise ValueError("Expected finite FP32 Nx10 model logits")
        outputs.append(z.cpu().numpy().copy())
    return np.concatenate(outputs)


def cache_identity(output, row, stage):
    output = Path(output)
    identity = {"kind": "external_FP32_logits_cache_v1", "stage": stage, "id": row["id"],
                "manifest_sha256": C.sha(output / "manifest.json"), "checkpoint_sha256": row["checkpoint_sha256"],
                "evaluator_sources": C.sources(), "policies": row["policies"]}
    if stage == "external":
        identity.update(seal_sha256=C.sha(output / "seal.json"),
                        external_file_sha256=C.read(output / "fetch.json")["file_sha256"])
    else:
        identity.update(original_identity=C.read(output / "manifest.json")["original_identity"],
                        validation_indices="first257_of_frozen_validation_order", scalar_indices=[0,127,256])
    return identity


def load_cache(output, row, stage):
    folder = Path(output) / ("logits" if stage == "external" else "preflight_logits")
    npz, receipt = folder / (row["id"]+".npz"), folder / (row["id"]+".json")
    if not npz.exists() and not receipt.exists():
        return None
    if not (npz.exists() and receipt.exists()):
        raise RuntimeError("Incomplete logit artifacts are preserved; do not overwrite them")
    saved = C.read(receipt)
    if saved["identity"] != cache_identity(output,row,stage) or C.sha(npz) != saved["npz_sha256"]:
        raise ValueError("Existing logit cache belongs to different model/source/data/policy")
    n = 2000 if stage == "external" else 257
    with np.load(npz, allow_pickle=False) as arrays:
        expected = {"raw", "flip"} if row["group"] == "main_KD_BEST" else {"raw"}
        if set(arrays.files) != expected:
            raise ValueError("Unexpected logit cache arrays")
        for name in expected:
            z = arrays[name]
            if z.shape != (n,10) or z.dtype != np.float32 or not np.isfinite(z).all():
                raise ValueError("Invalid saved FP32 logits")
    return saved


def store_cache(output, row, stage, arrays, checks):
    folder = Path(output) / ("logits" if stage == "external" else "preflight_logits")
    path = folder / (row["id"]+".npz"); receipt = folder / (row["id"]+".json")
    if path.exists() or receipt.exists():
        raise RuntimeError("Never replace an existing logit artifact")
    C.save_npz(path, **arrays)
    value = {"identity": cache_identity(output,row,stage), "npz_sha256": C.sha(path), "checks": checks}
    C.write(receipt, value); return value


def preflight(output):
    output = Path(output); C.require_no_access(output)
    if (output / "seal.json").exists():
        raise RuntimeError("Preflight may not mutate a sealed confirmation")
    manifest = current(output)
    x, _ = raw_training(manifest["data_dir"])
    metadata = C.read(Path(manifest["data_dir"]) / "dataset_manifest.json")
    indices = C.read(data_path(manifest["data_dir"],metadata["split_file"]))["validation_indices"][:257]
    images = x[np.asarray(indices)].copy(); del x
    # Explicit GPU stage: initialize CUDA before taking the all-RNG snapshot.
    # No CPU-only prepare/check command reaches this code.
    initialize_gpu()
    before = runtime_hash(); records = []; active_row=None; arrays=None
    try:
        with C.preserve_runtime():
            C.configure_fp32()
            for row in manifest["models"]:
                C.check_sources(manifest["evaluator_sources"])
                active_row=row;arrays=None
                existing = load_cache(output,row,"original_validation")
                if existing:
                    if not existing["checks"].get("passed"):
                        raise RuntimeError("A failed preflight cache must be retained, never promoted")
                    records.append(existing); continue
                model = load_model(row); state_before = C.state_hash(model)
                arrays = {"raw": logits(model,images,row["stats"])}
                if row["group"] == "main_KD_BEST":
                    arrays["flip"] = logits(model,images,row["stats"],flip=True)
                repeat = logits(model,images,row["stats"])
                if not np.array_equal(arrays["raw"],repeat):
                    raise ValueError("Identical-batch deterministic FP32 repeat differs")
                checks = {"passed": True, "device": "cuda", "precision": "float32_autocast_off_TF32_off",
                          "samples": 257, "batch_size": 128, "small_last_batch": 1,
                          "identical_batch_repeat_exact": True, "raw_scalar": [], "two_view_scalar": []}
                for probe in (0,127,256):
                    one = logits(model,images[probe:probe+1],row["stats"])
                    batch = arrays["raw"][probe:probe+1]
                    p1, _ = C.probability_policy(one); pb, _ = C.probability_policy(batch)
                    if not np.allclose(one,batch,atol=LOGIT_ATOL,rtol=LOGIT_RTOL) or not np.allclose(p1,pb,atol=PROB_ATOL,rtol=PROB_RTOL):
                        raise ValueError("Raw scalar/batch FP32 evaluator tolerance failed")
                    checks["raw_scalar"].append({"index": probe, "max_abs_logits": float(np.max(np.abs(one-batch))),
                                                  "max_abs_prob64": float(np.max(np.abs(p1-pb))),
                                                  "argmax_equal": bool(p1.argmax(1)[0] == pb.argmax(1)[0])})
                    if "flip" in arrays:
                        flip = logits(model,images[probe:probe+1],row["stats"],flip=True)
                        q1, _ = C.probability_policy(one,flip)
                        qb, _ = C.probability_policy(batch,arrays["flip"][probe:probe+1])
                        if not np.allclose(flip,arrays["flip"][probe:probe+1],atol=LOGIT_ATOL,rtol=LOGIT_RTOL) or not np.allclose(q1,qb,atol=PROB_ATOL,rtol=PROB_RTOL):
                            raise ValueError("Two-view scalar/batch FP32 evaluator tolerance failed")
                        checks["two_view_scalar"].append({"index":probe,"max_abs_prob64":float(np.max(np.abs(q1-qb))),
                                                          "argmax_equal":bool(q1.argmax(1)[0] == qb.argmax(1)[0])})
                if (C.state_hash(model) != state_before or state_before != row["model_state_sha256"]
                        or model.training or any(p.grad is not None or p.requires_grad for p in model.parameters())
                        or C.sha(row["checkpoint"]) != row["checkpoint_sha256"]):
                    raise ValueError("Preflight mutated weights/BN/grad state or checkpoint")
                checks["weights_BN_grad_metadata_unchanged"] = True
                records.append(store_cache(output,row,"original_validation",arrays,checks))
                del model; arrays=None; torch.cuda.empty_cache()
        after = runtime_hash()
        if after != before:
            raise ValueError("Preflight failed to restore all RNG/runtime flags")
        current(output)
        value = {"kind":"external_actual_original_validation_GPU_v1","passed":True,"at_utc":utc(),
                 "manifest_sha256":C.sha(output / "manifest.json"),"evaluator_sources":C.sources(),
                 "models":records,"raw_policy_checks":21,"two_view_policy_checks":3,"small_last_batch":1,
                 "logit_atol":LOGIT_ATOL,"logit_rtol":LOGIT_RTOL,"probability_atol":PROB_ATOL,"probability_rtol":PROB_RTOL,
                 "cross_batch_bitwise_identity_claimed":False,"all_RNG_runtime_unchanged":True,
                 "environment":environment(),
                 "external_data_read":False,"official_test_model_evaluation":False}
        C.write(output / "preflight.json",value); return value
    except Exception as exc:
        stamp=utc().replace(":","-")
        failed_npz=None
        if arrays:
            failed_npz=output/"failed_artifacts"/("preflight_"+active_row["id"]+"_"+stamp+".npz")
            C.save_npz(failed_npz,**arrays)
        C.write(output / ("preflight_failure_"+stamp+".json"),
                {"error":repr(exc),"completed_models":len(records),"all_existing_artifacts_retained":True,
                 "active_model":active_row["id"] if active_row else None,
                 "unverified_arrays":str(failed_npz) if failed_npz else None})
        raise


def evaluate(output):
    output = Path(output); manifest, _ = sealed(output)
    if (output / "retired.json").exists():
        raise RuntimeError("External evaluation is retired; only read the preserved fixed outputs")
    images, _ = load_external(output)
    initialize_gpu()
    if environment()!=C.read(output/"preflight.json")["environment"]:
        raise ValueError("External inference must use the preflight-verified software/GPU environment")
    before = runtime_hash(); records = [];active_row=None;arrays=None
    try:
        with C.preserve_runtime():
            C.configure_fp32()
            for row in manifest["models"]:
                C.check_sources(manifest["evaluator_sources"])
                active_row=row;arrays=None
                existing = load_cache(output,row,"external")
                if existing:
                    records.append(existing); continue
                model = load_model(row); state_before = C.state_hash(model)
                arrays = {"raw":logits(model,images,row["stats"])}
                if row["group"] == "main_KD_BEST":
                    arrays["flip"] = logits(model,images,row["stats"],flip=True)
                if (C.state_hash(model) != state_before or state_before != row["model_state_sha256"]
                        or model.training or any(p.grad is not None or p.requires_grad for p in model.parameters())
                        or C.sha(row["checkpoint"]) != row["checkpoint_sha256"]):
                    raise ValueError("Evaluation mutated weights/BN/grad state or checkpoint")
                records.append(store_cache(output,row,"external",arrays,{"passed":True,"samples":2000,
                               "batch_size":128,"last_batch":80,"weights_BN_grad_metadata_unchanged":True}))
                del model, arrays;arrays=None;torch.cuda.empty_cache()
    except Exception as exc:
        stamp=utc().replace(":","-");failed_npz=None
        if arrays:
            failed_npz=output/"failed_artifacts"/("external_"+active_row["id"]+"_"+stamp+".npz")
            C.save_npz(failed_npz,**arrays)
        C.write(output/("evaluation_failure_"+stamp+".json"),{"error":repr(exc),"completed_models":len(records),
                "unverified_arrays":str(failed_npz) if failed_npz else None,"all_existing_artifacts_retained":True})
        raise
    if runtime_hash() != before:
        raise ValueError("Evaluation did not restore all RNG/runtime flags")
    sealed(output)
    value = {"kind":"fixed_external_evaluation_complete_v1","at_utc":utc(),"checkpoints":len(records),
             "seed_policy_rows":24,"manifest_sha256":C.sha(output/"manifest.json"),"seal_sha256":C.sha(output/"seal.json"),
             "models":records,"all_RNG_runtime_unchanged":True,"analysis_and_manual_visual_review_pending":True}
    C.write(output/"evaluation_complete.json",value); return value
