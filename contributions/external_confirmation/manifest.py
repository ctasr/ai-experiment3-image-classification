"""Explicit portable checkpoint/source binding; sealing never downloads data."""
from pathlib import Path
import datetime
import json
import torch

from . import common as C


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def resolved(base, value):
    p = Path(value)
    return (p if p.is_absolute() else base / p).resolve()


def source_bindings(base, expected, bindings):
    """Historical source identifiers are labels, never implicitly resolved paths."""
    if set(expected) != set(bindings):
        raise ValueError("Every original source identifier needs exactly one explicit source_file_bindings entry")
    records, errors = {}, []
    for name, expected_sha in expected.items():
        path = resolved(base, bindings[name])
        actual = C.sha(path) if path.is_file() else None
        records[name] = {"path": str(path), "expected_sha256": expected_sha,
                         "actual_sha256": actual, "bytes": path.stat().st_size if path.is_file() else None}
        if actual != expected_sha:
            errors.append(f"Missing/mismatched source: {name}")
    return records, errors


def validate_config(group, seed, config, epoch):
    width = 32 if group.startswith("alignment_") else 48
    budget = 160 if width == 32 else 240
    expected = {"architecture": "resnet20", "width": width, "attention": False,
                "seed": seed, "epochs": budget, "batch_size": 256, "model": "cifar",
                "image_size": 32, "mix": "cutmix", "mix_alpha": .5,
                "distill_alpha": 0. if group == "noKD240_BEST" else .25,
                "distill_temperature": 1.,
                "alignment": "crop_mix" if group == "alignment_A_BEST" else "mixed_view"}
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"Wrong fixed model configuration {group}/{seed}: {name}")
    if not 1 <= epoch <= budget or (group.endswith("LAST") and epoch != budget):
        raise ValueError("Wrong BEST/LAST checkpoint epoch")
    if group == "weighted_single_LAST" and config.get("source_set_mode") != "weighted_single":
        raise ValueError("Expected the original weighted_single objective")
    if group == "confidence_mass_LAST" and config.get("source_set_mode") != "confidence_mass":
        raise ValueError("Expected the newly registered confidence_mass objective")


def prepare(spec_path, data_dir, output):
    from .dataset import original_identity, build_original_index
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    C.require_no_access(output)
    if (output / "seal.json").exists():
        raise RuntimeError("A sealed manifest cannot be replaced")
    spec_path = Path(spec_path).resolve(); spec = C.read(spec_path); base = spec_path.parent
    rows = spec.get("models", [])
    wanted = {C.model_id(g, s) for g in C.GROUPS for s in C.SEEDS}
    actual_ids = [C.model_id(r["group"], r["seed"]) for r in rows]
    if len(rows) != 21 or len(set(actual_ids)) != 21 or set(actual_ids) != wanted:
        raise ValueError("The manifest must register exactly the fixed 21 checkpoint identities")
    if output.is_relative_to(Path(data_dir).resolve()) or output.is_relative_to(C.HERE):
        raise ValueError("External outputs must be isolated from original data and evaluator sources")
    for row in rows:
        if output.is_relative_to(resolved(base,row["checkpoint"]).parent):
            raise ValueError("External outputs must be isolated from every historical/formal checkpoint run")
    if any((output/name).exists() for name in ("best.pt","last.pt","history.csv","status.json","result.json")):
        raise ValueError("The output directory contains unrelated training-run artifacts")
    identity = original_identity(data_dir)
    build_original_index(data_dir, output, identity)
    records, errors = [], []
    for row, mid in zip(rows, actual_ids):
        expected = row["expected_training_sources"]
        bound, source_errors = source_bindings(base, expected, row["source_file_bindings"])
        provenance = row.get("expected_provenance_sources", {})
        extra, extra_errors = source_bindings(base, provenance, row.get("provenance_source_file_bindings", {}))
        path = resolved(base, row["checkpoint"])
        record = {"id": mid, "group": row["group"], "seed": row["seed"],
                  "checkpoint": str(path), "policies": C.policies(row["group"]),
                  "expected_training_sources": expected, "source_file_bindings": bound,
                  "expected_provenance_sources": provenance, "provenance_source_file_bindings": extra,
                  "selector_evidence": row.get("selector_evidence", {}), "ready": False}
        row_errors = source_errors + extra_errors
        if not path.is_file():
            row_errors.append("Checkpoint has not been produced")
        else:
            actual_sha = C.sha(path)
            expected_sha = row.get("expected_weight_sha256")
            # New confidence weights acquire their identity before access. Old
            # checkpoints must have an externally supplied immutable SHA pin.
            if not expected_sha and row["group"] != "confidence_mass_LAST":
                row_errors.append("Historical checkpoint lacks a frozen SHA pin")
            elif expected_sha and expected_sha != actual_sha:
                row_errors.append("Checkpoint SHA mismatch")
            record.update(checkpoint_sha256=actual_sha, bytes=path.stat().st_size)
            if not row_errors:
                with C.preserve_runtime():
                    ckpt = torch.load(path, map_location="cpu", weights_only=False)
                    if ckpt["source_sha256"] != expected:
                        raise ValueError(f"Original training source dictionary differs: {mid}")
                    if ckpt["config"] != row["expected_config"]:
                        raise ValueError(f"Full checkpoint configuration differs: {mid}")
                    validate_config(row["group"], row["seed"], ckpt["config"], ckpt["epoch"])
                    if ckpt["stats"] != identity["stats"]:
                        raise ValueError(f"Training-only normalization/split differs: {mid}")
                    if expected.get("models.py") != C.sha(C.REPO / "models.py"):
                        raise ValueError("The inference architecture source is not the trained architecture")
                    import models as model_module
                    if Path(model_module.__file__).resolve()!=(C.REPO/"models.py").resolve():
                        raise ValueError("The imported architecture module is not the bound checkout source")
                    model = model_module.make_model(ckpt["config"])
                    model.load_state_dict(ckpt["model"], strict=True)
                    record.update(config=ckpt["config"], stats=ckpt["stats"], epoch=ckpt["epoch"],
                                  model_state_sha256=C.state_hash(model), ready=True)
                    del model, ckpt
        record["errors"] = row_errors
        errors.extend([f"{mid}: {e}" for e in row_errors]); records.append(record)
    unique_weights = [r.get("checkpoint_sha256") for r in records if r["ready"]]
    if len(unique_weights) != len(set(unique_weights)):
        raise ValueError("Different registered checkpoint identities share a weight file")
    value = {"kind": "external_confirmation_explicit_manifest_v1", "created_utc": utc(),
             "spec_path": str(spec_path), "spec_sha256": C.sha(spec_path),
             "data_dir": str(Path(data_dir).resolve()), "original_identity": identity,
             "evaluator_sources": C.sources(), "source_metadata_sha256": C.sha(C.HERE / "source_metadata.json"),
             "models": records, "ready_checkpoints": sum(r["ready"] for r in records),
             "seed_policy_rows": sum(len(r["policies"]) for r in records),
             "complete": not errors and len(records) == 21, "errors": errors,
             "external_dataset_accessed": False}
    existing_path=output/"manifest.json"
    if existing_path.exists():
        existing=C.read(existing_path)
        # A coordinator may restart the already completed prepare step. Keep
        # identical manifest bytes so its existing actual GPU proof remains
        # attached to the exact same binding, rather than a new timestamp.
        if ({k:v for k,v in existing.items() if k!="created_utc"}
                =={k:v for k,v in value.items() if k!="created_utc"}):
            return existing
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    C.write(output / "manifest_history" / (stamp + ".json"), value)
    C.write(output / "manifest.json", value)
    return value


def current(output, complete=True):
    output = Path(output); value = C.read(output / "manifest.json")
    C.check_sources(value["evaluator_sources"])
    rows=value["models"]
    wanted={C.model_id(g,s) for g in C.GROUPS for s in C.SEEDS}
    if (len(rows)!=21 or len({r["id"] for r in rows})!=21 or {r["id"] for r in rows}!=wanted
            or any(r["id"]!=C.model_id(r["group"],r["seed"]) or r["policies"]!=C.policies(r["group"]) for r in rows)):
        raise ValueError("Manifest model identities/policies differ from the fixed registration")
    if (value["ready_checkpoints"]!=sum(r["ready"] for r in rows)
            or value["seed_policy_rows"]!=sum(len(r["policies"]) for r in rows)):
        raise ValueError("Manifest row counts do not match its actual bindings")
    if complete and (not value["complete"] or value["ready_checkpoints"] != 21 or value["seed_policy_rows"] != 24
                     or any(not r["ready"] or r["errors"] for r in rows)):
        raise RuntimeError("All 21 original checkpoint/source bindings and 24 rows are required")
    for row in value["models"]:
        if row["ready"]:
            if C.sha(row["checkpoint"]) != row["checkpoint_sha256"]:
                raise ValueError("A frozen checkpoint changed")
        for key in ("source_file_bindings", "provenance_source_file_bindings"):
            for source in row[key].values():
                path = Path(source["path"])
                if complete and (not path.is_file() or C.sha(path) != source["expected_sha256"]):
                    raise ValueError("A bound original source changed or is unavailable")
    return value


def verify_original_preflight_cache_chain(output, manifest, proof):
    """Re-read the original-validation caches; never invoke sealed/fetch/GPU.

    Both initial sealing and every later sealed operation use this function.
    load_cache with stage=original_validation only reads manifest/cache bytes,
    so this deliberately has no sealed -> external-cache -> sealed recursion.
    """
    from .inference import load_cache
    output=Path(output)
    if (proof.get("kind") != "external_actual_original_validation_GPU_v1" or not proof.get("passed")
            or proof.get("manifest_sha256") != C.sha(output / "manifest.json")
            or proof.get("evaluator_sources") != manifest["evaluator_sources"]
            or len(proof.get("models", [])) != 21 or proof.get("raw_policy_checks") != 21
            or proof.get("two_view_policy_checks") != 3 or proof.get("small_last_batch") != 1
            or not proof.get("all_RNG_runtime_unchanged") or proof.get("external_data_read") is not False
            or proof.get("environment",{}).get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8"):
        raise RuntimeError("A complete actual original-validation CUDA evaluator preflight is required")
    cached=[]
    for row in manifest["models"]:
        cache=load_cache(output,row,"original_validation")
        if cache is None:
            raise RuntimeError("A real fixed original-validation logit cache is missing")
        checks=cache["checks"]
        if (not checks.get("passed") or checks.get("device")!="cuda" or checks.get("samples")!=257
                or checks.get("batch_size")!=128 or checks.get("small_last_batch")!=1
                or not checks.get("identical_batch_repeat_exact") or not checks.get("weights_BN_grad_metadata_unchanged")
                or [p["index"] for p in checks.get("raw_scalar",[])]!=[0,127,256]
                or [p["index"] for p in checks.get("two_view_scalar",[])]!=([0,127,256] if row["group"]=="main_KD_BEST" else [])):
            raise RuntimeError("GPU cache lacks required batch/scalar/weight guards")
        cached.append(cache)
    if proof["models"]!=cached:
        raise ValueError("GPU proof no longer matches its complete actual logit/check receipts")
    return cached


def seal(output):
    from .dataset import original_identity
    output = Path(output); C.require_no_access(output)
    manifest = current(output)
    if original_identity(manifest["data_dir"]) != manifest["original_identity"]:
        raise ValueError("Original data metadata or raw batches changed")
    proof = C.read(output / "preflight.json")
    verify_original_preflight_cache_chain(output,manifest,proof)
    index = C.read(output / "original_index.json")
    if C.sha(output / "original_index.npz") != index["npz_sha256"]:
        raise ValueError("Original 60000 pixel-overlap index changed")
    value = {"kind": "external_confirmation_seal_v1", "sealed_utc": utc(),
             "manifest_sha256": C.sha(output / "manifest.json"), "preflight_sha256": C.sha(output / "preflight.json"),
             "original_index_sha256": C.sha(output / "original_index.npz"),
             "original_index_metadata_sha256": C.sha(output / "original_index.json"),
             "evaluator_sources": manifest["evaluator_sources"], "checkpoints": 21, "seed_policy_rows": 24}
    if (output / "seal.json").exists() and C.read(output / "seal.json") != value:
        # UTC time must not allow overwriting an existing seal.
        existing = C.read(output / "seal.json")
        if {k:v for k,v in existing.items() if k != "sealed_utc"} != {k:v for k,v in value.items() if k != "sealed_utc"}:
            raise RuntimeError("An existing seal is immutable")
        return existing
    C.write(output / "seal.json", value)
    return value


def sealed(output):
    output = Path(output); value = C.read(output / "seal.json"); manifest = current(output)
    expected = {"manifest_sha256": "manifest.json", "preflight_sha256": "preflight.json",
                "original_index_sha256": "original_index.npz", "original_index_metadata_sha256": "original_index.json"}
    for key, filename in expected.items():
        if C.sha(output / filename) != value[key]:
            raise ValueError(f"Sealed artifact changed: {filename}")
    if value["evaluator_sources"] != manifest["evaluator_sources"]:
        raise ValueError("Seal/source identity mismatch")
    # A seal protects the original-validation proof's complete actual array and
    # metadata chain, not only its summary JSON. Recheck before access_started.
    verify_original_preflight_cache_chain(output,manifest,C.read(output/"preflight.json"))
    return manifest, value
