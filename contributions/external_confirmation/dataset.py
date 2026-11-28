"""Original-only preparation and a separately gated, single external fetch."""
from pathlib import Path, PurePosixPath
import datetime
import hashlib
import pickle
import urllib.request
import numpy as np

from . import common as C

DATA_SHA = {"manifest_sha256": "fcde007c02057463cc73e6024bcb2759a656e222e2f46219c3a5a75265013360",
            "split_sha256": "65b2461b3b2f1150516885f32fe224edfed12925b203b1fa8ed305329f4644ce",
            "normalization_sha256": "9582053fe89152dc5e78472cf4e1e20832bfafb48f539c5903e051b639955455"}


def data_path(data_dir, name):
    parts = PurePosixPath(str(name).replace("\\", "/")).parts
    if parts and parts[0] == "data":
        parts = parts[1:]
    if not parts or any(x in ("..", "") for x in parts):
        raise ValueError("Unsafe original-data path")
    result = (Path(data_dir) / Path(*parts)).resolve()
    if not result.is_relative_to(Path(data_dir).resolve()):
        raise ValueError("Original-data path escaped --data-dir")
    return result


def raw_training(data_dir):
    metadata = C.read(Path(data_dir) / "dataset_manifest.json")
    images, labels = [], []
    for item in metadata["official_train_batches"]:
        path = data_path(data_dir, item["file"])
        if C.sha(path) != item["sha256"]:
            raise ValueError("Original training pickle SHA differs")
        with path.open("rb") as stream:
            value = pickle.load(stream, encoding="bytes")
        images.append(value[b"data"]); labels.extend(value[b"labels"])
    return np.vstack(images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy(), np.asarray(labels, np.int64)


def original_identity(data_dir):
    data_dir = Path(data_dir); metadata = C.read(data_dir / "dataset_manifest.json")
    split_path = data_path(data_dir, metadata["split_file"])
    hashes = {"manifest_sha256": C.sha(data_dir / "dataset_manifest.json"), "split_sha256": C.sha(split_path),
              "normalization_sha256": C.sha(data_dir / "normalization.json")}
    if hashes != DATA_SHA or tuple(metadata["class_names"]) != C.CLASSES:
        raise ValueError("Expected the original fixed CIFAR10 metadata/split/normalization")
    split = C.read(split_path)
    train, val = split["train_indices"], split["validation_indices"]
    if (len(train) != 45000 or len(val) != 5000 or len(set(train)) != 45000 or len(set(val)) != 5000
            or set(train) & set(val) or set(train) | set(val) != set(range(50000))):
        raise ValueError("Expected a disjoint full 45k/5k original-training partition")
    raw = {}
    for item in metadata["official_train_batches"] + [metadata["test"]]:
        path = data_path(data_dir, item["file"])
        if C.sha(path) != item["sha256"]:
            raise ValueError("Original raw-pixel source changed")
        raw[path.name] = item["sha256"]
    normalization = C.read(data_dir / "normalization.json")
    class_meta=data_dir/"raw/cifar-10-batches-py/batches.meta"
    with class_meta.open("rb") as stream:
        names=pickle.load(stream,encoding="bytes")[b"label_names"]
    if tuple(name.decode("ascii") if isinstance(name,bytes) else name for name in names)!=C.CLASSES:
        raise ValueError("Original CIFAR10 class metadata order differs")
    stats = normalization.get("stats", normalization)
    # normalization.json contains bookkeeping as well as the trained stats.
    stats = {"source": "fixed 45000 training images only", "split_sha256": hashes["split_sha256"],
             "mean": stats["mean"], "std": stats["std"]}
    return {**hashes, "raw_files": raw, "class_metadata_sha256":C.sha(class_meta), "stats": stats, "class_names": list(C.CLASSES),
            "original_train_samples": 45000, "original_validation_samples": 5000,
            "official_test_pixels_read_for_overlap_only": True,
            "official_test_model_evaluation": False}


def pixel_hashes(images):
    x = np.asarray(images)
    if x.dtype != np.uint8 or x.ndim != 4 or x.shape[1:] != (32, 32, 3):
        raise ValueError("Pixel hashes require canonical HWC RGB uint8 images")
    return np.asarray([hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest().encode("ascii") for row in x], dtype="S64")


def build_original_index(data_dir, output, identity):
    output = Path(output); npz = output / "original_index.npz"; receipt = output / "original_index.json"
    if npz.exists() or receipt.exists():
        if not (npz.exists() and receipt.exists()):
            raise RuntimeError("An incomplete original index must be preserved and investigated")
        old = C.read(receipt)
        if old["identity"] != identity or C.sha(npz) != old["npz_sha256"]:
            raise ValueError("Existing original-pixel index identity differs")
        return old
    x, y = raw_training(data_dir)
    metadata = C.read(Path(data_dir) / "dataset_manifest.json")
    with data_path(data_dir, metadata["test"]["file"]).open("rb") as stream:
        test = pickle.load(stream, encoding="bytes")
    xt = test[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    yt = np.asarray(test[b"labels"], np.int64)
    if x.shape != (50000,32,32,3) or xt.shape != (10000,32,32,3) or y.shape != (50000,) or yt.shape != (10000,):
        raise ValueError("Original 60000-image schema differs")
    split = C.read(data_path(data_dir, metadata["split_file"]))
    partitions = np.zeros(60000, dtype=np.int8)
    partitions[np.asarray(split["validation_indices"])] = 1
    partitions[50000:] = 2
    if not np.array_equal(np.bincount(y[np.asarray(split["train_indices"])]), np.full(10,4500)) or not np.array_equal(np.bincount(y[np.asarray(split["validation_indices"])]), np.full(10,500)):
        raise ValueError("Original training/validation class count differs")
    C.save_npz(npz, pixel_sha256=np.concatenate([pixel_hashes(x),pixel_hashes(xt)]),
               labels=np.concatenate([y,yt]), partition=partitions)
    value = {"kind": "original60000_exact_pixel_index_v1", "identity": identity,
             "samples": 60000, "partition_names": ["train45000", "validation5000", "officialtest10000"],
             "npz_sha256": C.sha(npz), "no_model_prediction_or_new_test_evaluation": True}
    C.write(receipt, value); return value


def schema(images, labels):
    if images.dtype != np.uint8 or images.shape != (2000,32,32,3):
        raise ValueError("The pinned external image schema must be (2000,32,32,3) HWC RGB uint8")
    if labels.dtype != np.int64 or labels.shape != (2000,) or np.any((labels < 0) | (labels > 9)):
        raise ValueError("The pinned external labels must be int64(2000) in canonical 0..9 order")
    if np.any(np.bincount(labels, minlength=10) == 0):
        raise ValueError("All canonical classes must be present")


def fetch(output):
    """Only this explicit command may contact the pinned image-byte URL."""
    from .manifest import sealed, utc
    output = Path(output); manifest, seal = sealed(output)
    if (output / "retired.json").exists():
        raise RuntimeError("This external dataset has been retired from further access")
    meta = C.read(C.HERE / "source_metadata.json")
    start = {"kind": "external_access_started_v1", "seal_sha256": C.sha(output / "seal.json"),
             "manifest_sha256": C.sha(output / "manifest.json"), "source_metadata_sha256": C.sha(C.HERE / "source_metadata.json")}
    marker = output / "access_started.json"
    if marker.exists():
        old = C.read(marker)
        if {k:v for k,v in old.items() if k != "started_utc"} != start:
            raise ValueError("Access marker belongs to a different frozen confirmation")
    else:
        C.write(marker, {**start, "started_utc": utc()})
    target = output / "cifar102_test.npz"; receipt = output / "fetch.json"
    if target.exists() or receipt.exists():
        if not (target.exists() and receipt.exists()):
            raise RuntimeError("Incomplete external fetch artifacts are preserved; investigate them")
        loaded = C.read(receipt)
        if C.sha(target) != loaded["file_sha256"] or loaded["seal_sha256"] != start["seal_sha256"]:
            raise ValueError("Cached external dataset identity differs")
        load_external(output)
        return loaded
    # URL is constructed from the pinned metadata; no latest branch or fallback.
    commit = "521467a52ad494b78ff9c70051099c0d9ba74d40"
    blob = "7a3829681be56af70ea80ec99cf3822e2f5edff3"; size = 6393664
    url = "https://raw.githubusercontent.com/modestyachts/cifar-10.2/" + commit + "/cifar102_test.npz"
    if meta.get("commit", meta.get("commit_sha")) not in (None, commit):
        raise ValueError("Pinned metadata commit changed")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    partial = output / ("fetch_" + stamp + ".part")
    try:
        git = hashlib.sha1(b"blob " + str(size).encode("ascii") + b"\0")
        digest = hashlib.sha256(); count = 0
        with urllib.request.urlopen(url, timeout=60) as incoming, partial.open("xb") as outgoing:
            while True:
                block = incoming.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > size:
                    raise ValueError("External response exceeds the pinned byte count")
                outgoing.write(block); git.update(block); digest.update(block)
        if count != size or git.hexdigest() != blob:
            raise ValueError("External Git blob SHA1/byte count mismatch")
        # Parse only after pinned-byte identity passes, before model inference.
        with np.load(partial, allow_pickle=False) as source:
            images, labels = source["images"], source["labels"]
            schema(images, labels)
        partial.replace(target)
        value = {"kind": "single_fixed_external_fetch_v1", "fetched_utc": utc(), "commit": commit,
                 "git_blob_sha1": git.hexdigest(), "bytes": count, "file_sha256": digest.hexdigest(),
                 "seal_sha256": start["seal_sha256"], "manifest_sha256": start["manifest_sha256"],
                 "image_schema": [2000,32,32,3], "images_dtype": "uint8", "labels_dtype": "int64",
                 "class_counts": np.bincount(labels, minlength=10).tolist(), "semantic_order": list(C.CLASSES)}
        C.write(receipt, value); return value
    except Exception as exc:
        C.write(output / ("fetch_" + stamp + "_failure.json"), {"error": repr(exc), "partial_path": str(partial),
                "partial_exists": partial.exists(), "identical_frozen_protocol_only_may_resume": True})
        raise


def load_external(output):
    from .manifest import sealed
    output = Path(output); sealed(output)
    start = C.read(output / "access_started.json"); receipt = C.read(output / "fetch.json")
    if start["seal_sha256"] != C.sha(output / "seal.json") or receipt["seal_sha256"] != start["seal_sha256"]:
        raise ValueError("External data access/seal mismatch")
    path = output / "cifar102_test.npz"
    if C.sha(path) != receipt["file_sha256"] or path.stat().st_size != 6393664:
        raise ValueError("External file changed after pinned fetch")
    with np.load(path, allow_pickle=False) as value:
        images, labels = value["images"].copy(), value["labels"].copy()
    schema(images, labels); return images, labels
