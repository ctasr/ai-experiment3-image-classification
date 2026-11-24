"""下载并核验CIFAR-10，固定训练/验证划分，保留官方测试集。"""
from __future__ import annotations

import hashlib
import json
import pickle
import random
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
OFFICIAL_MD5 = "c58f30108f718f92721af3b95e74349a"
SEED = 42
VALIDATION_PER_CLASS = 500
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def digest(path: Path, algorithm: str = "sha256") -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save_json(path: Path, value: dict) -> None:
    """已有划分不静默覆盖，避免不同实验意外使用不同索引。"""
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"已有记录与当前配置不同：{path}")
        return
    path.write_text(content, encoding="utf-8", newline="\n")


def download() -> Path:
    archive = RAW / "cifar-10-python.tar.gz"
    if archive.exists():
        if digest(archive, "md5") != OFFICIAL_MD5:
            raise ValueError("已有数据包未通过官方MD5核验，请检查文件")
        print("数据包已存在，官方MD5核验通过。", flush=True)
        return archive
    partial = archive.with_suffix(archive.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "CIFAR10-coursework-data-download"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(URL, headers=headers)
    print("正在连接CIFAR-10官方服务器……", flush=True)
    with urllib.request.urlopen(request, timeout=45) as response:
        if offset and response.status != 206:
            offset = 0
        if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("服务器返回的续传位置不匹配")
        total = int(response.headers.get("Content-Length", "0")) + offset
        received = offset
        started = previous = time.monotonic()
        with partial.open("ab" if offset else "wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                received += len(chunk)
                now = time.monotonic()
                if now - previous >= 5:
                    percent = f"，{received / total:.1%}" if total else ""
                    speed = (received - offset) / max(now - started, .001) / 1024 ** 2
                    print(f"已下载{received / 1024 ** 2:.1f}MiB{percent}，平均{speed:.2f}MiB/s", flush=True)
                    previous = now
    if total and received != total:
        raise ValueError(f"下载长度不完整：{received}/{total}")
    if digest(partial, "md5") != OFFICIAL_MD5:
        raise ValueError("下载文件未通过官方MD5核验，不解压、不读取数据")
    partial.replace(archive)
    print(f"下载完成：{received / 1024 ** 2:.1f}MiB，官方MD5核验通过。", flush=True)
    return archive


def extract_verified(archive: Path) -> Path:
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            target = (RAW / member.name).resolve()
            if not target.is_relative_to(RAW.resolve()) or not (member.isdir() or member.isfile()):
                raise ValueError(f"压缩包中存在不支持的成员：{member.name}")
        for member in members:
            target = RAW / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            content = bundle.extractfile(member).read()
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError(f"已有解压文件与官方包不一致：{member.name}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    return RAW / "cifar-10-batches-py"


def read_batch(path: Path) -> tuple[list[int], int]:
    # 只读取已通过官方压缩包完整性核验的数据。
    with path.open("rb") as stream:
        batch = pickle.load(stream, encoding="bytes")
    images, labels = batch[b"data"], batch[b"labels"]
    if images.shape != (10000, 3072) or str(images.dtype) != "uint8":
        raise ValueError(f"图像形状或类型错误：{path.name}")
    if len(labels) != 10000 or any(label not in range(10) for label in labels):
        raise ValueError(f"标签数量或范围错误：{path.name}")
    return labels, len(images)


def make_split(labels: list[int]) -> dict:
    rng = random.Random(SEED)
    train, validation = [], []
    for label in range(10):
        indices = [i for i, value in enumerate(labels) if value == label]
        if len(indices) != 5000:
            raise ValueError(f"官方训练集类别{label}样本数错误")
        rng.shuffle(indices)
        validation.extend(indices[:VALIDATION_PER_CLASS])
        train.extend(indices[VALIDATION_PER_CLASS:])
    train.sort()
    validation.sort()
    assert len(train) == 45000 and len(validation) == 5000
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(range(50000))
    assert Counter(labels[i] for i in train) == Counter({i: 4500 for i in range(10)})
    assert Counter(labels[i] for i in validation) == Counter({i: 500 for i in range(10)})
    return {
        "seed": SEED,
        "method": "per-class shuffle with Python random.Random, 500 validation examples per class",
        "source": "CIFAR10(train=True); concatenate data_batch_1 through data_batch_5 in order",
        "index_range": [0, 49999],
        "train_indices": train,
        "validation_indices": validation,
        "test_source": "CIFAR10(train=False); official test_batch; independent index space 0..9999",
    }


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)
    archive = download()
    folder = extract_verified(archive)
    labels = []
    batches = []
    for number in range(1, 6):
        path = folder / f"data_batch_{number}"
        batch_labels, size = read_batch(path)
        batches.append({"file": path.relative_to(ROOT).as_posix(), "count": size, "sha256": digest(path)})
        labels.extend(batch_labels)
    test_path = folder / "test_batch"
    test_labels, test_count = read_batch(test_path)
    assert Counter(test_labels) == Counter({i: 1000 for i in range(10)})
    with (folder / "batches.meta").open("rb") as stream:
        meta = pickle.load(stream, encoding="bytes")
    names = [name.decode("utf-8") for name in meta[b"label_names"]]
    assert names == CLASSES
    split = make_split(labels)
    # 重建一次并比较，确认相同种子得到相同索引。
    assert split == make_split(labels)
    split_path = SPLITS / "train_validation_seed42.json"
    save_json(split_path, split)
    manifest = {
        "dataset": "CIFAR-10 Python version",
        "source_page": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "citation": "Alex Krizhevsky. Learning Multiple Layers of Features from Tiny Images. 2009.",
        "archive_url": URL,
        "archive_md5": OFFICIAL_MD5,
        "archive_sha256": digest(archive),
        "class_names": names,
        "image_shape": [3, 32, 32],
        "official_train_batches": batches,
        "split_file": split_path.relative_to(ROOT).as_posix(),
        "split_sha256": digest(split_path),
        "split_seed": SEED,
        "training": {"source": "official train", "count": 45000, "per_class": [4500] * 10, "purpose": "梯度更新、拟合模型参数"},
        "validation": {"source": "official train", "count": 5000, "per_class": [500] * 10, "purpose": "调参、早停及模型选择，不参与梯度更新"},
        "test": {"source": "official test_batch", "file": test_path.relative_to(ROOT).as_posix(), "sha256": digest(test_path), "count": test_count, "per_class": [1000] * 10, "purpose": "模型及超参数确定后进行最终评估，不用于调参或早停"},
        "checks": {"train_validation_disjoint_indices": True, "train_validation_cover_official_train": True, "test_uses_separate_official_file": True, "deterministic_split_rebuild": True},
        "preprocessing_policy": {"random_augmentation": "只用于训练集", "validation_and_test": "使用确定性预处理", "fitted_statistics": "若计算归一化统计量，仅由45000张训练图估计"},
    }
    save_json(ROOT / "data" / "dataset_manifest.json", manifest)
    print("数据准备完成：训练45000张、验证5000张、测试10000张。", flush=True)
    print("训练/验证索引互斥，合计覆盖全部官方训练数据；测试集来自独立官方文件。", flush=True)
    print(f"数据清单：{ROOT / 'data/dataset_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
