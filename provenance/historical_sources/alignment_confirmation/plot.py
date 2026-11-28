"""Render saved raw-BF16 validation evidence for the fixed A/B confirmation.

This module never runs a model or selects a checkpoint. All intervals are
conditional on the locked models and the common validation images.
"""
from datetime import datetime
from pathlib import Path
import csv
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from alignment_confirmation import common as c


CLASS_NAMES = ("飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "船", "卡车")
COLORS = {"A": "#777777", "B": "#28739d"}


def _local(value):
    root = Path(c.ROOT).resolve()
    path = (root / str(value).replace("\\", "/")).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Plot input escapes the project")
    return path


def _style():
    available = {font.name for font in font_manager.fontManager.ttflist}
    chinese = next((name for name in ("SimSun", "Microsoft YaHei", "Noto Sans CJK SC")
                    if name in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": [chinese, "DejaVu Sans"], "font.size": 10,
        "axes.unicode_minus": False, "axes.spines.top": False,
        "axes.spines.right": False, "savefig.dpi": 180,
    })
    Path(c.FIG).mkdir(parents=True, exist_ok=True)


def _save(fig, name, figures, error_gallery=False):
    if error_gallery:
        fig.tight_layout(h_pad=5, w_pad=1, rect=(0, 0, 1, .96))
    else:
        fig.tight_layout()
    path = Path(c.FIG) / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figures[name] = c.sha(path)


def _history(row):
    with _local(row["history_csv"]).open(encoding="utf-8-sig", newline="") as stream:
        result = list(csv.DictReader(stream))
    if [int(record["epoch"]) for record in result] != list(range(1, 161)):
        raise ValueError("Every displayed A/B history must contain the full160 epochs")
    for record in result:
        if int(record["train_samples"]) != 45000 or int(record["validation_samples"]) != 5000:
            raise ValueError("History does not use the fixed45000/5000 partition")
    return result


def _cache(row):
    with np.load(_local(row["cache"]), allow_pickle=False) as stored:
        probabilities = stored["probabilities"].copy()
        labels = stored["labels"].copy()
        indices = stored["indices"].copy()
    assert probabilities.shape == (5000, 10) and probabilities.dtype == np.float32
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(1), 1, atol=2e-6, rtol=0)
    assert labels.shape == indices.shape == (5000,) and len(np.unique(indices)) == 5000
    assert np.array_equal(np.bincount(labels, minlength=10), np.full(10, 500))
    return probabilities, labels, indices


def _official_training_images():
    """Read only the five original training batches, never test_batch."""
    manifest = json.loads((_local("data/dataset_manifest.json")).read_text(encoding="utf-8-sig"))
    images, labels = [], []
    assert len(manifest["official_train_batches"]) == 5
    for item in manifest["official_train_batches"]:
        path = _local(item["file"])
        assert path.name.startswith("data_batch_") and c.sha(path) == item["sha256"]
        with path.open("rb") as stream:
            data = pickle.load(stream, encoding="bytes")
        raw = data[b"data"]
        assert raw.dtype == np.uint8 and raw.shape == (10000, 3072)
        images.append(raw.reshape(-1, 3, 32, 32))
        labels.extend(data[b"labels"])
    return np.concatenate(images), np.asarray(labels, dtype=np.int64)


def generate(analysis):
    """Write seven figures and a SHA manifest from completed validation pairs."""
    _style()
    pairs = analysis["pairs"]
    seeds = sorted(pairs, key=lambda seed: int(seed))
    assert seeds and all(str(seed) in ("17", "42", "73") for seed in seeds)
    figures = {}
    histories = {(seed, role): _history(pairs[seed][role]) for seed in seeds for role in ("A", "B")}
    stage = "三种子完整配对" if len(seeds) == 3 else "当前已完成配对，三种子结论尚待"

    fig, axes = plt.subplots(len(seeds), 2, figsize=(12, 3.4 * len(seeds)), squeeze=False)
    for index, seed in enumerate(seeds):
        for role in ("A", "B"):
            history = histories[seed, role]
            epochs = [int(row["epoch"]) for row in history]
            axes[index, 0].plot(epochs, [float(row["validation_accuracy"]) * 100 for row in history],
                                label=role, color=COLORS[role], linewidth=1)
            axes[index, 1].plot(epochs, [float(row["validation_ce"]) for row in history],
                                label=role, color=COLORS[role], linewidth=1)
        axes[index, 0].set(title=f"种子{seed}：完整160轮原图验证", ylabel="准确率（%）", xlabel="轮次")
        axes[index, 1].set(title="BF16单前向硬标签CE", ylabel="CE", xlabel="轮次")
        for axis in axes[index]:
            axis.legend(fontsize=8)
    _save(fig, "learning_curves.png", figures)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, metric, ci_name, scale, label in (
        (axes[0], "accuracy", "accuracy_CI95", 100, "准确率差（B−A，百分点）"),
        (axes[1], "ce", "ce_CI95", 1, "CE差（B−A，负值较好）"),
    ):
        for index, seed in enumerate(seeds):
            pair = pairs[seed]
            center = float(pair["delta"][metric]) * scale
            low, high = np.asarray(pair["paired"][ci_name], dtype=float) * scale
            assert np.isfinite([center, low, high]).all() and low <= high
            axis.vlines(index, low, high, color=COLORS["B"], linewidth=1.5)
            axis.hlines([low, high], index - .05, index + .05, color=COLORS["B"])
            axis.scatter(index, center, color=COLORS["B"], zorder=3)
        axis.axhline(0, color="gray", linestyle="--", linewidth=.8)
        axis.set(xticks=np.arange(len(seeds)), xticklabels=[f"种子{seed}" for seed in seeds], ylabel=label,
                 title="固定模型的配对图像条件区间，非重训区间")
    _save(fig, "paired_validation_deltas.png", figures)

    differences = np.array([
        np.asarray(pairs[seed]["B"]["validation"]["per_class_accuracy"], dtype=float) -
        np.asarray(pairs[seed]["A"]["validation"]["per_class_accuracy"], dtype=float)
        for seed in seeds]) * 100
    assert differences.shape == (len(seeds), 10) and np.isfinite(differences).all()
    extent = max(1., float(np.abs(differences).max()))
    fig, axis = plt.subplots(figsize=(11, max(2.4, len(seeds) * .9 + 1.4)))
    heat = axis.imshow(differences, cmap="RdBu", vmin=-extent, vmax=extent, aspect="auto")
    axis.set(xticks=range(10), xticklabels=CLASS_NAMES, yticks=range(len(seeds)),
             yticklabels=[f"种子{seed}" for seed in seeds], title="原图验证每类准确率变化：B−A（百分点）")
    for y in range(len(seeds)):
        for x in range(10):
            value = differences[y, x]
            axis.text(x, y, f"{value:+.1f}", ha="center", va="center", fontsize=9,
                      color="white" if abs(value) > .6 * extent else "black")
    fig.colorbar(heat, ax=axis, shrink=.8)
    _save(fig, "per_class_changes.png", figures)

    fig, axes = plt.subplots(len(seeds), 2, figsize=(10.6, 4.6 * len(seeds)), squeeze=False)
    for index, seed in enumerate(seeds):
        for column, role in enumerate(("A", "B")):
            cm = np.asarray(pairs[seed][role]["validation"]["confusion_matrix"], dtype=np.int64)
            assert cm.shape == (10, 10) and np.array_equal(cm.sum(1), np.full(10, 500))
            axis = axes[index, column]
            axis.imshow(cm, cmap="Blues", vmin=0, vmax=500)
            axis.set(title=f"种子{seed}/{role}：原图验证5000张", xlabel="预测类", ylabel="真实类",
                     xticks=range(10), yticks=range(10), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
            axis.tick_params(axis="x", labelrotation=45, labelsize=8)
            axis.tick_params(axis="y", labelsize=8)
            for i in range(10):
                for j in range(10):
                    if cm[i, j]:
                        axis.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                                  color="white" if cm[i, j] > 250 else "black")
    _save(fig, "confusion_matrices.png", figures)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    bar_names, medians, totals = [], [], []
    line_styles = ("-", "--", ":")
    for index, seed in enumerate(seeds):
        for role in ("A", "B"):
            history = histories[seed, role]
            seconds = [float(row["seconds"]) for row in history]
            assert all(np.isfinite(seconds)) and all(value > 0 for value in seconds)
            axes[0].plot(range(1, 161), seconds, label=f"{seed}/{role}", color=COLORS[role],
                         linestyle=line_styles[index], linewidth=.9)
            bar_names.append(f"{seed}/{role}")
            medians.append(float(pairs[seed][role]["cost"]["median_epoch_seconds"]))
            totals.append(float(pairs[seed][role]["cost"]["raw_seconds"]) / 60)
    axes[0].set(xlabel="轮次", ylabel="原始秒/轮", title="原始墙钟时长与全部暂停保留")
    axes[0].legend(fontsize=8)
    axes[1].bar(range(len(medians)), medians, color=[COLORS[name[-1]] for name in bar_names])
    for index, (median, total) in enumerate(zip(medians, totals)):
        axes[1].text(index, median, f"{median:.2f}\n累计{total:.1f}分", ha="center", va="bottom", fontsize=8)
    axes[1].set(xticks=range(len(bar_names)), xticklabels=bar_names, ylabel="秒/轮，中位数",
                title="包含教师成本，非纯GPU或等FLOPs计时")
    axes[1].set_ylim(0, max(medians) * 1.3)
    _save(fig, "cost_and_pauses.png", figures)

    raw, all_labels = _official_training_images()
    gallery_rows = [(seed, role) for seed in seeds for role in ("A", "B")]
    fig, axes = plt.subplots(len(gallery_rows), 3, figsize=(10, 3.7 * len(gallery_rows)), squeeze=False)
    for row_index, (seed, role) in enumerate(gallery_rows):
        item = pairs[seed][role]
        probabilities, labels, indices = _cache(item)
        examples = item["error_examples"][:3]
        for column in range(3):
            axis = axes[row_index, column]
            axis.axis("off")
            if column >= len(examples):
                if column == 0:
                    axis.text(.5, .5, f"种子{seed}/{role}：无已记录错例", ha="center", va="center")
                continue
            example = examples[column]
            validation_row = int(example["validation_row"])
            official_index = int(example["official_train_index"])
            true, predicted = int(example["true"]), int(example["predicted"])
            assert int(indices[validation_row]) == official_index
            assert int(labels[validation_row]) == int(all_labels[official_index]) == true
            assert int(probabilities[validation_row].argmax()) == predicted and true != predicted
            assert abs(float(probabilities[validation_row].max()) - float(example["confidence"])) < 1e-6
            axis.imshow(raw[official_index].transpose(1, 2, 0))
            axis.set_title(f"种子{seed}/{role}　图{official_index}\n"
                           f"{CLASS_NAMES[true]}→{CLASS_NAMES[predicted]}　p={float(example['confidence']):.3f}", fontsize=9)
    fig.suptitle("锁定预测后各模型最高置信错误前3例；不是代表性随机样本", fontsize=10, y=.995)
    _save(fig, "fixed_error_examples.png", figures, error_gallery=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xx = np.arange(len(seeds))
    for offset, role in enumerate(("A", "B")):
        accuracies = [float(pairs[seed][role]["validation"]["accuracy"]) * 100 for seed in seeds]
        ces = [float(pairs[seed][role]["validation"]["ce"]) for seed in seeds]
        axes[0].bar(xx + offset * .35, accuracies, .35, label=role, color=COLORS[role])
        axes[1].bar(xx + offset * .35, ces, .35, label=role, color=COLORS[role])
        for index, value in enumerate(accuracies):
            axes[0].text(index + offset * .35, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        for index, value in enumerate(ces):
            axes[1].text(index + offset * .35, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    for axis in axes:
        axis.set_xticks(xx + .175, [f"种子{seed}" for seed in seeds])
        axis.legend(fontsize=8)
    axes[0].set(ylabel="原图验证准确率（%）", title="固定最佳权重，BF16单前向")
    axes[0].set_ylim(0, 104)
    axes[1].set(ylabel="原图验证硬标签CE", title=stage)
    axes[1].set_ylim(0, max(float(pairs[seed][role]["validation"]["ce"]) for seed in seeds
                          for role in ("A", "B")) * 1.2)
    _save(fig, "validation_summary.png", figures)

    c.write(Path(c.OUT) / "figure_manifest.json", {
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "figures": figures, "manual_visual_review_pending": True,
        "seeds_shown": [int(seed) for seed in seeds], "full_histories_epochs": 160,
        "scope": "Original5000 validation images only; BF16 raw single-forward; A/B matched training seeds.",
        "interval_scope": "Locked-model paired validation-image uncertainty, not independent retraining intervals.",
        "error_selection": "Top3 highest-confidence recorded errors per model after locking predictions; not random or representative.",
        "gate_not_modified": True, "plot_source_sha256": c.sha(Path(__file__)),
        "official_test_images": 0, "external_images": 0,
    })
    return figures
