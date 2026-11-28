"""Plot only completed, source-locked actual5000 results; manual review remains."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np

from . import protocol as p
from .analysis import read_full

CLASS_NAMES = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "船", "卡车"]


def plot(output, data_dir, font_family="SimSun"):
    output = Path(output)
    summary = p.read(output / "analysis.json")
    if summary["source_sha256"] != p.sources() or summary["identity_sha256"] != p.sha(output / "identity.json"):
        raise ValueError("Analysis is not bound to current frozen source/input identity")
    identity, data, receipts = read_full(output)
    if p.data_identity(data_dir) != identity["data"]:
        raise ValueError("Figure image directory differs from the frozen original training-batch data")
    font_manager.findfont(font_family, fallback_to_default=False)
    plt.rcParams.update({"font.family": font_family, "font.size": 10, "axes.unicode_minus": False})
    destination = output / "figures"; destination.mkdir(exist_ok=True)
    files = []
    def save(figure, name):
        path = destination / name; figure.savefig(path, dpi=180, bbox_inches="tight"); plt.close(figure); files.append(path)
    point = np.asarray(summary["overall"]["mean"]); interval = np.asarray(summary["overall"]["conditional_image_ci95"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].errorbar(range(4), point, yerr=np.stack((point - interval[:, 0], interval[:, 1] - point)), fmt="o", capsize=4)
    axes[0].axhline(0, color="gray", linewidth=.8); axes[0].set(xticks=range(4), xticklabels=["42", "17", "73", "三模型逐图均值"], ylabel="高响应窗降幅−随机窗降幅", title="同5000图的类别分层配对区间")
    for seed in p.SEEDS:
        value = np.sort(data[seed]["dq"])
        axes[1].plot(value, np.arange(1, 5001) / 5000, label=str(seed))
    axes[1].axvline(0, color="gray", linewidth=.8); axes[1].set(xlabel="逐图高响应窗−随机窗降幅", ylabel="累计比例", title="全部样本，包括负例与常量图"); axes[1].legend(title="种子")
    save(fig, "01_full5000_sensitivity.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    table = np.asarray([row["mean"][:3] for row in summary["classes"]]).T
    image = axes[0].imshow(table, aspect="auto", cmap="coolwarm"); fig.colorbar(image, ax=axes[0], label="平均降幅差")
    axes[0].set(xticks=range(10), xticklabels=CLASS_NAMES, yticks=range(3), yticklabels=[42, 17, 73], xlabel="真实类别（每类500图）", ylabel="训练种子", title="每类与每模型结果")
    means = np.asarray([row["mean"][3] for row in summary["classes"]]); ci = np.asarray([row["conditional_image_ci95"][3] for row in summary["classes"]])
    axes[1].errorbar(means, range(10), xerr=np.stack((means - ci[:, 0], ci[:, 1] - means)), fmt="o", capsize=3)
    axes[1].axvline(0, color="gray", linewidth=.8); axes[1].set(yticks=range(10), yticklabels=CLASS_NAMES, xlabel="三模型逐图平均降幅差", ylabel="真实类别", title="同图配对的类别条件区间")
    save(fig, "02_per_class.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    groups = [("目标预测正确", summary["cohorts"]["fixed_target_correct"]), ("目标预测错误", summary["cohorts"]["fixed_target_wrong"])]
    for index, (label, group) in enumerate(groups):
        if group:
            mean = group["mean"][3]; lo, hi = group["conditional_image_ci95"][3]
            axes[0].errorbar(index, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", capsize=4)
    axes[0].set(xticks=[0, 1], xticklabels=[f"{name}\n{group['n'] if group else 0}图" for name, group in groups], ylabel="三模型逐图平均降幅差", title="按原固定目标的正确性分组")
    axes[1].bar(range(3), [row["flat_count"] for row in summary["seeds"]]); axes[1].set(xticks=range(3), xticklabels=p.SEEDS, ylabel="样本数", title="常量正热图全部保留")
    for seed in (17, 73):
        row = summary["seeds"][list(p.SEEDS).index(seed)]
        axes[2].bar(str(seed), row["heatmap_TV_vs42_mean"] or 0, label=f"有效{row['heatmap_TV_vs42_valid_pairs']}图")
    axes[2].set(ylabel="正热图归一化TV距离", title="与种子42的图形差异"); axes[2].legend()
    save(fig, "03_errors_flat_and_map_distance.png")
    raw, _, original = p.load_validation(data_dir); lookup = {int(value): row for row, value in enumerate(original)}
    examples = []; seen = set()
    for item in summary["example_selection"]:
        if item["row"] not in seen:
            seen.add(item["row"]); examples.append(item)
    fig, axes = plt.subplots(len(examples), 4, figsize=(11, max(3, len(examples) * 2.7)), squeeze=False, layout="constrained")
    reasons = {"fixed_target_correct": "目标正确", "fixed_target_wrong": "目标错误", "negative_high_drop": "概率反升例", "flat_map": "常量图例"}
    for rr, item in enumerate(examples):
        row, official = item["row"], item["official_train_index"]
        image = raw[lookup[official]].permute(1, 2, 0).numpy(); axes[rr, 0].imshow(image); axes[rr, 0].axis("off")
        axes[rr, 0].set_title(f"{reasons[item['reason']]}，索引{official}\n真类{CLASS_NAMES[data[42]['labels'][row]]}，固定目标{CLASS_NAMES[data[42]['target'][row]]}")
        for cc, seed in enumerate(p.SEEDS, 1):
            value = data[seed]; heat = value["heat"][row]; yy, xx = value["positions"][row, 0]
            axes[rr, cc].imshow(image); axes[rr, cc].imshow(heat, cmap="jet", alpha=.4, vmin=0, vmax=float(heat.max()) or 1)
            axes[rr, cc].add_patch(Rectangle((xx-.5, yy-.5), 8, 8, fill=False, edgecolor="white", linewidth=1.3)); axes[rr, cc].axis("off")
            before = value["q"][row, value["target"][row]]; after = value["maskedq"][row, 0, value["target"][row]]
            axes[rr, cc].set_title(f"种子{seed}，遮挡前后{before:.3f}→{after:.3f}")
    fig.suptitle("预定最小索引规则选例；颜色逐图缩放，不能跨图比较深浅")
    save(fig, "04_fixed_rule_examples.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    groups = [("原200", summary["cohorts"]["original200"]), ("新增4800", summary["cohorts"]["remaining4800"]), ("全部5000", summary["overall"])]
    for index, (name, group) in enumerate(groups):
        mean = group["mean"][3]; lo, hi = group["conditional_image_ci95"][3]
        axes[0].errorbar(index, mean, yerr=[[mean-lo], [hi-mean]], fmt="o", capsize=4)
    axes[0].set(xticks=range(3), xticklabels=[name for name, _ in groups], ylabel="三模型逐图平均降幅差", title="覆盖范围；原200不是随机代表样本")
    bridge = {seed: p.read(output / "bridge" / f"seed{seed}.json") for seed in p.SEEDS}
    wall = [receipts[seed]["compute"]["wall_seconds_including_pauses"] + bridge[seed]["compute"]["wall_seconds_including_pauses"] + bridge[seed]["single_image_semantics"]["wall_seconds_including_CPU_autograd_and_pauses"] for seed in p.SEEDS]
    axes[1].bar(range(3), wall); axes[1].set(xticks=range(3), xticklabels=p.SEEDS, ylabel="批量推理与逐图/特征核查秒数（含CPU与暂停）", title="原200桥接与新增4800的计算记录")
    save(fig, "05_coverage_and_cost.png")
    manifest = {"state": "figures_generated_manual_pending", "source_sha256": p.sources(),
                "identity_sha256": p.sha(output / "identity.json"), "analysis_sha256": p.sha(output / "analysis.json"),
                "figures": {str(path.relative_to(output)): p.sha(path) for path in files},
                "actual_manual_visual_review_pending": True, "automatically_claimed_reviewed": False}
    p.write(output / "figures_manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True); parser.add_argument("--font-family", default="SimSun")
    args = parser.parse_args(argv); result = plot(args.output_dir, args.data_dir, args.font_family)
    print(json.dumps({"state": result["state"], "figure_count": len(result["figures"]), "actual_manual_review_pending": True}))


if __name__ == "__main__":
    main()
