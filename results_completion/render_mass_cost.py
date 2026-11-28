"""CPU-only presentation derivative of locked figure06; never changes science.

No model, PT, cache array, GPU, resampling, or clock measurement is imported or
performed. Historical costs remain exactly as recorded in analysis/results.json.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

MODES = ("original", "weighted_single", "confidence_mass")
SEEDS = (42, 17, 73)
LABELS = {"original": "原KL", "weighted_single": "条件结构损失", "confidence_mass": "显式系数对照"}
COLORS = {"original": "#777777", "weighted_single": "#28688b", "confidence_mass": "#bd673e"}
CONTRASTS = (
    ("confidence_mass_minus_original", "系数对照－原KL"),
    ("weighted_single_minus_original", "条件结构－原KL"),
    ("weighted_single_minus_confidence_mass", "条件结构－系数对照"),
)
OLD_TITLE = "训练原始轮时不能反推准确休眠时长；新只读拟合另记录系统唤醒/休眠计数\n系数总量对照不等于梯度范数匹配，也不等于纯GPU计时"
NEW_TITLE = "原始轮时与长暂停全部保留；只读拟合仅有墙钟记录，未取得系统唤醒或休眠计数\n系数总量对照不等于梯度范数匹配；训练轮时含教师开销，并非纯GPU计时"

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def reference(p):
    return {"path": str(Path(p).resolve()), "sha256": sha(p)}

def payload_sha(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def main():
    parser=argparse.ArgumentParser(description='Draw the reviewed cost/fixed-seed figure from published recorded values only; no model, test, resampling or clock measurement')
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing original/derived outputs')
    manifest=read(args.manifest)
    profile=manifest['presentation_revisions']['mass06']
    source=read(args.summary)
    if source['kind']!='derived_public_field_selection_of_original_generated_summary' or source['original_summary_sha256']!=profile['analysis_results_sha256']:
        raise ValueError('Published summary does not match the recorded source')
    result=source['data']
    fit_cost=result['fit_cost']
    for seed in SEEDS:
        for field in ('start','end'):
            if set(fit_cost[str(seed)][field])!={'monotonic','utc'}:raise ValueError('Clock fields differ; review the caption')
    payload={key:result[key] for key in ('training_cost','seed_effects','fit_cost')}
    if payload_sha(payload)!=profile['data_payload_sha256']:raise ValueError('Recorded values changed')
    target=args.output.resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    names, raw, point_rows = [], [], {}
    for mode in MODES:
        for seed in SEEDS:
            key = mode + "_" + str(seed)
            values = result["training_cost"][key]["raw_epoch_seconds"]
            if len(values) != 240 or not all(isinstance(x, (float, int)) and math.isfinite(x) and x >= 0 for x in values):
                raise ValueError("Expected all240 original finite epoch times, including every long pause")
            names.append(LABELS[mode] + str(seed))
            raw.append(values)
    for key, label in CONTRASTS:
        rows = result["seed_effects"][key]
        if len(rows) != 3 or {row["seed"] for row in rows} != set(SEEDS):
            raise ValueError("Fixed seed mapping is incomplete or duplicated")
        by_seed = {row["seed"]: row for row in rows}
        point_rows[key] = [by_seed[seed] for seed in SEEDS]

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    font_manager.findfont("SimSun", fallback_to_default=False)
    plt.rcParams.update({"font.family": "SimSun", "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # These two panels retain the original figure06 plotting operations, values,
    # order, colors, scale, fliers, titles and axis labels exactly.
    axes[0].boxplot(raw, showfliers=True)
    axes[0].set(xticks=range(1, 10), xticklabels=names, title="全部原始轮时，异常长轮保留", ylabel="秒/轮")
    axes[0].tick_params(axis="x", rotation=55)
    medians = [np.median(x) for x in raw]
    axes[1].bar(range(9), medians, color=[COLORS[m] for m in MODES for _ in SEEDS])
    axes[1].set(xticks=range(9), xticklabels=names, title="含教师开销的轮时中位数", ylabel="秒/轮")
    axes[1].tick_params(axis="x", rotation=55)
    for i, (key, label) in enumerate(CONTRASTS):
        axes[2].scatter(range(3), [r["FP64_logit_NLL_difference"] for r in point_rows[key]],
                        marker=("o", "s", "^")[i], s=38, label=label, color=("#1f77b4", "#ff7f0e", "#2ca02c")[i])
    axes[2].axhline(0, color="gray", ls=":")
    axes[2].set(xticks=range(3), xticklabels=[str(seed) for seed in SEEDS], xlabel="固定训练种子（分类标签）",
                ylabel="FP64描述NLL差", title="每个固定种子结果全部保留")
    axes[2].legend(fontsize=8)
    fig.suptitle(NEW_TITLE)
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({'figure':str(target),'sha256':sha(target),'data_payload_sha256':payload_sha(payload),'new_model_forwards':0,'new_clock_measurements':0,'new_resampling':0},ensure_ascii=False))
if __name__=='__main__':main()
