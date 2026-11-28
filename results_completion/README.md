# 实验结果

| 内容 | 数值 | 图 |
|---|---|---|
| 完整训练与拟合 | [结果](clean/summary.json) | [科学图](clean/figures/) |
| 增强与架构比较 | [结果](controlled/summary.json) | [科学图](controlled/figures/) |
| 损失与显式系数对照 | [结果](mass/summary.json) | [科学图](mass/figures/) |
| 全验证集解释 | [结果](explanation/summary.json) | [科学图](explanation/figures/) |
| 第二划分确认 | [结果](crosssplit/summary.json) | [科学图](crosssplit/figures/) |
| 固定外部确认 | [结果](external/summary.json) | [科学图](external/figures/) |

[逐轮记录与配置索引](cases.json) · [指标精度说明](weighted_precision.json)

各history文件保留完整逐轮数据，summary与cases注明来源SHA及字段。native BF16硬标签CE与FP64描述性NLL分别列出，区间对应其原指标。

损失对照的条件区间可由同5000图的派生对比量复算，不把三模型当15000个独立样本：

```powershell
python results_completion/recompute_mass.py --contrasts-csv results_completion/mass/paired_contrasts.csv --summary results_completion/mass/summary.json --output outputs/mass_intervals.json
```

解释缓存生成后，可用CPU入口绘制全验证集敏感性图：

```powershell
python -m contributions.full_explanation.presentation --output-dir outputs/explanation --data-dir data --destination outputs/explanation_presentation
```

成本与固定种子对比图：

```powershell
python results_completion/render_mass_cost.py --summary results_completion/mass/summary.json --manifest results_completion/manifest.json --output outputs/06_cost_and_seed_effects.png
```

外部固定对照区间图：

```powershell
python results_completion/render_external_intervals.py --summary results_completion/external/summary.json --summary-sha256 be7706976280752efc2fe498acc43d39cb1178bb32d80c582c58bf5601befa78 --output-dir outputs/external_intervals
```

第二划分图：

```powershell
python results_completion/render_crosssplit.py --payload results_completion/crosssplit/data_payload.json --payload-sha256 e3f3b1403518ee2da0ba18d5daefc52adfa29d9d3dd47c1c14f9a9484b76b9e2 --output-dir outputs/crosssplit_presentation
```

增强与架构拟合图：

```powershell
python results_completion/render_controlled_fit.py --payload results_completion/controlled/data_payload.json --payload-sha256 cedf97fe76a9097bd9f7f21811576a91162a1686ef326475aa11f533a43898ae --output-dir outputs/controlled_presentation
```
