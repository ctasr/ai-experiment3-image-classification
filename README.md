# CIFAR-10图像分类实验

比较自建CNN与残差网络，研究数据增强、教师目标对齐和损失设计。固定训练/验证划分为45000/5000张，测试使用官方10000张。

| 内容 | 入口 |
|---|---|
| 模型、训练与数据 | [models.py](models.py)、[train.py](train.py)、[data.py](data.py) |
| 目标对齐比较 | [运行说明](contributions/alignment/README.md) |
| 损失比较 | [运行说明](contributions/weighted_loss/README.md) |
| 两视图解释与遮挡 | [运行说明](contributions/full_explanation/README.md) |
| 增强与架构比较 | [训练配置](contributions/controlled_training.json) · [结果](results_completion/controlled/summary.json) |
| 第二划分对照 | [运行说明](contributions/crosssplit/README.md) |
| 固定外部确认 | [运行说明](contributions/external_confirmation/README.md) |
| 主要结果 | [预测结果](results_horizon_prediction/results.json)、[预算比较](results_budget_kd_interaction/results.json) |
| 实验结果 | [结果](results_completion/README.md) |

`configs*/`保存独立配置，`results*/`保存指标及来源，`runs/`保存逐轮记录，`data/`保存固定划分与归一化统计，`provenance/`保存历史源码字节副本，`tests/`保存评估计算测试。

[主模型训练过程](report_work/assets/five_page_curves.png)记录种子42完整240轮的CutMix面积加权训练命中率、混合CE和KL。
[官方测试混淆矩阵与误例](report_work/assets/five_page_errors.png)对应种子42验证选定的BEST权重、原图与水平翻转概率平均，以及官方10000张测试图。

## 运行

参考环境为Python3.12、PyTorch2.9.1+cu128和torchvision0.24.1。依赖见[requirements.txt](requirements.txt)。

```powershell
python -m pip install -r requirements.txt
python prepare_data.py
python download_checkpoints.py
python evaluate_release.py --seed 42 --split validation --device cuda --precision bfloat16 --output outputs/seed42_validation.json
```

默认下载原v1.0.0的三个最佳学生和两个教师权重，逐个校验大小与SHA。CUDA/BF16与CPU/FP32属于不同数值口径。

需要对照端点时按组获取，默认命令不会下载所有历史权重：

```powershell
python download_checkpoints.py --group alignment-best
python download_checkpoints.py --references-only --data-dir data
python download_checkpoints.py --verify-sources
```

可选组：`alignment-best`、`kl-last`、`weighted-last`、`budget-best`、`confidence-mass`、`references`。组名、种子、BEST/LAST端点和来源见`release_checkpoints.json`。种子42解释参考由无像素资产和本地训练数据重建，完整SHA必须等于原缓存。

历史来源副本只用于字节核对。旧Source10/15/16权重不能混入当前源码续训；其他历史方案仍按对应配置从头训练。

从头训练主模型：

```powershell
python download_pretrained.py
python -m horizon_study.train --config configs_horizon/horizon240_seed42.json --run-id reader_horizon240_seed42
```
