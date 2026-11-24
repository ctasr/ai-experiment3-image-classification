# CIFAR-10 图像分类实验

比较自建 CNN、残差网络、数据增强与知识蒸馏。最终学生使用宽48残差网络（2,436,346个参数），以宽32残差网络和ImageNet微调ResNet18对实际CutMix输入的四视图平均概率为教师目标，训练240轮。

官方训练集按类固定划分为45,000张训练、5,000张验证，另用官方10,000张测试图评估。三个训练种子42、17、73共用划分种子42。

三个学生分别进行原图与水平翻转图的概率平均，测试准确率为96.84%、96.67%、96.62%，均值 **96.71% ± 0.12个百分点**（样本标准差）；原图单次前向的均值为96.44%。

## 实验索引

| 主题 | 代码、配置与结果 |
|---|---|
| 基础CNN与残差 | [结构](models.py)、[三种子结果](results/test_results.json) |
| 增强与注意力 | [固定设置对照](results_rubric_review/controlled_improvements.json)、[配置](configs_round2/) |
| 迁移学习与教师 | [ResNet18迁移对照](results_round2/test_results.json)、[固定教师](results_round3/test_results.json)、[MobileNet](results_mobile_transfer/) |
| 教师目标对齐 | [实现](aligned_distillation/train.py)、[A/B结果与区间](results_aligned_distillation/) |
| 预算与最终训练 | [训练入口](horizon_study/train.py)、[主配置](configs_horizon/)、[预算对照](results_budget_kd_interaction/)、[逐轮记录](runs/horizon/)、[种子42复现](results_horizon/reproduction.json) |
| 最终预测与成本 | [测试汇总](results_horizon_prediction/results.json)、[校准](results_horizon_policy_calibration/)、[三成员组合](results_seed_ensemble/)、[同条件计时](results_seed_ensemble_cost/results.json) |
| 解释性与扰动 | [两视图解释](results_two_view_extension/)、[跨种子结果](results_seed_explanation/summary.json) |
| 其他训练策略 | [RandAugment](results_randaugment_v4/)、[一致性](results_clean_anchor_consistency/)、[SAM](results_sam_study/)、[EMA](results_online_ema/)、[Dropout](results_regularization/)、[早停回放](results_early_stopping/replay.json) |

源码按方法组织；`configs*/`保存独立配置，`results*/`保存指标及来源校验，`runs/`按运行名称保存逐轮记录。`data/`保存固定划分与归一化统计，`extensions/`、`distillation/`等提供共享模块，`tests/`为评估计算的合成测试。

图表：[改进对照](report_work/assets/improvement_visual_check.png)、[训练曲线](report_work/assets/five_page_curves.png)、[拟合诊断](report_work/assets/fit_visual_check.png)、[误分类](report_work/assets/five_page_errors.png)、[解释性分析](report_work/final_20261121/q_explanation_compact.png)。

## 运行

参考环境：Windows、Python 3.12、PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128，依赖见[requirements.txt](requirements.txt)。在PowerShell中执行：

```powershell
git clone https://github.com/ctasr/ai-experiment3-image-classification.git
cd ai-experiment3-image-classification
py -3.12 -m venv .venv
$experimentPython = ".\.venv\Scripts\python.exe"
& $experimentPython -m pip install -r requirements.txt
& $experimentPython prepare_data.py
& $experimentPython download_checkpoints.py
& $experimentPython evaluate_release.py --seed 42 --output outputs/seed42_test.json
```

数据从官方源下载；两个教师和三个学生权重从[v1.0.0](https://github.com/ctasr/ai-experiment3-image-classification/releases/tag/v1.0.0)下载并校验。将`--seed`改为17或73可评估其他学生，`--split validation`用于验证集。输出含准确率、每类准确率和混淆矩阵，已有文件不会覆盖。

评估默认在CUDA上使用BF16、CPU上使用FP32；上述参考结果来自CUDA/BF16，其他运行条件可能产生数值差异。

## 从头训练

完成上述数据和固定教师权重下载后，在同一PowerShell窗口中执行：

```powershell
& $experimentPython download_pretrained.py
& $experimentPython -m horizon_study.train --config configs_horizon/horizon240_seed42.json --run-id reader_horizon240_seed42
```

训练使用Windows+CUDA，共240轮，写入新的`runs/horizon/reader_horizon240_seed42/`目录。
