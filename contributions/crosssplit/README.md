# 第二划分固定对照

在官方训练集的第二个45,000/5,000划分上，比较标准KL与固定的面积加权条件KL。划分种子为20261127，每类4,500张训练、500张验证。全部配置在计算该划分的性能前封存。

两位教师重新训练：宽32 CutMix网络160轮、ImageNet微调ResNet18共60轮，训练种子均为42，固定使用最后一轮权重。学生使用宽48网络、240轮，两个损失分别运行42、17、73三个种子，共六组，全部固定使用LAST240。验证指标可以记录，但不用于选epoch、超参或追加种子。

教师和学生均不从旧CIFAR权重或旧断点继续。仅迁移教师保留原ImageNet初始化。教师与学生共享实际CutMix图像，四视图教师概率算法及条件损失数学实现保持原样。纯损失调用相邻`weighted_loss/legacy_loss.py`，其SHA与原实现一致。

新归一化只使用第二划分的45,000张训练图，以uint64通道和、平方和计算总体均值和标准差。它与旧NumPy float64中心化方差计算的舍入路径不同，不承诺逐位相同；新对照的两种方法共用封存的同一组统计量。

本对照不读取官方测试集或退休的CIFAR10.1，不拟合温度。官方训练图已经用于历史研究，第二划分属于内部确认，不能称为从未接触的独立盲测，也不能消除历史测试曝光。

## 运行

需要已有官方CIFAR-10训练批次、固定ImageNet初始化及主README所列环境。以下命令仅在专用目录封存划分、归一化和配置，不会启动训练，也不改原`data/`：

```powershell
python -m contributions.crosssplit.run --workspace outputs/crosssplit_20261127 prepare
```

训练每次只执行显式指定的一组。正式教师须先通过真实45k/5k前2轮连续与1+2轮恢复比较；正式学生须在新教师锁定后，通过完整240轮调度下前8轮连续与4+8轮恢复比较，以及两种损失的实际输入、标签、CutMix面积和教师概率哈希核对。`gate`只比较已实际生成的检查点，不以性能好坏判定是否运行其余组。

各入口及参数见：

```powershell
python -m contributions.crosssplit.run --help
python -m contributions.crosssplit.run train --help
python -m contributions.crosssplit.run gate --help
```

流程为`prepare`→教师预检与`gate`→两教师正式训练→`lock-teachers`→学生预检与`gate`→六组显式正式训练→`evaluate`。正式运行名固定为`main`，没有自动启动全部训练的入口。模型、原始逐轮记录、输入核对、来源SHA和分析均写入有标记的专用`crosssplit...`目录，旧研究目录不能作为输出。

`evaluate`仅评估固定LAST的原图45k/5k，输出准确率、硬标签CE、混淆矩阵、每类准确率和可配对的5k缓存。缓存中的逐图FP32 CE与原生批次平均CE分别保留，末位舍入差异不混作方法收益。完成程序不代表图表与科学检查已经全部完成。

`objective_metadata.json`记录实际完整目标与`train_kd`定义：标准方法为完整KL，weighted方法为完整KL加条件项。基础core生成的通用说明另存为`objective_metadata_base_core.json`，原始训练源码保持不变。
