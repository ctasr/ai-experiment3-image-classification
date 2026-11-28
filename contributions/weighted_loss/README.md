# 双来源加权蒸馏

本目录在冻结的主训练循环上提供三个独立模式，固定宽48、240轮、批量256、CutMix β=0.5、α=0.25、T=1。主比较使用第240轮权重，训练种子为42、17、73。每次命令只运行指定的一组，不自动增加种子、测试或复现。

- `original`：0.75 CE + 0.25 KL
- `weighted_single`：保留完整KL，按CutMix实际面积和教师对两来源类别的概率，加入两项排除该类别后的条件KL
- `confidence_mass`：0.75 CE + 0.25 mean([1+λq(yᵢ)+(1−λ)q(yⱼ)] KLᵢ)。系数在逐图KL上相乘后求平均。这是显式系数总量对照，不表示梯度强度严格相等，也不预设有益结果

`legacy_loss.py`与旧损失文件逐字节一致，原SHA及Source15-v1/Source16-v2字典见`provenance.json`。新的路径适配、配置和来源身份单独记录，不能用本入口接续旧断点。验证入口可识别这两套冻结旧来源，但不会接续训练。

在仓库根目录执行CPU数学和恢复检查：

```powershell
python -m contributions.weighted_loss.preflight --output outputs/cpu_preflight.json
```

数据目录应含`dataset_manifest.json`、`normalization.json`、`splits/`和`raw/`。教师路径指向原种子42的宽32 CutMix和ImageNet微调ResNet18权重，文件SHA必须与记录一致。初始化文件默认位于所给数据目录的`pretrained/resnet18-f37072fd.pth`，也可显式指定：

```powershell
python -m contributions.weighted_loss.train --mode weighted_single --seed 42 --data-dir path/to/data --teacher-cutmix path/to/cutmix/best.pt --teacher-transfer path/to/transfer/best.pt --imagenet-weights path/to/resnet18-f37072fd.pth --output-dir outputs/weighted_single
```

将`--mode`及`--output-dir`同时改为`confidence_mass`可运行独立对照。三个配置位于`configs/`，仅`--seed`可在42、17、73间改变。GPU预检可用`--pilot-samples 456 --stop-after 3`，其权重不能冒充正式45k训练。正式训练可用`--stop-after 8`分段核验，再对同目录运行原命令接续；全程仍按240轮调度。

每组输出位于`OUTPUT_DIR/runs/MODE_seedSEED/`，包括配置、逐轮记录、每批输入/教师目标校验和、BEST与完整LAST断点及来源回执。断点包含优化器、调度器和全部训练RNG。训练通过调用方数据路径加载原45k/5k划分，不读取官方测试批次，不拟合温度。`train_kd`随模式分别表示完整KL、完整KL加条件项或逐图系数加权KL，三者不能当作同一种训练指标。

输出目录须为空或带完全匹配的本入口身份记录，三模式分别使用目录。仅在完整轮次边界接续；若中断留下未提交批次的trace，入口会拒绝追加，须保留原始记录后明确恢复，不能让重复批次混入审计记录。

只评估冻结第240轮的原图验证集：

```powershell
python -m contributions.weighted_loss.validate --checkpoint outputs/weighted_single/runs/weighted_single_seed42/last.pt --mode weighted_single --data-dir path/to/data --output-dir outputs/weighted_single_validation
```

验证使用原BF16单次前向和硬标签CE，输出5000图的概率、标签、混淆矩阵、每类准确率及输入/权重/评估源码SHA。若另作BEST诊断，须显式使用`--endpoint best`和对应`best.pt`，不能与固定LAST比较混合。
