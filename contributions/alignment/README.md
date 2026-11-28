# 教师目标与实际输入对齐

固定宽32残差学生、160轮、45k/5k划分、种子42/17/73，比较两种目标：A先预测两张当前裁剪图再按CutMix实际面积混合概率；B直接预测学生实际看到的CutMix图。其余配置相同，教师为固定Teacher42的两个模型、各含原图与水平翻转，共四份未校准概率。

训练直接调用仓库冻结的Source10循环。适配器仅路由数据、教师权重与独立输出，新增源码身份另列；新checkpoint不冒充旧Source10。选点为完整160轮验证准确率最高、CE较低、首次完全并列的BEST。没有官方测试入口，也不改变历史筛选结论。

`evidence/index.json`索引原始三种子验证数值。`original_validation_analysis.json`为原文件逐字节副本，没有合并或改写原始实验记录。查看数值及CPU检查：

```powershell
python -m contributions.alignment.check evidence
python -m contributions.alignment.check cpu --data-dir C:/datasets/cifar10/data
```

在仓库根目录执行。`--data-dir`指含`raw/`、`splits/`、`dataset_manifest.json`、`normalization.json`的**data目录**，不是`raw`目录。两份教师权重必须与根目录`release_checkpoints.json`的`teachers`哈希一致。保留原ResNet18初始化文件，默认从`DATA/pretrained/resnet18-f37072fd.pth`读取，也可显式指定`--imagenet-weights`。

下面为PowerShell示例，路径换成实际位置。输出必须是全新目录，或本入口此前建立且来源完全相同的目录；不会在原实验目录上覆盖或混接断点。

```powershell
$inputs = @('--data-dir','C:/datasets/cifar10/data','--teacher-cutmix','C:/weights/teacher-cutmix-seed42.pt','--teacher-transfer','C:/weights/teacher-transfer-seed42.pt')
python -m contributions.alignment.train @inputs --output-dir C:/runs/alignment-cont3 --seeds 42 --stop-after 3
python -m contributions.alignment.train @inputs --output-dir C:/runs/alignment-resume3 --seeds 42 --stop-after 1
python -m contributions.alignment.train @inputs --output-dir C:/runs/alignment-resume3 --seeds 42 --stop-after 3
python -m contributions.alignment.check gpu-proof --continuous-root C:/runs/alignment-cont3 --resumed-root C:/runs/alignment-resume3 --output C:/runs/alignment-gpu-proof.json
```

这只是A/B各前三轮的GPU连续/恢复预检，使用完整45k/5k与原160轮调度，包含末批200训练样本。通过后才能完整训练；该短检不等于完整复现或所有GPU设备的证明。CPU检查不冒充GPU通过。

```powershell
python -m contributions.alignment.train @inputs --output-dir C:/runs/alignment-full --seeds 42 17 73 --preflight-proof C:/runs/alignment-gpu-proof.json
python -m contributions.alignment.validate --data-dir C:/datasets/cifar10/data --runs-dir C:/runs/alignment-full/runs --output-dir C:/runs/alignment-validation
```

两角色默认都执行，各种子不因上一组结果取消。所有计时、暂停及负结果保留。按原环境约16秒/轮，A/B各160轮约85分钟，六组约4.3小时，另加载入和验证，休眠顺延。

验证入口只读输入权重及记录，以原BF16单原图、批量256重放5k验证，核BEST与原记录，并输出CM、每类准确率、误差和原始计时。已有历史目录可用`--run-map`替代`--runs-dir`，JSON结构为`{"42":{"A":"相对或绝对目录","B":"目录"},"17":{...},"73":{...}}`，相对路径以该JSON所在目录为基准。数据、输入权重与历史文件均不修改。

完整重跑若要与原160轮检查点核对：

```powershell
python -m contributions.alignment.check compare --actual-run C:/runs/alignment-full/runs/A_seed42 --reference-run C:/original/A42 --epochs 160 --output C:/runs/alignment-A42-comparison.json
```

比较模型、优化器、调度器、随机状态和全部非计时history；新旧源码身份不同，因此不要求checkpoint整文件SHA相同。固定验证集反复使用、共用Teacher42等限制仍在，重跑一致不替代独立数据与种子效果检验。
