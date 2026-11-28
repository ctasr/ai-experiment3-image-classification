# 全验证集两视图解释

本模块将原固定200图的解释协议扩展到原验证集全部5000图，使用原种子42、17、73的宽48最佳权重。三模型始终解释原种子42在既有5000图缓存中的预测类别，目标在新推理前锁定。没有训练、测试集评估或温度拟合。

输入为原图与水平翻转图，取未校准概率平均。沿用GAP与线性分类头的解析概率梯度，两分支有符号图先映回原坐标，相加后再作ReLU。CPU autograd仅检查实际冻结特征与分类头，不向模型参数求梯度。遮挡8×8最高响应窗，并与10个同面积随机窗比较。先在归一化后的原图上填零，即训练通道均值，再生成翻转分支。随机窗由私有PCG64种子42产生，可重复或与最高响应窗重叠。

为保留原数值布局，原200图独立处理，新增4800图另作一段。干净前向批量32，遮挡按16图展开11个窗后分成64个输入。不同批量的FP32概率可能有微小差异，不声称逐图与批量概率逐位相同。原200同布局缓存桥必须逐值精确，逐图语义检查另保留原数值公差，并要求窗口和常量图标记一致。

数据目录须含原`raw/`、`splits/`、`dataset_manifest.json`与`normalization.json`。参考目录包含四个未经改写的原缓存副本：`legacy_42.npz`、`legacy_17.npz`、`legacy_73.npz`及`seed42_clean5000.npz`。这些文件与三权重的SHA见`provenance.json`，路径均由调用方明确给出。

独立命令在导入PyTorch前沿用原实验的`CUBLAS_WORKSPACE_CONFIG=:4096:8`，只设置本次Python进程。已有其他值时会在推理前报错，不改变用户环境或确定性要求。

双线性插值保留各缓存的原运行路径：种子42使用原生CUDA插值，17、73使用全局确定性模式的分解实现。两种路径在本机固定首32图的诊断中解释了热图的末位差异，未修改数学公式或精确缓存门槛。每个种子的200图桥与后续4800图使用同一路径，并保存实际运行状态，退出后恢复调用方状态。旧进程未记录实际cuBLAS环境值，首32图在未设置及`:4096:8`两条件下的逐位一致不作为历史环境证明。

在仓库根目录运行CPU合成、遮挡语义和旧缓存检查：

```powershell
python -m contributions.full_explanation.check --data-dir DATA --reference-dir REFERENCES --weights-42 BEST42 --weights-17 BEST17 --weights-73 BEST73 --output-dir OUTPUT
```

通过后，依次运行实际GPU桥接和全量扩展，参数保持相同：

```powershell
python -m contributions.full_explanation.run bridge --data-dir DATA --reference-dir REFERENCES --weights-42 BEST42 --weights-17 BEST17 --weights-73 BEST73 --output-dir OUTPUT
python -m contributions.full_explanation.run full --data-dir DATA --reference-dir REFERENCES --weights-42 BEST42 --weights-17 BEST17 --weights-73 BEST73 --output-dir OUTPUT
```

`full`要求三模型的实际GPU桥全部通过，原200缓存沿用本次桥接结果，只为其余4800图新增前向。常量热图、负降幅和模型间目标不一致均保留。输出记录每批实际索引、归一化输入校验和、布局、前向数量、原始耗时及权重/BN/梯度检查。

全量完成后生成分析和五张图：

```powershell
python -m contributions.full_explanation.analysis --output-dir OUTPUT
python -m contributions.full_explanation.figures --output-dir OUTPUT --data-dir DATA
```

区间在真实类别内配对重抽图像，同一批图像索引用于三个固定模型。5000张图不会当成15000个独立样本。原200与新增4800分别列出，原200不作为全体验证集的随机代表。图像区间不包含重训、教师或选参的不确定性，热图和遮挡差也不能证明物体边界或因果关系。生成图后仍须实际逐图核查，程序不会自动写已看图或全部完成标记。
