# 固定外部确认

这是一项预先登记的CIFAR10.2确认评估。它使用21份固定权重、24个种子与预测策略组合，统一保存FP32 logits，再用FP64计算概率和指标。旧CIFAR10.1不参与此流程。模型集合、预测规则、六项配对比较和像素重叠处理见[protocol.json](protocol.json)。

按顺序显式执行六个阶段。各命令只执行当前阶段，不会自动下载数据：

```powershell
python -m contributions.external_confirmation.run prepare-manifest --bindings BINDINGS.json --data-dir ORIGINAL_DATA --output-dir NEW_OUTPUT
python -m contributions.external_confirmation.run preflight --output-dir NEW_OUTPUT
python -m contributions.external_confirmation.run seal --output-dir NEW_OUTPUT
python -m contributions.external_confirmation.run fetch --output-dir NEW_OUTPUT
python -m contributions.external_confirmation.run evaluate --output-dir NEW_OUTPUT
python -m contributions.external_confirmation.run analyze --output-dir NEW_OUTPUT
```

`ORIGINAL_DATA`是含`raw/`、`splits/`、`dataset_manifest.json`和`normalization.json`的原CIFAR10数据目录。`NEW_OUTPUT`必须与数据、源码及各训练目录分开。`preflight`和`evaluate`需要CUDA，其余阶段使用CPU。预检仅使用原验证集前257张固定图，包括128、128、1的批次、逐例对照和原图/翻转均值规则，核查权重、BN、随机状态和来源。

`BINDINGS.json`必须显式给出全部21行，不能省略待训练模型。例如一行的字段如下，实际SHA、完整配置及来源字典须填入原记录：

```json
{
  "models": [{
    "group": "alignment_A_BEST",
    "seed": 42,
    "checkpoint": "/weights/original_A_seed42_best.pt",
    "expected_weight_sha256": "原权重完整SHA256",
    "expected_config": {"完整键值": "与原checkpoint一致"},
    "expected_training_sources": {"原来源标识": "原源码SHA256"},
    "source_file_bindings": {"原来源标识": "/sources/原文件副本.py"},
    "expected_provenance_sources": {"原分析来源标识": "原分析源码SHA256"},
    "provenance_source_file_bindings": {"原分析来源标识": "/sources/原分析副本.py"},
    "selector_evidence": {"selector": "原记录中的BEST或LAST规则"}
  }]
}
```

全部组名见`protocol.json`，每组固定种子42、17、73。主模型BEST同时评估原图和翻转概率均值，其余每份权重只评估原图。旧加权损失使用原Source15/16的三份LAST，新精确重跑不作为第四个种子。新confidence_mass三份LAST可在产生后、外部访问前绑定实际SHA。所有旧来源字典保持原键值，旧`../tmp/...`只作为来源标识，通过`source_file_bindings`指定任意位置的原字节副本，程序不会推断本机历史目录。

缺少任一权重或来源时只能生成不完整清单，不能封存或下载。封存还要求21份权重均通过实际GPU验证集预检。下载前先写`access_started.json`，随后核验固定Git blob、文件字节数、SHA256及图像标签schema；出现失败保留原产物，只能恢复完全相同的封存任务。

原60000张像素只用于预先规定的重叠索引，原官方测试集不做新增模型评估。主分析保留外部全部2000行，次分析剔除与原60000图精确像素重叠的行。内部重复及标签冲突单独披露。10000次分层Bootstrap在每个人口内共享图像抽样索引，先计算三种子的配对均值，不把三种子的6000次预测当独立样本。

固定分析完成后写退休记录，禁止继续用该外部集选择模型、温度、阈值或新方向。`analysis.json`保存全部固定结果、条件区间及五张图SHA；实际看图仍须另外完成，程序分析完成不代表人工核查完成。
