"""仅重定向已冻结容量引擎；新12源，旧核心与权重不改。"""
import argparse
from pathlib import Path
from data import ROOT
from train import write_json
from distillation.targets import read
import capacity_study.train as capacity

OUT=ROOT/"results_capacity_control"
DIRECTORY=ROOT/"runs/capacity_control"
BASE=tuple(read(ROOT/"results_capacity/source_manifest.json")["source_sha256"])
capacity.OUT,capacity.DIRECTORY=OUT,DIRECTORY
capacity.core.OUT,capacity.core.DIRECTORY=OUT,DIRECTORY
capacity.core.SOURCES=BASE+("capacity_control/train.py",)
source_hashes=capacity.core.source_hashes


def run(cfg,name,stop_after=None):
    result=capacity.run(cfg,name,stop_after)
    write_json(DIRECTORY/name/"control_metadata.json",{"distill_alpha":cfg["distill_alpha"],"teacher_loaded":cfg["distill_alpha"]>0,
               "external_ImageNet_knowledge_used":cfg["distill_alpha"]>0,"note":"通用objective_metadata的教师/外部知识字段描述潜在蒸馏协议；本次alpha0分支直接调用原硬标签CutMix循环，不加载教师、不使用教师目标；本文件和teacher_check为实际角色证据。"})
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--run-id",required=True);parser.add_argument("--stop-after",type=int)
    args=parser.parse_args();run(read(Path(args.config)),args.run_id,args.stop_after)
