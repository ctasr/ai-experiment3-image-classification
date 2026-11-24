"""复用同一通用在线蒸馏引擎；仅独立目录和适配器来源版本。"""
import argparse
from pathlib import Path
import torch
from data import ROOT
from train import write_json
from distillation.targets import read
import aligned_distillation.train as core

OUT=ROOT/"results_capacity"
DIRECTORY=ROOT/"runs/capacity"
BASE=tuple(read(ROOT/"results_aligned_distillation/source_manifest.json")["source_sha256"])
core.OUT,core.DIRECTORY=OUT,DIRECTORY
core.SOURCES=BASE+("capacity_study/train.py",)
source_hashes=core.source_hashes


def run(cfg,name,stop_after=None):
    torch.cuda.reset_peak_memory_stats()
    result=core.run(cfg,name,stop_after)
    write_json(DIRECTORY/name/"runtime_capacity.json",{"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),"width":cfg["width"],"parameters":result.get("parameters"),"note":"本进程训练峰值，含固定教师，不等于学生推理占用；轮次计时保留原值，长暂停不当纯GPU时间"})
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--run-id",required=True);parser.add_argument("--stop-after",type=int)
    args=parser.parse_args();run(read(Path(args.config)),args.run_id,args.stop_after)
