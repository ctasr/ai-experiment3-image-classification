"""保持原四视图教师与学生算法，隔离更长调度终点的配置/目录。"""
import argparse
from pathlib import Path
from data import ROOT
from distillation.targets import read
import capacity_study.train as capacity
from aligned_distillation.teacher import Teacher,identity

OUT=ROOT/"results_horizon"
DIRECTORY=ROOT/"runs/horizon"
BASE=tuple(read(ROOT/"results_capacity/source_manifest.json")["source_sha256"])
capacity.OUT,capacity.DIRECTORY=OUT,DIRECTORY
capacity.core.OUT,capacity.core.DIRECTORY=OUT,DIRECTORY
capacity.core.SOURCES=BASE+("horizon_study/train.py",)
capacity.core.Teacher,capacity.core.identity=Teacher,identity
source_hashes=capacity.core.source_hashes
run=capacity.run


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--run-id",required=True);parser.add_argument("--stop-after",type=int)
    args=parser.parse_args();run(read(Path(args.config)),args.run_id,args.stop_after)
