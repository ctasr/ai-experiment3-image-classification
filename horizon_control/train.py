"""Isolated alpha-zero 240-horizon control; reuse frozen training computation."""
import argparse
from pathlib import Path
from data import ROOT
from train import write_json
from distillation.targets import read
from aligned_distillation.teacher import Teacher,identity
import capacity_control.train as original

OUT=ROOT/'results_horizon_control'
DIRECTORY=ROOT/'runs/horizon_control'
BASE=tuple(read(ROOT/'results_horizon/source_manifest.json')['source_sha256'])+('capacity_control/train.py',)
capacity=original.capacity


def configure():
    original.OUT,original.DIRECTORY=OUT,DIRECTORY
    capacity.OUT,capacity.DIRECTORY=OUT,DIRECTORY
    capacity.core.OUT,capacity.core.DIRECTORY=OUT,DIRECTORY
    capacity.core.SOURCES=BASE+('horizon_control/train.py',)
    capacity.core.Teacher,capacity.core.identity=Teacher,identity


configure()
source_hashes=capacity.core.source_hashes


def run(cfg,name,stop_after=None):
    assert cfg['distill_alpha']==0. and cfg.get('model')=='cifar'
    configure()
    result=original.run(cfg,name,stop_after)
    actual=read(DIRECTORY/name/'teacher_check.json')
    assert not actual['teacher_loaded'] and actual['weights_bn_gradients_unchanged']
    runtime=read(DIRECTORY/name/'runtime_capacity.json')
    write_json(DIRECTORY/name/'runtime_control.json',{**runtime,'actual_teacher_loaded':False,'actual_external_ImageNet_knowledge_used':False,'note':'本进程纯监督训练峰值，无教师目标或教师模型；runtime_capacity原通用注释描述潜在协议，不代表实际教师存在。所有墙钟/暂停保留。'})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--run-id',required=True);parser.add_argument('--stop-after',type=int)
    args=parser.parse_args();run(read(Path(args.config)),args.run_id,args.stop_after)
