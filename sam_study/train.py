"""One fixed non-adaptive L2 SAM update around the original online KD240 objective."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import hashlib
import math
import numpy as np
import torch
from data import ROOT
from train import write_json
from distillation.targets import read, sha
from distillation.train import loss_parts
from aligned_distillation.teacher import Teacher, identity
import horizon_study.train as original

OUT=ROOT/'results_sam_study'
DIRECTORY=ROOT/'runs/sam_study'
BASE=tuple(read(ROOT/'results_horizon/source_manifest.json')['source_sha256'])
core=original.capacity.core
OLD_MIXED=core.mixed_epoch
EPOCH=0
FOLDER=None


def source_hashes():
    return {n:sha(ROOT/n) for n in BASE+('sam_study/train.py',)}


def bn_modules(model):
    return [m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]


def bn_buffers(model):
    return {n:b.detach().clone() for n,b in model.named_buffers() if n.endswith(('running_mean','running_var','num_batches_tracked'))}


@contextmanager
def no_running_update(model):
    modules=bn_modules(model);flags=[m.track_running_stats for m in modules]
    assert model.training and all(m.training for m in modules)
    try:
        for m in modules:m.track_running_stats=False
        yield
    finally:
        for m,flag in zip(modules,flags):m.track_running_stats=flag


def sam_step(model,x,y,order,lam,soft,optimizer,amp,rho=.05,audit=False):
    params=[p for p in model.parameters() if p.requires_grad]
    cuda=x.device if x.is_cuda else None
    cpu_before=torch.get_rng_state().clone()
    cuda_before=torch.cuda.get_rng_state(cuda).clone() if cuda is not None else None
    buffers_before=bn_buffers(model) if audit else None
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
        logits=model(x)
        first,ce,kd=loss_parts(logits,y,order,lam,soft,1.,.25)
    first.backward()
    cpu_after=torch.get_rng_state().clone()
    cuda_after=torch.cuda.get_rng_state(cuda).clone() if cuda is not None else None
    assert all(p.grad is not None for p in params)
    norm=torch.linalg.vector_norm(torch.stack([p.grad.detach().float().norm(2) for p in params]),2)
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite SAM first-gradient norm')
    originals=[p.detach().clone() for p in params]
    scale=float(rho)/(norm+1e-12)
    second=None
    clean_buffers=bn_buffers(model) if audit else None
    try:
        with torch.no_grad():
            for p in params:p.add_(p.grad.detach()*scale)
        perturb_norm=torch.linalg.vector_norm(torch.stack([(p.detach()-w).float().norm(2) for p,w in zip(params,originals)]),2)
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_before)
        if cuda is not None:torch.cuda.set_rng_state(cuda_before,cuda)
        with no_running_update(model):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                perturbed_logits=model(x)
                second,second_ce,second_kd=loss_parts(perturbed_logits,y,order,lam,soft,1.,.25)
            second.backward()
        if audit:
            after=bn_buffers(model)
            for n,b in after.items():
                assert torch.equal(b,clean_buffers[n]),n
                if n.endswith('num_batches_tracked'):assert torch.equal(b,buffers_before[n]+1)
    finally:
        # Subtracting a rounded perturbation need not restore FP32 weights exactly.
        with torch.no_grad():
            for p,w in zip(params,originals):p.copy_(w)
        torch.set_rng_state(cpu_after)
        if cuda is not None:torch.cuda.set_rng_state(cuda_after,cuda)
    if audit:assert all(torch.equal(p.detach(),w) for p,w in zip(params,originals))
    optimizer.step()
    hit=lam*(logits.argmax(1)==y).sum()+(1-lam)*(logits.argmax(1)==y[order]).sum()
    return dict(loss=first.detach(),ce=ce.detach(),kd=kd.detach(),hit=hit.detach(),
        second_loss=second.detach(),second_ce=second_ce.detach(),second_kd=second_kd.detach(),
        gradient_norm=norm.detach(),actual_perturb_norm=perturb_norm.detach())


def mixed_epoch(model,loader,transform,device,optimizer,method,mix_alpha,amp):
    global EPOCH
    EPOCH+=1
    cfg=core.ACTIVE
    if not cfg['sam_enabled']:return OLD_MIXED(model,loader,transform,device,optimizer,method,mix_alpha,amp)
    assert method=='cutmix' and cfg['alignment']=='mixed_view' and cfg['sam_rho']==.05 and cfg['sam_bn_policy']=='first_only'
    assert cfg['distill_alpha']==.25 and cfg['distill_temperature']==1.
    model.train();modules=bn_modules(model);assert len(modules)==21
    counts_before=[int(m.num_batches_tracked.item()) for m in modules]
    totals=torch.zeros(10,device=device,dtype=torch.float64)
    hashes={}
    for batch,(raw,y) in enumerate(loader):
        raw,y=raw.to(device,non_blocking=True),y.to(device,non_blocking=True)
        view=core.augment_raw(raw)
        order=torch.randperm(len(y),device=device)
        lam=float(np.random.beta(mix_alpha,mix_alpha));cx,cy=int(np.random.randint(32)),int(np.random.randint(32))
        view,lam=core.cutmix_patch(view,order,lam,cx,cy)
        soft=core.target_for_view(core.TEACHER,view,order,lam,1.,'mixed_view')
        if batch==0 or batch==len(loader)-1:
            hashes['first' if batch==0 else 'last']=dict(samples=len(y),view_sha256=hashlib.sha256(view.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                teacher_target_sha256=hashlib.sha256(soft.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),actual_lambda=lam)
        result=sam_step(model,transform(view,training=False),y,order,lam,soft,optimizer,amp,cfg['sam_rho'],audit=batch in (0,len(loader)-1))
        totals+=torch.stack((result['loss'].double()*len(y),result['ce'].double()*len(y),result['kd'].double()*len(y),result['hit'].double(),
            result['second_loss'].double()*len(y),result['gradient_norm'].double()*len(y),result['actual_perturb_norm'].double()*len(y),
            torch.tensor(len(y),device=device).double(),torch.tensor(lam*len(y),device=device).double(),
            (result['second_loss']-result['loss']).double()*len(y)))
    loss,ce,kd,hit,second,gnorm,pnorm,count,lambda_sum,delta=totals.cpu().tolist()
    assert count==len(loader.dataset)
    counts_after=[int(m.num_batches_tracked.item()) for m in modules]
    assert all(b-a==len(loader) for a,b in zip(counts_before,counts_after)) and all(m.track_running_stats for m in modules)
    assert all(math.isfinite(v) for v in [loss,ce,kd,hit,second,gnorm,pnorm])
    path=FOLDER/'sam_trace.json';trace=read(path) if path.exists() else {}
    trace[str(EPOCH)]=dict(epoch=EPOCH,samples=int(count),batches=len(loader),student_forward_batches=2*len(loader),
        teacher_query_batches=len(loader),optimizer_steps=len(loader),first_BN_update_batches=len(loader),second_BN_update_batches=0,
        all21_BN_counter_deltas=[b-a for a,b in zip(counts_before,counts_after)],first_last_BN_and_exact_weight_restore_checked=True,
        rho=cfg['sam_rho'],mean_gradient_norm=gnorm/count,mean_actual_perturb_norm=pnorm/count,
        first_loss=loss/count,second_loss=second/count,second_minus_first_loss=delta/count,mean_actual_lambda=lambda_sum/count,
        first_and_last_batch=hashes,scope='First loss/CE/KD/accuracy logged in history. Second loss is diagnostic, not validation or selected checkpoint sharpness proof.')
    write_json(path,trace)
    return dict(loss=loss/count,ce=ce/count,kd=kd/count,accuracy=hit/count,samples=int(count))


def configure():
    original.OUT,original.DIRECTORY=OUT,DIRECTORY
    original.capacity.OUT,original.capacity.DIRECTORY=OUT,DIRECTORY
    core.OUT,core.DIRECTORY=OUT,DIRECTORY
    core.SOURCES=BASE+('sam_study/train.py',)
    core.Teacher,core.identity,core.mixed_epoch=Teacher,identity,mixed_epoch


def run(cfg,name,stop_after=None):
    global FOLDER,EPOCH
    assert cfg['sam_rho']==.05 and cfg['sam_bn_policy']=='first_only'
    assert cfg['epochs']==240 or name.startswith('audit_')
    configure();FOLDER=DIRECTORY/name;EPOCH=0
    if (FOLDER/'last.pt').exists():
        saved=torch.load(FOLDER/'last.pt',map_location='cpu',weights_only=False)
        assert saved['config']==cfg and saved['source_sha256']==source_hashes();EPOCH=saved['epoch']
    result=original.capacity.run(cfg,name,stop_after)
    write_json(FOLDER/'sam_metadata.json',dict(enabled=cfg['sam_enabled'],rho=.05,adaptive=False,
        algorithm='g1 atw -> e=.05*g1/(globalL2norm+1e-12), g2 atw+e -> EXACT copy originalw -> one originalSGD update withg2; no weightdecay in innergradient, baseSGD weightdecay once atoriginalw.',
        BN='Firstpass updates buffers; secondpass ownbatchstatistics with track_running_stats=False; restoreflags. Same actual CutMix input and one frozen teacher target for both passes.',
        RNG='Secondpass replays first TorchCPU/CUDA state then restores post-first state; original crop/flip/CutMix RNG untouched by SAM.',
        metrics='Unperturbed firstpass areaCutMix hardCE/KL/accuracy; not raw fit. Perturbed secondloss diagnostic only.',
        extra_student_gradient_evaluations=True,optimizer_steps_per_full_epoch=176,source_sha256=source_hashes(),indirect_ImageNet_knowledge=True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-id',required=True);p.add_argument('--stop-after',type=int)
    a=p.parse_args();run(read(Path(a.config)),a.run_id,a.stop_after)
