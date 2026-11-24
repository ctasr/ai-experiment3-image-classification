"""Clean hard supervision + noisy-input KD, one SGD update, clean-only BN running updates."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import math
import numpy as np
import torch
from torch.nn import functional as F
from data import ROOT
from train import write_json
from distillation.targets import read, sha
from aligned_distillation.teacher import Teacher, identity
import clean_teacher_noise.train as previous

OUT = ROOT / 'results_clean_anchor_consistency'
DIRECTORY = ROOT / 'runs/clean_anchor_consistency'
BASE = tuple(read(ROOT / 'results_clean_teacher_noise/source_manifest.json')['source_sha256'])
core = previous.core
noise_seed, apply_noise = previous.noise_seed, previous.apply_noise


def source_hashes():
    return {n: sha(ROOT / n) for n in BASE + ('clean_anchor_consistency/train.py',)}


def bn_modules(model):
    return [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


@contextmanager
def batch_statistics_without_running_updates(model):
    modules = bn_modules(model)
    flags = [m.track_running_stats for m in modules]
    assert model.training and all(m.training for m in modules)
    try:
        for m in modules:
            m.track_running_stats = False
        yield
    finally:
        for m, flag in zip(modules, flags):
            m.track_running_stats = flag


def bn_buffers(model):
    return {n: t.detach().clone() for n, t in model.named_buffers() if n.endswith(('running_mean','running_var','num_batches_tracked'))}


def branch_step(model, clean_x, noisy_x, labels, order, lam, soft, optimizer, amp, audit=False):
    """Two backward passes accumulate the weighted gradients before exactly one optimizer step."""
    optimizer.zero_grad(set_to_none=True)
    before = bn_buffers(model) if audit else None
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
        clean_logits = model(clean_x)
        hard = lam * F.cross_entropy(clean_logits, labels) + (1-lam) * F.cross_entropy(clean_logits, labels[order])
        weighted_hard = .75 * hard
    weighted_hard.backward()
    after_clean = bn_buffers(model) if audit else None
    with batch_statistics_without_running_updates(model):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            noisy_logits = model(noisy_x)
            kd = F.kl_div(F.log_softmax(noisy_logits.float(), dim=1), soft.float(), reduction='batchmean')
            weighted_kd = .25 * kd
        weighted_kd.backward()
    if audit:
        after_noisy = bn_buffers(model)
        assert before.keys() == after_clean.keys() == after_noisy.keys()
        for n in before:
            assert torch.equal(after_clean[n], after_noisy[n]), n
            if n.endswith('num_batches_tracked'):
                assert torch.equal(before[n] + 1, after_clean[n]), n
    optimizer.step()
    clean_hit = lam*(clean_logits.argmax(1)==labels).sum() + (1-lam)*(clean_logits.argmax(1)==labels[order]).sum()
    noisy_hit = lam*(noisy_logits.argmax(1)==labels).sum() + (1-lam)*(noisy_logits.argmax(1)==labels[order]).sum()
    return dict(loss=.75*hard.detach()+.25*kd.detach(), ce=hard.detach(), kd=kd.detach(),
        clean_hit=clean_hit.detach(), noisy_hit=noisy_hit.detach())


def mixed_epoch(model, loader, transform, device, optimizer, method, mix_alpha, amp):
    cfg = core.ACTIVE
    if not cfg['clean_anchor_enabled']:
        # A compatibility check only; actual dual-branch loop is checked separately.
        return previous.mixed_epoch(model, loader, transform, device, optimizer, method, mix_alpha, amp)
    assert cfg['teacher_input'] == 'clean' and cfg['bn_running_policy'] == 'clean_only'
    assert cfg['distill_alpha'] == .25 and cfg['distill_temperature'] == 1 and method == 'cutmix'
    previous.EPOCH += 1
    epoch = previous.EPOCH
    previous.GENERATOR = torch.Generator(device=device).manual_seed(previous.noise_seed(cfg['seed'], epoch))
    previous.TRACE = dict(epoch=epoch, private_seed=previous.noise_seed(cfg['seed'], epoch), batches=0, samples=0,
        selected_images=0, mask_weighted_checksum=0, squared_pixel_change_sum=0., actual_lambda_sample_sum=0., teacher_matched_geometry_queries=0)
    model.train()
    assert len(bn_modules(model)) == 21
    counts_before = [int(m.num_batches_tracked.item()) for m in bn_modules(model)]
    totals = torch.zeros(6, device=device, dtype=torch.float64)
    for batch, (raw, y) in enumerate(loader):
        raw, y = raw.to(device, non_blocking=True), y.to(device, non_blocking=True)
        noisy = previous.augment(raw)
        order = torch.randperm(len(y), device=device)
        lam = float(np.random.beta(mix_alpha, mix_alpha))
        cx, cy = int(np.random.randint(32)), int(np.random.randint(32))
        noisy, lam = previous.patch(noisy, order, lam, cx, cy)
        clean = previous.CLEAN_VIEW
        soft = previous.target(core.TEACHER, noisy, order, lam, 1., 'mixed_view')
        audit = batch == 0 or batch == len(loader)-1
        result = branch_step(model, transform(clean, training=False), transform(noisy, training=False),
            y, order, lam, soft, optimizer, amp, audit=audit)
        totals += torch.stack((result['loss'].double()*len(y), result['ce'].double()*len(y), result['kd'].double()*len(y),
            result['clean_hit'].double(), result['noisy_hit'].double(), torch.tensor(len(y),device=device).double()))
    counts_after = [int(m.num_batches_tracked.item()) for m in bn_modules(model)]
    assert all(b-a == len(loader) for a,b in zip(counts_before,counts_after))
    assert all(m.track_running_stats for m in bn_modules(model))
    loss, hard, kd, hit, noisy_hit, count = totals.cpu().tolist()
    assert count == len(loader.dataset) == previous.TRACE['samples']
    assert previous.TRACE['batches'] == previous.TRACE['teacher_matched_geometry_queries'] == len(loader)
    assert all(math.isfinite(v) for v in (loss,hard,kd,hit,noisy_hit))
    t = previous.TRACE
    t.update(selected_fraction=t['selected_images']/count,
        effective_RMS_over_all_pixels=(t['squared_pixel_change_sum']/(count*3*32*32))**.5,
        mean_actual_lambda=t['actual_lambda_sample_sum']/count, student_forward_batches=2*len(loader),
        optimizer_steps=len(loader), clean_BN_running_update_batches=len(loader), noisy_BN_running_update_batches=0,
        all21_BN_counter_deltas=[b-a for a,b in zip(counts_before,counts_after)], first_last_noisy_buffers_unchanged=True,
        clean_hard_label_accuracy=hit/count, noisy_area_label_accuracy_diagnostic=noisy_hit/count,
        training_accuracy_scope='clean CutMix area-weighted labels; differs from previous noisy-view training accuracy')
    path = previous.FOLDER / 'noise_trace.json'
    rows = read(path) if path.exists() else {}
    rows[str(epoch)] = dict(t)
    write_json(path, rows)
    return dict(loss=loss/count, ce=hard/count, kd=kd/count, accuracy=hit/count, samples=int(count))


def configure():
    previous.original.OUT, previous.original.DIRECTORY = OUT, DIRECTORY
    previous.original.capacity.OUT, previous.original.capacity.DIRECTORY = OUT, DIRECTORY
    core.OUT, core.DIRECTORY = OUT, DIRECTORY
    core.SOURCES = BASE + ('clean_anchor_consistency/train.py',)
    core.Teacher, core.identity, core.mixed_epoch = Teacher, identity, mixed_epoch


def run(cfg, name, stop_after=None):
    assert cfg['teacher_input']=='clean' and cfg['bn_running_policy']=='clean_only'
    assert cfg['noise_probability']==.5 and cfg['noise_sigma_uint8']==8.
    assert cfg['distill_alpha']==.25 and cfg['distill_temperature']==1.
    assert cfg['epochs']==240 or name.startswith('audit_')
    configure()
    previous.FOLDER = DIRECTORY / name
    previous.EPOCH = 0
    if (previous.FOLDER/'last.pt').exists():
        saved=torch.load(previous.FOLDER/'last.pt',map_location='cpu',weights_only=False)
        assert saved['config']==cfg and saved['source_sha256']==source_hashes()
        previous.EPOCH=saved['epoch']
    result=previous.original.capacity.run(cfg,name,stop_after)
    write_json(previous.FOLDER/'objective_branch_metadata.json',dict(
        enabled=cfg['clean_anchor_enabled'], teacher_input='same geometry clean crop/flip/CutMix; fixed uncalibrated original4viewTeacher42',
        objective='.75 area hard CE(student(clean)) + .25 KL(teacher(clean) || student(noisy)), T1',
        backward='Clean weighted CE backward, noisy weighted KL backward, gradients accumulate; exactly one optimizer.step per original batch',
        BN='Clean train-mode forward updates running statistics once; noisy train-mode forward uses its own batch statistics with track_running_stats=False temporarily, floating/integer buffers unchanged; flags restored',
        training_accuracy='Clean CutMix actual-lambda weighted hit rate; noisy hit stored only in trace; not original-image fit',
        student_image_exposures_per_epoch=90000 if cfg['clean_anchor_enabled'] else 45000,
        independent_training_images=45000, optimizer_steps_per_epoch=176, extra_training_compute=True,
        mechanism_package='Hard-loss view and clean-only BN running policy both belong to fixed dual-branch design; does not isolate their separate causal contributions.',
        generic_shared_view_objective_metadata_superseded=True, external_ImageNet_knowledge=True, source_sha256=source_hashes()))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-id',required=True);p.add_argument('--stop-after',type=int)
    a=p.parse_args();run(read(Path(a.config)),a.run_id,a.stop_after)
