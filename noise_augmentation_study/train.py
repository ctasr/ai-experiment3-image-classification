"""One fixed noisy-input augmentation. Original engines remain unchanged."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from data import ROOT
from train import write_json
from distillation.targets import read, sha
import horizon_study.train as original
from aligned_distillation.teacher import Teacher, identity

OUT = ROOT / 'results_noise_augmentation'
DIRECTORY = ROOT / 'runs/noise_augmentation'
BASE = tuple(read(ROOT / 'results_horizon/source_manifest.json')['source_sha256'])
core = original.capacity.core
OLD_MIXED, OLD_AUGMENT, OLD_PATCH, OLD_TARGET = core.mixed_epoch, core.augment_raw, core.cutmix_patch, core.target_for_view
EPOCH = 0
GENERATOR = None
TRACE = None
FOLDER = None
LAST_VIEW = None


def source_hashes():
    return {n: sha(ROOT / n) for n in BASE + ('noise_augmentation_study/train.py',)}


def noise_seed(seed, epoch):
    return int(np.random.SeedSequence([int(seed), int(epoch), 53191]).generate_state(1, dtype=np.uint64)[0]) % (2**63 - 1)


def apply_noise(view, mask, normal, sigma):
    assert view.dtype == torch.uint8 and mask.shape == (len(view),)
    changed = (view.float() + float(sigma) * normal).clamp(0, 255).round().to(torch.uint8)
    return torch.where(mask[:, None, None, None], changed, view)


def augment(raw):
    view = OLD_AUGMENT(raw)
    mask = torch.rand(len(view), device=view.device, generator=GENERATOR) < core.ACTIVE['noise_probability']
    normal = torch.randn(view.shape, device=view.device, generator=GENERATOR, dtype=torch.float32)
    result = apply_noise(view, mask, normal, core.ACTIVE['noise_sigma_uint8'])
    TRACE['batches'] += 1
    TRACE['samples'] += len(view)
    TRACE['selected_images'] += int(mask.sum().item())
    TRACE['mask_weighted_checksum'] += int((mask.long() * torch.arange(TRACE['samples'] - len(view) + 1, TRACE['samples'] + 1, device=view.device)).sum().item())
    TRACE['squared_pixel_change_sum'] += float(((result.float() - view.float()) ** 2).sum().item())
    TRACE['last_batch_samples'] = len(view)
    return result


def patch(view, order, lam, cx, cy):
    global LAST_VIEW
    result, actual = OLD_PATCH(view, order, lam, cx, cy)
    LAST_VIEW = result
    TRACE['actual_lambda_sample_sum'] += actual * len(view)
    if TRACE['batches'] == 1:
        TRACE['first_actual_teacher_input_sha256'] = hashlib.sha256(result.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return result, actual


def target(teacher, view, order, lam, temperature, alignment):
    assert view is LAST_VIEW and alignment == 'mixed_view'
    TRACE['teacher_shared_view_queries'] += 1
    return OLD_TARGET(teacher, view, order, lam, temperature, alignment)


def mixed_epoch(model, loader, transform, device, optimizer, method, mix_alpha, amp):
    global EPOCH, GENERATOR, TRACE
    EPOCH += 1
    cfg = core.ACTIVE
    if cfg['noise_probability'] == 0 or cfg['noise_sigma_uint8'] == 0:
        return OLD_MIXED(model, loader, transform, device, optimizer, method, mix_alpha, amp)
    assert cfg['distill_alpha'] == .25 and cfg['alignment'] == 'mixed_view'
    GENERATOR = torch.Generator(device=device).manual_seed(noise_seed(cfg['seed'], EPOCH))
    TRACE = dict(epoch=EPOCH, private_seed=noise_seed(cfg['seed'], EPOCH), batches=0, samples=0, selected_images=0,
        mask_weighted_checksum=0, squared_pixel_change_sum=0., actual_lambda_sample_sum=0., teacher_shared_view_queries=0)
    old = core.augment_raw, core.cutmix_patch, core.target_for_view
    core.augment_raw, core.cutmix_patch, core.target_for_view = augment, patch, target
    try:
        result = OLD_MIXED(model, loader, transform, device, optimizer, method, mix_alpha, amp)
    finally:
        core.augment_raw, core.cutmix_patch, core.target_for_view = old
    assert TRACE['batches'] == TRACE['teacher_shared_view_queries'] == len(loader)
    assert TRACE['samples'] == result['samples'] == len(loader.dataset)
    TRACE['selected_fraction'] = TRACE['selected_images'] / TRACE['samples']
    TRACE['effective_RMS_over_all_pixels'] = (TRACE['squared_pixel_change_sum'] / (TRACE['samples'] * 3 * 32 * 32)) ** .5
    TRACE['mean_actual_lambda'] = TRACE['actual_lambda_sample_sum'] / TRACE['samples']
    path = FOLDER / 'noise_trace.json'
    rows = read(path) if path.exists() else {}
    rows[str(EPOCH)] = dict(TRACE)
    write_json(path, rows)
    return result


def configure():
    original.OUT, original.DIRECTORY = OUT, DIRECTORY
    original.capacity.OUT, original.capacity.DIRECTORY = OUT, DIRECTORY
    core.OUT, core.DIRECTORY = OUT, DIRECTORY
    core.SOURCES = BASE + ('noise_augmentation_study/train.py',)
    core.Teacher, core.identity, core.mixed_epoch = Teacher, identity, mixed_epoch


def run(cfg, name, stop_after=None):
    global EPOCH, FOLDER
    assert 0 <= cfg['noise_probability'] <= 1 and cfg['noise_sigma_uint8'] >= 0
    assert cfg['mix'] == 'cutmix' and cfg['mix_alpha'] == .5 and cfg['distill_alpha'] == .25 and cfg['alignment'] == 'mixed_view'
    assert cfg['epochs'] == 240 or name.startswith('audit_')
    configure()
    FOLDER = DIRECTORY / name
    EPOCH = 0
    if (FOLDER / 'last.pt').exists():
        saved = torch.load(FOLDER / 'last.pt', map_location='cpu', weights_only=False)
        assert saved['config'] == cfg and saved['source_sha256'] == source_hashes()
        EPOCH = saved['epoch']
    result = original.capacity.run(cfg, name, stop_after)
    write_json(FOLDER / 'noise_metadata.json', dict(probability=cfg['noise_probability'], sigma_uint8=cfg['noise_sigma_uint8'],
        order='originalcrop/flip then independentperimage Gaussian noise+round/clip, then originalCutMix; sameactualview teacher+student',
        private_RNG='CUDA TorchGenerator seed derived by NumPySeedSequence([trainingseed,onebasedepoch,53191]); no global noise RNG draws',
        restore='Private generator reconstructed at epoch boundary; original globalPythonNumPyTorchCUDA/loader state still saved/restored by originalengine',
        target='Original .75CutMixCE+.25KL, T1 and originalfourview uncalibratedTeacher42 unchanged; teacher sees noisy actualview',
        training_accuracy='actual CutMix area lambda-weighted hit rate, not original-image fit',
        validation='Clean original raw singleforward hardCE/accuracy; best chosen accuracy then CE only, no noisy selection during240epochs',
        external_ImageNet_knowledge=True, source_sha256=source_hashes()))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--stop-after', type=int)
    args = p.parse_args()
    run(read(Path(args.config)), args.run_id, args.stop_after)
