"""Isolated RA hooks around the unchanged online KD loop and checkpoint engine.

No private RNG state is advanced: epoch is restored from the old engine checkpoint,
and each per-image policy is a pure function of seed/epoch/batch/image coordinate.
"""
import argparse
import hashlib
import math
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from data import ROOT
from train import write_json,save_checkpoint
from distillation.targets import read,sha
import aligned_distillation.train as core
from aligned_distillation.teacher import Teacher,identity
from randaugment_study_v4 import augment

OUT=ROOT/'results_randaugment_v4'
DIRECTORY=ROOT/'runs/randaugment_v4'
BASE=tuple(read(ROOT/'results_horizon/source_manifest.json')['source_sha256'])
SOURCES=BASE+('randaugment_study_v4/train.py','randaugment_study_v4/augment.py')
ORIGINAL_MIXED=core.mixed_epoch
ORIGINAL_AUGMENT=core.augment_raw
ORIGINAL_CUTMIX=core.cutmix_patch
ORIGINAL_TARGET=core.target_for_view
EPOCH=0
FOLDER=None
AUDIT=None
BATCH=-1


def source_hashes():return {name:sha(ROOT/name) for name in SOURCES}


def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def validate_cfg(cfg):
    if not cfg.get('randaugment',False):return
    assert cfg['ra_num_ops']==2 and cfg['ra_magnitude']==9 and cfg['ra_bins']==31
    assert cfg['ra_fill']==[125,123,114] and cfg['ra_interpolation']=='bilinear' and cfg['ra_seed_tag']==augment.SEED_TAG
    assert cfg['alignment']=='mixed_view' and cfg['distill_alpha']==.25 and cfg['distill_temperature']==1
    assert cfg['mix']=='cutmix' and cfg['mix_alpha']==.5 and not cfg.get('cutout',0)


def augmented_raw(images):
    global BATCH
    BATCH+=1
    view=ORIGINAL_AUGMENT(images)
    if EPOCH==0 and BATCH==0 and FOLDER is not None:
        AUDIT['examples']={'raw':images[:8].detach().cpu().numpy().copy(),
                           'crop_flip':view[:8].detach().cpu().numpy().copy()}
    operations,signs=augment.policy(core.ACTIVE['seed'],EPOCH,BATCH,len(images))
    AUDIT['policy_hasher'].update(operations.astype('<i8').tobytes()+signs.astype('<i8').tobytes())
    AUDIT['op_counts']+=np.bincount(operations.ravel(),minlength=len(augment.NAMES))
    AUDIT['images']+=len(images);AUDIT['batch_sizes'].append(len(images))
    before=time.perf_counter()
    view=augment.grouped(view,operations,signs)
    if EPOCH==0 and BATCH==0 and 'examples' in AUDIT:AUDIT['examples']['randaugment']=view[:8].detach().cpu().numpy().copy()
    AUDIT['augmentation_dispatch_seconds']+=time.perf_counter()-before
    AUDIT['current']={'batch':BATCH,'size':len(images),'first_policy':operations[0].tolist(),
                      'first_signs':signs[0].tolist(),'policy_sha256':augment.digest(operations,signs)}
    return view


def cutmix_checked(view,order,nominal,cx,cy):
    width,height=view.shape[-1],view.shape[-2]
    cw,ch=int(width*math.sqrt(1-nominal)),int(height*math.sqrt(1-nominal))
    l,r=max(0,cx-cw//2),min(width,cx+(cw+1)//2)
    t,b=max(0,cy-ch//2),min(height,cy+(ch+1)//2)
    output,lam=ORIGINAL_CUTMIX(view,order,nominal,cx,cy)
    assert lam==1-(r-l)*(b-t)/(width*height)
    AUDIT['area_checks']+=1;AUDIT['lambda_values'].append(lam)
    AUDIT['current'].update({'actual_lambda':lam,'patch':[t,b,l,r]})
    if EPOCH==0 and BATCH==0 and 'examples' in AUDIT:
        example={**AUDIT['examples'],'cutmix':output[:8].detach().cpu().numpy().copy(),
                 'labels':AUDIT['labels'][:8].numpy().copy(),
                 'partner_labels':AUDIT['labels'][order.detach().cpu()[:8]].numpy().copy(),
                 'actual_lambda':np.array(lam),'patch':np.array([t,b,l,r]),
                 'operation_indices':augment.policy(core.ACTIVE['seed'],EPOCH,BATCH,len(view))[0][:8],
                 'operation_signs':augment.policy(core.ACTIVE['seed'],EPOCH,BATCH,len(view))[1][:8]}
        path=FOLDER/'augmentation_examples.npz'
        if path.exists():
            previous=np.load(path);assert set(previous.files)==set(example) and all(np.array_equal(previous[k],v) for k,v in example.items())
        else:np.savez_compressed(path,**example)
    return output,lam


def target_checked(teacher,view,order,lam,temperature,alignment):
    assert alignment=='mixed_view'
    version=view._version
    AUDIT['teacher_view']=(view.data_ptr(),tuple(view.shape),version)
    if BATCH in (0,1) or len(view)<core.ACTIVE['batch_size']:
        AUDIT['current']['actual_mixed_input_sha256']=tensor_hash(view)
    target=ORIGINAL_TARGET(teacher,view,order,lam,temperature,alignment)
    assert view._version==version and target.shape==(len(view),10)
    assert torch.isfinite(target).all() and torch.allclose(target.sum(1),torch.ones(len(view),device=view.device),atol=1e-6)
    return target


class AuditedTransform:
    def __init__(self,transform):self.transform=transform
    def __call__(self,view,training=False):
        assert not training
        assert AUDIT['teacher_view']==(view.data_ptr(),tuple(view.shape),view._version)
        if 'actual_mixed_input_sha256' in AUDIT['current']:
            assert tensor_hash(view)==AUDIT['current']['actual_mixed_input_sha256']
            AUDIT['input_examples'].append(dict(AUDIT['current']))
        AUDIT['shared_input_batches']+=1
        return self.transform(view,training=False)


class AuditedLoader:
    def __init__(self,loader):self.loader=loader;self.dataset=loader.dataset
    def __len__(self):return len(self.loader)
    def __iter__(self):
        for x,y in self.loader:
            AUDIT['labels']=y.detach().cpu()
            yield x,y


def mixed_epoch(model,loader,transform,device,optimizer,method,mix_alpha,amp):
    global EPOCH,AUDIT,BATCH
    if not core.ACTIVE.get('randaugment',False):
        return ORIGINAL_MIXED(model,loader,transform,device,optimizer,method,mix_alpha,amp)
    AUDIT={'policy_hasher':hashlib.sha256(),'op_counts':np.zeros(len(augment.NAMES),dtype=np.int64),
           'images':0,'batch_sizes':[],'area_checks':0,'lambda_values':[],
           'shared_input_batches':0,'augmentation_dispatch_seconds':0.,'input_examples':[]}
    BATCH=-1
    core.augment_raw,core.cutmix_patch,core.target_for_view=augmented_raw,cutmix_checked,target_checked
    try:result=ORIGINAL_MIXED(model,AuditedLoader(loader),AuditedTransform(transform),device,optimizer,method,mix_alpha,amp)
    finally:core.augment_raw,core.cutmix_patch,core.target_for_view=ORIGINAL_AUGMENT,ORIGINAL_CUTMIX,ORIGINAL_TARGET
    assert AUDIT['images']==len(loader.dataset) and AUDIT['shared_input_batches']==len(loader)==AUDIT['area_checks']
    audit={'epoch':EPOCH+1,'zero_based_epoch':EPOCH,'seed':core.ACTIVE['seed'],
           'private_policy_sha256':AUDIT['policy_hasher'].hexdigest(),
           'operator_counts':dict(zip(augment.NAMES,map(int,AUDIT['op_counts']))),
           'sample_count':AUDIT['images'],'operations_count':int(AUDIT['op_counts'].sum()),
           'batch_sizes':AUDIT['batch_sizes'],'actual_area_checked_batches':AUDIT['area_checks'],
           'same_actual_RA_CutMix_teacher_student_batches':AUDIT['shared_input_batches'],
           'actual_lambda_min':min(AUDIT['lambda_values']),'actual_lambda_max':max(AUDIT['lambda_values']),
           'input_examples':AUDIT['input_examples'],
           'augmentation_dispatch_seconds':AUDIT['augmentation_dispatch_seconds'],
           'timing_note':'CPU dispatch wall interval, includes possible operator synchronization; not pure GPU kernel time.'}
    if FOLDER is not None:write_json(FOLDER/'augmentation_audits'/f'epoch_{EPOCH+1:04d}.json',audit)
    EPOCH+=1
    return {**result,'ra_images':AUDIT['images'],'ra_operations':int(AUDIT['op_counts'].sum()),
            'ra_batches':len(AUDIT['batch_sizes']),'ra_area_checks':AUDIT['area_checks'],
            'ra_shared_input_batches':AUDIT['shared_input_batches'],'ra_last_batch_samples':AUDIT['batch_sizes'][-1]}


def configure():
    core.OUT,core.DIRECTORY=OUT,DIRECTORY
    core.SOURCES=SOURCES;core.Teacher,core.identity=Teacher,identity
    core.mixed_epoch=mixed_epoch
    core.engine.DIRECTORY=DIRECTORY


def ensure_best_commit(folder,cfg,expected_sources):
    """Recover only the provable last->best transaction window; never guess older weights."""
    last=folder/'last.pt';best=folder/'best.pt'
    if not last.exists():
        assert not best.exists(),'best exists without its last/history; inspect rather than overwrite.'
        return None
    state=torch.load(last,map_location='cpu',weights_only=False)
    assert state['config']==cfg and state['source_sha256']==expected_sources
    assert len(state['history'])==state['epoch'] and state['history']
    chosen=max(state['history'],key=lambda r:(r['validation_accuracy'],-r['validation_ce']))
    assert state['best_accuracy']==chosen['validation_accuracy'] and state['best_ce']==chosen['validation_ce']
    previous=None
    if best.exists():
        previous=torch.load(best,map_location='cpu',weights_only=False)
        assert previous['config']==cfg and previous['source_sha256']==expected_sources and previous['stats']==state['stats']
        assert previous['epoch']<=chosen['epoch'],'best is ahead of committed history; inspect rollback.'
        if previous['epoch']==chosen['epoch']:
            assert previous['best_accuracy']==chosen['validation_accuracy'] and previous['best_ce']==chosen['validation_ce']
            return {'best_epoch':chosen['epoch'],'recovered':False,'best_sha256':sha(best)}
    # The last model can reconstruct best only if that exact completed epoch is the new winner.
    assert chosen['epoch']==state['epoch'],'Missing historical best cannot be reconstructed from a later last model.'
    old_sha=sha(best) if best.exists() else None
    if old_sha:
        backup=folder/f'best_before_commit_recovery_{old_sha}.pt'
        if not backup.exists():shutil.copy2(best,backup)
        assert sha(backup)==old_sha
    payload={k:v for k,v in state.items() if k not in ('optimizer','scheduler','rng','history')}
    save_checkpoint(best,payload)
    proof={'reason':'last checkpoint committed an improved epoch before best checkpoint transaction finished',
           'best_epoch':chosen['epoch'],'last_epoch':state['epoch'],'previous_best_epoch':None if previous is None else previous['epoch'],
           'old_best_sha256':old_sha,'reconstructed_best_sha256':sha(best),
           'reconstruction':'Exact model/state from samewinninglast epoch; optimizer/RNG/history omitted as oldenginebest format.',
           'source_sha256':expected_sources}
    write_json(folder/f'checkpoint_commit_recovery_epoch_{state["epoch"]:04d}.json',proof)
    return {**proof,'recovered':True}


def run_locked(cfg,name,stop_after=None):
    global EPOCH,FOLDER
    validate_cfg(cfg);configure()
    FOLDER=DIRECTORY/name;FOLDER.mkdir(parents=True,exist_ok=True)
    (FOLDER/'augmentation_audits').mkdir(exist_ok=True)
    library=augment.library_manifest()
    assert read(OUT/'source_manifest.json')['source_sha256']==source_hashes()
    assert read(OUT/'source_manifest.json')['augmentation_library']==library
    metadata={'enabled':bool(cfg.get('randaugment',False)),'sampling':'private stateless PCG64 SeedSequence([seed,zero_based_epoch,batch,58193]); fixed256 policy rows, image-in-batch index selects row',
              'num_ops':2,'magnitude':9,'bins':31,'interpolation':'bilinear','fill':[125,123,114],
              'order':'crop/flip uint8 -> per-image RA -> CutMix actual-area -> teacher/student sharedrawview',
              'library':library,'training_source_sha256':source_hashes(),
              'checkpoint_determinism':'No advanced private RNG; config seed, checkpoint completed epoch and frozen sampler source uniquely determine all next policies.'}
    path=FOLDER/'randaugment_protocol.json'
    if path.exists():assert read(path)==metadata
    else:write_json(path,metadata)
    EPOCH=0
    if (FOLDER/'last.pt').exists():
        saved=torch.load(FOLDER/'last.pt',map_location='cpu',weights_only=False)
        assert saved['config']==cfg and saved['source_sha256']==source_hashes()
        EPOCH=saved['epoch']
    ensure_best_commit(FOLDER,cfg,source_hashes())
    torch.cuda.reset_peak_memory_stats()
    result=core.run(cfg,name,stop_after)
    commit=ensure_best_commit(FOLDER,cfg,source_hashes())
    if result.get('state')=='completed':
        assert commit['best_epoch']==result['best_epoch']
        chosen=torch.load(FOLDER/'best.pt',map_location='cpu',weights_only=False)
        assert chosen['best_accuracy']==result['best_validation_accuracy'] and chosen['best_ce']==result['best_validation_ce']
    write_json(FOLDER/'checkpoint_commit_check.json',commit)
    write_json(FOLDER/'runtime_randaugment.json',{'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'parameters':result.get('parameters'),
        'note':'Training process peak includes original fixed teacher; not student deploymentmemory or walltime.'})
    return result


def run(cfg,name,stop_after=None):
    from randaugment_study_v4.common import file_lease,process_info,write,now
    folder=DIRECTORY/name
    with file_lease(folder/'train.lock'):
        record={**process_info(__import__('os').getpid()),'state':'running','run_id':name,
                'source_sha256':source_hashes(),'started_at_utc':now()}
        write(folder/'train_process.json',record)
        try:return run_locked(cfg,name,stop_after)
        finally:write(folder/'train_process.json',{**record,'state':'exited','ended_at_utc':now()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--run-id',required=True);parser.add_argument('--stop-after',type=int)
    a=parser.parse_args();run(read(Path(a.config)),a.run_id,a.stop_after)
