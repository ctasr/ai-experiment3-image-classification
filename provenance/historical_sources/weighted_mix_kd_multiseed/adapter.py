"""Loss-only routing over the original loop; the weighted loss is frozen elsewhere."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import traceback
import aligned_distillation.train as CORE

ENGINE=CORE.engine
ROOT=CORE.ROOT
HERE=Path(__file__).resolve().parent
torch=CORE.torch
TEACHER_MODULE=sys.modules[CORE.Teacher.__module__]
ORIGINAL_LOSS_PARTS=CORE.loss_parts
ORIGINAL_AUGMENT=CORE.augment_raw
ORIGINAL_TEACHER=CORE.Teacher
LOSS_PATH=HERE.parent/'source_set_kd_pilot/kd_loss.py'
LOSS_SHA='272ebb7e19cc34023ea7d060972b1d57d2f2a65c39179cd528960e97582a206d'
assert hashlib.sha256(LOSS_PATH.read_bytes()).hexdigest()==LOSS_SHA
_spec=importlib.util.spec_from_file_location('confirmation_frozen_weighted_loss',LOSS_PATH)
LOSS=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(LOSS)
assert Path(LOSS.__file__).resolve()==LOSS_PATH.resolve()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def private(path):
    path=Path(path).resolve();assert path.is_relative_to(HERE);return path

def write(path,value):
    path=private(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)

def tensor_sha(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def source_hashes(reference=False):
    old=read(HERE/'plan.json')['original_training_source_dictionary']
    assert len(old)==12 and all(sha(ROOT/name)==digest for name,digest in old.items())
    assert sha(LOSS_PATH)==LOSS_SHA
    if reference:return old
    additions={Path(os.path.relpath(path,ROOT)).as_posix():sha(path) for path in (LOSS_PATH,Path(__file__),HERE/'run.py',HERE/'postprocess.py')}
    assert len(additions)==4 and not set(old)&set(additions)
    return {**old,**additions}

def configuration(cfg,mode):
    assert mode in ('original','weighted_single')
    cfg=dict(cfg);assert cfg.get('source_set_mode',mode)==mode;cfg['source_set_mode']=mode
    assert cfg['epochs']==240 and cfg['warmup_epochs']==5 and cfg['width']==48 and cfg['seed'] in (42,17,73)
    assert cfg['architecture']=='resnet20' and cfg['model']=='cifar' and cfg['batch_size']==256
    assert cfg['mix']=='cutmix' and cfg['mix_alpha']==.5 and cfg['alignment']=='mixed_view'
    assert cfg['distill_alpha']==.25 and cfg['distill_temperature']==1 and cfg['amp'] is True
    assert not cfg.get('attention',False) and not cfg.get('cutout',0)
    assert cfg.get('pilot_samples') in (None,456)
    return cfg

def bindings():
    return ([(CORE,key) for key in ('OUT','DIRECTORY','SOURCES','Teacher','identity','ACTIVE','TEACHER','mixed_epoch','loss_parts','augment_raw','cutmix_patch','run','source_hashes')]
      +[(ENGINE,key) for key in ('OUT','DIRECTORY','SOURCES','mixed_epoch','build','run')]
      +[(TEACHER_MODULE,key) for key in ('OUT','build')])

@contextmanager
def runtime():
    initialized=torch.cuda.is_initialized()
    saved=(random.getstate(),CORE.np.random.get_state(),torch.get_rng_state().clone(),torch.cuda.get_rng_state_all() if initialized else None,
      torch.get_num_threads(),torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),
      torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic,torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
      torch.utils.deterministic.fill_uninitialized_memory)
    record={'CUDA_initialized_before':initialized}
    try:yield record
    finally:
        random.setstate(saved[0]);CORE.np.random.set_state(saved[1]);torch.set_rng_state(saved[2])
        if saved[3] is not None:torch.cuda.set_rng_state_all(saved[3])
        torch.set_num_threads(saved[4]);torch.use_deterministic_algorithms(saved[5],warn_only=saved[6])
        torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic=saved[7:9]
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=saved[9:11]
        torch.utils.deterministic.fill_uninitialized_memory=saved[11]
        record.update({'CPU_RNG_threads_flags_restored':True,'prior_CUDA_RNG_restored':initialized,
          'first_CUDA_initialization_cannot_be_reversed':not initialized and torch.cuda.is_initialized()})

def combined(logits,labels,order,lam,q,T,alpha,mode):
    original=ORIGINAL_LOSS_PARTS(logits,labels,order,lam,q,T,alpha)
    if mode=='original':return original,original[2].detach()*0,None,original
    assert mode=='weighted_single'
    extra,details=LOSS.extras(logits,labels,order,lam,q,T,mode='weighted_single')
    return (original[0]+alpha*extra,original[1],original[2]+extra),extra,details,original

@contextmanager
def scope(cfg,name,mode,out_dir,run_dir,trace_file=None,reference=False,start_epoch=0):
    cfg=configuration(cfg,mode);assert not reference or mode=='original'
    out_dir,run_dir=private(out_dir),private(run_dir);trace_path=private(trace_file or HERE/'traces'/(name+'.jsonl'))
    missing=object();previous=[(module,key,getattr(module,key,missing)) for module,key in bindings()]
    current={};counter=0;trace=None;state={}
    audit={'mode':mode,'reference_source12':reference,'loss_path':str(LOSS_PATH),'loss_sha256':LOSS_SHA,'original_loop_reused':True,'exception':None}
    try:
        with runtime() as state:
            try:
                identity=read(ROOT/'results_horizon/teacher_identity.json');assert CORE.identity()[1]==identity
                sources=source_hashes(reference);out_dir.mkdir(parents=True,exist_ok=True);run_dir.mkdir(parents=True,exist_ok=True)
                for path,value in ((out_dir/'teacher_identity.json',identity),(out_dir/'source_manifest.json',{'source_sha256':sources})):
                    if path.exists():assert read(path)==value
                    else:write(path,value)
                CORE.OUT,CORE.DIRECTORY,CORE.SOURCES=out_dir,run_dir,tuple(sources)
                TEACHER_MODULE.OUT=out_dir
                trace_path.parent.mkdir(parents=True,exist_ok=True);trace=trace_path.open('a',encoding='utf-8')
                def augmented(images):
                    view=ORIGINAL_AUGMENT(images);current['view_sha256']=tensor_sha(view);return view
                class RecordedTeacher(ORIGINAL_TEACHER):
                    def probabilities(self,images):
                        result=super().probabilities(images)
                        current['teacher_input_sha256']=tensor_sha(images);current['teacher_raw_q_sha256']=tensor_sha(result)
                        return result
                def loss_parts(logits,labels,order,lam,q,T,alpha):
                    nonlocal counter
                    result,extra,details,original=combined(logits,labels,order,lam,q,T,alpha,mode)
                    counter+=1;per_epoch=math.ceil(cfg.get('pilot_samples',45000)/cfg['batch_size'])
                    epoch=start_epoch+(counter-1)//per_epoch+1;batch=(counter-1)%per_epoch+1
                    row={'epoch':epoch,'batch_in_epoch':batch,'global_batch':(epoch-1)*per_epoch+batch,'batch_size':len(labels),'actual_lambda':float(lam),
                      'order_sha256':tensor_sha(order),'labels_sha256':tensor_sha(labels),'teacher_t1_q_sha256':tensor_sha(q),
                      'original_KL':float(original[2].detach()),'new_extra':float(extra.detach()),'modified_KD':float(result[2].detach()),
                      'hard_CE':float(original[1].detach()),'returned_total':float(result[0].detach()),'mode':mode,**current}
                    if details is not None:row.update({key:value.detach().cpu().tolist() if isinstance(value,torch.Tensor) else value for key,value in details.items()})
                    assert all(math.isfinite(row[key]) for key in ('original_KL','new_extra','modified_KD','hard_CE','returned_total'))
                    trace.write(json.dumps(row,allow_nan=False)+'\n');trace.flush();current.clear()
                    return result
                CORE.augment_raw,CORE.Teacher,CORE.loss_parts=augmented,RecordedTeacher,loss_parts
                assert TEACHER_MODULE.build is next(value for module,key,value in previous if module is TEACHER_MODULE and key=='build')
                yield audit
            except BaseException as exc:
                audit['exception']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};raise
            finally:
                try:
                    if trace is not None:trace.close()
                finally:
                    current.clear()
                    for module,key,value in reversed(previous):
                        if value is missing:
                            if hasattr(module,key):delattr(module,key)
                        else:setattr(module,key,value)
                    audit['all_bindings_restored']=all((not hasattr(module,key)) if value is missing else getattr(module,key) is value for module,key,value in previous)
                    audit['batches_seen']=counter
    finally:
        audit['runtime']=state

def run(cfg,name,mode,stop_after,out_dir=None,run_dir=None,trace_file=None,reference=False):
    cfg=configuration(cfg,mode)
    assert type(stop_after) is int and ((cfg.get('pilot_samples')==456 and 1<=stop_after<=6) or (not cfg.get('pilot_samples') and mode=='weighted_single' and stop_after in (12,240)))
    assert not reference or (mode=='original' and cfg.get('pilot_samples')==456)
    out_dir=private(out_dir or HERE/'outputs');run_dir=private(run_dir or HERE/'runs');folder=private(run_dir/name)
    assert folder.name==name
    expected=source_hashes(reference);epoch=0
    if (folder/'config.json').exists():assert read(folder/'config.json')==cfg
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        assert saved['config']==cfg and saved['source_sha256']==expected
        epoch=int(saved['epoch']);assert len(saved['history'])==epoch
        if epoch>=stop_after:return {'state':'already_at_bounded_target','core_called':False,'completed_epochs':epoch}
    audit={};failure=None
    try:
        with scope(cfg,name,mode,out_dir,run_dir,trace_file,reference,epoch) as audit:
            return CORE.run(cfg,name,stop_after)
    except BaseException as exc:
        failure={'exception_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};raise
    finally:
        receipt={**audit,'start_epoch':epoch,'stop_after':stop_after,'source_sha256':expected,'failure':failure}
        try:write(folder/('adapter_failure.json' if failure else 'adapter_audit.json'),receipt)
        except BaseException:
            if failure is None:raise
            print('Could not persist adapter failure receipt; original exception preserved',file=sys.stderr,flush=True)

def cpu_loss_check():
    assert not torch.cuda.is_initialized();generator=torch.Generator().manual_seed(2026112643)
    rows=[];cpu_rng=torch.get_rng_state().clone()
    specs=[('two_sources',.35,.25,'weighted_single'),('same_label',.35,.25,'weighted_single'),('self_order',.7,.25,'weighted_single'),
      ('empty_i',0.,.25,'weighted_single'),('empty_j',1.,.25,'weighted_single'),('extreme',.42,.25,'weighted_single'),
      ('alpha_zero_diagnostic',.42,0.,'weighted_single'),('disabled_original',.42,.25,'original')]
    for name,lam,alpha,mode in specs:
        z=torch.randn((4,10),generator=generator);q=torch.randn((4,10),generator=generator).softmax(1)
        labels=torch.tensor([0,1,2,3]);order=torch.tensor([3,2,1,0])
        if name=='same_label':labels.fill_(2)
        if name=='self_order':order=torch.arange(4)
        if name=='extreme':z=z*500;q=torch.tensor([[1e-30]*9+[1.]]*4,dtype=torch.float32)
        saved_q=q.clone();a=z.clone().requires_grad_();b=z.clone().requires_grad_()
        actual,extra,details,base=combined(a,labels,order,lam,q,1,alpha,mode)
        expected=ORIGINAL_LOSS_PARTS(b,labels,order,lam,q,1,alpha)
        if mode!='original':
            frozen_extra,_=LOSS.extras(b,labels,order,lam,q,1,mode='weighted_single')
            expected=(expected[0]+alpha*frozen_extra,expected[1],expected[2]+frozen_extra)
        assert all(torch.equal(x,y) for x,y in zip(actual,expected)) and all(torch.isfinite(v).all() for v in actual)
        grad_a=torch.autograd.grad(actual[0],a)[0];grad_b=torch.autograd.grad(expected[0],b)[0]
        assert torch.equal(grad_a,grad_b) and torch.equal(q,saved_q)
        if mode=='original':assert actual is base and details is None
        rows.append({'name':name,'loss_tuple_exact':True,'gradient_exact':True,'q_unchanged':True,'new_student_training':False})
    assert torch.equal(cpu_rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    assert Path(LOSS.__file__).resolve()==LOSS_PATH.resolve() and sha(LOSS_PATH)==LOSS_SHA
    return {'state':'passed','cases':rows,'case_count':8,'frozen_loss_sha256':LOSS_SHA,'CPU_synthetic_loss_grad_only':True,'CUDA_initialized':False,'model_forwards':0}

def cpu_route_check(cfg):
    import tempfile
    cfg=dict(cfg);cfg.pop('source_set_mode',None);assert not torch.cuda.is_initialized()
    missing=object();baseline=[(module,key,getattr(module,key,missing)) for module,key in bindings()];rng=torch.get_rng_state().clone();cases=[]
    with tempfile.TemporaryDirectory(dir=HERE,prefix='cpu_scope_') as temp:
        root=Path(temp)
        for mode in ('original','weighted_single'):
            for fail in (False,True):
                active=None
                try:
                    with scope(cfg,mode,mode,root/(mode+'_out'),root/'runs',root/(mode+'.jsonl')) as active:
                        assert CORE.loss_parts is not ORIGINAL_LOSS_PARTS and CORE.Teacher is not ORIGINAL_TEACHER
                        assert TEACHER_MODULE.OUT==root/(mode+'_out') and TEACHER_MODULE.build is next(value for module,key,value in baseline if module is TEACHER_MODULE and key=='build')
                        CORE.ACTIVE={'mock':True};CORE.TEACHER=object();random.random();CORE.np.random.random();torch.rand(1)
                        if fail:raise RuntimeError('intentional CPU scope failure')
                except RuntimeError as exc:assert fail and str(exc)=='intentional CPU scope failure'
                assert active['all_bindings_restored'] and bool(active['exception'])==fail
                for module,key,value in baseline:assert (not hasattr(module,key)) if value is missing else getattr(module,key) is value
                cases.append({'mode':mode,'intentional_exception':fail,'externally_verified_bindings_restored':True})
        with scope(cfg,'outer','weighted_single',root/'outer',root/'runs',root/'outer.jsonl'):
            outer=CORE.loss_parts
            with scope(cfg,'inner','original',root/'inner',root/'runs',root/'inner.jsonl'):assert CORE.loss_parts is not outer
            assert CORE.loss_parts is outer
        for module,key,value in baseline:assert (not hasattr(module,key)) if value is missing else getattr(module,key) is value
        original_run=CORE.run
        def failed_run(*args,**kwargs):
            CORE.ACTIVE={'mock_failure':True};CORE.TEACHER=object();torch.rand(1)
            raise RuntimeError('intentional CPU run failure')
        try:
            CORE.run=failed_run
            try:
                run(cfg,'run_failure','weighted_single',stop_after=2 if cfg.get('pilot_samples') else 12,
                  out_dir=root/'failure_out',run_dir=root/'failure_runs',trace_file=root/'failure.jsonl')
            except RuntimeError as exc:assert str(exc)=='intentional CPU run failure'
            else:raise AssertionError('Expected the original run exception')
            receipt=read(root/'failure_runs/run_failure/adapter_failure.json')
            assert receipt['failure']['message']=='intentional CPU run failure' and receipt['all_bindings_restored']
            assert receipt['runtime']['CPU_RNG_threads_flags_restored'] and CORE.run is failed_run
        finally:CORE.run=original_run
        for module,key,value in baseline:assert (not hasattr(module,key)) if value is missing else getattr(module,key) is value
    assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    return {'state':'passed','cases':cases,'nested_restore':True,'CPU_RNG_restored':True,'Teacher_build_role_unchanged':True,
      'run_failure_receipt_verified':True,'original_exception_preserved':True,'old_SourceSet_adapter_imported':False,'model_forwards':0,'CUDA_initialized':False}
