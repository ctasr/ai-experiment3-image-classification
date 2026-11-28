"""Frozen two-seed confirmation, separate42bridge and autonomous fixed analysis."""
from pathlib import Path
from contextlib import contextmanager
from itertools import zip_longest
import argparse,ast,ctypes,hashlib,importlib.util,json,math,os,statistics,sys,time,traceback
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]/"实验三-图像分类"
PARENT=HERE.parent/"weighted_mix_kd_confirmation"
OLD=HERE.parent/"source_set_kd_pilot"
DECISION_SHA="bc78434c073a125a154f816d6d2d865cbde22e4828b156eaa5e610a94cc80c99"
SEEDS=(17,73)
INPUT_KEYS=("epoch","batch_in_epoch","global_batch","batch_size","actual_lambda","order_sha256","labels_sha256","view_sha256","teacher_input_sha256","teacher_raw_q_sha256","teacher_t1_q_sha256")
LOSS_KEYS=("original_KL","new_extra","modified_KD","hard_CE","returned_total")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write(name, value):
    path = (HERE / name).resolve()
    assert path.is_relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)

def load_adapter():
    old_path = list(sys.path)
    old_bytecode = sys.dont_write_bytecode
    try:
        sys.path.insert(0, str(PROJECT))
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location("private_weighted_confirmation_adapter", HERE / "adapter.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode

@contextmanager
def awake():
    assert os.name == "nt"
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel.SetThreadExecutionState
    function.argtypes = [ctypes.c_uint]; function.restype = ctypes.c_uint
    previous = int(function(0x80000003))
    if not previous:
        raise ctypes.WinError(ctypes.get_last_error())
    record = {"previous_request": previous, "requested": 0x80000003, "restored": False, "powercfg_changed": False}
    try:
        yield record
    finally:
        if not function(previous):
            raise ctypes.WinError(ctypes.get_last_error())
        record["restored"] = True

@contextmanager
def exclusive():
    assert os.name == "nt"
    import msvcrt
    stream = (HERE / "driver.lock").open("a+b")
    locked = False
    try:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        locked = True
        yield
    finally:
        if locked:
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        stream.close()

def checkpoint(adapter, folder, kind="last.pt"):
    return adapter.torch.load(Path(folder) / kind, map_location="cpu", weights_only=False)

def equal(adapter, a, b):
    if isinstance(a, adapter.torch.Tensor):
        return isinstance(b, adapter.torch.Tensor) and adapter.torch.equal(a, b)
    if isinstance(a, adapter.CORE.np.ndarray):
        return isinstance(b, adapter.CORE.np.ndarray) and adapter.CORE.np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(adapter, a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(adapter, x, y) for x, y in zip(a, b))
    return a == b

def compare(adapter, left_folder, right_folder):
    left, right = checkpoint(adapter, left_folder), checkpoint(adapter, right_folder)
    for key in ("epoch", "model", "optimizer", "scheduler", "rng", "best_accuracy", "best_ce", "stats"):
        assert equal(adapter, left[key], right[key]), "Checkpoint identity: " + key
    histories = [[{key: value for key, value in row.items() if key != "seconds"} for row in saved["history"]]
                 for saved in (left, right)]
    assert histories[0] == histories[1]
    first, second = checkpoint(adapter, left_folder, "best.pt"), checkpoint(adapter, right_folder, "best.pt")
    for key in ("epoch", "model", "best_accuracy", "best_ce", "stats"):
        assert equal(adapter, first[key], second[key]), "Best checkpoint identity: " + key

def audit_trace(path, epochs, bpe):
    count = 0
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            epoch, batch = count // bpe + 1, count % bpe + 1
            size = 200 if batch == bpe else 256
            assert (row["epoch"], row["batch_in_epoch"], row["global_batch"], row["batch_size"]) == (epoch, batch, count + 1, size)
            assert all(key in row for key in INPUT_KEYS + LOSS_KEYS)
            assert all(math.isfinite(row[key]) for key in LOSS_KEYS)
            for key in INPUT_KEYS:
                if key.endswith("sha256"):
                    assert isinstance(row[key], str) and len(row[key]) == 64
                    assert all(character in "0123456789abcdef" for character in row[key].lower())
            count += 1
    assert count == epochs * bpe
    return {"records": count, "sample_counts_indices_losses_finite": True}

def shared_trace_equal(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert all(a[key] == b[key] for key in INPUT_KEYS + LOSS_KEYS)
        shared = set(a) & set(b)
        assert all(a[key] == b[key] for key in shared)


def implementation():
    assert sha(HERE/'implementation_decision.json')==DECISION_SHA
    decision=read(HERE/'implementation_decision.json');plan=read(HERE/'plan.json')
    assert decision['implementation_approved'] is True and decision['GPU_or_training_approved'] is False
    assert plan['decision_sha256']==DECISION_SHA and plan['training_seed_order']==list(SEEDS)
    assert sha(PARENT/'stage_closed.json')==decision['parent_closed_stage_sha256']==plan['parent_closed_stage_sha256']
    assert sha(PARENT/'source_manifest.json')==decision['parent_source_manifest_sha256']
    parent_manifest=read(PARENT/'source_manifest.json')
    assert all(sha(PROJECT/name)==digest for name,digest in parent_manifest['training_sources'].items())
    return decision,plan

def load_postprocess():
    old=list(sys.path)
    try:
        sys.path.insert(0,str(PROJECT));spec=importlib.util.spec_from_file_location('private_multiseed_postprocess',HERE/'postprocess.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    finally:sys.path[:]=old

def verify(adapter=None):
    implementation();frozen=read(HERE/'source_manifest.json')
    assert frozen['plan_sha256']==sha(HERE/'plan.json') and frozen['implementation_decision_sha256']==DECISION_SHA
    for name,digest in frozen['private_sources'].items():assert sha(HERE/name)==digest
    for name,digest in frozen['analysis_sources'].items():assert sha(HERE/name)==digest
    for path,digest in frozen['protected_inputs'].items():assert sha(path)==digest,'Protected file changed: '+path
    adapter=adapter or load_adapter();assert adapter.source_hashes()==frozen['training_sources'] and len(frozen['training_sources'])==16
    return {'passed':True,'protected_count':len(frozen['protected_inputs']),'complete_execution_source_count':16}

def ready(name,adapter):
    item=read(HERE/name);assert item['approved'] is True
    for key,filename in (('plan_sha256','plan.json'),('source_manifest_sha256','source_manifest.json'),('prepare_CPU_sha256','prepare_CPU.json')):assert item[key]==sha(HERE/filename)
    if name=='CONFIRM_READY.json':
        assert item['bridge_GPU_sha256']==sha(HERE/'bridge_GPU.json') and read(HERE/'bridge_GPU.json')['state']=='passed'
        assert item['seed_order']==list(SEEDS) and item['automatic_test_reproduction'] is False
    verify(adapter);return item

def baseline(seed):return next(row for row in read(HERE/'plan.json')['baseline_last'] if row['seed']==seed)

def cfg(adapter,seed):
    assert seed in (42,17,73);value=read(PROJECT/'runs/horizon'/baseline(seed)['run_id']/'config.json')
    assert value['seed']==seed;value['source_set_mode']='weighted_single';return adapter.configuration(value,'weighted_single')

def unchanged_bodies():
    def functions(path):return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(Path(path).read_text(encoding='utf-8')).body if isinstance(node,ast.FunctionDef)}
    old,new=functions(PARENT/'adapter.py'),functions(HERE/'adapter.py')
    expected={'configuration','source_hashes'};changed={name for name in old if old[name]!=new.get(name)}
    assert changed==expected and set(old)==set(new)
    names=('runtime','combined','scope','run','bindings','cpu_loss_check','cpu_route_check')
    assert all(old[name]==new[name] for name in names)
    return {'state':'passed','changed_functions':sorted(changed),'actual_train_loss_context_resume_bodies_unchanged':list(names),
      'reused_parent_CPU_sha256':sha(PARENT/'prepare_CPU.json'),'reused_parent_actual_GPU_preflight_sha256':sha(PARENT/'preflight_GPU.json')}

def snapshot(adapter):
    import random
    torch=adapter.torch;np=adapter.CORE.np
    return {'bindings':[(module,key,hasattr(module,key),getattr(module,key,None)) for module,key in adapter.bindings()],
      'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state().clone(),
      'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
      'threads':torch.get_num_threads(),'deterministic':torch.are_deterministic_algorithms_enabled(),'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
      'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
      'TF32matmul':torch.backends.cuda.matmul.allow_tf32,'TF32cudnn':torch.backends.cudnn.allow_tf32,'fill':torch.utils.deterministic.fill_uninitialized_memory}

def check_snapshot(adapter,old):
    now=snapshot(adapter)
    for module,key,present,value in old['bindings']:assert hasattr(module,key)==present and (not present or getattr(module,key) is value),'Postprocess leaked binding: '+key
    for key in old:
        if key!='bindings':assert equal(adapter,old[key],now[key]),'Postprocess leaked runtime: '+key

def checked_postprocess(adapter,callback,label):
    previous=snapshot(adapter);original_failure=None
    try:return callback()
    except BaseException:
        original_failure=traceback.format_exc();raise
    finally:
        try:check_snapshot(adapter,previous)
        except BaseException as leaked:
            write('postprocess_state_failure.json',{'state':'postprocess_integrity_failure_no_next_seed','label':label,
              'integrity_error':repr(leaked),'original_exception':original_failure,'original_exception_preserved':original_failure is not None})
            if original_failure is None:raise

def prepare():
    implementation();assert not (HERE/'source_manifest.json').exists() and not (HERE/'prepare_CPU.json').exists()
    adapter=load_adapter();assert not adapter.torch.cuda.is_initialized()
    body_proof=unchanged_bodies();routes={};roles={}
    with adapter.runtime():
        adapter.torch.set_num_threads(1)
        reference=cfg(adapter,42)
        for seed in (42,17,73):
            value=cfg(adapter,seed);assert {key for key in value if value[key]!=reference[key]}==({'seed'} if seed!=42 else set())
            routes[str(seed)]=adapter.cpu_route_check(value)
            assert routes[str(seed)]['state']=='passed'
            roles[str(seed)]={'config_seed':value['seed'],'baseline_run':baseline(seed)['run_id'],'data_split_seed':42,'teacher_seed':42}
        bad=dict(reference);bad['seed']=43
        try:adapter.configuration(bad,'weighted_single')
        except AssertionError:pass
        else:raise AssertionError('Unexpected training seed accepted')
        original=snapshot(adapter);postprocess=load_postprocess();post_proof=postprocess.cpu_import_preflight();check_snapshot(adapter,original)
        assert post_proof['state']=='passed'
        import random
        post_runtime=[]
        for fail in (False,True):
            previous=snapshot(adapter);runtime_record=None
            try:
                with postprocess._runtime(adapter.torch,adapter.CORE.np) as runtime_record:
                    random.random();adapter.CORE.np.random.random();adapter.torch.rand(3)
                    adapter.torch.set_num_threads(2)
                    adapter.torch.use_deterministic_algorithms(not previous['deterministic'])
                    adapter.torch.backends.cudnn.benchmark=not previous['cudnn_benchmark']
                    adapter.torch.backends.cudnn.deterministic=not previous['cudnn_deterministic']
                    adapter.torch.backends.cuda.matmul.allow_tf32=not previous['TF32matmul']
                    adapter.torch.backends.cudnn.allow_tf32=not previous['TF32cudnn']
                    adapter.torch.utils.deterministic.fill_uninitialized_memory=not previous['fill']
                    if fail:raise RuntimeError('intentional postprocess CPU restore test')
            except RuntimeError as exc:assert fail and str(exc)=='intentional postprocess CPU restore test'
            check_snapshot(adapter,previous);assert runtime_record['CPU_RNG_threads_flags_restored']
            post_runtime.append({'intentional_exception':fail,'CORE_Teacher_RNG_threads_flags_restored':True})
        import tempfile
        with tempfile.TemporaryDirectory(dir=HERE,prefix='cpu_source_reject_') as temp:
            root=Path(temp);value=cfg(adapter,17);old_source=read(PARENT/'source_manifest.json')['training_sources']
            for name,saved_cfg,sources in (('old_source',value,old_source),('wrong_seed',cfg(adapter,42),adapter.source_hashes())):
                folder=root/name;folder.mkdir();adapter.torch.save({'config':saved_cfg,'source_sha256':sources,'epoch':12,'history':[{}]*12},folder/'last.pt')
                try:adapter.run(value,name,'weighted_single',stop_after=240,out_dir=root/'out',run_dir=root)
                except AssertionError:pass
                else:raise AssertionError('Mismatched source/seed checkpoint accepted')
    parent_manifest=read(PARENT/'source_manifest.json');protected=dict(parent_manifest['protected_inputs'])
    parent_finish=read(PARENT/'finish_manifest.json');protected.update(parent_finish.get('extra_protected_inputs',{}))
    for path in PARENT.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.lock','.tmp') and path.name not in ('worker.json','finisher_worker.json','progress.json'):
            protected[str(path.resolve())]=sha(path)
    protected[str(HERE/'implementation_decision.json')]=DECISION_SHA
    for document in HERE.glob('ROOT*LOCK*.json'):
        protected[str(document.resolve())]=sha(document);item=read(document)
        for group in (item,item.get('locks',{}),item.get('protected_inputs',{})):
            if isinstance(group,dict):
                for path,digest in group.items():
                    if isinstance(digest,str) and len(digest)==64 and Path(path).is_absolute():assert path not in protected or protected[path]==digest;protected[path]=digest
    assert all(sha(path)==digest for path,digest in protected.items())
    write('source_manifest.json',{'plan_sha256':sha(HERE/'plan.json'),'implementation_decision_sha256':DECISION_SHA,
      'private_sources':{name:sha(HERE/name) for name in ('adapter.py','run.py','postprocess.py')},'analysis_sources':{'postprocess.py':sha(HERE/'postprocess.py')},
      'training_sources':adapter.source_hashes(),'old_reference_sources':adapter.source_hashes(True),'protected_inputs':protected,
      'source_role':'Complete Source16-v2 execution dictionary, distinct from Source15-V1; postprocess frozen because it executes between two fresh training cases.'})
    proof={'state':'passed','unchanged_bodies':body_proof,'routes_per_seed':routes,'roles_per_seed':roles,'invalid_seed_rejected':True,
      'oldSource15_and_wrong_seed_checkpoint_rejected':True,'postprocess':post_proof,'postprocess_CORE_Teacher_RNG_flags_unchanged':True,
      'actual_postprocess_runtime_checks':post_runtime,
      'loss_math_proof_reused':read(PARENT/'prepare_CPU.json')['loss'],'protected_after':verify(adapter),
      'model_forwards':0,'CUDA_initialized':False,'new_training':0,'extra_tiny_GPU_epochs':0}
    write('prepare_CPU.json',proof)
    return {'state':'CPU_prepared_waits_root_GPU_READY','plan_sha256':sha(HERE/'plan.json'),'source_manifest_sha256':sha(HERE/'source_manifest.json'),
      'prepare_CPU_sha256':sha(HERE/'prepare_CPU.json'),'execution_source_count':len(adapter.source_hashes()),'GPU':0}

def bridge(adapter):
    ready('GPU_READY.json',adapter);assert not (HERE/'bridge_runs').exists() and not (HERE/'bridge_GPU.json').exists()
    name='weighted_bridge_seed42';folder=HERE/'bridge_runs'/name;trace=HERE/'bridge_traces'/(name+'.jsonl');began=time.monotonic()
    with awake() as power:
        write('worker.json',{'pid':os.getpid(),'state':'seed42_full12_routing_bridge','current_run':name,'stop_after':12,'formal240_completed_added':0})
        adapter.run(cfg(adapter,42),name,'weighted_single',stop_after=12,out_dir=HERE/'bridge_out',run_dir=HERE/'bridge_runs',trace_file=trace)
        saved=checkpoint(adapter,folder);old_folder=OLD/'pilot_runs/pilot_weighted_single_seed42';old=checkpoint(adapter,old_folder)
        assert saved['epoch']==len(saved['history'])==12 and saved['config']==old['config']==cfg(adapter,42)
        assert saved['source_sha256']==adapter.source_hashes() and old['source_sha256']==read(OLD/'source_manifest.json')['training_sources']
        compare(adapter,folder,old_folder)
        def rows(path):
            with Path(path).open(encoding='utf-8') as stream:
                for line in stream:
                    if line.strip():yield json.loads(line)
        count=0;old_trace=OLD/'pilot_traces/pilot_weighted_single_seed42.jsonl'
        for a,b in zip_longest(rows(trace),rows(old_trace)):
            assert a is not None and b is not None and a==b;count+=1
        assert count==2112 and all(row['train_samples']==45000 and row['validation_samples']==5000 for row in saved['history'])
        trace_proof=audit_trace(trace,12,176);verify(adapter)
    result={'state':'passed','seed':42,'epochs':12,'SGD_steps':2112,'train_exposures':540000,'validation_exposures':60000,
      'original_schedule_horizon':240,'full_model_optimizer_scheduler_RNG_best_last_non_timehistory_exact':True,'config_exact':True,
      'all2112_input_q_loss_records_exact':True,'own_training_sources':adapter.source_hashes(),'old_reference_training_sources':old['source_sha256'],
      'parentV1_first12_identity_sha256':sha(PARENT/'first12_identity.json'),'last_sha256':sha(folder/'last.pt'),'best_sha256':sha(folder/'best.pt'),
      'trace_sha256':sha(trace),'reference_trace_sha256':sha(old_trace),'trace_proof':trace_proof,'power_thread_request':power,
      'elapsed_monotonic_seconds_including_pauses':time.monotonic()-began,'formal240_completed_added':0,'automatic_continue240':False,'overall_goal_achieved':False}
    write('bridge_GPU.json',result);write('worker.json',{'pid':os.getpid(),'state':'bridge_passed_waits_root_CONFIRM_READY','formal240_completed_added':0})
    return {'state':result['state'],'bridge_GPU_sha256':sha(HERE/'bridge_GPU.json'),'formal240_completed_added':0}

def train_seed(adapter,seed):
    assert seed in SEEDS;name='weighted240_seed'+str(seed);folder=HERE/'formal_runs'/name;trace=HERE/'formal_traces'/(name+'.jsonl')
    result_file=HERE/'results'/('seed'+str(seed)+'.json')
    if result_file.exists():
        result=read(result_file);assert result['state']=='program_complete_manual_pending' and result['training_sources']==adapter.source_hashes() and result['seed']==seed
        assert sha(folder/'last.pt')==result['last_sha256'] and sha(folder/'best.pt')==result['best_sha256'] and sha(trace)==result['trace_sha256'];return result
    write('worker.json',{'pid':os.getpid(),'state':'formal240_training','seed':seed,'current_run':name,'stop_after':240,'fixed_order':list(SEEDS)})
    adapter.run(cfg(adapter,seed),name,'weighted_single',stop_after=240,out_dir=HERE/'formal_out',run_dir=HERE/'formal_runs',trace_file=trace)
    saved=checkpoint(adapter,folder);assert saved['epoch']==len(saved['history'])==240 and saved['source_sha256']==adapter.source_hashes() and saved['config']==cfg(adapter,seed)
    for i,row in enumerate(saved['history']):
        assert row['epoch']==i+1 and row['train_samples']==45000 and row['validation_samples']==5000 and all(math.isfinite(value) for value in row.values())
        factor=(i+1)/5 if i<5 else .5*(1+math.cos(math.pi*(i-5)/235));assert abs(row['learning_rate']-.1*factor)<1e-12
    trace_proof=audit_trace(trace,240,176);base=baseline(seed);last=saved['history'][-1];correct=round(last['validation_accuracy']*5000)
    median=statistics.median(row['seconds'] for row in saved['history'])
    result={'state':'program_complete_manual_pending','seed':seed,'run_id':name,'training_sources':adapter.source_hashes(),'last240':last,'last_correct':correct,
      'baseline_last':base,'quality_comparison':{'correct_delta':correct-base['correct'],'accuracy_delta':correct/5000-base['accuracy'],'CE_delta':last['validation_ce']-base['hard_validation_CE']},
      'raw_epoch_seconds':[row['seconds'] for row in saved['history']],'median_epoch_seconds':median,'median_wall_ratio':median/base['median_raw_epoch_seconds'],
      'resource_policy':'descriptive only; no new per-seed/posthoc cost gate','trace_proof':trace_proof,'last_sha256':sha(folder/'last.pt'),'best_sha256':sha(folder/'best.pt'),
      'history_sha256':sha(folder/'history.csv'),'trace_sha256':sha(trace),'formal240_completed_added':1,'official_test':0,'automatic_test_reproduction':False,'overall_goal_achieved':False}
    write('results/seed'+str(seed)+'.json',result);verify(adapter);return result

def derive_evaluation_locks(adapter,seed,result):
    directory=HERE/'analyses'/('seed'+str(seed));path=directory/'evaluation_locks.json'
    if path.exists():
        old=read(path);assert old['seed_results_sha256']==sha(HERE/'results'/('seed'+str(seed)+'.json')) and old['candidate_sources']==adapter.source_hashes();return
    candidate=HERE/'formal_runs'/result['run_id'];base_path=PROJECT/'runs/horizon'/baseline(seed)['run_id']/'last.pt'
    candidates=(candidate/'best.pt',candidate/'last.pt',base_path);locks={str(item.resolve()):sha(item) for item in candidates}
    assert locks[str((candidate/'best.pt').resolve())]==result['best_sha256'] and locks[str((candidate/'last.pt').resolve())]==result['last_sha256']
    frozen=read(HERE/'source_manifest.json');assert frozen['protected_inputs'][str(base_path.resolve())]==locks[str(base_path.resolve())]
    write('analyses/seed'+str(seed)+'/evaluation_locks.json',{'state':'auto_derived_under_root_preapproval','seed':seed,'CONFIRM_READY_sha256':sha(HERE/'CONFIRM_READY.json'),
      'source_manifest_sha256':sha(HERE/'source_manifest.json'),'seed_results_sha256':sha(HERE/'results'/('seed'+str(seed)+'.json')),
      'checkpoint_locks':locks,'candidate_sources':adapter.source_hashes(),'baseline_sources':frozen['old_reference_sources'],'manual_root_weight_check_claimed':False,'manual_figures_opened':False})

def confirm(adapter):
    ready('CONFIRM_READY.json',adapter);assert not (HERE/'confirmation_results.json').exists();began=time.monotonic();postprocess=load_postprocess();summaries={}
    with awake() as power:
        for seed in SEEDS:
            verify(adapter);result=train_seed(adapter,seed);derive_evaluation_locks(adapter,seed,result)
            write('worker.json',{'pid':os.getpid(),'state':'seed_scientific_postprocess','seed':seed,'completed_full240':len(summaries)+1})
            summaries[str(seed)]=checked_postprocess(adapter,lambda:postprocess.execute_seed(seed),'seed'+str(seed))
            verify(adapter)
        write('worker.json',{'pid':os.getpid(),'state':'shared5000_three_seed_summary','completed_full240':2})
        joint=checked_postprocess(adapter,postprocess.summarize_three_seeds,'three_seed_summary')
        verify(adapter)
    result={'state':'program_complete_manual_pending','seed_order':list(SEEDS),'per_seed':summaries,'joint':joint,'power_thread_request':power,
      'formal240_completed_added':2,'full12_bridge_counted_as_formal':False,'elapsed_monotonic_seconds_including_pauses':time.monotonic()-began,
      'manual_figures_opened':False,'automatic_test_reproduction':False,'overall_goal_achieved':False}
    write('confirmation_results.json',result);write('worker.json',{'pid':os.getpid(),'state':result['state'],'completed_full240':2,'manual_pending':True,'overall_goal_achieved':False})
    return {'state':result['state'],'confirmation_results_sha256':sha(HERE/'confirmation_results.json'),'automatic_test_reproduction':False}

def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--prepare',action='store_true');group.add_argument('--bridge',action='store_true');group.add_argument('--confirm',action='store_true');args=parser.parse_args()
    try:
        with exclusive():
            if args.prepare:result=prepare()
            else:
                adapter=load_adapter()
                with adapter.runtime():
                    adapter.torch.set_num_threads(1);result=bridge(adapter) if args.bridge else confirm(adapter)
    except BaseException as exc:
        phase='prepare' if args.prepare else 'bridge' if args.bridge else 'confirm'
        write('failure_'+phase+'.json',{'pid':os.getpid(),'state':'technical_failure_no_automatic_retry','reason':repr(exc),'traceback':traceback.format_exc(),'overall_goal_achieved':False});raise
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
