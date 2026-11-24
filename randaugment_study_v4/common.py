"""Paths, immutable protocol, protected evidence and child-process orchestration."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,os,subprocess,sys,time
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_randaugment_v4';RUNS=ROOT/'runs/randaugment_v4';CONFIGS=ROOT/'configs_randaugment_v4';FIG=ROOT/'figures_randaugment_v4'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_name(p.name+'.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temporary.replace(p)


def process_info(pid):
    """Read-only Windows process query; never use Windows os.kill(pid,0)."""
    if os.name!='nt':
        path=Path('/proc')/str(int(pid))
        return {'pid':int(pid),'alive':path.exists(),'creation':None}
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000|0x00100000,False,int(pid))
    if not handle:
        error=ctypes.get_last_error()
        if error in (87,1168):return {'pid':int(pid),'alive':False,'creation':None}
        raise OSError(error,'Cannot query process safely; refuse duplicate launch.')
    try:
        code=wintypes.DWORD();assert kernel.GetExitCodeProcess(handle,ctypes.byref(code))
        created,exited,kernel_time,user_time=(wintypes.FILETIME() for _ in range(4))
        assert kernel.GetProcessTimes(handle,ctypes.byref(created),ctypes.byref(exited),ctypes.byref(kernel_time),ctypes.byref(user_time))
        stamp=(int(created.dwHighDateTime)<<32)|int(created.dwLowDateTime)
        return {'pid':int(pid),'alive':code.value==259,'creation':stamp,'exit_code':int(code.value)}
    finally:kernel.CloseHandle(handle)


def same_live_process(record):
    current=process_info(record['pid'])
    return current['alive'] and (record.get('creation') is None or record['creation']==current.get('creation'))


@contextmanager
def file_lease(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    handle=path.open('a+b')
    try:
        if path.stat().st_size==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            handle.seek(0)
            if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
    finally:handle.close()


def wait_existing_children():
    records=[]
    active=OUT/'active_child.json'
    if active.exists():records.append(read(active))
    for p in RUNS.glob('*/train_process.json'):
        row=read(p)
        if row.get('state')=='running':records.append(row)
    observed=set()
    for row in records:
        key=(row.get('pid'),row.get('creation'))
        if key in observed or not row.get('pid') or row['pid']==os.getpid():continue
        observed.add(key)
        while same_live_process(row):
            write(OUT/'progress.json',{'state':'waiting_for_existing_training','current_run':row.get('run_id'),
                                      'child_pid':row['pid'],'updated_at_utc':now()})
            time.sleep(10)  # Backgrounddriver waits; it does not launch another GPUworker.
        if row.get('source_sha256'):
            from randaugment_study_v4.train import source_hashes
            assert row['source_sha256']==source_hashes(),'Existing child is a different frozen version.'
def config(seed,enabled=True):
    cfg=read(ROOT/f'configs_horizon/{"horizon240_seed42" if seed==42 else "repeat_horizon240_seed"+str(seed)}.json')
    if enabled:cfg.update(randaugment=True,ra_num_ops=2,ra_magnitude=9,ra_bins=31,ra_interpolation='bilinear',ra_fill=[125,123,114],ra_seed_tag=58193)
    return cfg
def baseline_rows():
    rows=read(ROOT/'results_horizon/validation_confirmation.json')['students']
    return [next(r for r in rows if r['config']['seed']==s) for s in [42,17,73]]
def history(row):
    with (ROOT/row['artifact']/'history.csv').open(encoding='utf-8-sig') as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
def preserved():
    from randaugment_study_v4.train import source_hashes
    from randaugment_study_v4.augment import library_manifest
    seal=read(OUT/'source_manifest.json')
    assert seal['source_sha256']==source_hashes() and seal['augmentation_library']==library_manifest()
    assert seal['plan_sha256']==sha(OUT/'plan.json')
    for name,value in read(OUT/'previous_results_snapshot.json').items():assert sha(ROOT/name)==value,name
    for name,value in read(ROOT/'results_reopened_execution/execution_scope.json')['report_sha256'].items():assert sha(ROOT/name)==value,name
def launch(name,cfg,stop_after=None):
    if (ROOT/'PAUSE_ROUND2').exists() or (ROOT/'PAUSE_RANDAUGMENT').exists():raise InterruptedError('User pause; checkpoint retained.')
    path=CONFIGS/(name+'.json')
    if path.exists():assert read(path)==cfg
    else:write(path,cfg)
    from randaugment_study_v4.train import source_hashes
    if stop_after and (RUNS/name/'last.pt').exists():
        import torch
        saved=torch.load(RUNS/name/'last.pt',map_location='cpu',weights_only=False)
        assert saved['config']==cfg and saved['source_sha256']==source_hashes()
        if saved['epoch']>=stop_after:return
    state='reproduction_check' if name.startswith('audit_full_ra_N2_M9') else ('preflight' if name.startswith('audit_') else 'running')
    write(OUT/'progress.json',{'state':state,'current_run':name,'updated_at_utc':now()})
    args=[sys.executable,'-u','-X','utf8','-m','randaugment_study_v4.train','--config',str(path),'--run-id',name]
    if stop_after:args+=['--stop-after',str(stop_after)]
    wait_existing_children()
    completed=RUNS/name/'result.json'
    if completed.exists():
        result=read(completed)
        assert result['config']==cfg and result['state']=='completed'
        assert read(RUNS/name/'environment.json')['source_sha256']==source_hashes()
        return
    with (ROOT/'randaugment_v4_training.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        identity={**process_info(process.pid),'state':'running','run_id':name,'argv':args,
                  'config_sha256':sha(path),'source_sha256':source_hashes(),'started_at_utc':now()}
        write(OUT/'active_child.json',identity)
        code=process.wait()
        write(OUT/'active_child.json',{**identity,'state':'exited','exit_code':code,'ended_at_utc':now()})
    assert code==0,'Inspect randaugment_v4_training.log; never truncate the frozen budget.'
    if (ROOT/'PAUSE_ROUND2').exists() or (ROOT/'PAUSE_RANDAUGMENT').exists():raise InterruptedError('Paused at complete epoch.')
def one(seed):
    preserved();name=f'ra_N2_M9_seed{seed}';cfg=config(seed)
    if not (RUNS/name/'result.json').exists():launch(name,cfg)
    row=read(RUNS/name/'result.json')
    assert row['state']=='completed' and row['epochs_completed']==240 and row['parameters']==2436346 and row['config']==cfg
    row={**row,'artifact':str((RUNS/name).relative_to(ROOT))};write(OUT/'catalog'/(name+'.json'),row)
    return row
