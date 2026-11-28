"""Source16-v2 autonomous clean postprocessing; no work is launched at import.
BF16 single-view network outputs are stored FP32. Native AMP history CE is the
prospective gate; cached FP64 NLL is a separate fixed-model description.
"""
from contextlib import contextmanager
from pathlib import Path
import ast
import csv
import ctypes
import functools
import hashlib
import itertools
import json
import math
import os
import pickle
import random
import statistics
import sys
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]/'实验三-图像分类'
PARENT=HERE.parent/'weighted_mix_kd_confirmation'
PROFILES=('candidate_best','candidate_last','baseline_last')
CLASSES=['飞机','汽车','鸟','猫','鹿','狗','蛙','马','船','卡车']
INPUT_KEYS=('epoch','batch_in_epoch','global_batch','batch_size','actual_lambda','order_sha256','labels_sha256','view_sha256','teacher_input_sha256','teacher_raw_q_sha256','teacher_t1_q_sha256')
LOSS_KEYS=('original_KL','new_extra','modified_KD','hard_CE','returned_total')


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1048576),b''):value.update(block)
    return value.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path).resolve();assert path.is_relative_to(HERE)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def _write_output(output,name,value):write(output/name,value)


def _npz(output,name,arrays):
    path=(output/name).resolve();assert path.is_relative_to(HERE) and not path.exists()
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
    os.replace(temporary,path)


def _seed(seed):
    assert isinstance(seed,int) and not isinstance(seed,bool) and seed in (17,73)
    return seed


def seed_dir(seed):return HERE/'analyses'/('seed'+str(_seed(seed)))


def paths(seed):
    seed=_seed(seed);candidate=HERE/'formal_runs'/('weighted240_seed'+str(seed))
    baseline=next(row for row in read(HERE/'plan.json')['baseline_last'] if row['seed']==seed)
    return {'candidate_best':(candidate/'best.pt').resolve(),'candidate_last':(candidate/'last.pt').resolve(),
            'baseline_last':(PROJECT/'runs/horizon'/baseline['run_id']/'last.pt').resolve()}


def _sources(protected=False):
    manifest=read(HERE/'source_manifest.json');plan=read(HERE/'plan.json')
    assert sha(HERE/'plan.json')==manifest['plan_sha256']
    sources=manifest['training_sources'];assert len(sources)==16
    expected=plan['original_training_source_dictionary'];assert len(expected)==12
    assert all(sources.get(name)==digest for name,digest in expected.items())
    assert sources.get(os.path.relpath(__file__,PROJECT).replace('\\','/'))==sha(__file__)
    for name,digest in sources.items():assert sha(PROJECT/name)==digest,'Changed training source: '+name
    for name,digest in manifest.get('analysis_sources',{}).items():
        path=Path(name) if Path(name).is_absolute() else (HERE/name)
        if not path.exists():path=PROJECT/name
        assert sha(path)==digest
    for name,digest in manifest.get('private_sources',{}).items():assert sha(HERE/name)==digest
    if protected:
        for name,digest in manifest['protected_inputs'].items():assert sha(name)==digest,'Changed protected input: '+name
    return manifest,expected


def _approval():
    ready=read(HERE/'CONFIRM_READY.json');assert ready['approved'] is True
    for key,name in (('plan_sha256','plan.json'),('source_manifest_sha256','source_manifest.json'),
                     ('prepare_CPU_sha256','prepare_CPU.json'),('bridge_GPU_sha256','bridge_GPU.json')):
        assert ready[key]==sha(HERE/name),'Stale CONFIRM_READY: '+key
    assert ready['seed_order']==[17,73] and ready['automatic_test_reproduction'] is False
    assert read(HERE/'prepare_CPU.json')['state']=='passed' and read(HERE/'bridge_GPU.json')['state']=='passed'
    return ready


def _ready(seed):
    seed=_seed(seed);_approval();manifest,baseline=_sources(protected=True)
    result_path=HERE/'results'/('seed'+str(seed)+'.json');formal=read(result_path)
    derived=read(seed_dir(seed)/'evaluation_locks.json');assert derived['state']=='auto_derived_under_root_preapproval'
    for key,path in (('CONFIRM_READY_sha256',HERE/'CONFIRM_READY.json'),('source_manifest_sha256',HERE/'source_manifest.json'),('seed_results_sha256',result_path)):
        assert derived[key]==sha(path)
    assert derived['candidate_sources']==manifest['training_sources'] and derived['baseline_sources']==baseline
    assert all(Path(name).is_absolute() for name in derived['checkpoint_locks'])
    locks={Path(name).resolve():digest for name,digest in derived['checkpoint_locks'].items()}
    assert len(locks)==3 and set(locks)==set(paths(seed).values())
    assert all(sha(path)==digest for path,digest in locks.items())
    assert formal['state']=='program_complete_manual_pending' and formal['run_id']=='weighted240_seed'+str(seed) and formal['seed']==seed
    assert formal['last240']['epoch']==240 and formal['training_sources']==manifest['training_sources'] and formal['formal240_completed_added']==1
    assert formal['best_sha256']==locks[paths(seed)['candidate_best']] and formal['last_sha256']==locks[paths(seed)['candidate_last']]
    return locks,manifest['training_sources'],baseline


def unchanged(locks):
    for path,digest in locks.items():assert sha(path)==digest,'Changed input: '+str(path)


def records(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def cpu_import_preflight():
    """Only own source/schema and synthetic NumPy; no model/raw/weight/CUDA read."""
    tree=ast.parse(Path(__file__).read_text(encoding='utf-8'))
    assert not any(isinstance(n,ast.ImportFrom) and n.module and ('adapter' in n.module or 'weighted_mix' in n.module or 'source_set' in n.module) for n in ast.walk(tree))
    assert (176+20)*2+20==412 and 50000*2+5000==105000
    z=np.array([[10000.,9999.,-10000.],[-10000.,0.,10000.]]);p,nll=probability_nll(z,np.array([0,2]))
    assert np.isfinite(nll).all() and np.allclose(p.sum(1),1) and (nll>=0).all()
    assert _seed(17)==17 and _seed(73)==73
    try:_seed(42)
    except AssertionError:pass
    else:raise AssertionError('Unexpected formal seed role')
    # No global RNG is consumed by private-PCG paired sampling.
    before=np.random.get_state();local=np.random.Generator(np.random.PCG64(2026112701));local.integers(0,2,4)
    after=np.random.get_state();assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    return {'state':'passed','postprocess_source_sha256':sha(__file__),'planned_profiles':list(PROFILES),'image_forwards_per_seed':105000,'forward_calls_per_seed':412,
      'source_role':'self-contained Source16-v2 analysis; no adapter/driver import','scope':'static AST + tiny NumPy only; no model/weights/raw data/CUDA','files_written':0}


def _data():
    import numpy as np
    manifest = read(PROJECT / "data/dataset_manifest.json")
    split_path = PROJECT / manifest["split_file"]
    assert sha(split_path) == manifest["split_sha256"]
    split = read(split_path)
    ti, vi = np.asarray(split["train_indices"], dtype=np.int64), np.asarray(split["validation_indices"], dtype=np.int64)
    assert len(ti) == 45000 and len(vi) == 5000 and not set(ti) & set(vi)
    assert set(ti) | set(vi) == set(range(50000))
    images, labels, raw_hashes = [], [], {}
    assert len(manifest["official_train_batches"]) == 5
    for number, item in enumerate(manifest["official_train_batches"], 1):
        path = PROJECT / item["file"]
        assert path.name == f"data_batch_{number}" and sha(path) == item["sha256"]
        with path.open("rb") as stream:
            batch = pickle.load(stream, encoding="bytes")
        assert batch[b"data"].dtype == np.uint8 and batch[b"data"].shape == (10000, 3072)
        images.append(batch[b"data"].reshape(-1, 3, 32, 32)); labels.extend(batch[b"labels"])
        raw_hashes[item["file"]] = item["sha256"]
    images, labels = np.concatenate(images), np.asarray(labels, dtype=np.int64)
    assert np.array_equal(np.bincount(labels[ti], minlength=10), np.full(10, 4500))
    assert np.array_equal(np.bincount(labels[vi], minlength=10), np.full(10, 500))
    return {"train": (images[ti].copy(), labels[ti], ti), "val": (images[vi].copy(), labels[vi], vi)}, {
        "raw_training_files_sha256": raw_hashes, "split_file": manifest["split_file"], "split_sha256": manifest["split_sha256"],
        "dataset_manifest_sha256": sha(PROJECT / "data/dataset_manifest.json"),
        "normalization_sha256": sha(PROJECT / "data/normalization.json")}

@contextmanager
def _runtime(torch, np):
    initialized = torch.cuda.is_initialized()
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
             torch.cuda.get_rng_state_all() if initialized else None, torch.get_num_threads(),
             torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.utils.deterministic.fill_uninitialized_memory)
    record = {"CUDA_initialized_before": initialized}
    try:
        yield record
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])
        torch.set_num_threads(state[4]); torch.use_deterministic_algorithms(state[5], warn_only=state[6])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = state[7:9]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = state[9:11]
        torch.utils.deterministic.fill_uninitialized_memory = state[11]
        record.update({"CPU_RNG_threads_flags_restored": True, "prior_CUDA_RNG_restored": initialized,
                       "first_CUDA_initialization_cannot_be_reversed": not initialized and torch.cuda.is_initialized()})


def _preserve_analysis_runtime(function):
    @functools.wraps(function)
    def guarded(*args,**kwargs):
        import torch
        with _runtime(torch,np):
            return function(*args,**kwargs)
    return guarded

@contextmanager
def _awake():
    assert os.name == "nt"
    function = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
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

def _predict(torch, np, model, transform, raw, labels):
    output, sizes = [], []
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0)
    totals = torch.zeros(3, dtype=torch.float64, device="cuda")
    with torch.no_grad():
        for start in range(0, len(raw), 256):
            x = torch.from_numpy(raw[start:start + 256]).to("cuda", non_blocking=True)
            y = torch.from_numpy(labels[start:start + 256]).to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(transform(x, training=False)); ce = criterion(z, y)
            n = len(y)
            totals += torch.stack((ce.double() * n, (z.argmax(1) == y).sum().double(),
                                   torch.tensor(n, dtype=torch.float64, device="cuda")))
            output.append(z.float().cpu().numpy()); sizes.append(n)
    ce, correct, count = totals.cpu().tolist(); logits = np.concatenate(output)
    assert logits.dtype == np.float32 and logits.shape == (len(labels), 10) and np.isfinite(logits).all()
    assert count == len(labels) and int(correct) == int((logits.argmax(1) == labels).sum())
    z = logits.astype(np.float64); maximum = z.max(1)
    nll = maximum + np.log(np.exp(z - maximum[:, None]).sum(1)) - z[np.arange(len(z)), labels]
    return logits, {"accuracy": correct / count, "correct": int(correct), "samples": int(count),
                    "CE_runtime": ce / count, "CE_CPU_FP64_from_BF16_logits_descriptive": float(nll.mean()),
                    "batch_count": len(sizes), "batch_sizes": sizes}

def evaluate_seed(seed):
    seed=_seed(seed);OUTPUT=seed_dir(seed)/"evaluation"
    if (OUTPUT/"results.json").exists():
        _ready(seed);result=read(OUTPUT/"results.json");_verify_evaluation(seed,result)
        return result
    assert not OUTPUT.exists(), "Partial evaluation exists; explicit reviewed recovery required"
    locks, candidate_sources, baseline_sources = _ready(seed)
    assert not (OUTPUT / "inputs.npz").exists() and not (OUTPUT / "results.json").exists()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    search_path = list(sys.path)
    try:
        sys.path.insert(0, str(PROJECT))
        from extensions.models import build, Transform
        from train import seed_everything
    finally:
        sys.path[:] = search_path
    assert Path(sys.modules[build.__module__].__file__).resolve() == PROJECT / "extensions/models.py"
    cases, runtime, power, failure = {}, {}, {}, None
    model = transform = None
    try:
        with _runtime(torch, np) as runtime, _awake() as power:
            torch.set_num_threads(8); seed_everything(seed)
            assert torch.cuda.is_available() and torch.get_default_dtype() == torch.float32
            data, inputs = _data()
            _npz(OUTPUT, "inputs.npz", {"val_images": data["val"][0], "train_ids": data["train"][2], "train_labels": data["train"][1],
                                "val_ids": data["val"][2], "val_labels": data["val"][1]})
            inputs["shared_inputs_sha256"] = sha(OUTPUT / "inputs.npz"); _write_output(OUTPUT, "inputs.json", inputs)
            for profile in PROFILES:
                path = paths(seed)[profile]
                assert sha(path) == locks[path]
                saved = torch.load(path, map_location="cpu", weights_only=False)
                assert sha(path) == locks[path]
                expected_sources = baseline_sources if profile == "baseline_last" else candidate_sources
                assert saved["source_sha256"] == expected_sources
                cfg = saved["config"]
                assert (cfg["architecture"], cfg["model"], cfg["width"], cfg["epochs"], cfg["seed"]) == ("resnet20", "cifar", 48, 240, seed)
                assert cfg["amp"] is True and cfg["batch_size"] == 256 and not cfg.get("pilot_samples")
                if profile.startswith("candidate"):
                    assert cfg["source_set_mode"] == "weighted_single"
                else:
                    assert "source_set_mode" not in cfg
                if profile.endswith("last"):
                    assert saved["epoch"] == 240 and len(saved["history"]) == 240
                assert saved["stats"] == read(PROJECT / "data/normalization.json")
                model = build(cfg).to("cuda", memory_format=torch.channels_last)
                model.load_state_dict(saved["model"]); model.eval().requires_grad_(False)
                transform = Transform(saved["stats"], cfg).to("cuda")
                assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
                before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                arrays, metrics = {}, {}
                _write_output(OUTPUT, "progress.json", {"pid": os.getpid(), "state": "BF16_clean_evaluation", "profile": profile,
                                         "completed_profiles": len(cases), "planned_profiles": 3})
                for split in (("val",) if profile == "baseline_last" else ("train", "val")):
                    raw, y, ids = data[split]
                    z, metrics[split] = _predict(torch, np, model, transform, raw, y)
                    arrays[split + "_logits"] = z; arrays[split + "_ids"] = ids; arrays[split + "_labels"] = y
                    assert metrics[split]["batch_count"] == (176 if split == "train" else 20)
                    assert metrics[split]["batch_sizes"][-1] == (200 if split == "train" else 136)
                assert not model.training and all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
                assert all(torch.equal(value.detach().cpu(), before[name]) for name, value in model.state_dict().items())
                assert sha(path) == locks[path]
                historical = {"CE": saved["best_ce"], "accuracy": saved["best_accuracy"]} if profile.endswith("best") else {
                    "CE": saved["history"][-1]["validation_ce"], "accuracy": saved["history"][-1]["validation_accuracy"]}
                history_comparison = {"historical_CE": historical["CE"], "historical_accuracy": historical["accuracy"],
                                      "CE_runtime_abs_difference": abs(metrics["val"]["CE_runtime"] - historical["CE"]),
                                      "CE_within_1e_minus6": abs(metrics["val"]["CE_runtime"] - historical["CE"]) <= 1e-6,
                                      "accuracy_equal": metrics["val"]["accuracy"] == historical["accuracy"],
                                      "role": "description/technical audit only; never replace the frozen LAST-history gate"}
                _npz(OUTPUT, profile + ".npz", arrays)
                row = {"profile": profile, "epoch": int(saved["epoch"]), "metrics": metrics,
                       "cache_file": profile + ".npz", "cache_sha256": sha(OUTPUT / (profile + ".npz")),
                       "file_path": str(OUTPUT / (profile + ".npz")), "file_sha256": sha(OUTPUT / (profile + ".npz")),
                       "checkpoint": str(path), "checkpoint_sha256": locks[path], "training_sources": expected_sources,
                       "history_comparison": history_comparison, "W_BN_gradients_unchanged": True, "no_mask_or_extra_head_hooks": True}
                cases[profile] = row; _write_output(OUTPUT, profile + ".json", row)
                model = transform = None
                del saved, before, arrays
        _ready(seed)
        total_images = sum(row["samples"] for case in cases.values() for row in case["metrics"].values())
        total_calls = sum(row["batch_count"] for case in cases.values() for row in case["metrics"].values())
        assert total_images == 105000 and total_calls == 412 and len(cases) == 3
        result = {"state": "evaluation_complete_manual_pending", "cases": cases, "actual_image_forwards": total_images,
                  "actual_forward_calls": total_calls, "inputs_file": str(OUTPUT / "inputs.npz"),
                  "inputs_sha256": inputs["shared_inputs_sha256"], "input_identity": inputs,
                  "runtime": runtime, "power_thread_request": power, "evaluation_source_sha256": sha(Path(__file__)),
                  "stored_logits": "float32 storage from BF16 network calls, not FP32 inference",
                  "CE_runtime_policy": "BF16 autocast CE batch mean weighted by actual count, accumulated FP64",
                  "CPU_CE_policy": "descriptive FP64 logsumexp of saved BF16-derived logits, separate from runtime CE",
                  "primary_profiles": ["candidate_last", "baseline_last"], "candidate_best_secondary_only": True,
                  "official_test_images": 0, "new_training_or_optimizer_steps": 0, "TTA_or_calibration": False,
                  "manual_visual_review_pending": True, "quality_gate_modified": False, "overall_goal_achieved": False}
        _write_output(OUTPUT, "results.json", result)
        return result
    except BaseException as exc:
        failure = {"state": "evaluation_failed", "reason": repr(exc), "traceback": traceback.format_exc(),
                   "completed_profiles": list(cases), "runtime": runtime, "power_thread_request": power,
                   "evaluation_source_sha256": sha(Path(__file__)), "overall_goal_achieved": False}
        raise
    finally:
        if failure is not None:
            try:
                _ready(seed); failure["post_exception_sources_and_three_weights_unchanged"] = True
            except BaseException as error:
                failure["post_exception_protection_attention"] = repr(error)
            if model is not None and "before" in locals():
                try:
                    failure["model_state_unchanged"] = all(torch.equal(value.detach().cpu(), before[name]) for name, value in model.state_dict().items())
                    failure["eval_and_no_grad"] = not model.training and all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
                except BaseException as error:
                    failure["model_state_check_attention"] = repr(error)
            _write_output(OUTPUT, "exception.json", failure)
        model = transform = None

def probability_nll(z,y):
    z=np.asarray(z,dtype=np.float64);maximum=z.max(1,keepdims=True);e=np.exp(z-maximum);p=e/e.sum(1,keepdims=True)
    nll=maximum[:,0]+np.log(e.sum(1))-z[np.arange(len(z)),y];return p,nll

def plotter():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    return plt

@_preserve_analysis_runtime
def analyze_seed(seed,training=None):
    seed=_seed(seed);BASE=seed_dir(seed)
    if (BASE/"results/analysis_results.json").exists():
        value=read(BASE/"results/analysis_results.json");_verify_analysis(seed,value);return value
    training=audit_training(seed) if training is None else training
    assert training['state']=='passed';unchanged(training['locks'])
    locks=dict(training['locks']);result=read(HERE/'results'/('seed'+str(seed)+'.json'));evaluation=read(BASE/'evaluation/results.json')
    _verify_evaluation(seed,evaluation);assert evaluation['state']=='evaluation_complete_manual_pending' and evaluation['actual_image_forwards']==105000
    locks[str(BASE/'evaluation/results.json')]=sha(BASE/'evaluation/results.json')
    shared=Path(evaluation['inputs_file']);assert shared.resolve()==(BASE/'evaluation/inputs.npz').resolve() and sha(shared)==evaluation['inputs_sha256'];locks[str(shared)]=evaluation['inputs_sha256']
    with np.load(shared,allow_pickle=False) as saved:inputs={key:saved[key] for key in saved.files}
    y,ids=inputs['val_labels'],inputs['val_ids'];assert np.array_equal(np.bincount(y,minlength=10),np.full(10,500)) and len(np.unique(ids))==5000
    assert inputs['val_images'].shape==(5000,3,32,32)
    cases={};metrics={};assert set(evaluation['cases'])=={'candidate_best','candidate_last','baseline_last'}
    for key,row in evaluation['cases'].items():
        path=Path(row['file_path']);assert path.resolve()==(BASE/'evaluation'/(key+'.npz')).resolve() and sha(path)==row['file_sha256'];locks[str(path)]=row['file_sha256']
        checkpoint=Path(row['checkpoint']);assert sha(checkpoint)==row['checkpoint_sha256'];locks[str(checkpoint)]=row['checkpoint_sha256']
        with np.load(path,allow_pickle=False) as saved:arrays={name:saved[name] for name in saved.files}
        metrics[key]={'epoch':row['epoch'],'metrics':row['metrics'],'history_comparison':row['history_comparison']}
        for split in (('val',) if key=='baseline_last' else ('train','val')):
            labels=inputs[split+'_labels'];z=arrays[split+'_logits'];assert np.array_equal(arrays[split+'_ids'],inputs[split+'_ids']) and np.array_equal(arrays[split+'_labels'],labels)
            assert z.shape==(len(labels),10) and z.dtype==np.float32 and np.isfinite(z).all()
            p,nll=probability_nll(z,labels);pred=z.argmax(1);correct=pred==labels;m=row['metrics'][split]
            assert int(correct.sum())==m['correct'] and float(correct.mean())==m['accuracy'] and abs(float(nll.mean())-m['CE_CPU_FP64_from_BF16_logits_descriptive'])<1e-10
            if split=='val':
                cm=np.zeros((10,10),dtype=np.int64);np.add.at(cm,(labels,pred),1);cases[key]={'probability':p,'nll':nll,'pred':pred,'correct':correct,'CM':cm}
    candidate,baseline=cases['candidate_last'],cases['baseline_last'];delta_correct=candidate['correct'].astype(np.float64)-baseline['correct'];delta_nll=candidate['nll']-baseline['nll']
    differences=np.column_stack((delta_correct,delta_nll));rng=np.random.default_rng(2026112644);draws=np.zeros((2000,2))
    for k in range(10):
        indexes=np.flatnonzero(y==k);selection=rng.integers(0,500,(2000,500));draws+=differences[indexes][selection].mean(1)/10
    intervals=np.quantile(draws,[.025,.975],axis=0,method='linear')
    corrected=np.flatnonzero(~baseline['correct']&candidate['correct']);regressed=np.flatnonzero(baseline['correct']&~candidate['correct'])
    change={'accuracy_delta':float(delta_correct.mean()),'NLL_FP64_delta':float(delta_nll.mean()),'accuracy_conditional95':intervals[:,0].tolist(),'NLL_FP64_conditional95':intervals[:,1].tolist(),
      'corrected_count':len(corrected),'regressed_count':len(regressed),'perclass_accuracy_delta':[float(delta_correct[y==k].mean()) for k in range(10)]}
    output=BASE/'figures';assert not output.exists();output.mkdir();plt=plotter();figures={}
    def save(fig,name):
        path=output/name;fig.savefig(path,facecolor='white');plt.close(fig);figures[name]=sha(path)
    history=training['history'];parts=np.asarray(training['epoch_weighted_loss_parts']);epochs=np.arange(1,241)
    fig,axes=plt.subplots(2,2,figsize=(12.5,7.5),constrained_layout=True)
    for ax,key,scale,title in ((axes[0,0],'validation_accuracy',100,'原图验证准确率（%）'),(axes[0,1],'validation_ce',1,'原图验证硬标签CE'),(axes[1,0],'train_accuracy',100,'CutMix面积加权训练命中率（%）')):
        ax.plot(epochs,[row[key]*scale for row in history],color='#4C78A8');ax.set(title=title,xlabel='完整训练轮次');ax.grid(alpha=.2)
    for column,label in ((0,'总目标'),(1,'混合硬CE'),(3,'原full KL'),(4,'新增weighted extra')):axes[1,1].plot(epochs,parts[:,column],label=label)
    axes[1,1].set(title='训练目标与分量（不同于原图验证CE）',xlabel='完整训练轮次');axes[1,1].legend(fontsize=8,frameon=False);axes[1,1].grid(alpha=.2)
    fig.suptitle(f'固定weighted双单类KD确认：种子{seed}、240轮；主结论仅LAST240\n旧fullKL和新增extra均保留，train_kd为二者之和；未测试、未校准、未自动扩种子',fontsize=11);save(fig,'01_full240_curves.png')
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
    keys=['candidate_best','candidate_last','baseline_last'];ticks=[f"候选BEST e{metrics[keys[0]]['epoch']}",'候选LAST240','原LAST240']
    for split,color,offset in (('train','#4C78A8',-.16),('val','#D26C32',.16)):
        for j,key in enumerate(keys):
            if split not in metrics[key]['metrics']:continue
            m=metrics[key]['metrics'][split];axes[0].bar(j+offset,m['accuracy']*100,width=.30,color=color,label=split if j==0 else None);axes[1].bar(j+offset,m['CE_runtime'],width=.30,color=color)
    for ax,title in ((axes[0],'同尺度原图拟合准确率（%）'),(axes[1],'同尺度原AMP硬标签CE')):ax.set(title=title,xticks=range(3),xticklabels=ticks);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylim(0,100);axes[0].legend(frameon=False,fontsize=8)
    axes[2].axis('off');axes[2].text(0,1,'LAST240同种子差（描述）\n'+json.dumps(result['quality_comparison'],ensure_ascii=False,indent=1)+'\n不设单种子或新资源门槛'+'\n\nBEST仅描述，不用于救主门槛。\n原模型仅val5k，不新增train推理。',va='top',fontsize=9)
    fig.suptitle('BF16网络/B256、原图不增强无TTA；候选best/last各50k+原LASTval5k，共105k\nFP32保存的是BF16输出，CPU FP64描述NLL不替换history AMP CE',fontsize=11);save(fig,'02_clean_fit_and_frozen_gates.png')
    fig,axes=plt.subplots(1,3,figsize=(13.3,5),constrained_layout=True)
    for ax,key,title in ((axes[0],'baseline_last','原LAST240'),(axes[1],'candidate_last','候选LAST240')):
        im=ax.imshow(cases[key]['CM']/5,vmin=0,vmax=100,cmap='Blues');ax.set(title=title,xticks=range(10),yticks=range(10),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='预测类',ylabel='真实类');ax.tick_params(axis='x',rotation=55,labelsize=8)
        for k in range(10):ax.text(k,k,str(cases[key]['CM'][k,k]),color='white' if cases[key]['CM'][k,k]>250 else 'black',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=axes[:2],label='每真实类比例（%）',shrink=.65)
    axes[2].bar(range(10),np.array(change['perclass_accuracy_delta'])*100,color='#D26C32');axes[2].axhline(0,color='gray',lw=.8);axes[2].set(title='候选−原LAST每类准确率',ylabel='百分点',xticks=range(10),xticklabels=CLASSES);axes[2].grid(axis='y',alpha=.2)
    fig.suptitle('同一验证5000原图，每类500；固定LAST权重，非官方测试结果',fontsize=11);save(fig,'03_confusion_and_perclass.png')
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.3),constrained_layout=True)
    for ax,j,title,scale in ((axes[0],0,'配对准确率差（百分点）',100),(axes[1],1,'配对描述NLL差',1)):
        mean=float(differences[:,j].mean())*scale;lo,hi=intervals[:,j]*scale;ax.errorbar([0],[mean],yerr=np.array([[mean-lo],[hi-mean]]),fmt='o',capsize=5,color='#D26C32');ax.axhline(0,color='gray',ls='--');ax.set(title=title,xticks=[0],xticklabels=['候选LAST−原LAST']);ax.grid(axis='y',alpha=.2)
    axes[2].bar([0,1],[len(corrected),len(regressed)],color=['#4C78A8','#D26C32']);axes[2].set(title='同图错误转移',xticks=[0,1],xticklabels=['原错→候选对','原对→候选错'],ylabel='图数')
    fig.suptitle('固定模型/验证图的条件区间：分层500图/类、2000次共同配对Bootstrap\n单种子且复用验证集，不含重训/Teacher/方案选择不确定性；只描述，不新增门槛',fontsize=11);save(fig,'04_paired_intervals_and_errors.png')
    fig,axes=plt.subplots(2,4,figsize=(13,7.5),constrained_layout=True);examples=[]
    for row,(positions,wrong,caption) in enumerate(((corrected,baseline,'原错→候选对'),(regressed,candidate,'原对→候选错'))):
        confidence=wrong['probability'].max(1);chosen=positions[np.lexsort((ids[positions],-confidence[positions]))][:4]
        for col,ax in enumerate(axes[row]):
            ax.axis('off')
            if col>=len(chosen):ax.text(.5,.5,'无该类转移样本',ha='center',va='center');continue
            i=int(chosen[col]);ax.imshow(inputs['val_images'][i].transpose(1,2,0));bp=int(baseline['pred'][i]);cp=int(candidate['pred'][i]);ax.set_title(caption+f"\nID{int(ids[i])} 真类{CLASSES[int(y[i])]}\n原:{CLASSES[bp]} {baseline['probability'][i,bp]:.3f}\n候选:{CLASSES[cp]} {candidate['probability'][i,cp]:.3f}",fontsize=8)
            examples.append({'position':i,'ID':int(ids[i]),'direction':caption,'wrong_confidence':float(confidence[i])})
    fig.suptitle('固定展示：出错模型置信度降序、ID升序，各组至多4图\n复用原图缓存，不根据图调参或选权重；高置信错例不构成独立因果证据',fontsize=11);save(fig,'05_high_confidence_transfers.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True);seconds=np.array(result['raw_epoch_seconds']);axes[0].plot(epochs,seconds,color='#4C78A8');axes[0].set(title='全部240轮原始墙钟时间（含暂停）',xlabel='轮次',ylabel='秒');axes[0].grid(alpha=.2)
    axes[1].bar([0,1],[result['median_epoch_seconds'],result['baseline_last']['median_raw_epoch_seconds']],color=['#D26C32','#777777']);axes[1].set(title=f"墙钟中位数比 {result['median_wall_ratio']:.4f}（仅描述，无新门槛）",xticks=[0,1],xticklabels=['候选240轮','历史原240轮'],ylabel='秒/轮')
    fig.suptitle('不删任何轮时；历史时钟参照非同session纯GPU/等FLOPs性能对照\n该方法目标为分类质量，不能把时钟比例冒充部署速度或确定训练节省',fontsize=11);save(fig,'06_training_cost_and_pauses.png')
    unchanged(locks);proof={'state':'CPU_full_audit_and_figures_complete_manual_pending','seed':seed,'analysis_source_sha256':sha(__file__),'training':training,'evaluation_cases':metrics,'last240_changes_descriptive':change,'example_selection':examples,'figures':figures,
      'validation_statistics':{key:{'confusion_matrix':value['CM'].tolist(),'per_class_accuracy':(np.diag(value['CM'])/500).tolist()} for key,value in cases.items()},
      'conditional_interval':{'PCG64_seed':2026112644,'replicates':2000,'unit':'paired same5000 raw validation images; stratified500/class','quantile_method':'linear','new_gate':False},
      'evaluation_results_sha256':sha(BASE/'evaluation/results.json'),'seed_results_sha256':sha(HERE/'results'/('seed'+str(seed)+'.json')),'new_GPU_or_model_forward':0,'new_training':0,'official_test':0,'overall_goal_achieved':False,'manual_figures_opened':False}
    write(BASE/'results/analysis_results.json',proof)
    return {'state':proof['state'],'analysis_results_sha256':sha(BASE/'results/analysis_results.json'),'figures':figures,'manual_pending':True}



def _static_locks():
    manifest,_=_sources(protected=True);_approval()
    locks={str(HERE/name):sha(HERE/name) for name in ('plan.json','source_manifest.json','CONFIRM_READY.json','prepare_CPU.json','bridge_GPU.json')}
    locks.update({str((PROJECT/name).resolve()):digest for name,digest in manifest['training_sources'].items()})
    locks.update(manifest['protected_inputs'])
    return manifest,locks


def audit_training(seed):
    """CPU checkpoint audit plus streaming 42240-row telemetry; no forwards."""
    seed=_seed(seed);_ready(seed);manifest,locks=_static_locks()
    result_path=HERE/'results'/('seed'+str(seed)+'.json');result=read(result_path);locks[str(result_path)]=sha(result_path)
    folder=HERE/'formal_runs'/result['run_id'];trace=HERE/'formal_traces'/(result['run_id']+'.jsonl')
    for filename,key in (('last.pt','last_sha256'),('best.pt','best_sha256'),('history.csv','history_sha256')):
        assert sha(folder/filename)==result[key];locks[str(folder/filename)]=result[key]
    assert sha(trace)==result['trace_sha256'];locks[str(trace)]=result['trace_sha256']
    baseline_path=paths(seed)['baseline_last'];locks[str(baseline_path)]=sha(baseline_path)
    import torch
    with _runtime(torch,np) as runtime:
        torch.set_num_threads(1)
        last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        baseline=torch.load(baseline_path,map_location='cpu',weights_only=False)
        assert last['source_sha256']==best['source_sha256']==manifest['training_sources']
        assert baseline['source_sha256']==read(HERE/'plan.json')['original_training_source_dictionary']
        cfg=last['config'];assert cfg==best['config']
        clean_cfg={k:v for k,v in cfg.items() if k!='source_set_mode'}
        assert clean_cfg==baseline['config'] and cfg['source_set_mode']=='weighted_single'
        assert cfg['epochs']==240 and cfg['seed']==seed and cfg['width']==48 and not cfg.get('pilot_samples')
        assert cfg['distill_alpha']==.25 and cfg['distill_temperature']==1 and cfg['batch_size']==256 and cfg['amp'] is True
        assert last['stats']==best['stats']==baseline['stats']==read(PROJECT/'data/normalization.json')
        history=last['history'];assert last['epoch']==len(history)==baseline['epoch']==240
        for i,row in enumerate(history):
            assert row['epoch']==i+1 and row['train_samples']==45000 and row['validation_samples']==5000
            assert all(math.isfinite(value) for value in row.values())
            factor=(i+1)/5 if i<5 else .5*(1+math.cos(math.pi*(i-5)/235))
            assert abs(row['learning_rate']-.1*factor)<1e-12
        with (folder/'history.csv').open(encoding='utf-8-sig',newline='') as stream:csvrows=list(csv.DictReader(stream))
        assert len(csvrows)==240
        for row,csvrow in zip(history,csvrows):
            assert set(row)==set(csvrow) and all(float(csvrow[key])==value for key,value in row.items())
        selected=max(history,key=lambda row:(row['validation_accuracy'],-row['validation_ce']))
        assert best['epoch']==selected['epoch'] and best['best_accuracy']==selected['validation_accuracy'] and best['best_ce']==selected['validation_ce']
        assert history[-1]==result['last240'] and result['raw_epoch_seconds']==[row['seconds'] for row in history]
        assert result['median_epoch_seconds']==statistics.median(row['seconds'] for row in history)
        base=next(row for row in read(HERE/'plan.json')['baseline_last'] if row['seed']==seed)
        bfinal=baseline['history'][-1]
        assert bfinal['validation_accuracy']==base['accuracy'] and bfinal['validation_ce']==base['hard_validation_CE']
        assert round(bfinal['validation_accuracy']*5000)==base['correct'] and result['baseline_last']==base
        final=history[-1];correct=round(final['validation_accuracy']*5000)
        comparison={'correct_delta':correct-base['correct'],'accuracy_delta':final['validation_accuracy']-base['accuracy'],'CE_delta':final['validation_ce']-base['hard_validation_CE']}
        assert result['quality_comparison']==comparison and result['last_correct']==correct
        assert result['median_wall_ratio']==result['median_epoch_seconds']/base['median_raw_epoch_seconds']
        del last,best,baseline
    sums=np.zeros((240,5),dtype=np.float64);count=0
    errors_max={'modified_KD':0.,'returned_total':0.,'per_row_extra_mean':0.,'q_mass_sum':0.}
    for row in records(trace):
        epoch,batch=count//176+1,count%176+1;size=200 if batch==176 else 256
        assert (row['epoch'],row['batch_in_epoch'],row['global_batch'],row['batch_size'])==(epoch,batch,count+1,size)
        assert row['mode']=='weighted_single' and 0<=row['actual_lambda']<=1 and all(math.isfinite(row[key]) for key in LOSS_KEYS)
        for key in INPUT_KEYS:
            if key.endswith('sha256'):assert isinstance(row[key],str) and len(row[key])==64 and all(c in '0123456789abcdef' for c in row[key])
        a={key:np.asarray(row[key]) for key in ('source_cardinality','qS','qN','per_row_extra','conditional_KL_non_source','direct_non_source_mass','finite')}
        assert all(value.shape==(size,) and np.isfinite(value).all() for value in a.values())
        assert row['all_finite'] and a['finite'].all() and np.isin(a['source_cardinality'],(1,2)).all()
        assert (a['qS']>=0).all() and (a['qN']>0).all()
        errors={'modified_KD':abs(row['modified_KD']-row['original_KL']-row['new_extra']),
          'returned_total':abs(row['returned_total']-(.75*row['hard_CE']+.25*row['modified_KD'])),
          'per_row_extra_mean':abs(float(a['per_row_extra'].mean())-row['new_extra']),
          'q_mass_sum':float(np.max(np.abs(a['qS']+a['qN']-1)))}
        for key,value in errors.items():errors_max[key]=max(errors_max[key],value)
        assert errors['modified_KD']<=1e-5*max(1,abs(row['modified_KD'])) and errors['returned_total']<=1e-5*max(1,abs(row['returned_total']))
        assert errors['per_row_extra_mean']<=1e-5*max(1,abs(row['new_extra'])) and errors['q_mass_sum']<=1e-6
        sums[epoch-1]+=np.array([row['returned_total'],row['hard_CE'],row['modified_KD'],row['original_KL'],row['new_extra']])*size
        count+=1
    assert count==42240
    for i,row in enumerate(history):assert np.allclose(sums[i,:3]/45000,[row['train_loss'],row['train_ce'],row['train_kd']],atol=1e-10,rtol=1e-10)
    unchanged(locks)
    return {'state':'passed','seed':seed,'locks':locks,'history':history,'epoch_weighted_loss_parts':(sums/45000).tolist(),
      'trace_records':count,'trace_streaming':True,'teacher_actual_input_q_hashes':'all 42240 recorded shape/format/source audited; no extra teacher queries or independent value recomputation',
      'max_loss_diagnostic_identity_error':errors_max,'runtime':runtime,'best_epoch_secondary_only':int(selected['epoch']),
      'quality_comparison':comparison,'no_perseed_quality_or_new_resource_gate':True,
      'pauses_descriptive':[{'epoch':row['epoch'],'seconds':row['seconds']} for row in history if row['seconds']>max(60,3*result['median_epoch_seconds'])]}


def _verify_evaluation(seed,result):
    locks,candidate,baseline=_ready(seed);output=seed_dir(seed)/'evaluation'
    assert result['state']=='evaluation_complete_manual_pending' and set(result['cases'])==set(PROFILES)
    assert result['evaluation_source_sha256']==sha(__file__) and result['actual_image_forwards']==105000 and result['actual_forward_calls']==412
    shared=Path(result['inputs_file']);assert shared.resolve()==(output/'inputs.npz').resolve() and sha(shared)==result['inputs_sha256']
    for profile,row in result['cases'].items():
        path=Path(row['file_path']);assert path.resolve()==(output/(profile+'.npz')).resolve() and sha(path)==row['file_sha256']==row['cache_sha256']
        assert Path(row['checkpoint']).resolve()==paths(seed)[profile] and row['checkpoint_sha256']==locks[paths(seed)[profile]]
        assert row['training_sources']==(baseline if profile=='baseline_last' else candidate)
        assert row['W_BN_gradients_unchanged'] and row['no_mask_or_extra_head_hooks']
        assert row['history_comparison']['accuracy_equal'] and row['history_comparison']['CE_within_1e_minus6']
    return True


def _verify_analysis(seed,value):
    _ready(seed);assert value['state']=='CPU_full_audit_and_figures_complete_manual_pending' and value['seed']==seed
    assert value['analysis_source_sha256']==sha(__file__)
    assert value['evaluation_results_sha256']==sha(seed_dir(seed)/'evaluation/results.json')
    assert value['seed_results_sha256']==sha(HERE/'results'/('seed'+str(seed)+'.json'))
    unchanged(value['training']['locks'])
    _verify_evaluation(seed,read(seed_dir(seed)/'evaluation/results.json'))
    assert len(value['figures'])==6
    for filename,digest in value['figures'].items():assert sha(seed_dir(seed)/'figures'/filename)==digest
    return True


def execute_seed(seed):
    """Exactly-once complete evaluation/analysis; partial output requires review."""
    seed=_seed(seed);result_path=seed_dir(seed)/'results/analysis_results.json'
    if result_path.exists():
        result=read(result_path);_verify_analysis(seed,result)
        return {'state':result['state'],'seed':seed,'cached_verified':True,'new_image_forwards':0,'manual_pending':True,'analysis_results_sha256':sha(result_path)}
    training=audit_training(seed)
    try:
        evaluate_seed(seed);analyze_seed(seed,training)
        result=read(result_path);_verify_analysis(seed,result)
        return {'state':result['state'],'seed':seed,'cached_verified':False,'new_image_forwards':105000,'manual_pending':True,'analysis_results_sha256':sha(result_path)}
    except BaseException as exc:
        receipt={'state':'postprocess_failed','seed':seed,'reason':repr(exc),'traceback':traceback.format_exc(),'retry_automatic':False,'overall_goal_achieved':False}
        try:_ready(seed);unchanged(training['locks']);receipt['protected_inputs_and_locked_weights_unchanged']=True
        except BaseException as attention:receipt['protection_attention']=repr(attention)
        write(seed_dir(seed)/'postprocess_exception.json',receipt)
        raise


def _paired_interval(differences,labels,seed):
    assert differences.shape==(5000,2) and labels.shape==(5000,)
    assert np.array_equal(np.bincount(labels,minlength=10),np.full(10,500))
    rng=np.random.Generator(np.random.PCG64(seed));draws=np.zeros((2000,2),dtype=np.float64)
    for cls in range(10):
        index=np.flatnonzero(labels==cls);selection=rng.integers(0,500,(2000,500))
        draws+=differences[index][selection].mean(1)/10
    return np.quantile(draws,[.025,.975],axis=0,method='linear')


@_preserve_analysis_runtime
def summarize_three_seeds():
    """Native LAST gate and shared-image conditional CI; never select on test."""
    manifest,locks=_static_locks();output=HERE/'analyses/joint';target=output/'summary.json'
    if target.exists():
        value=read(target);assert value['postprocess_source_sha256']==sha(__file__);unchanged(value['input_locks'])
        for filename,digest in value['figures'].items():assert sha(output/'figures'/filename)==digest
        assert sha(output/'paired_shared_image_effects.npz')==value['paired_effect_cache_sha256']
        return value
    assert not output.exists(),'Partial joint output requires explicit reviewed recovery'
    for seed in (17,73):_verify_analysis(seed,read(seed_dir(seed)/'results/analysis_results.json'))
    parent_manifest=read(PARENT/'source_manifest.json');parent_closed=read(PARENT/'stage_closed.json')
    decision=read(HERE/'implementation_decision.json')
    assert sha(PARENT/'stage_closed.json')==decision['parent_closed_stage_sha256']
    assert sha(PARENT/'source_manifest.json')==decision['parent_source_manifest_sha256']
    assert parent_closed['all_bounded_phase_required_checks_closed'] is True
    assert sha(PARENT/'formal_results.json')==parent_closed['evidence_SHA']['formal_results.json']
    locks[str(PARENT/'stage_closed.json')]=sha(PARENT/'stage_closed.json');locks[str(PARENT/'source_manifest.json')]=sha(PARENT/'source_manifest.json')
    cases={};per_image=[];labels=ids=None;rows=[]
    for seed in (42,17,73):
        base=PARENT if seed==42 else seed_dir(seed)
        result_path=PARENT/'formal_results.json' if seed==42 else HERE/'results'/('seed'+str(seed)+'.json')
        formal=read(result_path);evaluation=read(base/'evaluation/results.json')
        assert formal['state']=='program_complete_manual_pending' and formal['last240']['epoch']==240
        assert formal['Source15' if seed==42 else 'training_sources']==(parent_manifest if seed==42 else manifest)['training_sources']
        assert evaluation['state']=='evaluation_complete_manual_pending' and evaluation['actual_image_forwards']==105000
        locks[str(result_path)]=sha(result_path);locks[str(base/'evaluation/results.json')]=sha(base/'evaluation/results.json')
        shared=Path(evaluation['inputs_file']);assert shared.resolve()==(base/'evaluation/inputs.npz').resolve() and sha(shared)==evaluation['inputs_sha256'];locks[str(shared)]=evaluation['inputs_sha256']
        with np.load(shared,allow_pickle=False) as z:current_labels=z['val_labels'];current_ids=z['val_ids']
        if labels is None:labels=current_labels;ids=current_ids
        else:assert np.array_equal(labels,current_labels) and np.array_equal(ids,current_ids)
        values={}
        for profile in ('candidate_last','baseline_last'):
            row=evaluation['cases'][profile];path=Path(row['file_path'])
            assert path.resolve()==(base/'evaluation'/(profile+'.npz')).resolve() and sha(path)==row['file_sha256'];locks[str(path)]=row['file_sha256']
            assert sha(row['checkpoint'])==row['checkpoint_sha256'];locks[row['checkpoint']]=row['checkpoint_sha256']
            expected=parent_manifest['training_sources'] if seed==42 and profile=='candidate_last' else (manifest['training_sources'] if profile=='candidate_last' else read(HERE/'plan.json')['original_training_source_dictionary'])
            assert row['training_sources']==expected and row['epoch']==240 and row['history_comparison']['accuracy_equal'] and row['history_comparison']['CE_within_1e_minus6']
            with np.load(path,allow_pickle=False) as z:
                logits=z['val_logits'];assert logits.shape==(5000,10) and logits.dtype==np.float32 and np.isfinite(logits).all()
                assert np.array_equal(z['val_labels'],labels) and np.array_equal(z['val_ids'],ids)
            p,nll=probability_nll(logits,labels);correct=logits.argmax(1)==labels
            assert int(correct.sum())==row['metrics']['val']['correct']
            assert abs(float(nll.mean())-row['metrics']['val']['CE_CPU_FP64_from_BF16_logits_descriptive'])<1e-10
            values[profile]={'nll':nll,'correct':correct,'pred':logits.argmax(1)}
        candidate,baseline=values['candidate_last'],values['baseline_last']
        difference=np.column_stack((candidate['correct'].astype(float)-baseline['correct'],candidate['nll']-baseline['nll']))
        per_image.append(difference)
        native=next(row for row in read(HERE/'plan.json')['baseline_last'] if row['seed']==seed)
        assert formal['baseline_last']==native
        accuracy_delta=formal['last240']['validation_accuracy']-native['accuracy'];ce_delta=formal['last240']['validation_ce']-native['hard_validation_CE']
        assert abs(float(difference[:,0].mean())-accuracy_delta)<1e-12
        row={'seed':seed,'native_LAST_accuracy_delta':accuracy_delta,'native_LAST_CE_delta':ce_delta,
          'native_candidate_LAST_accuracy':formal['last240']['validation_accuracy'],'native_candidate_LAST_CE':formal['last240']['validation_ce'],
          'native_baseline_LAST_accuracy':native['accuracy'],'native_baseline_LAST_CE':native['hard_validation_CE'],
          'descriptive_FP64_NLL_delta':float(difference[:,1].mean()),'candidate_correct':int(candidate['correct'].sum()),'baseline_correct':int(baseline['correct'].sum()),
          'perclass_accuracy_delta':[float(difference[labels==k,0].mean()) for k in range(10)],
          'corrected':int((~baseline['correct']&candidate['correct']).sum()),'regressed':int((baseline['correct']&~candidate['correct']).sum()),
          'median_epoch_seconds':formal['median_epoch_seconds'],'median_wall_ratio_descriptive':formal['median_wall_ratio'],'raw_epoch_seconds':formal['raw_epoch_seconds']}
        rows.append(row);cases[str(seed)]=values
    shared_effect=np.mean(np.stack(per_image),axis=0)
    intervals=_paired_interval(shared_effect,labels,2026112701)
    mean_accuracy=float(np.mean([row['native_LAST_accuracy_delta'] for row in rows]));mean_ce=float(np.mean([row['native_LAST_CE_delta'] for row in rows]));nonnegative=sum(row['native_LAST_accuracy_delta']>=-1e-12 for row in rows)
    thresholds=read(HERE/'plan.json')['joint_gate'];assert thresholds['mean_accuracy_delta_min']==.001 and thresholds['mean_hard_CE_delta_max']==-.005 and thresholds['individual_nonnegative_accuracy_seeds_min']==2
    criteria={'mean_accuracy_ge001':mean_accuracy>=.001-1e-12,'mean_native_CE_le_minus005':mean_ce<=-.005,'at_least2_nonnegative_accuracy':nonnegative>=2}
    gate={'criteria':criteria,'supported':all(criteria.values()),'mean_accuracy_delta':mean_accuracy,'mean_CE_delta':mean_ce,'nonnegative_accuracy_seeds':nonnegative,'no_new_resource_gate':True}
    output.mkdir(parents=True);figdir=output/'figures';figdir.mkdir();plt=plotter();figures={}
    def save(fig,name):
        path=figdir/name;fig.savefig(path,facecolor='white');plt.close(fig);figures[name]=sha(path)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True);x=np.arange(3)
    axes[0].bar(x,[row['native_LAST_accuracy_delta']*100 for row in rows],color='#4C78A8');axes[0].axhline(.1,color='#D26C32',ls=':',label='均值阈值参考0.1pp');axes[0].set(xticks=x,xticklabels=['42','17','73'],title='同种子LAST准确率差（pp）',xlabel='训练种子');axes[0].axhline(0,color='gray',lw=.8)
    axes[1].bar(x,[row['native_LAST_CE_delta'] for row in rows],color='#D26C32');axes[1].axhline(-.005,color='gray',ls=':');axes[1].axhline(0,color='gray',lw=.8);axes[1].set(xticks=x,xticklabels=['42','17','73'],title='同种子LAST原AMP CE差',xlabel='训练种子')
    axes[2].axis('off');axes[2].text(0,1,'预定三种子联合判断\n'+json.dumps(gate,ensure_ascii=False,indent=1)+'\n共享5000图条件95%区间\n准确率(pp): '+str((intervals[:,0]*100).tolist())+'\n描述NLL: '+str(intervals[:,1].tolist()),va='top',fontsize=8)
    fig.suptitle('完整预定42/17/73；主判断仅原生LAST准确率与AMP CE\n共享图条件区间不含重训/Teacher/选择不确定性，非额外门槛；不将15000视作独立',fontsize=11);save(fig,'01_three_seed_joint_metrics.png')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    heat=np.asarray([row['perclass_accuracy_delta'] for row in rows])*100;limit=max(.1,float(np.abs(heat).max()));im=axes[0].imshow(heat,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto');axes[0].set(xticks=range(10),xticklabels=CLASSES,yticks=range(3),yticklabels=['42','17','73'],title='每类LAST差（pp）');axes[0].tick_params(axis='x',rotation=55,labelsize=8);fig.colorbar(im,ax=axes[0],shrink=.7)
    axes[1].bar(x-.17,[row['corrected'] for row in rows],.32,label='原错→候选对');axes[1].bar(x+.17,[row['regressed'] for row in rows],.32,label='原对→候选错');axes[1].set(xticks=x,xticklabels=['42','17','73'],title='固定同图错误转移',ylabel='图数');axes[1].legend(frameon=False,fontsize=8)
    axes[2].bar(x,[row['median_wall_ratio_descriptive'] for row in rows],color='#777777');axes[2].set(xticks=x,xticklabels=['42','17','73'],title='原始训练墙钟中位数比（描述）',ylabel='候选/历史原KD');axes[2].axhline(1,color='gray',ls=':')
    fig.suptitle('每类、错误与成本均保留，不按单类或种子符号取消预定实验\n历史时钟非同session/等FLOPs对照；不宣称部署或训练节省',fontsize=11);save(fig,'02_three_seed_classes_errors_cost.png')
    _npz(output,'paired_shared_image_effects.npz',{'val_ids':ids,'val_labels':labels,'per_seed_effects':np.stack(per_image),'mean_effect':shared_effect})
    unchanged(locks)
    value={'state':'three_seed_summary_complete_manual_pending','postprocess_source_sha256':sha(__file__),'input_locks':locks,'seeds':[42,17,73],'per_seed':rows,'joint_gate':gate,
      'conditional_interval':{'accuracy95':intervals[:,0].tolist(),'NLL_FP64_descriptive95':intervals[:,1].tolist(),'PCG64_seed':2026112701,'replicates':2000,'unit':'paired same5000 original validation images; mean per-image across3seeds, stratified500/class','quantile_method':'linear','used_for_gate':False},
      'paired_effect_cache_sha256':sha(output/'paired_shared_image_effects.npz'),'figures':figures,'manual_visual_review_pending':True,'new_model_forwards':0,'official_test':0,'automatic_test_reproduction':False,'overall_goal_achieved':False}
    write(target,value);return value
