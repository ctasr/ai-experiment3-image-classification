"""Fixed-image descriptive NLL and stratified shared-image bootstrap."""
import numpy as np

def probability_nll(logits,labels):
    logits=np.asarray(logits)
    labels=np.asarray(labels)
    if logits.ndim!=2 or logits.shape[1]!=10 or not len(logits) or labels.shape!=(len(logits),) or labels.dtype.kind not in 'iu' or not np.isfinite(logits).all():
        raise ValueError('Expected finite Nx10 logits and integer labels')
    if np.any(labels<0)or np.any(labels>=10):raise ValueError('Labels must be integer class indices0..9')
    z=logits.astype(np.float64);maximum=z.max(1);exp=np.exp(z-maximum[:,None]);denom=exp.sum(1)
    nll=maximum+np.log(denom)-z[np.arange(len(z)),labels]
    return exp/denom[:,None],nll

def summarize(logits,labels):
    probability,nll=probability_nll(logits,labels);prediction=np.asarray(logits).argmax(1);correct=prediction==labels
    confusion=np.zeros((10,10),dtype=np.int64);np.add.at(confusion,(labels,prediction),1)
    counts=confusion.sum(1)
    return {'probability':probability,'nll':nll,'prediction':prediction,'correct':correct,'confusion':confusion,
            'accuracy':float(correct.mean()),'NLL_FP64_from_BF16_logits':float(nll.mean()),
            'per_class_accuracy':np.divide(confusion.diagonal(),counts,out=np.full(10,np.nan),where=counts!=0)}

def paired_bootstrap(values,labels,repeats=2000,seed=2026112705):
    """N images x K seed-averaged contrasts; share sampled image rows across K.

    Three training models are fixed, not resampled or treated as 15k IID rows.
    True-class strata preserve each class count. Quantiles are descriptive
    fixed-model image uncertainty and exclude selection/retraining uncertainty.
    """
    values=np.asarray(values,dtype=np.float64);labels=np.asarray(labels)
    if values.ndim!=2 or values.shape[0]!=len(labels) or not np.isfinite(values).all() or repeats<1:
        raise ValueError('Expected finite image-paired contrasts')
    groups=[np.flatnonzero(labels==label) for label in range(10)]
    if any(len(group)==0 for group in groups):raise ValueError('Every true class must be present')
    rng=np.random.Generator(np.random.PCG64(seed));draws=np.empty((repeats,values.shape[1]),dtype=np.float64)
    class_draws=np.empty((repeats,10,values.shape[1]),dtype=np.float64)
    for number in range(repeats):
        selected_groups=[group[rng.integers(0,len(group),len(group))]for group in groups]
        selected=np.concatenate(selected_groups)
        draws[number]=values[selected].mean(0)
        for label,selected_class in enumerate(selected_groups):class_draws[number,label]=values[selected_class].mean(0)
    return {'mean':values.mean(0),'lower':np.quantile(draws,.025,axis=0),'upper':np.quantile(draws,.975,axis=0),'draws':draws,
            'class_mean':np.stack([values[group].mean(0)for group in groups]),'class_lower':np.quantile(class_draws,.025,axis=0),
            'class_upper':np.quantile(class_draws,.975,axis=0),'class_draws':class_draws}

def high_confidence_errors(summary,labels,ids,count=8):
    wrong=np.flatnonzero(~summary['correct'])
    confidence=summary['probability'][wrong,summary['prediction'][wrong]]
    chosen=wrong[np.lexsort((ids[wrong],-confidence))[:count]]
    return [{'row':int(row),'official_training_id':int(ids[row]),'true_class':int(labels[row]),
             'predicted_class':int(summary['prediction'][row]),'predicted_probability':float(summary['probability'][row,summary['prediction'][row]])}for row in chosen]


def main():
    import argparse,csv,hashlib,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description="Recompute the fixed-image conditional mass-control intervals from the explicitly derived contrast table; no model/test/training")
    parser.add_argument('--contrasts-csv',type=Path,required=True);parser.add_argument('--summary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Do not overwrite prior results')
    summary=json.loads(args.summary.read_text(encoding='utf-8'))['data']
    with args.contrasts_csv.open(encoding='utf-8',newline='')as stream:
        reader=csv.reader(stream);header=next(reader);rows=list(reader)
    names=list(summary['paired_contrasts'])
    if header!=['image_id','true_class',*names]or len(rows)!=5000:raise ValueError('Actual common5000 contrast CSV required')
    ids=np.asarray([int(row[0])for row in rows],dtype=np.int64);labels=np.asarray([int(row[1])for row in rows],dtype=np.int64)
    values=np.asarray([[float(value)for value in row[2:]]for row in rows],dtype=np.float64)
    if len(np.unique(ids))!=5000 or values.shape!=(5000,len(names))or not np.array_equal(np.bincount(labels,minlength=10),np.full(10,500)):raise ValueError('Common image identity/true-class strata differ')
    result=paired_bootstrap(values,labels,summary['bootstrap']['repetitions'],summary['bootstrap']['seed'])
    checked={}
    for i,name in enumerate(names):
        actual={'mean':float(result['mean'][i]),'lower95':float(result['lower'][i]),'upper95':float(result['upper'][i])}
        expected=summary['paired_contrasts'][name]
        if any(actual[key]!=expected[key]for key in actual):raise ValueError('Exact original interval differs; inspect dependency version/CSV/source, do not relabel')
        checked[name]=actual
    output={'kind':'recomputed_fixed5000_conditional_image_intervals','contrasts':checked,'three_models_fixed_not15000_IID':True,'CI_excludes_retraining_Teacher_selection':True,'new_model_forwards':0,'new_training':0,'metrics':'accuracy fractions and FP64-logit descriptive NLL, never native batch-mean CE','source_CSV_sha256':hashlib.sha256(args.contrasts_csv.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8')as stream:json.dump(output,stream,ensure_ascii=False,indent=2);stream.write('\n')
if __name__=='__main__':main()
