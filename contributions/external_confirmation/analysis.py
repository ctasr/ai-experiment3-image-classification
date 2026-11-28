"""Fixed primary/secondary scoring and common-image conditional intervals."""
from pathlib import Path
import numpy as np

from . import common as C
from .dataset import load_external, pixel_hashes
from .manifest import sealed, utc
from .inference import load_cache

# (name, left B, B policy, right A, A policy). Every interval is B minus A.
CONTRASTS = (
    ("alignment_B_minus_A","alignment_B_BEST",C.RAW,"alignment_A_BEST",C.RAW),
    ("weighted_minus_main_LAST","weighted_single_LAST",C.RAW,"main_KD_LAST",C.RAW),
    ("confidence_minus_main_LAST","confidence_mass_LAST",C.RAW,"main_KD_LAST",C.RAW),
    ("weighted_minus_confidence","weighted_single_LAST",C.RAW,"confidence_mass_LAST",C.RAW),
    ("main_BEST_two_view_minus_raw","main_KD_BEST",C.TWO,"main_KD_BEST",C.RAW),
    ("main_BEST_raw_minus_noKD","main_KD_BEST",C.RAW,"noKD240_BEST",C.RAW))


def metrics(q, lp, labels):
    labels = np.asarray(labels); n = len(labels)
    if n == 0:
        return {"samples":0,"state":"empty_population_no_estimand"}
    if q.dtype != np.float64 or lp.dtype != np.float64 or q.shape != (n,10) or lp.shape != (n,10):
        raise ValueError("Scoring must use the declared FP64 probability/log-probability policy")
    pred = q.argmax(1); correct = pred == labels
    nll = -lp[np.arange(n),labels]
    cm = np.bincount(labels*10+pred,minlength=100).reshape(10,10)
    count = cm.sum(1)
    confidence = q.max(1); bins = np.minimum((confidence*15).astype(np.int64),14)
    calibration = []; ece = 0.
    for b in range(15):
        selection = bins == b; number = int(selection.sum())
        a = float(correct[selection].mean()) if number else None
        conf = float(confidence[selection].mean()) if number else None
        calibration.append({"bin":b,"lower":b/15,"upper":(b+1)/15,"samples":number,"accuracy":a,"confidence":conf})
        if number:
            ece += number/n*abs(a-conf)
    onehot = np.eye(10,dtype=np.float64)[labels]
    return {"samples":n,"accuracy":float(correct.mean()),"NLL":float(nll.mean()),
            "multiclass_Brier_sum_over10":float(np.square(q-onehot).sum(1).mean()),"ECE15":float(ece),
            "per_class_accuracy":[float(cm[k,k]/count[k]) if count[k] else None for k in range(10)],
            "class_counts":count.tolist(),"confusion_matrix":cm.tolist(),"calibration_bins":calibration}


def paired_intervals(effect_matrix, labels, repetitions=10000):
    """Each row is a three-seed mean per-image effect, not 3N observations."""
    effects = np.asarray(effect_matrix,np.float64); labels = np.asarray(labels)
    if effects.ndim != 2 or effects.shape[1] != len(labels) or repetitions != 10000:
        raise ValueError("Fixed 10000-draw common-image bootstrap expected")
    if not len(labels):
        return None
    classes = [np.flatnonzero(labels==k) for k in range(10)]
    rng = np.random.Generator(np.random.PCG64(20261127))
    draws = np.empty((repetitions,len(effects)),np.float64)
    for start in range(0,repetitions,100):
        size = min(100,repetitions-start); chunk = np.zeros((size,len(effects)),np.float64)
        # Exactly one sampled index matrix is used for every contrast/metric,
        # therefore for every constituent model, seed and prediction policy.
        for indices in classes:
            if len(indices):
                chosen = indices[rng.integers(0,len(indices),size=(size,len(indices)))]
                chunk += effects[:,chosen].mean(2).T * (len(indices)/len(labels))
        draws[start:start+size] = chunk
    return {"point":effects.mean(1),"lower":np.quantile(draws,.025,axis=0),"upper":np.quantile(draws,.975,axis=0),
            "draws":draws}


def overlap(output, images, labels):
    output = Path(output)
    with np.load(output/"original_index.npz",allow_pickle=False) as value:
        hashes, original_labels, partition = value["pixel_sha256"],value["labels"],value["partition"]
    lookup = {}
    for i,h in enumerate(hashes):
        lookup.setdefault(h,[]).append(i)
    external = pixel_hashes(images); codes=np.zeros(len(images),np.uint8); conflicts=np.zeros(len(images),bool)
    duplicate_first=np.full(len(images),-1,np.int64); seen={}; duplicate_conflicts=[]
    for i,h in enumerate(external):
        matches=lookup.get(h,[])
        for index in matches:
            codes[i] |= 1 << int(partition[index])
            conflicts[i] |= original_labels[index] != labels[i]
        if h in seen:
            duplicate_first[i]=seen[h]
            if labels[seen[h]] != labels[i]:
                duplicate_conflicts.append([int(seen[h]),i])
        else:
            seen[h]=i
    excluded=codes!=0
    C.save_npz(output/"overlap.npz",external_pixel_sha256=external,original_partition_match_bits=codes,
               original_label_conflict=conflicts,external_duplicate_first=duplicate_first,secondary_include=~excluded)
    value={"kind":"fixed_pixel_overlap_report_v1","primary_samples":len(images),"secondary_samples":int((~excluded).sum()),
           "overlap_any_original60000":int(excluded.sum()),"overlap_train45000":int(((codes&1)!=0).sum()),
           "overlap_validation5000":int(((codes&2)!=0).sum()),"overlap_officialtest10000":int(((codes&4)!=0).sum()),
           "overlap_original_label_conflict_rows":int(conflicts.sum()),"external_duplicate_rows_after_first":int((duplicate_first>=0).sum()),
           "external_unique_pixel_images":len(seen),"external_duplicate_label_conflict_pairs":duplicate_conflicts,
           "primary_retains_all_internal_duplicates":True,"secondary_exclusion_uses_pixels_only":True,
           "npz_sha256":C.sha(output/"overlap.npz")}
    C.write(output/"overlap.json",value);return ~excluded,value


def figures(output, populations, contrasts, predictions, images, labels, overlap_info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(output); folder=output/"figures"; folder.mkdir(exist_ok=True)
    names=[]
    rows=[(g,p) for g in C.GROUPS for p in C.policies(g)]
    short=[g.replace("_BEST", " best").replace("_LAST"," last").replace("_"," ")+(" +flip" if p==C.TWO else "") for g,p in rows]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for axis,metric,scale,title in ((axes[0],"accuracy",100,"Accuracy (%)"),(axes[1],"NLL",1,"Stable FP64 NLL")):
        for i,(group,policy) in enumerate(rows):
            values=[populations["primary"][C.model_id(group,s)+"/"+policy][metric]*scale for s in C.SEEDS]
            axis.scatter(values,[i]*3,s=18,color="steelblue",alpha=.65)
            axis.scatter(np.mean(values),i,s=42,color="black",marker="|")
        axis.set_yticks(range(len(rows)),short if axis is axes[0] else []);axis.invert_yaxis();axis.set_xlabel(title);axis.grid(axis="x",alpha=.2)
    fig.suptitle("All 2000 external rows: fixed models, three training seeds (dots)")
    name="01_fixed_metrics.png";fig.savefig(folder/name,dpi=180);plt.close(fig);names.append(name)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for pi,pop in enumerate(("primary","secondary")):
        for mi,metric in enumerate(("accuracy","NLL")):
            axis=axes[pi,mi];records=contrasts[pop]
            if records is None:
                axis.text(.5,.5,"Empty prespecified population",ha="center",transform=axis.transAxes);continue
            for i,record in enumerate(records):
                entry=record[metric];factor=100 if metric=="accuracy" else 1
                point,lo,hi=[entry[k]*factor for k in ("difference","lower95","upper95")]
                axis.plot([lo,hi],[i,i],color="steelblue")
                axis.plot([lo,lo],[i-.08,i+.08],color="steelblue")
                axis.plot([hi,hi],[i-.08,i+.08],color="steelblue")
                axis.plot(point,i,"o",color="steelblue")
            axis.set_yticks(range(6),[r[0].replace("_"," ") for r in CONTRASTS] if mi==0 else [])
            axis.invert_yaxis();axis.axvline(0,color="gray",linestyle="--");axis.grid(axis="x",alpha=.2)
            axis.set_title(pop+": B-A "+("accuracy (pp)" if metric=="accuracy" else "NLL"))
    fig.suptitle("Shared-image class-paired bootstrap: conditional descriptive 95% intervals")
    name="02_fixed_paired_intervals.png";fig.savefig(folder/name,dpi=180);plt.close(fig);names.append(name)
    key=C.model_id("main_KD_BEST",42)+"/"+C.TWO;q,_=predictions[key]
    entry=populations["primary"][key];cm=np.asarray(entry["confusion_matrix"])
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    m=axes[0].imshow(cm,cmap="Blues");fig.colorbar(m,ax=axes[0],shrink=.7)
    axes[0].set_xticks(range(10),C.CLASSES,rotation=60,ha="right");axes[0].set_yticks(range(10),C.CLASSES)
    axes[0].set_xlabel("Predicted");axes[0].set_ylabel("Label")
    for i in range(10):
        for j in range(10):
            if cm[i,j]:axes[0].text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=7,color="white" if cm[i,j]>cm.max()/2 else "black")
    axes[1].barh(C.CLASSES,np.asarray(entry["per_class_accuracy"])*100,color="steelblue");axes[1].set_xlim(0,100);axes[1].set_xlabel("Per-class accuracy (%)")
    fig.suptitle("Prespecified example model: main BEST seed42, probability-mean two-view")
    name="03_confusion_per_class.png";fig.savefig(folder/name,dpi=180);plt.close(fig);names.append(name)
    prediction=q.argmax(1);wrong=np.flatnonzero(prediction!=labels)
    wrong=wrong[np.argsort(-q[wrong].max(1),kind="stable")][:6]
    fig,axes=plt.subplots(2,3,figsize=(10,7),layout="constrained")
    for axis in axes.flat:axis.axis("off")
    for axis,index in zip(axes.flat,wrong):
        axis.imshow(images[index]);axis.set_title(f"row {index}: {C.CLASSES[labels[index]]}\npred {C.CLASSES[prediction[index]]}, q={q[index].max():.3f}",fontsize=10)
    if not len(wrong):axes.flat[0].text(.5,.5,"No misclassified row",ha="center")
    fig.suptitle("Fixed highest-confidence errors; descriptive, no model/policy selection")
    name="04_fixed_errors.png";fig.savefig(folder/name,dpi=180);plt.close(fig);names.append(name)
    fig,axis=plt.subplots(figsize=(9,4),layout="constrained")
    categories=["Primary all rows","Secondary no old pixel match","Old train match","Old validation match","Old test match","Repeated external rows"]
    counts=[2000,overlap_info["secondary_samples"],overlap_info["overlap_train45000"],overlap_info["overlap_validation5000"],overlap_info["overlap_officialtest10000"],overlap_info["external_duplicate_rows_after_first"]]
    bars=axis.barh(categories,counts,color="steelblue");axis.bar_label(bars);axis.set_xlim(0,2200);axis.set_xlabel("Rows (overlap categories may intersect)")
    axis.set_title("Prespecified exact-pixel overlap disclosure")
    name="05_overlap_disclosure.png";fig.savefig(folder/name,dpi=180);plt.close(fig);names.append(name)
    return {name:C.sha(folder/name) for name in names}


def analyze(output):
    output=Path(output);manifest,_=sealed(output)
    if (output/"retired.json").exists():
        if C.sha(output/"analysis.json")!=C.read(output/"retired.json")["analysis_sha256"]:
            raise ValueError("The retired fixed analysis record changed")
        value=C.read(output/"analysis.json")
        for name,expected in value["figures_sha256"].items():
            if C.sha(output/"figures"/name)!=expected:raise ValueError("A retired scientific figure changed")
        return value
    done=C.read(output/"evaluation_complete.json")
    if done.get("checkpoints")!=21 or done.get("seed_policy_rows")!=24 or done["manifest_sha256"]!=C.sha(output/"manifest.json"):
        raise RuntimeError("The fixed complete evaluation is required before analysis")
    images,labels=load_external(output);predictions={};scores={}
    for row in manifest["models"]:
        if load_cache(output,row,"external") is None:raise RuntimeError("Missing fixed logit cache")
        with np.load(output/"logits"/(row["id"]+".npz"),allow_pickle=False) as value:
            z=value["raw"].copy();flip=value["flip"].copy() if "flip" in value else None
        for policy in row["policies"]:
            key=row["id"]+"/"+policy
            q,lp=C.probability_policy(z,flip if policy==C.TWO else None)
            predictions[key]=(q,lp)
            scores[key]=(q.argmax(1)==labels,-lp[np.arange(len(labels)),labels])
    secondary,overlap_info=overlap(output,images,labels)
    populations={};comparisons={};bootstrap_shas={}
    for population,mask in (("primary",np.ones(len(labels),bool)),("secondary",secondary)):
        populations[population]={key:metrics(q[mask],lp[mask],labels[mask]) for key,(q,lp) in predictions.items()}
        if not mask.any():comparisons[population]=None;continue
        effect=[]
        for _,g1,p1,g0,p0 in CONTRASTS:
            for mi in (0,1):
                effect.append(np.mean([scores[C.model_id(g1,s)+"/"+p1][mi].astype(np.float64)-scores[C.model_id(g0,s)+"/"+p0][mi].astype(np.float64) for s in C.SEEDS],axis=0)[mask])
        result=paired_intervals(effect,labels[mask]);records=[]
        for i,(name,*_) in enumerate(CONTRASTS):
            record={"name":name}
            for j,metric in enumerate(("accuracy","NLL")):
                index=2*i+j;record[metric]={"difference":float(result["point"][index]),"lower95":float(result["lower"][index]),"upper95":float(result["upper"][index])}
            records.append(record)
        comparisons[population]=records
        path=output/(population+"_bootstrap.npz");C.save_npz(path,draws=result["draws"],effects=np.asarray(effect),included_external_rows=np.flatnonzero(mask))
        bootstrap_shas[path.name]=C.sha(path)
    images_sha=figures(output,populations,comparisons,predictions,images,labels,overlap_info)
    value={"kind":"prespecified_external_fixed_analysis_v1","at_utc":utc(),"manifest_sha256":C.sha(output/"manifest.json"),
           "seal_sha256":C.sha(output/"seal.json"),"external_file_sha256":C.read(output/"fetch.json")["file_sha256"],
           "populations":populations,"fixed_contrasts":comparisons,"overlap":overlap_info,"bootstrap_sha256":bootstrap_shas,
           "bootstrap_repetitions":10000,"bootstrap_seed":20261127,"training_seeds":list(C.SEEDS),
           "interval_scope":"Conditional image-row resampling only; no retraining, Teacher, selection or familywise uncertainty claim",
           "all_24_seed_policy_rows_retained":True,"all_fixed_results_retained":True,
           "figures_sha256":images_sha,"manual_all_figure_review_pending":True,
           "no_new_model_policy_or_development_choice_from_external_outcomes":True}
    C.write(output/"analysis.json",value)
    C.write(output/"retired.json",{"kind":"external_dataset_retired_after_fixed_analysis_v1","at_utc":utc(),
            "analysis_sha256":C.sha(output/"analysis.json"),"all_further_model_selection_and_development_prohibited":True,
            "manual_visual_review_pending":True})
    return value
