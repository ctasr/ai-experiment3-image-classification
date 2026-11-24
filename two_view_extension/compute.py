import csv
import time
import numpy as np
import torch
from .common import (ROOT, OUT, CASES, read, write, progress, normalise, js_distance,
                     per_image, transition, bootstrap, state_hash, assert_frozen, sha)
from .math_checks import analytic_maps, cpu_autograd, signed_map_distance


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def branches(model, x, capture=False):
    features = {}
    handle = model.features.register_forward_hook(lambda m, i, o: features.update(value=o.detach())) if capture else None
    try:
        z0 = model(x)
        f0 = features["value"].clone() if capture else None
        z1 = model(x.flip(-1).contiguous(memory_format=torch.channels_last))
        f1 = features["value"].clone() if capture else None
    finally:
        if handle is not None:
            handle.remove()
    p0, p1 = z0.softmax(1), z1.softmax(1)
    return p0, p1, (p0+p1)/2, f0, f1, z0, z1


def interpretability(model, stats, raw, labels, indices, rows, old, device):
    before = state_hash(model)
    parts = {k: [] for k in ["p0", "p1", "q", "map0", "map1", "signed", "heat", "flat", "locations", "feature0", "feature1", "norm0", "norm1"]}
    max_identity = 0.0
    sync(device); began = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, 200, 32):
            select = rows[start:start+32]
            x = normalise(raw[select].to(device), stats)
            p0,p1,q,f0,f1,z0,z1 = branches(model,x,True)
            c = q.argmax(1)
            a0,a1,m0,m1,signed,heat,flat,loc = analytic_maps(f0,f1,model.classifier.weight,p0,p1,c)
            for feature,z in [(f0,z0),(f1,z1)]:
                ordinary_cam = (feature*model.classifier.weight[c,:,None,None]).sum(1)
                identity = ordinary_cam.mean((1,2)) + model.classifier.bias[c]
                selected = z.gather(1,c[:,None]).squeeze(1)
                max_identity=max(max_identity,float((identity-selected).abs().max()))
                assert torch.allclose(identity,selected,atol=2e-5,rtol=2e-5)
            vals = dict(p0=p0,p1=p1,q=q,map0=m0,map1=m1,signed=signed,heat=heat,flat=flat,locations=loc,
                        feature0=f0,feature1=f1,norm0=a0.flatten(1).norm(dim=1),norm1=a1.flatten(1).norm(dim=1))
            for key,value in vals.items(): parts[key].append(value.cpu())
            progress("interpretability_clean", completed=min(start+32,200), total=200, current_model="kd")
    data={key:torch.cat(value) for key,value in parts.items()}
    p0,p1,q=[data[k].numpy() for k in ("p0","p1","q")]
    c=q.argmax(1)
    old_p=np.load(ROOT/"results_final_interpretability_v1/figure_data.npz",allow_pickle=False)["probabilities"]
    assert np.allclose(p0,old_p,atol=2e-5,rtol=2e-5)
    # CPU autograd: copies are constructed outside inference_mode and use only detached head parameters.
    selected=np.argsort(indices[rows])[:4]
    real_check=cpu_autograd(data["feature0"][selected].clone(),data["feature1"][selected].clone(),
                            model.classifier.weight.detach().cpu().clone(),model.classifier.bias.detach().cpu().clone(),
                            torch.as_tensor(c[selected],dtype=torch.long))
    real_check["official_train_indices"]=indices[rows[selected]].tolist()
    write("real_feature_cpu_autograd_preflight.json",real_check)
    masked0,masked1,positions=[],[],[]
    with torch.inference_mode():
        for start in range(0,200,16):
            select=rows[start:start+16]
            x=normalise(raw[select].to(device),stats)
            variants=[]; boxes=[]
            for local in range(len(x)):
                n=start+local
                loc=int(data["locations"][n]); high=[loc//25,loc%25]
                chosen=[high,*old["records"][n]["positions"][1:]]
                assert len(chosen)==11
                boxes.append(chosen)
                for yy,xx in chosen:
                    v=x[local].clone();v[:,yy:yy+8,xx:xx+8]=0;variants.append(v)
            b0,b1=[],[]
            for v in torch.stack(variants).split(64):
                a,b,*_=branches(model,v)
                b0.append(a.cpu());b1.append(b.cpu())
            masked0.append(torch.cat(b0).reshape(len(x),11,10))
            masked1.append(torch.cat(b1).reshape(len(x),11,10));positions.extend(boxes)
            progress("interpretability_occlusion",completed=min(start+16,200),total=200,current_model="kd")
    masked0=torch.cat(masked0).numpy();masked1=torch.cat(masked1).numpy();maskedq=(masked0+masked1)/2
    select=np.arange(200)
    scores0=masked0[select,:,c];scores1=masked1[select,:,c];scoresq=maskedq[select,:,c]
    drops0=p0[select,c,None]-scores0;drops1=p1[select,c,None]-scores1;dropsq=q[select,c,None]-scoresq
    drop_identity=float(np.max(np.abs(dropsq-(drops0+drops1)/2)))
    assert drop_identity < 2e-6
    d0=drops0[:,0]-drops0[:,1:].mean(1);d1=drops1[:,0]-drops1[:,1:].mean(1);dq=dropsq[:,0]-dropsq[:,1:].mean(1)
    distances,valid=signed_map_distance(data["map0"].numpy(),data["map1"].numpy())
    records=[]
    for i in range(200):
        records.append({"official_train_index":int(indices[rows[i]]),"true":int(labels[rows[i]]),"target_q_clean":int(c[i]),
            "raw_predicted":int(p0[i].argmax()),"flip_predicted":int(p1[i].argmax()),"q_correct":bool(c[i]==labels[rows[i]]),
            "q_probability":float(q[i,c[i]]),"q_high_drop":float(dropsq[i,0]),"q_random_mean_drop":float(dropsq[i,1:].mean()),
            "q_high_minus_random":float(dq[i]),"raw_same_masks_high_minus_random":float(d0[i]),
            "flip_same_masks_high_minus_random":float(d1[i]),"positions":positions[i],"flat_positive_heatmap":bool(data["flat"][i]),
            "branch_positive_map_distance":float(distances[i]) if valid[i] else None,
            "branch_gradient_coefficient_l2":[float(data["norm0"][i]),float(data["norm1"][i])]})
    examples=[]
    for correct in (True,False):
        candidates=[i for i,v in enumerate(records) if v["q_correct"]==correct]
        if candidates:examples.append(min(candidates,key=lambda i:records[i]["official_train_index"]))
    y=labels[rows]
    sync(device);elapsed=time.perf_counter()-began
    np.savez_compressed(OUT/"interpretability_data.npz",images=raw[rows].permute(0,2,3,1).numpy(),labels=y,indices=indices[rows],
        p0=p0,p1=p1,q=q,map0=data["map0"].numpy(),map1=data["map1"].numpy(),signed=data["signed"].numpy(),heat=data["heat"].numpy(),
        masked0=masked0,masked1=masked1,maskedq=maskedq,positions=np.asarray(positions),example_rows=np.asarray(examples),dq=dq,d0=d0,d1=d1)
    assert_frozen(model,before)
    result={"state":"computed","model":"kd seed42 width48","target":"fixed uncalibrated q(clean).argmax","samples":200,
        "new_validation_forward_images":4800,"new_training":0,"new_gpu_autograd":0,"new_official_test_inference":0,
        "wall_seconds_compute_including_cpu_autograd":elapsed,"q_high_minus_random_mean":float(dq.mean()),"q_ci95":bootstrap(dq),
        "raw_same_masks_high_minus_random_mean":float(d0.mean()),"q_minus_raw_sensitivity_mean":float((dq-d0).mean()),
        "q_minus_raw_sensitivity_ci95":bootstrap(dq-d0),"branch_disagreement_count":int((p0.argmax(1)!=p1.argmax(1)).sum()),
        "branch_js_mean":float(js_distance(p0,p1).mean()),"flat_positive_heatmap_count":int(data["flat"].sum()),
        "undefined_branch_positive_map_distance_count":int((~valid).sum()),"branch_positive_map_distance_mean":float(distances[valid].mean()) if valid.any() else None,
        "raw_to_q_correctness":transition(p0.argmax(1),c,y),"per_class":[{"class":k,"n":int((y==k).sum()),
            "q_accuracy":float((c[y==k]==k).mean()),"branch_disagreement_count":int((p0[y==k].argmax(1)!=p1[y==k].argmax(1)).sum()),
            "q_sensitivity_mean":float(dq[y==k].mean())} for k in range(10)],
        "checks":{"raw_matches_v1_fp32":True,"ordinary_cam_logit_identity_max_error":max_identity,
                  "q_drop_branch_mean_identity_max_error":drop_identity,"real_feature_cpu_autograd_passed":True,
                  "weights_bn_gradients_unchanged":True,"random_positions_exactly_reused":True,
                  "q_branch_swap_identity_is_algebraic_not_extra_inference":True},
        "records":records,"example_official_indices":[records[i]["official_train_index"] for i in examples],
        "data_sha256":sha(OUT/"interpretability_data.npz"),"source_manifest_sha256":sha(OUT/"source_manifest.json"),"manual_pending":True,
        "limitations":"q-target local gradient coarse localization, not exact additive q decomposition/causal/segmentation proof. Probability saturation and flat maps retained. FP32 uncalibrated single-seed validation diagnostics."}
    write("interpretability.json",result)
    return result


def stability(model,name,stats,raw,labels,indices,device):
    before=state_hash(model);arrays={};results={};sync(device);began=time.perf_counter()
    for case in CASES:
        out0,out1=[],[]
        with torch.inference_mode():
            for start in range(0,5000,128):
                x=normalise(raw[start:start+128].to(device),stats,case)
                p0,p1,*_=branches(model,x)
                out0.append(p0.cpu().numpy());out1.append(p1.cpu().numpy())
                progress("stability_inference",current_model=name,current_case=case,completed=min(start+128,5000),total=5000)
        p0,p1=np.concatenate(out0),np.concatenate(out1);q=(p0+p1)/2
        arrays[f"{case}_raw"],arrays[f"{case}_flip"],arrays[f"{case}_q"]=p0,p1,q
        results[case]={"branch_disagreement_count":int((p0.argmax(1)!=p1.argmax(1)).sum()),"branch_js_mean":float(js_distance(p0,p1).mean()),
                       "raw_to_q_correctness":transition(p0.argmax(1),q.argmax(1),labels),"policies":{}}
        for policy,p in [("raw",p0),("q",q)]:
            pred,acc,ce=per_image(p,labels)
            basepred,baseacc,basece=per_image(arrays[f"clean_{policy}"],labels)
            cm=np.zeros((10,10),dtype=int);np.add.at(cm,(labels,pred),1)
            results[case]["policies"][policy]={"accuracy":float(acc.mean()),"hard_ce":float(ce.mean()),
                "accuracy_change_from_clean_pp":float((acc-baseacc).mean()*100),"hard_ce_change_from_clean":float((ce-basece).mean()),
                "accuracy_change_ci95_pp":(np.asarray(bootstrap(acc-baseacc,labels))*100).tolist(),
                "hard_ce_change_ci95":bootstrap(ce-basece,labels),"prediction_flip_count":int((pred!=basepred).sum()),
                "prediction_flip_rate":float((pred!=basepred).mean()),"correctness_transitions":transition(basepred,pred,labels),
                "confusion_matrix":cm.tolist(),"error_indices":indices[pred!=labels].tolist(),
                "per_class":[{"class":k,"n":500,"accuracy":float(acc[labels==k].mean()),"hard_ce":float(ce[labels==k].mean()),
                    "prediction_flip_rate":float((pred[labels==k]!=basepred[labels==k]).mean()),
                    "branch_disagreement_rate":float((p0[labels==k].argmax(1)!=p1[labels==k].argmax(1)).mean())} for k in range(10)]}
        _,a0,_=per_image(p0,labels);_,aq,_=per_image(q,labels)
        _,bc0,_=per_image(arrays["clean_raw"],labels);_,bcq,_=per_image(arrays["clean_q"],labels)
        results[case]["q_minus_raw_accuracy_pp"]=float((aq-a0).mean()*100)
        results[case]["q_minus_raw_accuracy_ci95_pp"]=(np.asarray(bootstrap(aq-a0,labels))*100).tolist()
        interaction=(aq-a0)-(bcq-bc0)
        results[case]["q_minus_raw_additional_gain_vs_clean_pp"]=float(interaction.mean()*100)
        results[case]["q_minus_raw_additional_gain_ci95_pp"]=(np.asarray(bootstrap(interaction,labels))*100).tolist()
    sync(device);elapsed=time.perf_counter()-began
    assert_frozen(model,before)
    arrays["labels"],arrays["official_train_indices"]=labels,indices
    np.savez_compressed(OUT/f"stability_{name}_data.npz",**arrays)
    result={"state":"computed","model":name,"seed":42,"samples":5000,"cases":results,
        "new_validation_forward_images":40000,"wall_seconds_compute_including_bootstrap":elapsed,
        "new_training":0,"new_teacher_queries":0,"new_official_test_inference":0,"new_temperature_fits":0,
        "checks":{"weights_bn_gradients_unchanged":True,"all_cases_retained":True,"500_per_class":True},
        "data_sha256":sha(OUT/f"stability_{name}_data.npz"),"source_manifest_sha256":sha(OUT/"source_manifest.json"),"manual_pending":True,
        "limitations":"One fixed seed per condition; custom predetermined uint8 +/-8 brightness and right1 mean-fill shift. Not adversarial/benchmark robustness or pure ImageNet causal proof. Conditional image intervals exclude training/selection uncertainty; no multi-case adjustment."}
    write(f"stability_{name}.json",result)
    with (OUT/f"stability_{name}_per_class.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=["case","policy","class","n","accuracy","hard_ce","prediction_flip_rate","branch_disagreement_rate"])
        writer.writeheader()
        for case,row in results.items():
            for policy,v in row["policies"].items():
                writer.writerows({"case":case,"policy":policy,**item} for item in v["per_class"])
    return result


def compare_stability(labels):
    kd=np.load(OUT/"stability_kd_data.npz",allow_pickle=False)
    plain=np.load(OUT/"stability_no_kd_data.npz",allow_pickle=False)
    assert np.array_equal(kd["labels"],plain["labels"]) and np.array_equal(kd["official_train_indices"],plain["official_train_indices"])
    results={}
    for case in CASES:
        results[case]={}
        for policy in ("raw","q"):
            pk,ak,cek=per_image(kd[f"{case}_{policy}"],labels)
            pn,an,cen=per_image(plain[f"{case}_{policy}"],labels)
            _,basek,basecek=per_image(kd[f"clean_{policy}"],labels)
            _,basen,basecen=per_image(plain[f"clean_{policy}"],labels)
            difference=ak-an; delta=difference-(basek-basen)
            results[case][policy]={"kd_minus_no_kd_accuracy_pp":float(difference.mean()*100),
                "accuracy_ci95_pp":(np.asarray(bootstrap(difference,labels))*100).tolist(),
                "kd_minus_no_kd_ce":float((cek-cen).mean()),"ce_ci95":bootstrap(cek-cen,labels),
                "accuracy_gain_change_from_clean_pp":float(delta.mean()*100),
                "accuracy_gain_change_ci95_pp":(np.asarray(bootstrap(delta,labels))*100).tolist(),
                "ce_gap_change_from_clean":float(((cek-cen)-(basecek-basecen)).mean()),
                "control_to_kd_correctness":transition(pn,pk,labels)}
    result={"state":"computed","samples":5000,"direction":"KD minus matched no-KD seed42; same images per case/policy",
        "cases":results,"bootstrap":"1000 seed42 stratified-by-label paired image bootstrap; fixed two weights; no multiple-case correction",
        "causal_limit":"This is a matched training-recipe KD control, not an isolated ImageNet-knowledge intervention, and not training-seed uncertainty.","manual_pending":True}
    write("stability_comparison.json",result)
    with (OUT/"stability_case_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.writer(f);writer.writerow(["model","case","policy","accuracy","hard_ce","accuracy_change_pp","prediction_flip_rate","branch_disagreement_count"])
        for name in ("kd","no_kd"):
            row=read(OUT/f"stability_{name}.json")
            for case,v in row["cases"].items():
                for policy,m in v["policies"].items():
                    writer.writerow([name,case,policy,m["accuracy"],m["hard_ce"],m["accuracy_change_from_clean_pp"],m["prediction_flip_rate"],v["branch_disagreement_count"]])
    return result
