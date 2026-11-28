"""Two presentation-only corrections from locked controlled-analysis numbers.

Summary mode requires Root's actual opening of all8 original PNGs first. Payload
mode only redraws an already frozen derived numeric payload. Neither mode reads
model/data caches, samples intervals, trains, or performs a model forward.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

SEEDS=(42,17,73)
METHODS=('baseline','se','cutout','smoothing','combined','cutmix')
ARCH=('cnn','resnet20')
CLASSES=('飞机','汽车','鸟','猫','鹿','狗','蛙','马','船','卡车')
LABELS={'baseline':'基线','se':'通道注意力','cutout':'Cutout','smoothing':'标签平滑',
        'combined':'注意力＋Cutout＋平滑','cutmix':'CutMix','cnn':'CNN32','resnet20':'残差16'}
TARGETS=('03_augmentation_LAST_fit_and_classes.png','06_matched_architecture_curves_fit_cost.png')
ORIGINAL_ANALYSIS_SHA='3e7d13391aeaeb0e654073fdf6f33c5db7510185688183d5f533fda0ee8bfe64'
TRAIN={'color':'#28688b','marker':'o','label':'原图训练45k'}
VALIDATION={'color':'#bd673e','marker':'x','label':'原图验证5k'}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def need(value,reason):
    if not value:raise ValueError(reason)
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def ref(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def write(path,value):
    with Path(path).open('x',encoding='utf-8')as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')

def original_gate(summary_path,audit_path,review_path):
    summary,audit,review=read(summary_path),read(audit_path),read(review_path)
    need(summary['state']=='analysis_complete_manual_review_pending'and
         summary['analysis_sha256']==audit['analysis_sha256']==ORIGINAL_ANALYSIS_SHA,
         'A matching actual frozen analysis summary/audit is required')
    need(audit['passed']is True and audit['case_count']==24 and audit['augmentation_cases']==18 and
         audit['matched_architecture_cases']==6 and audit['official_test_images']==0 and audit['new_training']==0,
         'All24 actual completed-case audit is required, not a prefix or synthetic fixture')
    figures=summary['figure_manifest']
    need(len(figures)==8 and len({Path(row['path']).name for row in figures})==8,'Exactly8 original scientific PNGs required')
    need(review['kind']=='root_controlled_original8_visual_review_before_presentation_v1'and
         review['reviewer_role']=='root'and review['actual_opened_all8_by_root']is True,
         'Root must actually open all8 original PNGs before correction')
    need(review['source_summary_sha256']==sha(summary_path)and review['source_audit_sha256']==sha(audit_path),
         'Root opened figures for different summary/audit bytes')
    need(set(review['figures_requiring_presentation_revision'])==set(TARGETS),
         'Only the reviewed original03/06 may be replaced by this renderer')
    old={}
    for row in figures:
        path=Path(row['path']);name=path.name
        need(sha(path)==row['sha256']==review['original_figure_sha256'][name],
             'Original PNG or Root original-opened identity differs: '+name)
        old[name]=ref(path)
    need(set(review['original_figure_sha256'])==set(old),'Root review must cover all8 originals')
    need(set(TARGETS)<=set(old),'Expected original03/06 absent')
    return summary,old,{'summary':ref(summary_path),'audit':ref(audit_path),'root_original8_visual_review':ref(review_path)}

def prepare_payload(summary,source_refs,originals):
    import numpy as np
    cases=summary['cases']
    need(len(cases)==24 and len({row['id']for row in cases})==24,'Expected24 distinct actual cases')
    result={'kind':'controlled03_06_frozen_numeric_presentation_payload_v1','seeds':list(SEEDS),
            'source_summary_sha256':source_refs['summary']['sha256'],
            'source_audit_sha256':source_refs['audit']['sha256'],
            'root_original8_visual_review_sha256':source_refs['root_original8_visual_review']['sha256'],
            'original_analysis_sha256':ORIGINAL_ANALYSIS_SHA,
            'original_figure_sha256':{name:row['sha256']for name,row in originals.items()},
            'selection':'Exact existing LAST160 values/intervals/history; only original plotting mean/sampleSD/percent scaling, no new bootstrap',
            'augmentation':{},'architecture':{}}
    def group(study,method):
        rows=sorted([row for row in cases if row['study']==study and row['method']==method],key=lambda row:SEEDS.index(row['seed']))
        need([row['seed']for row in rows]==list(SEEDS),'Exactly fixed42/17/73 required per method')
        for row in rows:
            need(len(row['history'])==160 and [h['epoch']for h in row['history']]==list(range(1,161)),
                 'Actual complete160 histories required')
            need(row['LAST160']['train']['samples']==45000 and row['LAST160']['validation']['samples']==5000,
                 'Actual original train45k/validation5k required')
        return rows
    def fit(rows):
        # Preserve the original figure operation order: first scale all three
        # accuracies by100, then mean/std(ddof1); do not change numeric units.
        output={}
        for part in('train','validation'):
            values=np.array([row['LAST160'][part]['accuracy']for row in rows])*100
            output[part]={'values_percent':values.tolist(),'mean_percent':float(values.mean()),
                          'sample_SD_pp':float(values.std(ddof=1))}
        return output
    aug=summary['augmentation_BEST_and_LAST_separate']
    for method in METHODS:
        rows=group('augmentation',method);interval=aug[method]['LAST160_vs_reference_accuracy_fraction']
        lo,hi=np.array(interval['conditional_image_bootstrap_95_interval'])*100
        result['augmentation'][method]={'fit':fit(rows),
           'paired_accuracy_interval_pp':{'point':interval['point']*100,'lower95':float(lo),'upper95':float(hi)},
           'per_class_mean_accuracy_percent':(np.array(aug[method]['LAST160_per_class_accuracy_mean'])*100).tolist()}
    for method in ARCH:
        rows=group('architecture',method)
        result['architecture'][method]={'fit':fit(rows),
           'validation_accuracy_curve_mean_percent':(np.mean([[h['validation_accuracy']for h in row['history']]for row in rows],axis=0)*100).tolist(),
           'native_hard_CE_curve_mean':np.mean([[h['validation_ce']for h in row['history']]for row in rows],axis=0).tolist(),
           'recorded_epoch_median_seconds_by_seed':[row['raw_training_cost']['median_recorded_epoch_seconds']for row in rows]}
    need(np.isfinite(np.asarray([result['augmentation'][method]['fit'][part]['mean_percent']
          for method in METHODS for part in('train','validation')])).all(),'Nonfinite actual fit values')
    return result

def draw(payload,output):
    import numpy as np
    need(payload['kind']=='controlled03_06_frozen_numeric_presentation_payload_v1'and
         payload['seeds']==list(SEEDS)and payload['original_analysis_sha256']==ORIGINAL_ANALYSIS_SHA,
         'Unexpected frozen numeric presentation payload')
    for key in('source_summary_sha256','source_audit_sha256','root_original8_visual_review_sha256'):
        value=payload.get(key)
        need(isinstance(value,str)and len(value)==64 and all(c in'0123456789abcdef'for c in value),
             'Prepared payload must retain its source and actual original8 Root-review fingerprints')
    need(set(payload['augmentation'])==set(METHODS)and set(payload['architecture'])==set(ARCH),
         'Complete six augmentation/two architecture numeric groups required')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    font=Path('C:/Windows/Fonts/simsun.ttc')
    need(font.is_file(),'SimSun font required, as in the original figures')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'SimSun','font.size':10,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':170})
    figures={}
    fig,axes=plt.subplots(1,3,figsize=(14,4));xx=np.arange(6)
    for i,method in enumerate(METHODS):
        row=payload['augmentation'][method]
        for part,shift,style in(('train',-.08,TRAIN),('validation',.08,VALIDATION)):
            values=row['fit'][part]
            axes[0].errorbar(i+shift,values['mean_percent'],yerr=values['sample_SD_pp'],
                fmt=style['marker'],color=style['color'],label=style['label']if i==0 else None)
        interval=row['paired_accuracy_interval_pp'];p=interval['point'];lo=interval['lower95'];hi=interval['upper95']
        # The existing precomputed conditional interval is neither resampled
        # nor replaced by the descriptive three-seed sampleSD.
        axes[1].errorbar(i,p,yerr=[[max(0,p-lo)],[max(0,hi-p)]],fmt='o',capsize=3)
    im=axes[2].imshow(np.array([payload['augmentation'][method]['per_class_mean_accuracy_percent']for method in METHODS]),
                     vmin=75,vmax=100,aspect='auto',cmap='Blues')
    axes[2].set_xticks(range(10),CLASSES,rotation=45);axes[2].set_yticks(xx,[LABELS[m]for m in METHODS])
    fig.colorbar(im,ax=axes[2]);axes[2].set_title('LAST每类准确率，三模型均值（%）')
    for ax in axes[:2]:ax.set_xticks(xx,[LABELS[m]for m in METHODS],rotation=40,ha='right');ax.grid(alpha=.2)
    axes[0].set_ylabel('LAST准确率（%）');axes[0].legend();axes[0].set_title('LAST原图：三种子均值±样本SD')
    axes[1].set_ylabel('相对基线百分点');axes[1].axhline(0,c='gray',lw=.7);axes[1].set_title('LAST条件图像抽样区间，非种子推断')
    fig.tight_layout();path=output/TARGETS[0];fig.savefig(path,bbox_inches='tight');plt.close(fig);figures[TARGETS[0]]=ref(path)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for i,method in enumerate(ARCH):
        row=payload['architecture'][method]
        axes[0,0].plot(np.arange(1,161),row['validation_accuracy_curve_mean_percent'],label=LABELS[method])
        axes[0,1].plot(np.arange(1,161),row['native_hard_CE_curve_mean'],label=LABELS[method])
        for part,shift,style in(('train',-.08,TRAIN),('validation',.08,VALIDATION)):
            values=row['fit'][part]
            axes[1,0].errorbar(i+shift,values['mean_percent'],yerr=values['sample_SD_pp'],
                fmt=style['marker'],color=style['color'],label=style['label']if i==0 else None)
        axes[1,1].scatter([i]*3,row['recorded_epoch_median_seconds_by_seed'],label=LABELS[method])
    for ax in axes[0]:ax.set_xlabel('训练轮次');ax.legend();ax.grid(alpha=.2)
    for ax in axes[1]:ax.set_xticks(range(2),['CNN32','残差16']);ax.grid(alpha=.2)
    axes[0,0].set_ylabel('验证准确率（%）');axes[0,1].set_ylabel('验证硬标签CE')
    axes[0,0].set_title('42/17/73逐轮验证均值');axes[0,1].set_title('42/17/73逐轮验证均值')
    axes[1,0].set_ylabel('LAST准确率（%）');axes[1,0].legend();axes[1,0].set_title('原图拟合：三种子均值±样本SD')
    axes[1,1].set_ylabel('原训练轮时中位数（秒）')
    fig.tight_layout();path=output/TARGETS[1];fig.savefig(path,bbox_inches='tight');plt.close(fig);figures[TARGETS[1]]=ref(path)
    return figures

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--summary',type=Path);source.add_argument('--payload',type=Path)
    parser.add_argument('--audit',type=Path);parser.add_argument('--original-review',type=Path)
    parser.add_argument('--payload-sha256');parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--check-ready',action='store_true')
    args=parser.parse_args()
    renderer_before=sha(__file__)
    required=[args.summary,args.audit,args.original_review]if args.summary else[args.payload]
    if any(path is None for path in required):raise ValueError('Summary requires explicit audit and Root original8 review')
    missing=[str(path)for path in required if not path.is_file()]
    if missing:
        print(json.dumps({'state':'pending_actual_locked_inputs_no_figure_created','missing':missing,
                          'scientific_passed':False,'new_images':0},ensure_ascii=False));return 3
    source_refs={};old=None
    if args.summary:
        summary,old,source_refs=original_gate(args.summary,args.audit,args.original_review)
        payload=prepare_payload(summary,source_refs,old)
    else:
        need(args.payload_sha256 and sha(args.payload)==args.payload_sha256,'Explicit actual frozen payload SHA required')
        payload=read(args.payload);source_refs={'frozen_numeric_payload':ref(args.payload)}
    if args.check_ready:
        print(json.dumps({'state':'actual_input_gate_ready_NO_figures_generated','data_payload_sha256':digest(payload),
                          'new_images':0,'not_manual_visual_acceptance':True},ensure_ascii=False));return 0
    need(args.output_dir is not None,'A new isolated output directory is required')
    output=args.output_dir.resolve()
    need(not output.exists(),'Preserve previous original or derived output; choose a new directory')
    for path in required:
        need(not path.resolve().is_relative_to(output),'Output may not contain/replace its frozen inputs')
    if args.summary:need(not output.is_relative_to(args.summary.resolve().parent),'Keep original producer results and8PNG unchanged')
    output.mkdir(parents=True)
    frozen=digest(payload);write(output/'data_payload.json',payload)
    figures=draw(payload,output)
    need(digest(payload)==frozen,'Presentation numeric payload changed')
    need(sha(__file__)==renderer_before,'Renderer source changed during drawing')
    for row in source_refs.values():need(sha(row['path'])==row['sha256'],'A frozen presentation input changed during drawing')
    if old:
        for row in old.values():need(sha(row['path'])==row['sha256'],'An original PNG changed during rendering')
    manifest={'kind':'controlled03_06_presentation_only_derivative_v1',
      'state':'two_derivative_figures_generated_root_actual_visual_review_pending',
      'renderer':ref(__file__),'source_inputs':source_refs,
      'numeric_payload':ref(output/'data_payload.json'),'numeric_payload_canonical_sha256':frozen,
      'original8_figures':old,'new2_figures':figures,
      'changes':{'train_validation_legend':'All methods use fixed train blue/circle and validation orange/cross',
                 'fit_errorbars':'Three fixed seeds42/17/73, mean±sampleSD(ddof1), not95%CI',
                 'existing_architecture_curves':'Same160 recorded epoch means, explicitly labelled42/17/73',
                 'other_panels':'Original numbers, precomputed conditional intervals, class order/color limits and recorded cost values retained'},
      'scientific_figures':8,'preserved_original_versions':8,'new_presentation_versions':2,
      'new_model_forward_training_bootstrap_or_clock_measurement':False,
      'original_producer_source_plan_data_or8PNG_changed':False,
      'actual_new2_root_visual_review_complete':False,'whole_scope_acceptance':False}
    write(output/'provenance_manifest.json',manifest)
    print(json.dumps({'state':manifest['state'],'manifest':ref(output/'provenance_manifest.json'),'new_images':2},ensure_ascii=False))
    return 0
if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,FileNotFoundError,KeyError)as exc:
        print(json.dumps({'state':'presentation_refused','reason':str(exc),'scientific_passed':False},ensure_ascii=False));raise SystemExit(2)
