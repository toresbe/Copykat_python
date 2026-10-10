"""Paired accuracy, coverage, study bootstrap and scientific diagnostics only."""
import fcntl
import json
import os
from pathlib import Path
import time
from collections import Counter,defaultdict
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/home/toresbe/cancer_research/navin_accuracy_2026-10-10/scratch/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from navin_accuracy_data import ROOT,save
from navin_accuracy_worker import metrics

DOCS=Path(__file__).resolve().parent
lock=open(ROOT/'results/accuracy_report.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
assets=DOCS/'navin-accuracy-assets';assets.mkdir(exist_ok=True)
manifest_path=ROOT/'dataset_manifest.json'
manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(cases=[],excluded=[])
cases=manifest['cases'];rows=[];paired=[]
for case in cases:
    results={}
    for variant in ['main','gpu_anchor']:
        p=ROOT/'results'/(case['id']+'__'+variant+'.json')
        results[variant]=json.loads(p.read_text()) if p.exists() else dict(status='pending')
    row=dict(case=case,main_status=results['main']['status'],gpu_status=results['gpu_anchor']['status'])
    for v in ['main','gpu_anchor']:row[v]=results[v].get('metrics')
    if all(results[v]['status']=='ok' for v in results):
        a=np.load(ROOT/'results'/(case['id']+'__main.cells.npz'),allow_pickle=False)
        b=np.load(ROOT/'results'/(case['id']+'__gpu_anchor.cells.npz'),allow_pickle=False)
        if not np.array_equal(a['cells'],b['cells']) or not np.array_equal(a['truth'],b['truth']) or results['main']['input']['input_sha256']!=results['gpu_anchor']['input']['input_sha256']:
            row['comparison_error']='Input/cell alignment mismatch'
        else:
            truth=a['truth'];common=(a['calls']>=0)&(b['calls']>=0)
            masks=dict(all_input=np.ones(len(truth),dtype=bool),common_defined=common,
                       non_anchor=~b['reference'],non_marker=(b['immune_counts']<3)&(b['endothelial_counts']<3))
            row['paired_metrics']={stratum:dict(main=metrics(truth,a['calls'],mask),gpu_anchor=metrics(truth,b['calls'],mask)) for stratum,mask in masks.items()}
            row['common_defined_call_agreement']=float((a['calls'][common]==b['calls'][common]).mean()) if common.any() else None
            row['anchor_path']=results['gpu_anchor'].get('anchor_path')
            row['anchor_malignant_fraction']=results['gpu_anchor'].get('anchor_malignant_fraction')
            row['cell_type_errors']={}
            for typ in np.unique(a['cell_types']):
                mask=a['cell_types']==typ
                if mask.sum()<10:continue
                row['cell_type_errors'][str(typ)]={v:metrics(truth,z['calls'],mask) for v,z in [('main',a),('gpu_anchor',b)]}
            paired.append(row)
    rows.append(row)
save(ROOT/'results/paired_accuracy.json',rows)

def aggregate(stratum,metric):
    studies=defaultdict(list);samples=[]
    for row in paired:
        pair=row['paired_metrics'][stratum]
        a=pair['main'].get(metric);b=pair['gpu_anchor'].get(metric)
        if a is not None and b is not None:
            samples.append([a,b]);studies[row['case']['study']].append([a,b])
    if not studies:return dict(samples=0,studies=0)
    values=np.array([np.mean(studies[k],axis=0) for k in sorted(studies)])
    rng=np.random.default_rng(20261010)
    deltas=values[:,1]-values[:,0]
    boot=np.mean(deltas[rng.integers(0,len(values),size=(10000,len(values)))],axis=1)
    return dict(samples=len(samples),studies=len(studies),sample_mean=np.mean(samples,axis=0).tolist(),
                study_mean=values.mean(axis=0).tolist(),study_delta=float(deltas.mean()),
                study_delta_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),
                by_study={k:np.mean(studies[k],axis=0).tolist() for k in sorted(studies)})
summary={s:aggregate(s,'coverage_adjusted_balanced_recall') for s in ['all_input','non_anchor','non_marker']}
summary['common_defined']=aggregate('common_defined','balanced_accuracy_defined')
healthy=[r for r in paired if r['case']['family']=='healthy controls']
healthy_summary={}
if healthy:
    for variant in ['main','gpu_anchor']:
        healthy_summary[variant]=dict(samples=len(healthy),mean_specificity=float(np.mean([r['paired_metrics']['all_input'][variant]['coverage_adjusted_specificity'] for r in healthy])),
                                     mean_coverage=float(np.mean([r['paired_metrics']['all_input'][variant]['coverage'] for r in healthy])))
summary['healthy_only']=healthy_summary
save(ROOT/'results/accuracy_summary.json',summary)

def value(x):return '—' if x is None else f'{x:.3f}'
lines=['# Main versus GPU + marker anchoring: installed-data accuracy','',
       'Last refreshed: '+time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())+'.','',
       'This accuracy-only phase is queued after the serial performance repeats. '
       'No benchmark performance counters, timings or speed comparisons are collected. '
       'Reference: Navin Lab upstream main `ea1a15c`. Candidate: pinned `5401668`, '
       'GPU backend, B6 markers and F3 arm correlation, default Monte Carlo KS. '
       'Both use the same installed Python dependencies, four requested threads and '
       'identical raw inputs. Labels never enter the inference pipeline.','',
       'Author/OpenScPCA malignant labels are proxies for aneuploidy, not independent DNA '
       'ground truth. Many author annotations are partly CNA/marker-derived. These datasets '
       'were often used in prior development/evaluation; this is not a new sealed holdout. '
       'No method, anchor rule or threshold is tuned on these results.','',
       f"Selected {len(cases)} samples from {len({c['study'] for c in cases})} studies; {len(paired)} completed aligned pairs. "
       'The frozen selection takes up to two fraction-extreme eligible samples per 3CA study, '
       'all eligible converted ScPCA libraries, eligible README samples, and up to two healthy '
       'donors per count-valid h5ad. Whole selected samples are used, without cell subsampling. '
       'Selection uses only installed data/labels and frozen size/class criteria, never model results.','',
       '## Paired study-level results','',
       'Coverage-adjusted balanced recall = half of correct malignant calls / all labelled '
       'malignant input cells plus correct non-malignant calls / all labelled non-malignant '
       'input cells. Unknown labels are unscored; filtered/unclassified cells cannot inflate '
       'this measure. Conditional balanced accuracy and coverage are shown separately. '
       'Normal-only controls have specificity/false positives rather than balanced accuracy.','',
       '| Stratum | Paired samples / studies | Main study mean | GPU+anchor study mean | Paired delta | Study-bootstrap 95% interval |',
       '|---|---:|---:|---:|---:|---|']
for s in ['all_input','non_anchor','non_marker','common_defined']:
    d=summary[s]
    if not d['studies']:lines.append(f'| {s} | 0 / 0 | — | — | — | — |');continue
    ci=d['study_delta_bootstrap_95ci']
    lines.append(f"| {s} | {d['samples']} / {d['studies']} | {d['study_mean'][0]:.3f} | {d['study_mean'][1]:.3f} | {d['study_delta']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
lines+=['','Non-anchor excludes the candidate’s actual reference cells from BOTH implementations. '
        'Non-marker excludes immune/endothelial marker-positive cells from BOTH. Common-defined '
        'scores use cells classified by BOTH, so they can still conceal QC attrition; read them '
        'with coverage-adjusted scores. Bootstrap resamples studies 10,000 times with seed '
        '20261010; it does not remove annotation circularity or selection bias. Failures and '
        'missing pairs are not silently counted as successful results.','',
        '## Per-sample results','',
        '| Study / sample | Family; prior-use cohort | Main status | GPU status | Main / GPU conditional BA | Main / GPU coverage | Coverage-adjusted BA delta | Anchor path |',
        '|---|---|---|---|---|---|---:|---|']
for row in rows:
    c=row['case'];a=row['main'] or {};b=row['gpu_anchor'] or {}
    delta=None
    if row.get('paired_metrics'):
        p=row['paired_metrics']['all_input'];x=p['main']['coverage_adjusted_balanced_recall'];y=p['gpu_anchor']['coverage_adjusted_balanced_recall']
        if x is not None and y is not None:delta=y-x
    lines.append(f"| {c['study']} / {c['sample']} | {c['family']}; {c['cohort']} | {row['main_status']} | {row['gpu_status']} | "
                 +value(a.get('balanced_accuracy_defined'))+' / '+value(b.get('balanced_accuracy_defined'))+' | '
                 +value(a.get('coverage'))+' / '+value(b.get('coverage'))+' | '+value(delta)+' | '+str(row.get('anchor_path','—'))+' |')
lines+=['','## Healthy controls','', '| Implementation | Completed samples | Mean specificity including abstentions | Mean classified fraction |','|---|---:|---:|---:|']
for v,d in healthy_summary.items():lines.append(f"| {v} | {d['samples']} | {d['mean_specificity']:.3f} | {d['mean_coverage']:.3f} |")
reasons=Counter(e['reason'] for e in manifest['excluded'])
lines+=['','## Selection exclusions','', '| Reason | Records |','|---|---:|']
for reason,count in sorted(reasons.items()):lines.append(f'| {reason} | {count} |')
lines+=['','Each exclusion, source path and selected sample is retained in `dataset_manifest.json`. '
        'Unlabelled Xenium, mouse data, unsupported/non-UMI formats and unmapped DNA/RNA '
        'profiles are outside this cell-call panel; no genomic-profile accuracy claim is made. '
        'The cell-type error tables, paired per-cell calls, reference membership, full-precision '
        f'final CNA arm means and linkage matrices are under `{ROOT}/results`. '
        'Compact diagnostics use original-precision calculations; the candidate’s own F3 '
        'precision policy remains unchanged. Calculation I/O and caches use the SSD; '
        'the final evidence archive uses the NAS.']
# Summary figures: points represent studies, so giant studies cannot dominate.
d=summary['all_input']
if d.get('studies'):
    by=d['by_study'];values=np.array(list(by.values()))
    fig,axes=plt.subplots(1,2,figsize=(14,max(5,.19*len(values)+2)));axes[0].scatter(values[:,0],values[:,1],s=25,alpha=.75)
    axes[0].plot([0,1],[0,1],color='gray',linestyle='--');axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='Main study mean',ylabel='GPU + anchor study mean')
    diff=values[:,1]-values[:,0];order=np.argsort(diff)
    axes[1].barh(np.arange(len(order)),diff[order]);axes[1].set_yticks(np.arange(len(order)),np.array(list(by))[order],fontsize=8)
    axes[1].axvline(0,color='gray');axes[1].set_xlabel('Paired coverage-adjusted BA difference')
    fig.suptitle('Installed annotation accuracy — study means');fig.tight_layout();fig.savefig(assets/'study-accuracy.png',dpi=200);plt.close(fig)
    lines+=['','![Study-level paired accuracy](navin-accuracy-assets/study-accuracy.png)']
# Preset plus both extremes are descriptive figures, never used to change the method.
mixed=[r for r in paired if r['paired_metrics']['all_input']['main']['coverage_adjusted_balanced_recall'] is not None]
selected=[]
if mixed:
    def delta(r):return r['paired_metrics']['all_input']['gpu_anchor']['coverage_adjusted_balanced_recall']-r['paired_metrics']['all_input']['main']['coverage_adjusted_balanced_recall']
    selected=[min(mixed,key=delta),max(mixed,key=delta)]
    kidney=next((r for r in mixed if 'Bi2021_Kidney' in r['case']['study']),None)
    if kidney:selected.append(kidney)
for row in {r['case']['id']:r for r in selected}.values():
    case=row['case'];cid=case['id']
    output=assets/(cid+'.png')
    if output.exists():
        lines+=['',f"![CNA/tree diagnostics: {case['study']} / {case['sample']}](navin-accuracy-assets/{output.name})"];continue
    a=np.load(ROOT/'results'/(cid+'__main.cells.npz'),allow_pickle=False);b=np.load(ROOT/'results'/(cid+'__gpu_anchor.cells.npz'),allow_pickle=False)
    da=np.load(ROOT/'results'/(cid+'__main.diagnostics.npz'),allow_pickle=False);db=np.load(ROOT/'results'/(cid+'__gpu_anchor.diagnostics.npz'),allow_pickle=False)
    lookup={cell:i for i,cell in enumerate(a['cells'])}
    fig,axes=plt.subplots(3,2,figsize=(13,11))
    for column,(z,diag,title) in enumerate([(a,da,'Main'),(b,db,'GPU + anchor')]):
        Z=diag['Z'];n=len(diag['cells']);count=np.ones(2*n-1,dtype=int);known=np.zeros(2*n-1);tumour=np.zeros(2*n-1);called=np.zeros(2*n-1);aneuploid=np.zeros(2*n-1)
        ix=np.array([lookup[cell] for cell in diag['cells']]);known[:n]=z['truth'][ix]>=0;tumour[:n]=z['truth'][ix]==1;called[:n]=z['calls'][ix]>=0;aneuploid[:n]=z['calls'][ix]==1
        for i,merge in enumerate(Z):
            left,right=map(int,merge[:2]);node=n+i
            for arr in [count,known,tumour,called,aneuploid]:arr[node]=arr[left]+arr[right]
        def leaf(node):return f'n={count[node]}\nT={tumour[node]/max(known[node],1):.0%}\nA={aneuploid[node]/max(called[node],1):.0%}'
        dendrogram(Z,truncate_mode='lastp',p=12,leaf_label_func=leaf,leaf_rotation=90,leaf_font_size=7,ax=axes[0,column],color_threshold=0)
        axes[0,column].set_title(title+' final Ward tree');axes[0,column].set_ylabel('Merge height')
        m=row['paired_metrics']['common_defined']['main' if column==0 else 'gpu_anchor'];conf=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
        axes[1,column].imshow(conf,cmap='Blues');axes[1,column].set_xticks([0,1],['diploid','aneuploid']);axes[1,column].set_yticks([0,1],['non-malignant','malignant']);axes[1,column].set_title('Same jointly classified labelled cells')
        for i in range(2):
            for j in range(2):axes[1,column].text(j,i,str(conf[i,j]),ha='center',va='center')
    shared=np.intersect1d(da['cells'],db['cells']);shared=sorted(shared,key=lambda cell:(a['truth'][lookup[cell]],cell))
    if len(shared)>600:shared=np.array(shared)[np.linspace(0,len(shared)-1,600).astype(int)].tolist()
    arm_ids=np.intersect1d(da['arm_ids'],db['arm_ids'])
    values=[]
    for diag in [da,db]:
        ci={cell:i for i,cell in enumerate(diag['cells'])};ai={arm:i for i,arm in enumerate(diag['arm_ids'])}
        values.append(diag['arms'][np.ix_([ai[i] for i in arm_ids],[ci[cell] for cell in shared])])
    scale=max(.01,float(np.quantile(np.abs(np.concatenate([v.ravel() for v in values])),.995)))
    for column,value_matrix in enumerate(values):
        axes[2,column].imshow(value_matrix,aspect='auto',vmin=-scale,vmax=scale,cmap='RdBu_r',interpolation='nearest')
        axes[2,column].set_title('Final CNA arm means; identical matched cell order');axes[2,column].set_xlabel('Matched cells, sorted by truth then barcode');axes[2,column].set_ylabel('Chromosome arm')
    fig.suptitle(case['study']+' / '+case['sample']+' — descriptive whole-endpoint differences; T=truth fraction, A=called fraction')
    fig.tight_layout();fig.savefig(output,dpi=160);plt.close(fig)
    lines+=['',f"![CNA/tree diagnostics: {case['study']} / {case['sample']}](navin-accuracy-assets/{output.name})"]
lines+=['','Trees have independent leaf orders and are truncated to 12 aggregates. Candidate F3 '
        'calls are not determined by these final Ward trees. Heatmaps share color limits '
        'and matched cell order; displayed cells are deterministically thinned to at most '
        '600, while metrics and stored arm data use every cell. Extremes are selected only '
        'for descriptive diagnostics, not for changing any rule.']
target=DOCS/'navin-accuracy-results.md';temporary=target.with_suffix('.md.tmp');temporary.write_text('\n'.join(lines)+'\n');temporary.replace(target)
