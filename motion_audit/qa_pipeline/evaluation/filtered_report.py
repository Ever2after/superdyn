"""Rescore a fixed evaluation on QA IDs retained by a separate quality audit.

No model inference, image modification, reference rewriting or HTML generation.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

GROUPS = ['LL','LH','HL','HH']
GROUP4 = {'LL':'LLL','LH':'LLH','HL':'HLL+LHL+HHL','HH':'HLH+LHH+HHH'}
COLORS = ['#5279B6','#43A89B','#D7A34E','#CB6268']

def read(path):
    return json.loads(path.read_text())

def save(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def csv_save(path,rows):
    if not rows:return
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def chance(examples):
    return float(np.mean([1/len(e['options']) for e in examples])) if examples else None

def grouped_accuracy(values,cell,n_cells):
    counts=np.bincount(cell,minlength=n_cells)
    if not np.all(counts>0):raise ValueError('Task standardization is not identifiable: empty task cell')
    sums=np.zeros((n_cells,values.shape[1]));np.add.at(sums,cell,values)
    return sums/counts[:,None],counts

def standardized_groups(examples,values,tasks,weights):
    group_index=np.array([GROUPS.index(next(g for g,canonical in GROUP4.items() if canonical==e['group4'])) for e in examples])
    task_index=np.array([tasks.index(e['task']) for e in examples])
    cells=group_index*len(tasks)+task_index
    mean,n=grouped_accuracy(values,cells,len(GROUPS)*len(tasks))
    videos,vi=np.unique([e['video_id'] for e in examples],return_inverse=True)
    influence=np.zeros((len(videos),24,values.shape[1]))
    np.add.at(influence,(vi,cells),(values-mean[cells])/n[cells,None])
    factor=len(videos)/(len(videos)-1)*(len(examples)-1)/(len(examples)-24)
    estimates=[];ses=[]
    for gi in range(4):
        estimates.append(weights@mean[gi*6:(gi+1)*6])
        score=np.einsum('vcm,c->vm',influence[:,gi*6:(gi+1)*6],weights)
        ses.append(np.sqrt((score**2).sum(axis=0)*factor))
    return np.array(estimates),np.array(ses),n.reshape(4,6)

def plot_comparison(out,rows,old_chance,new_chance,weights,before_info,after_info):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,axes=plt.subplots(2,1,figsize=(16,10.5),facecolor='white')
    fig.suptitle('Baseline accuracy before and after QA filtering',fontsize=23,weight='bold',y=.98)
    fig.text(.5,.935,f'Before: {before_info[0]:,} QA / {before_info[1]:,} videos    |    Keep only: {after_info[0]:,} QA / {after_info[1]:,} videos',ha='center',color='#56677c')
    x=np.arange(len(rows));width=.34
    names=[r['model'].replace(' (7B backbone)','\n7B backbone').replace(' SFT','') for r in rows]
    for ax,field,title in zip(axes,['after_percent','after_fixed_task_mix_percent'],
                              ['Observed QA composition','Same task weights before and after (original dataset mix)']):
        before=[r['before_percent'] for r in rows];after=[r[field] for r in rows]
        for pos,vals,color,label in [(x-width/2,before,'#8c9eb6','Before filtering'),
                                     (x+width/2,after,'#287d9a','Keep only')]:
            bars=ax.bar(pos,vals,width,color=color,label=label,zorder=3)
            ax.bar_label(bars,labels=[f'{v:.2f}' for v in vals],padding=3,fontsize=9)
        random_label='Random before' if field=='after_percent' else 'Random (common mix)'
        ax.axhline(100*old_chance,color='#b07b3d',ls='--',lw=1.2,label=f'{random_label}: {old_chance*100:.2f}%')
        if field=='after_percent':
            ax.axhline(100*new_chance,color='#4e8f7b',ls=':',lw=1.3,label=f'Random keep: {new_chance*100:.2f}%')
        ax.set_title(title,fontsize=13,pad=12);ax.set_xticks(x,names,fontsize=9.5)
        ax.set_ylim(0,72);ax.set_ylabel('Accuracy (%)');ax.set_axisbelow(True);ax.yaxis.grid(color='#e6ebf1')
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(length=0)
        ax.legend(loc='upper right',ncol=2,fontsize=9,frameon=False)
    fig.text(.04,.047,'T1-T6 common weights: '+' / '.join(f'{100*w:.2f}%' for w in weights),fontsize=10,color='#56677c')
    fig.text(.04,.023,'Original predictions rescored on identical inputs and automatic references. Only audit decision = keep is retained.',fontsize=10,color='#56677c')
    fig.subplots_adjust(left=.07,right=.98,top=.865,bottom=.12,hspace=.43)
    for ext in ['png','pdf']:fig.savefig(out/f'before_after.{ext}',dpi=180)
    plt.close(fig)

def plot_standardized(out,examples,models,labels,estimates,ses,weights,counts,random_accuracy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    names=[labels[m].replace(' (7B backbone)','\n7B backbone').replace(' SFT','') for m in models]
    fig=plt.figure(figsize=(17,10.8),facecolor='white');ax=fig.add_axes([.065,.39,.91,.40])
    fig.text(.065,.957,'QA-filtered accuracy with a common task distribution',fontsize=23,weight='bold',color='#182b43')
    fig.text(.065,.915,f'{len(examples):,} keep QA / {len({e["video_id"] for e in examples}):,} videos. All groups use the original 2,387-QA task mix.',fontsize=12,color='#56677c')
    handles=[Patch(facecolor=c,label=f'{g}: camera {"high" if g[0]=="H" else "low"}, dynamics {"high" if g[1]=="H" else "low"}') for g,c in zip(GROUPS,COLORS)]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.055,.887),ncol=2,frameon=False,fontsize=11,columnspacing=2.4)
    x=np.arange(len(models));width=.18
    for gi,color in enumerate(COLORS):
        ax.bar(x+(gi-1.5)*width,100*estimates[gi],width=width*.89,color=color,zorder=3,
               yerr=100*1.96*ses[gi],error_kw=dict(ecolor='#35455a',elinewidth=1,capsize=2))
    ax.axhline(100*random_accuracy,color='#7c8795',ls='--',lw=1.3)
    ax.text(6.48,100*random_accuracy+.8,f'Random {random_accuracy*100:.2f}%',ha='right',fontsize=10,
            color='#667385',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
    ax.set_xticks(x,names,fontsize=10.5);ax.tick_params(axis='x',length=0,pad=13)
    ax.set_ylabel('Task-standardized accuracy (%)',labelpad=15);ax.set_ylim(0,85);ax.set_yticks(np.arange(0,81,10));ax.set_xlim(-.6,6.6)
    ax.set_axisbelow(True);ax.yaxis.grid(color='#e5eaf0',lw=.8);ax.spines[['top','right','left','bottom']].set_visible(False)
    fig.text(.065,.314,'Accuracy (%)',fontsize=12,weight='bold',color='#182b43')
    ta=fig.add_axes([.065,.15,.91,.145]);ta.axis('off')
    table=ta.table(cellText=[[f'{100*v:.2f}' for v in row] for row in estimates],rowLabels=GROUPS,
                   colLabels=[n.replace('\n',' ') for n in names],cellLoc='center',rowLoc='center',bbox=[.025,0,.975,1])
    table.auto_set_font_size(False);table.set_fontsize(9.5)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('white');cell.set_linewidth(2)
        if row==0:cell.set_facecolor('#edf2f7');cell.set_text_props(weight='bold',color='#182b43')
        else:
            cell.set_facecolor(COLORS[row-1]+'14');cell.set_text_props(color=COLORS[row-1],weight='bold' if col==-1 else 'normal')
    fig.text(.065,.112,'T1-T6 common weights: '+' / '.join(f'{100*w:.2f}%' for w in weights),fontsize=10.5,color='#56677c')
    fig.text(.065,.084,'Error bars: video-cluster-robust 95% CIs. Camera high = translation high OR rotation high.',fontsize=10.5,color='#56677c')
    fig.text(.065,.057,'Retained LL / LH / HL / HH QA: '+' / '.join(str(sum(e['group4']==GROUP4[g] for e in examples)) for g in GROUPS)+'.',fontsize=10,color='#56677c')
    fig.text(.065,.031,f'HL has only {counts[2,4]} T5 and {counts[2,5]} T6 QA. Estimates for sparse cells are less precise. References are automatic geometry labels.',fontsize=9.5,color='#56677c')
    for ext in ['png','pdf']:fig.savefig(out/f'task_standardized_groups.{ext}',dpi=180)
    plt.close(fig)

def run(source,filter_path,out):
    source=source.resolve();filter_path=filter_path.resolve();out=out.resolve()
    if out==source or source.is_relative_to(out):raise ValueError('Use a separate subset output directory')
    if (out/'manifest.json').exists():raise ValueError('Cannot overwrite a frozen benchmark; use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    from .runner import parse_answer,stats,stamp
    from .reports import draw,TASKS,LABELS as default_labels
    LABELS={m['key']:m.get('label',default_labels.get(m['key'],m['key'])) for m in read(source/'models.json')}
    parent=read(source/'summary.json');assert parent['status']=='completed'
    full=read(source/'inputs.json');refs=read(source/'reference_answers.json');tasks=list(TASKS)
    audit=read(filter_path);assert len({r['qa_id'] for r in audit})==len(audit)
    assert {r['qa_id'] for r in audit}=={e['qa_id'] for e in full}
    assert all(r['status']=='ok' and r['decision'] in {'keep','exclude','review'} for r in audit)
    filter_config=read(filter_path.parent/'config.json')
    assert filter_config['inputs_sha256']==digest(source/'inputs.json')
    keep={r['qa_id'] for r in audit if r['decision']=='keep'}
    examples=[e for e in full if e['qa_id'] in keep];mask=np.array([e['qa_id'] in keep for e in full])
    if not examples:raise ValueError('No keep QA')
    protected=[source/n for n in ['inputs.json','reference_answers.json','summary.json','leaderboard.csv','models.json','protocol.json']]+[filter_path]
    hashes={str(p):digest(p) for p in protected}
    models=list(LABELS);weights=np.array([sum(e['task']==t for e in full)/len(full) for t in tasks])
    old_chance=chance(full);new_chance=chance(examples)
    chance_tasks=np.array([chance([e for e in examples if e['task']==t]) for t in tasks])
    standard_chance=float(weights@chance_tasks)
    summary=dict(status='completed',created_at=stamp(),source_evaluation=str(source),filter_results=str(filter_path),
                 filter_prompt_version=filter_config['prompt_version'],retained_decisions=['keep'],
                 source_qa=len(full),qa=len(examples),videos=len({e['video_id'] for e in examples}),
                 filter_counts=dict(collections.Counter(r['decision'] for r in audit)),models={})
    summary['standardization']=dict(task_weights=dict(zip(tasks,weights.tolist())),
                                    reference='Original full evaluation task composition; unchanged from the pre-filter graph',
                                    uncertainty='CR1 cluster robust normal 95% CI; clustering unit = video')
    leaderboard=[];task_rows=[];group4_rows=[];group8_rows=[];family_rows=[];before_after=[];pred_manifest=hashlib.sha256()
    yfull=np.zeros((len(full),len(models)))
    for mi,key in enumerate(models):
        predictions=[]
        for i,e in enumerate(full):
            path=source/'predictions'/key/f'{e["qa_id"]}.json';blob=path.read_bytes();p=json.loads(blob)
            pred_manifest.update(str(path).encode());pred_manifest.update(hashlib.sha256(blob).digest())
            assert p['qa_id']==e['qa_id'] and p['model']==key and p['video_id']==e['video_id'] and p['task']==e['task']
            assert p['status']=='ok' and p['prompt_sha256']==e['prompt_sha256'] and p['group']==e['group']
            assert p['usage']['frames']==9
            choice=parse_answer(p['raw_output'],e['options'])
            assert choice==p['predicted_option_index'] and bool(p['parsed'])==(choice is not None)
            correct=choice is not None and choice==refs[e['qa_id']]['answer_index']
            assert correct==p['correct'];yfull[i,mi]=correct;predictions.append(p)
        original=stats(predictions,len(full))
        for field in ['correct','invalid','errors','processed']:assert original[field]==parent['models'][key][field]
        rows=[p for p in predictions if p['qa_id'] in keep];removed=[p for p in predictions if p['qa_id'] not in keep]
        model=stats(rows,len(examples));assert model['correct']+sum(p['correct'] for p in removed)==original['correct']
        assert model['invalid']+sum(not p['parsed'] for p in removed)==original['invalid']
        for field,name,values in [('task','per_task',tasks),('group4','per_group4',list(GROUP4.values())),('group','per_group8',sorted({e['group'] for e in full}))]:
            model[name]={v:stats([p for p in rows if p[field]==v],sum(e[field]==v for e in examples)) for v in values}
        model['task_macro_accuracy']=float(np.mean([s['accuracy'] for s in model['per_task'].values()]))
        model['status']='completed';summary['models'][key]=model
        task_acc=np.array([model['per_task'][t]['accuracy'] for t in tasks]);fixed=float(weights@task_acc)
        leaderboard.append(dict(model=LABELS[key],key=key,correct=model['correct'],total=len(examples),
                                accuracy_percent=100*model['accuracy'],before_percent=100*original['accuracy'],
                                delta_pp=100*(model['accuracy']-original['accuracy']),
                                task_macro_percent=100*model['task_macro_accuracy'],random_percent=100*new_chance,
                                gap_vs_random_pp=100*(model['accuracy']-new_chance),invalid=model['invalid'],errors=model['errors']))
        task_rows.extend(dict(model=key,task=t,**v) for t,v in model['per_task'].items())
        group4_rows.extend(dict(model=key,group=g,**v) for g,v in model['per_group4'].items())
        group8_rows.extend(dict(model=key,group=g,**v) for g,v in model['per_group8'].items())
        for family,ts in [('tasks_1_2',tasks[:2]),('tasks_3_4_5_6',tasks[2:])]:
            selected=[p for p in rows if p['task'] in ts]
            family_rows.append(dict(model=key,family=family,**stats(selected),random_accuracy=chance([e for e in examples if e['task'] in ts])))
        before_after.append(dict(model=LABELS[key],key=key,before_percent=100*original['accuracy'],after_percent=100*model['accuracy'],
                                 delta_pp=100*(model['accuracy']-original['accuracy']),after_fixed_task_mix_percent=100*fixed,
                                 fixed_task_mix_delta_pp=100*(fixed-original['accuracy'])))
        print(key,model['correct'],'/',len(examples),f'{100*model["accuracy"]:.2f}%',flush=True)
    retained_y=yfull[mask]
    est,se,cell_counts=standardized_groups(examples,retained_y,tasks,weights)
    std_rows=[]
    for gi,g in enumerate(GROUPS):
        for mi,key in enumerate(models):
            a,s=100*est[gi,mi],100*se[gi,mi]
            std_rows.append(dict(model=key,group=g,accuracy_percent=a,se_pp=s,ci95_low_percent=a-1.96*s,
                                 ci95_high_percent=a+1.96*s,random_percent=100*standard_chance))
    for filename,rows in [('leaderboard.csv',leaderboard),('per_task.csv',task_rows),('per_group4.csv',group4_rows),
                          ('per_group8.csv',group8_rows),('per_task_family.csv',family_rows),('before_after.csv',before_after),
                          ('task_standardized_groups.csv',std_rows)]:csv_save(out/filename,rows)
    csv_save(out/'accuracy_by_group.csv',[dict(model=LABELS[key],**{g:100*summary['models'][key]['per_group4'][canonical]['accuracy'] for g,canonical in GROUP4.items()},overall=100*summary['models'][key]['accuracy']) for key in models])
    csv_save(out/'accuracy_by_task.csv',[dict(model=LABELS[key],**{f'task_{i+1}':100*summary['models'][key]['per_task'][t]['accuracy'] for i,t in enumerate(tasks)},overall=100*summary['models'][key]['accuracy']) for key in models])
    save(out/'summary.json',summary)
    save(out/'selection.json',dict(qa_ids=sorted(keep),qa=len(keep),videos=summary['videos'],filter_results=str(filter_path),
                                   retained_decisions=['keep'],filter_counts=summary['filter_counts'],
                                   task_qa_counts=dict(collections.Counter(e['task'] for e in examples)),
                                   group4_qa_counts=dict(collections.Counter(e['group4'] for e in examples)),
                                   group8_qa_counts=dict(collections.Counter(e['group'] for e in examples))))
    save(out/'random_baseline.json',dict(overall=new_chance,before_filter=old_chance,
                                       per_task=dict(zip(tasks,chance_tasks.tolist())),
                                       per_group4={g:chance([e for e in examples if e['group4']==g]) for g in GROUP4.values()},
                                       standardized_common_task_mix=standard_chance))
    save(out/'models.json',read(source/'models.json'))
    draw(summary,leaderboard,examples,output=out,model_labels=LABELS,title='Temporal Geometry: QA-filtered Evaluation',
         groups=list(GROUP4.values()),group_labels={canonical:g for g,canonical in GROUP4.items()},
         group_title='Accuracy by camera motion / scene dynamics',save_pdf=True,
         footer_extra='Keep only from Qwen3.8 QA-conditioned audit; exclude/review removed. LL/LH/HL/HH = camera motion / scene dynamics.')
    plot_comparison(out,before_after,old_chance,new_chance,weights,
                    (len(full),len({e['video_id'] for e in full})),(len(examples),summary['videos']))
    plot_standardized(out,examples,models,LABELS,est,se,weights,cell_counts,standard_chance)
    assert all(digest(Path(path))==value for path,value in hashes.items()),'Original data changed during reporting'
    save(out/'validation.json',dict(status='passed',checked_at=stamp(),qa=len(examples),videos=summary['videos'],
                                   models=len(models),retained_predictions=len(examples)*len(models),
                                   all_parent_predictions_reparsed=len(full)*len(models),
                                   removed_qa=len(full)-len(examples),original_data_hashes_unchanged=hashes,
                                   parsed_prediction_manifest_sha256=pred_manifest.hexdigest(),
                                   task_group_cell_counts=cell_counts.tolist(),new_inference=False,
                                   checks=['exact keep-ID coverage','same QA across models','all raw responses reparsed and rescored',
                                           'retained + removed correct/invalid counts equal original statistics',
                                           'source/filter hashes unchanged','nonempty task x cohort cells','video-cluster uncertainty'],
                                   code_sha256=digest(Path(__file__))))
    (out/'README.txt').write_text(f'''QA-filtered evaluation: keep only

{len(examples)} QA / {summary['videos']} videos; {len(full)-len(examples)} exclude/review QA omitted.
Source audit: {filter_path}
Source predictions: {source}/predictions
Existing responses are reparsed and rescored against the same automatic reference
answers. No new inference. Original source QA, predictions and full reports are preserved.
Invalid model answers remain wrong; all models use exactly the same retained QA.

results.png: overall accuracy, task accuracy, and raw LL/LH/HL/HH accuracy.
before_after.png/pdf: observed and fixed-task-mix comparisons before/after filtering.
task_standardized_groups.png/pdf: common task mix across four cohorts, with
video-cluster-robust normal 95% confidence intervals. Common weights remain the
original 2,387-QA task distribution, matching the previous unfiltered graph.
HL contains only {cell_counts[2,4]} T5 and {cell_counts[2,5]} T6 questions; interpret sparse cells cautiously.

leaderboard.csv, per_task.csv, per_group4.csv, per_group8.csv,
per_task_family.csv (tasks 1-2 versus 3-6), accuracy_by_group.csv (overall at right),
accuracy_by_task.csv, before_after.csv and task_standardized_groups.csv.
Raw random expectation changes from {100*old_chance:.4f}% to {100*new_chance:.4f}%.
References are model-estimated geometry labels. Filtering is an automatic Qwen
quality audit, not a human verification of reference correctness. No HTML generated.
''')
    print(json.dumps(dict(output=str(out),qa=len(examples),videos=summary['videos'],random_percent=100*new_chance,
                         leaderboard=leaderboard),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--filter-results',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.evaluation,args.filter_results,args.out)
