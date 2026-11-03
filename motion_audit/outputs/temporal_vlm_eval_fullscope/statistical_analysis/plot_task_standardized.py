"""Export model-by-cohort accuracy under the same fixed task distribution."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
analysis = json.loads((OUT/'analysis.json').read_text())
protocol = analysis['protocol']
models = protocol['models']
tasks = list(protocol['global_task_weights'])
weights = np.array(list(protocol['global_task_weights'].values()))
groups = ['LL','LH','HL','HH']
names = ['Qwen2.5-VL\n3B','Qwen2.5-VL\n7B','VG-LLM\n4B','VG-LLM\n8B',
         'MLLM-4D\n8B','4D-RGPT\n8B','DSR Suite\n7B backbone']
colors = ['#5279B6','#43A89B','#D7A34E','#CB6268']
labels = ['LL  |  camera low, dynamics low','LH  |  camera low, dynamics high',
          'HL  |  camera high, dynamics low','HH  |  camera high, dynamics high']
qs = json.loads((ROOT/'inputs.json').read_text())
video_names, video_indices = np.unique([q['video_id'] for q in qs],return_inverse=True)
canonical = {g:key for key, gs in protocol['groups'].items() for g in gs}
cells = np.array([groups.index(canonical[q['group']])*6+tasks.index(q['task']) for q in qs])
counts = np.bincount(cells,minlength=24)
y=[]
for q in qs:
    row=[]
    for model in models:
        pred=json.loads((ROOT/'predictions'/model/(q['qa_id']+'.json')).read_text())
        assert pred['status']=='ok' and pred['video_id']==q['video_id']
        row.append(float(pred['correct']))
    y.append(row)
y=np.array(y)
sums=np.zeros((24,len(models)))
np.add.at(sums,cells,y)
cell_means=sums/counts[:,None]
scores=np.zeros((len(video_names),24,len(models)))
np.add.at(scores,(video_indices,cells),(y-cell_means[cells])/counts[cells,None])
factor=len(video_names)/(len(video_names)-1)*(len(qs)-1)/(len(qs)-24)
estimates=np.zeros((4,len(models)))
ses=np.zeros_like(estimates)
for gi,g in enumerate(groups):
    estimates[gi]=weights@cell_means[gi*6:gi*6+6]
    influence=np.einsum('vcm,c->vm',scores[:,gi*6:gi*6+6],weights)
    ses[gi]=np.sqrt((influence**2).sum(axis=0)*factor)
    expected=[analysis['groups'][g]['standardized_accuracy'][m] for m in models]
    np.testing.assert_allclose(estimates[gi],expected,atol=1e-12,rtol=0)

chance_by_task={task:1/len(next(q for q in qs if q['task']==task)['options']) for task in tasks}
chance=sum(weights[i]*chance_by_task[task] for i,task in enumerate(tasks))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.spines.left':False,'axes.spines.bottom':False})
fig=plt.figure(figsize=(17,10.8),facecolor='white')
ax=fig.add_axes([.065,.39,.91,.40])
fig.text(.065,.956,'Model accuracy with a common task distribution',fontsize=24,weight='bold',color='#182b43')
fig.text(.065,.916,'Same task weights in every group: the full 2,387-QA task mix. All existing QA retained; no subsampling.',
         fontsize=12,color='#56677c')
handles=[Patch(facecolor=c,label=l) for c,l in zip(colors,labels)]
fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.055,.887),ncol=2,frameon=False,
           fontsize=11,columnspacing=2.4,handlelength=1.3)
x=np.arange(len(models));width=.18
for gi,(g,c) in enumerate(zip(groups,colors)):
    offset=(gi-1.5)*width
    ax.bar(x+offset,100*estimates[gi],width=width*.89,color=c,zorder=3,
           yerr=100*1.96*ses[gi],error_kw=dict(ecolor='#35455a',elinewidth=1.0,capsize=2,capthick=1.0))
ax.axhline(chance*100,color='#7c8795',linestyle=(0,(5,4)),linewidth=1.4,zorder=2)
ax.text(6.48,chance*100+.7,f'Random {chance*100:.2f}%',ha='right',fontsize=10,color='#667385',
        bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
ax.set_xticks(x,names,fontsize=12)
ax.tick_params(axis='x',length=0,pad=13)
ax.tick_params(axis='y',length=0)
ax.set_ylabel('Task-standardized accuracy (%)',labelpad=15)
ax.set_ylim(0,85);ax.set_yticks(np.arange(0,81,10));ax.set_xlim(-.6,6.6)
ax.set_axisbelow(True);ax.yaxis.grid(True,color='#e5eaf0',linewidth=.8)
fig.text(.065,.314,'Accuracy (%)',fontsize=12,weight='bold',color='#182b43')
table_ax=fig.add_axes([.065,.15,.91,.145]);table_ax.axis('off')
table=table_ax.table(cellText=[[f'{100*a:.2f}' for a in row] for row in estimates],
                    rowLabels=groups,colLabels=[n.replace('\n',' ') for n in names],
                    cellLoc='center',rowLoc='center',bbox=[.025,0,.975,1])
table.auto_set_font_size(False);table.set_fontsize(10)
for (row,col),cell in table.get_celld().items():
    cell.set_edgecolor('white');cell.set_linewidth(2)
    if row==0:
        cell.set_facecolor('#edf2f7');cell.set_text_props(weight='bold',color='#182b43')
    else:
        cell.set_facecolor(colors[row-1]+'14')
        cell.set_text_props(color=colors[row-1],weight='bold' if col==-1 else 'normal')
fig.text(.065,.112,'Task weights T1-T6: '+ ' / '.join(f'{100*w:.2f}%' for w in weights),fontsize=10.5,color='#56677c')
fig.text(.065,.084,'Bars: sum of (common task weight x within-group task accuracy). Error bars: video-cluster-robust 95% CIs.',fontsize=10.5,color='#56677c')
fig.text(.065,.057,'Groups use camera motion x scene dynamics. Camera high = translation high OR rotation high. LL / LH / HL / HH QA: 176 / 848 / 406 / 957.',fontsize=9.5,color='#56677c')
fig.text(.065,.031,'HL has only 3 T5 and 6 T6 QA; its estimates are less precise. References are automatic geometry annotations. DSR includes spatial modules.',fontsize=9.5,color='#56677c')
fig.savefig(OUT/'task_standardized_model_comparison.png',dpi=200,facecolor='white')
fig.savefig(OUT/'task_standardized_model_comparison.pdf',facecolor='white')
plt.close(fig)
rows=[]
for gi,g in enumerate(groups):
    for mi,model in enumerate(models):
        a,s=100*estimates[gi,mi],100*ses[gi,mi]
        rows.append(dict(group=g,model=model,accuracy_percent=a,se_pp=s,ci95_low_percent=a-1.96*s,
                         ci95_high_percent=a+1.96*s,n_qa=analysis['groups'][g]['n_qa'],
                         n_videos=analysis['groups'][g]['n_videos'],random_accuracy_percent=chance*100))
with (OUT/'task_standardized_model_comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
validation=dict(passed=True,n_models=len(models),n_groups=4,n_qa=len(qs),n_videos=len(video_names),
                common_weights=protocol['global_task_weights'],matches_previous_analysis=True,
                uncertainty=protocol['uncertainty'],qa_subsampled=False,
                common_random_accuracy_percent=chance*100)
(OUT/'task_standardized_plot_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(validation,indent=2))
print('Saved PNG, PDF and CSV in',OUT)
