"""Reproducible descriptive and video-cluster uncertainty analysis of fixed baselines."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import norm

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
inputs = json.loads((ROOT / 'inputs.json').read_text())
models = [m['key'] for m in json.loads((ROOT / 'models.json').read_text())]
tasks = ['camera_translation_direction', 'camera_view_direction_change',
         'object_world_motion_direction', 'camera_object_distance_change',
         'object_pair_distance_change', 'object_travel_distance_comparison']
groups = ['LL', 'LH', 'HL', 'HH']
def group(q):
    return ('H' if 'H' in q['group'][:2] else 'L') + q['group'][2]

ys = []
for q in inputs:
    row = []
    for model in models:
        pred = json.loads((ROOT / 'predictions' / model / (q['qa_id'] + '.json')).read_text())
        assert pred['qa_id'] == q['qa_id'] and pred['video_id'] == q['video_id']
        assert pred['status'] == 'ok' and pred['group'] == q['group']
        row.append(float(pred['correct']))
    ys.append(row)
y = np.asarray(ys)
y = np.column_stack([y, y.mean(axis=1)])
labels = models + ['fixed_seven_model_mean']
g = np.array([groups.index(group(q)) for q in inputs])
t = np.array([tasks.index(q['task']) for q in inputs])
video_names, v = np.unique([q['video_id'] for q in inputs], return_inverse=True)
chance = np.array([1 / len(q['options']) for q in inputs])
global_task_weights = np.bincount(t, minlength=6) / len(t)
equal_task_weights = np.ones(6) / 6

def estimate(values, cell_index, n_cells):
    """Saturated cell means and CR1 covariance, clustering all QA of each video.

    Models are fixed columns; mean-model inference resamples videos, not models.
    """
    counts = np.bincount(cell_index, minlength=n_cells)
    sums = np.zeros((n_cells, values.shape[1]))
    np.add.at(sums, cell_index, values)
    beta = sums / counts[:, None]
    residual = values - beta[cell_index]
    scores = np.zeros((len(video_names), n_cells, values.shape[1]))
    np.add.at(scores, (v, cell_index), residual / counts[cell_index, None])
    n, k, clusters = len(values), n_cells, len(video_names)
    factor = clusters / (clusters - 1) * (n - 1) / (n - k)
    covariance = np.einsum('vcm,vdm->mcd', scores, scores) * factor
    return beta, covariance, counts

def contrast(beta, cov, weights):
    delta = weights @ beta
    se = np.sqrt(np.maximum(0, np.einsum('c,mcd,d->m', weights, cov, weights)))
    return {label: dict(difference_pp=float(100*d), se_pp=float(100*s),
                        ci95_pp=[float(100*(d-1.96*s)), float(100*(d+1.96*s))],
                        two_sided_p=float(2 * norm.sf(abs(d/s))) if s else None)
            for label, d, s in zip(labels, delta, se)}

raw_beta, raw_cov, _ = estimate(y, g, 4)
cell_beta, cell_cov, cell_n = estimate(y, g*6+t, 24)
summary = dict(
    protocol=dict(n_qa=len(inputs), n_videos=len(video_names), models=models,
                  groups={'LL': ['LLL'], 'LH': ['LLH'], 'HL': ['HLL','LHL','HHL'],
                          'HH': ['HLH','LHH','HHH']},
                  uncertainty='CR1 cluster robust normal 95% CI; independent unit = video',
                  model_average='Equal average of seven fixed model correctness indicators per QA; models are not resampled',
                  task_standardization='Direct standardization of 24 group x task means using common weights',
                  global_task_weights=dict(zip(tasks, global_task_weights.tolist())),
                  equal_task_weights=dict(zip(tasks, equal_task_weights.tolist())),
                  significance='Two sided unadjusted exploratory p values; no random assignment or causal identification'),
    groups={}, comparisons={}, task_cell_mean_accuracy={}, source_counts={})
for i, name in enumerate(groups):
    idx = g == i
    summary['groups'][name] = dict(n_qa=int(idx.sum()), n_videos=len(set(v[idx])),
        chance_accuracy=float(chance[idx].mean()),
        raw_accuracy=dict(zip(labels, raw_beta[i].tolist())),
        standardized_accuracy=dict(zip(labels, (global_task_weights @ cell_beta[i*6:(i+1)*6]).tolist())),
        macro_task_accuracy=dict(zip(labels, (equal_task_weights @ cell_beta[i*6:(i+1)*6]).tolist())),
        task_counts=dict(zip(tasks, cell_n[i*6:(i+1)*6].tolist())))
    summary['task_cell_mean_accuracy'][name] = dict(zip(tasks, cell_beta[i*6:(i+1)*6, -1].tolist()))
    summary['source_counts'][name] = dict(Counter(q['video_id'].split('__')[0] for q in inputs if group(q) == name))
    if i == 3:
        continue
    raw_weights = np.zeros(4); raw_weights[3] = 1; raw_weights[i] = -1
    contrasts = {'raw': contrast(raw_beta, raw_cov, raw_weights)}
    for label, task_weights in [('task_standardized', global_task_weights), ('macro_task', equal_task_weights)]:
        weights = np.zeros(24); weights[18:24] = task_weights; weights[i*6:(i+1)*6] = -task_weights
        contrasts[label] = contrast(cell_beta, cell_cov, weights)
    summary['comparisons']['HH_minus_' + name] = contrasts

# A separate pooled comparator permits direct standardization without rare HL task cells.
h = (g == 3).astype(int)
pooled_beta, pooled_cov, _ = estimate(y, h*6+t, 12)
pooled_raw_beta, pooled_raw_cov, _ = estimate(y, h, 2)
rw = np.array([-1.,1.])
summary['comparisons']['HH_minus_all_other'] = {'raw': contrast(pooled_raw_beta, pooled_raw_cov, rw)}
for label, tw in [('task_standardized', global_task_weights), ('macro_task', equal_task_weights)]:
    summary['comparisons']['HH_minus_all_other'][label] = contrast(pooled_beta, pooled_cov, np.r_[-tw,tw])

# Nonparametric cluster bootstrap cross-check for HH versus pooled remainder.
# Each replicate samples complete videos with replacement separately in the two arms.
rng = np.random.default_rng(20261022)
B = 10000
boot = []
for arm in [0, 1]:
    vids = np.unique(v[h == arm])
    counts = np.zeros((len(vids), 6))
    sums = np.zeros((len(vids), 6, y.shape[1]))
    for j, vid in enumerate(vids):
        mask = v == vid
        assert np.all(h[mask] == arm)
        np.add.at(counts[j], t[mask], 1)
        np.add.at(sums[j], t[mask], y[mask])
    probabilities = np.full(len(vids), 1/len(vids))
    all_raw, all_std = [], []
    for start in range(0, B, 200):
        multiplicities = rng.multinomial(len(vids), probabilities, size=min(200, B-start))
        denominator = multiplicities @ counts
        numerator = (multiplicities @ sums.reshape(len(vids), -1)).reshape(-1,6,y.shape[1])
        assert np.all(denominator > 0)
        all_raw.append(numerator.sum(axis=1) / denominator.sum(axis=1)[:,None])
        all_std.append(np.einsum('btm,t->bm', numerator / denominator[:,:,None], global_task_weights))
    boot.append((np.concatenate(all_raw), np.concatenate(all_std)))
summary['bootstrap_HH_minus_all_other'] = dict(seed=20261022, replicates=B, intervals={})
for k, method in enumerate(['raw','task_standardized']):
    delta = 100*(boot[1][k] - boot[0][k])
    summary['bootstrap_HH_minus_all_other']['intervals'][method] = {
        label: dict(ci95_pp=np.quantile(delta[:,j],[.025,.975]).tolist())
        for j,label in enumerate(labels)}

(OUT / 'analysis.json').write_text(json.dumps(summary, indent=2)+'\n')
print('GROUPS (seven-model mean)')
for name, row in summary['groups'].items():
    print(name, 'QA',row['n_qa'],'video',row['n_videos'], 'raw',round(100*row['raw_accuracy'][labels[-1]],3),
          'chance',round(100*row['chance_accuracy'],3),'standardized',round(100*row['standardized_accuracy'][labels[-1]],3),
          'macro',round(100*row['macro_task_accuracy'][labels[-1]],3))
for comp, row in summary['comparisons'].items():
    print(comp)
    for method, vals in row.items():
        print(' ',method, vals[labels[-1]])
print('MODEL HH vs rest')
for label in labels:
    print(label, {method: vals[label] for method,vals in summary['comparisons']['HH_minus_all_other'].items()})
print('TASK CELLS',summary['task_cell_mean_accuracy'])
print('BOOTSTRAP mean', {m: row[labels[-1]] for m,row in summary['bootstrap_HH_minus_all_other']['intervals'].items()})
print('SOURCES',summary['source_counts'])
