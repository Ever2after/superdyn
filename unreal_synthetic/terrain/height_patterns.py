"""Independent high/low zones, repaired for ramp connectivity without a focus gradient."""
from functools import lru_cache
from itertools import combinations
import numpy as np


def components(regions,edges):
    neighbours=[set() for _ in regions]
    for e in edges:neighbours[e['a']].add(e['b']);neighbours[e['b']].add(e['a'])
    seen=set();groups=[]
    for i,p in enumerate(regions):
        if i in seen:continue
        group={i};seen.add(i);stack=[i]
        while stack:
            u=stack.pop()
            for v in neighbours[u]:
                if v not in seen and regions[v]['height']==p['height']:seen.add(v);group.add(v);stack.append(v)
        groups.append((group,set().union(*(neighbours[u] for u in group))-group))
    return groups


def planar_trend(regions,heights=None):
    areas=np.array([(p['bounds'][1]-p['bounds'][0])*(p['bounds'][3]-p['bounds'][2]) for p in regions],float)
    xy=np.array([[(p['bounds'][0]+p['bounds'][1])/2,(p['bounds'][2]+p['bounds'][3])/2] for p in regions])
    values=np.asarray(heights if heights is not None else [p['height'] for p in regions],float)
    design=np.c_[np.ones(len(regions)),xy];w=np.sqrt(areas);coefficient=np.linalg.lstsq(design*w[:,None],values*w,rcond=None)[0]
    total=float(np.sum(areas*(values-np.average(values,weights=areas))**2))
    return 0. if total<1e-12 else float(np.clip(1-np.sum(areas*(values-design@coefficient)**2)/total,0,1))


def metrics(regions,spatial_edges):
    heights=[p['height'] for p in regions];low=min(heights);high=max(heights);peaks=[];lows=[];high_groups=[];low_groups=[]
    for group,adjacent in components(regions,spatial_edges):
        h=heights[min(group)];outside=[heights[v] for v in adjacent];ids=sorted(regions[v]['id'] for v in group)
        if outside and h>low and all(v<=h for v in outside) and any(v<h for v in outside):peaks.append(ids)
        if outside and h<high and all(v>=h for v in outside) and any(v>h for v in outside):lows.append(ids)
        if h==high:high_groups.append(ids)
        if h==low:low_groups.append(ids)
    area=[(p['bounds'][1]-p['bounds'][0])*(p['bounds'][3]-p['bounds'][2]) for p in regions];total=sum(area)
    return dict(local_peak_components=len(peaks),local_low_components=len(lows),global_high_components=len(high_groups),global_low_components=len(low_groups),
        peak_regions=peaks,low_regions=lows,global_high_regions=high_groups,global_low_regions=low_groups,
        high_area_fraction=sum(a for a,z in zip(area,heights) if z==high)/total,low_area_fraction=sum(a for a,z in zip(area,heights) if z==low)/total,
        planar_trend_r2=planar_trend(regions))


@lru_cache(maxsize=256)
def permutations(bag):
    counts={v:bag.count(v) for v in sorted(set(bag))};out=[]
    def visit(prefix):
        if len(prefix)==len(bag):out.append(tuple(prefix));return
        for v in counts:
            if counts[v]:counts[v]-=1;visit(prefix+[v]);counts[v]+=1
    visit([]);return tuple(out)


def connected(tiers,edges,step,settings):
    n=len(tiers);parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for e in edges:
        delta=abs(tiers[e['a']]-tiers[e['b']])*step
        if delta and (e['span'][1]-e['span'][0]<settings['ramp_width_range'][0]+2*settings['edge_margin'] or delta>settings['maximum_grade']*.84*settings['ramp_length_range'][1]+1e-9):continue
        parent[find(e['a'])]=find(e['b'])
    return len({find(i) for i in range(n)})==1


def assign(rng,regions,edges,spatial_edges,settings):
    rule=settings['height_assignment'];n=len(regions)
    if rule['minimum_peak_components']!=2 or rule['minimum_low_components']!=2:raise ValueError('This height sampler requires two high and two low anchors')
    options={count:[step for step in settings['height_steps'] if settings['minimum_height_span']<=(count-1)*step<=settings['maximum_height']]
             for count in range(settings['level_count_range'][0],min(settings['level_count_range'][1],n-2)+1)}
    options={count:steps for count,steps in options.items() if steps}
    if not options:raise ValueError('insufficient_height_bands')
    count=rng.choice(list(options));step=rng.choice(options[count]);highest=count-1
    adjacent={frozenset((e['a'],e['b'])) for e in spatial_edges}
    pairs=[pair for pair in combinations(range(n),2) if frozenset(pair) not in adjacent]
    area=np.array([(p['bounds'][1]-p['bounds'][0])*(p['bounds'][3]-p['bounds'][2]) for p in regions]);total=area.sum()
    evaluated=0
    for trial in range(1,rule['anchor_search_attempts']+1):
        if not pairs:break
        low=rng.choice(pairs);choices=[p for p in pairs if not set(p)&set(low)]
        if not choices:continue
        high=rng.choice(choices)
        if min(area[list(low)].sum(),area[list(high)].sum())/total<rule['minimum_extreme_area_fraction']:continue
        free=[i for i in range(n) if i not in low+high]
        bag=list(range(1,highest))+[rng.randrange(1,highest) for _ in range(len(free)-(highest-1))];rng.shuffle(bag)
        targets=[None]*n
        for i in low:targets[i]=0
        for i in high:targets[i]=highest
        for i,v in zip(free,bag):targets[i]=v
        alternatives=list(permutations(tuple(sorted(bag))));rng.shuffle(alternatives)
        alternatives.sort(key=lambda p:sum(abs(a-b) for a,b in zip(p,bag)))
        for middle in alternatives:
            evaluated+=1
            if evaluated>rule['assignment_search_limit']:raise ValueError('mixed_height_search_limit')
            tiers=targets[:]
            for i,v in zip(free,middle):tiers[i]=v
            if not connected(tiers,edges,step,settings):continue
            if planar_trend(regions,tiers)>rule['maximum_planar_trend_r2']:continue
            for region,tier in zip(regions,tiers):region['height']=tier*step
            result=metrics(regions,spatial_edges)
            if result['global_high_components']<2 or result['global_low_components']<2:continue
            return sorted(set(p['height'] for p in regions)),dict(height_step=step,bands=count,pattern='mixed_extrema',
                height_assignment='independent_extrema_with_connectivity_repair',low_anchors=list(low),high_anchors=list(high),
                sampled_tiers=targets,assigned_tiers=tiers,height_repair_changed_regions=sum(a!=b for a,b in zip(targets,tiers)),
                height_anchor_trials=trial,height_assignment_candidates=evaluated,height_metrics=result)
    raise ValueError('mixed_height_assignment_infeasible')
