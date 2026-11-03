"""Seeded space partition, graph height assignment and constrained ramp placement."""
from collections import Counter,deque
from copy import deepcopy
import hashlib,json,math,random
from pathlib import Path
import numpy as np
from terrain import CONFIG_DIR


def canonical(value):
    """Nanometre-rounded coordinates give stable layout hashes across NumPy builds."""
    if isinstance(value,float):return round(value,9)
    if isinstance(value,dict):return {k:canonical(v) for k,v in value.items()}
    if isinstance(value,list):return [canonical(v) for v in value]
    return value


def load_settings(path=None):return json.loads((Path(path) if path else CONFIG_DIR/'terraced_sampling.json').read_text())


def snap(value,step=.25):return round(value/step)*step


def partition(rng,length,width,count,settings):
    leaves=[[-length/2,length/2,-width/2,width/2]];tree=[];minimum=settings['minimum_region_side'];q=settings['dimension_snap']
    while len(leaves)<count:
        choices=[]
        for i,(a,b,c,d) in enumerate(leaves):
            for axis,lo,hi in [('x',a,b),('y',c,d)]:
                positions=[]
                for j in range(math.ceil(minimum/q),math.floor((hi-lo-minimum)/q)+1):
                    cut=lo+j*q
                    children=[[a,cut,c,d],[cut,b,c,d]] if axis=='x' else [[a,b,c,cut],[a,b,cut,d]]
                    if all(max(v[1]-v[0],v[3]-v[2])/min(v[1]-v[0],v[3]-v[2])<=settings['maximum_region_aspect'] for v in children):positions.append(cut)
                if positions:choices.append((i,axis,(b-a)*(d-c),positions))
        if not choices:raise ValueError('partition_capacity')
        i,axis,_,positions=rng.choices(choices,weights=[v[2] for v in choices])[0];a,b,c,d=leaves.pop(i);cut=rng.choice(positions)
        children=[[a,cut,c,d],[cut,b,c,d]] if axis=='x' else [[a,b,c,cut],[a,b,cut,d]]
        tree.append(dict(bounds=[a,b,c,d],axis=axis,cut=cut));leaves.extend(children)
    return [dict(id=f'region_{i:02d}',bounds=bounds) for i,bounds in enumerate(leaves)],tree


def adjacency(regions,minimum_width=2.5):
    edges=[]
    for i,p in enumerate(regions):
        a,b,c,d=p['bounds']
        for j in range(i+1,len(regions)):
            e,f,g,h=regions[j]['bounds'];edge=None
            if b==e or f==a:
                lo,hi=max(c,g),min(d,h)
                if hi-lo>=minimum_width:edge=dict(a=i,b=j,axis='x',boundary=b if b==e else a,span=[lo,hi])
            elif d==g or h==c:
                lo,hi=max(a,e),min(b,f)
                if hi-lo>=minimum_width:edge=dict(a=i,b=j,axis='y',boundary=d if d==g else c,span=[lo,hi])
            if edge:edges.append(edge)
    return edges


def distances(n,edges,roots):
    graph=[[] for _ in range(n)]
    for e in edges:graph[e['a']].append(e['b']);graph[e['b']].append(e['a'])
    result=[None]*n;queue=deque(roots)
    for root in roots:result[root]=0
    while queue:
        i=queue.popleft()
        for j in graph[i]:
            if result[j] is None:result[j]=result[i]+1;queue.append(j)
    if None in result:raise ValueError('partition_adjacency_disconnected')
    return result


def assign_heights(rng,regions,edges,settings):
    # Graph distance is used only to check that the partition is connected.
    distances(len(regions),edges,[0])
    from terrain.height_patterns import assign
    return assign(rng,regions,edges,adjacency(regions,1e-9),settings)


def overlap(a,b,margin=0.):return not (a[1]+margin<=b[0] or b[1]+margin<=a[0] or a[3]+margin<=b[2] or b[3]+margin<=a[2])


def reserved(r,landing):
    a,b,c,d=r['bounds'];return [a-landing,b+landing,c,d] if r['axis']=='x' else [a,b,c-landing,d+landing]


def propose_ramp(rng,e,regions,placed,settings):
    first,second=regions[e['a']],regions[e['b']];axis=e['axis'];boundary=e['boundary'];lo,hi=e['span'];edge_margin=settings['edge_margin'];landing=settings['landing_length']
    p,q=first,second
    if (p['bounds'][0] if axis=='x' else p['bounds'][2])>(q['bounds'][0] if axis=='x' else q['bounds'][2]):p,q=q,p
    delta=abs(p['height']-q['height']);minimum=max(settings['ramp_length_range'][0],delta/(settings['maximum_grade']*.84))
    for _ in range(70):
        quantum=settings['walk_grid_step'];width=snap(rng.uniform(*settings['ramp_width_range']),quantum)
        if not settings['ramp_width_range'][0]<=width<=settings['ramp_width_range'][1]:continue
        if hi-lo<width+2*edge_margin:continue
        if minimum>settings['ramp_length_range'][1]:return None
        length=snap(rng.uniform(minimum,settings['ramp_length_range'][1]),quantum)
        if not minimum-1e-9<=length<=settings['ramp_length_range'][1]+1e-9:continue
        if delta/(length*.84)>settings['maximum_grade']:continue
        split=snap(length*rng.uniform(.38,.62),quantum);other=length-split
        pa,pb,pc,pd=p['bounds'];qa,qb,qc,qd=q['bounds']
        if axis=='x':available0=boundary-pa;available1=qb-boundary
        else:available0=boundary-pc;available1=qd-boundary
        if min(split,other)<1 or split+landing>=available0-edge_margin or other+landing>=available1-edge_margin:continue
        lateral=lo+snap(rng.uniform(edge_margin,hi-lo-edge_margin-width),quantum)
        if lateral<lo+edge_margin or lateral+width>hi-edge_margin:continue
        centre=lateral+width/2
        bounds=[boundary-split,boundary+other,centre-width/2,centre+width/2] if axis=='x' else [centre-width/2,centre+width/2,boundary-split,boundary+other]
        candidate=dict(id=f'ramp_{len(placed):02d}',bounds=bounds,axis=axis,z0=p['height'],z1=q['height'],regions=[p['id'],q['id']])
        if any(overlap(reserved(candidate,landing),reserved(r,landing),settings['ramp_clearance']) for r in placed):continue
        return candidate
    return None


def connect(rng,regions,edges,settings):
    parent=list(range(len(regions)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):parent[find(a)]=find(b)
    for e in edges:
        if regions[e['a']]['height']==regions[e['b']]['height']:union(e['a'],e['b'])
    shuffled=edges[:];rng.shuffle(shuffled);ramps=[];used=[]
    for e in shuffled:
        if find(e['a'])==find(e['b']):continue
        r=propose_ramp(rng,e,regions,ramps,settings)
        if r:ramps.append(r);used.append(e);union(e['a'],e['b'])
    if len({find(i) for i in parent})!=1:raise ValueError('ramp_graph_disconnected')
    extras=rng.randint(*settings['extra_ramp_links_range'])
    for e in shuffled:
        if extras==0:break
        if e in used or regions[e['a']]['height']==regions[e['b']]['height']:continue
        r=propose_ramp(rng,e,regions,ramps,settings)
        if r:ramps.append(r);used.append(e);extras-=1
    return ramps


def context(rng,layout,report,settings):
    rects=report['flat_rectangles'];chosen=[]
    # Show every elevation before filling the remaining region examples.
    for level in layout['levels']:chosen.append(max((r for r in rects if r['height']==level),key=lambda r:r['area_m2']))
    rest=[r for r in rects if r not in chosen];rng.shuffle(rest);chosen+=rest
    count=max(settings['people_count'],len(rects));people=[];occupied=[]
    for i in range(count):
        r=chosen[i%len(chosen)];a,b,c,d=r['bounds']
        for _ in range(100):
            x=round(rng.uniform(a+.6,b-.6),3);y=round(rng.uniform(c+.6,d-.6),3)
            if all(np.hypot(x-u,y-v)>=1.25 for u,v in occupied):break
        else:raise ValueError('figure_clearance')
        people.append([x,y,round(rng.uniform(-180,180),3)]);occupied.append((x,y))
    ramps=layout['ramps'][:];rng.shuffle(ramps);rovers=[]
    for r in ramps[:settings['rover_count']]:
        a,b,c,d=r['bounds'];yaw=0 if r['axis']=='x' else 90
        if r['z1']<r['z0']:yaw+=180
        x,y=(a+b)/2,(c+d)/2;rovers.append([x,y,yaw]);occupied.append((x,y))
    props=[]
    for j in range(settings['prop_count']):
        for _ in range(100):
            r=rng.choice(rects);a,b,c,d=r['bounds'];sx=snap(rng.uniform(.5,1.2));sy=snap(rng.uniform(.5,1.8));sz=snap(rng.uniform(.7,1.6))
            if b-a<sx+.5 or d-c<sy+.5:continue
            x=round(rng.uniform(a+sx/2+.15,b-sx/2-.15),3);y=round(rng.uniform(c+sy/2+.15,d-sy/2-.15),3)
            if any(np.hypot(x-u,y-v)<math.hypot(sx,sy)/2+1.1 for u,v in occupied):continue
            props.append(dict(id=f'context_{j:02d}',center=[x,y],size=[sx,sy,sz]));occupied.append((x,y));break
    labels=[]
    for level in layout['levels']:
        r=max((p for p in rects if p['height']==level),key=lambda p:p['area_m2']);a,b,c,d=r['bounds'];labels.append(dict(height=level,position=[a+.25,c+.25]))
    x0,x1,y0,y1=layout['bounds'];entries=[]
    for r in layout['plateaus']:
        a,b,c,d=r['bounds'];z=r['height']
        if a==x0:entries.append(dict(side='west',point=[x0,(c+d)/2],inward=[1,0],height=z,span=d-c))
        if b==x1:entries.append(dict(side='east',point=[x1,(c+d)/2],inward=[-1,0],height=z,span=d-c))
        if c==y0:entries.append(dict(side='south',point=[(a+b)/2,y0],inward=[0,1],height=z,span=b-a))
        if d==y1:entries.append(dict(side='north',point=[(a+b)/2,y1],inward=[0,-1],height=z,span=b-a))
    entry=min(entries,key=lambda p:(p['height'],-p['span']))
    layout.update(people=people,rovers=rovers,props=props,level_labels=labels,camera_entry=entry)


def generate(seed,settings=None):
    settings=deepcopy(settings or load_settings());rng=random.Random(int(seed));failures=Counter()
    for key in ['length_range','width_range','ramp_width_range','ramp_length_range']:
        low,high=settings[key]
        if not (0<low<=high and math.isfinite(high)):raise ValueError(f'Invalid {key}')
    if settings['walk_grid_step']<=0 or settings['mesh_grid_step']<=0:raise ValueError('Grid steps must be positive')
    for ratio in [settings['walk_grid_step']/settings['mesh_grid_step'],settings['dimension_snap']/settings['walk_grid_step']]:
        if abs(ratio-round(ratio))>1e-9:raise ValueError('Geometry and analysis grids must align')
    if settings['minimum_region_side']<=0 or settings['maximum_attempts']<1:raise ValueError('Invalid partition constraints')
    if not settings['height_steps'] or min(settings['height_steps'])<=0:raise ValueError('Height steps must be positive')
    if settings['minimum_flat_fraction']<=0 or settings['minimum_flat_fraction']>=1:raise ValueError('minimum_flat_fraction must be between 0 and 1')
    if 'maximum_flat_fraction' in settings and not settings['minimum_flat_fraction']<=settings['maximum_flat_fraction']<1:raise ValueError('Flat fraction bounds must be ordered below one')
    if settings['maximum_grade']<=0:raise ValueError('maximum_grade must be positive')
    from terrain.validation import validate
    for attempt in range(1,settings['maximum_attempts']+1):
        try:
            q=settings['dimension_snap'];length=snap(rng.uniform(*settings['length_range']),q);width=snap(rng.uniform(*settings['width_range']),q)
            regions,tree=partition(rng,length,width,rng.randint(*settings['region_count_range']),settings)
            edges=adjacency(regions);levels,pattern=assign_heights(rng,regions,edges,settings);ramps=connect(rng,regions,edges,settings)
            layout=dict(title='Random connected terraced ground',title_ko=f'랜덤 단차 지면 · seed {seed}',bounds=[-length/2,length/2,-width/2,width/2],
                grid_step=settings['mesh_grid_step'],bottom_z=settings['bottom_z'],levels=levels,plateaus=regions,ramps=ramps,
                generation=dict(seed=int(seed),algorithm='bsp_graph_terraces_v3',settings=settings,partition_tree=tree,adjacency=edges,**pattern))
            report=validate(layout,settings)
            if report['status']!='passed':raise ValueError(next(c['name'] for c in report['checks'] if not c['passed']))
            context(rng,layout,report,settings)
            layout['generation'].update(attempt=attempt,rejections=dict(failures),flat_fraction=report['flat_fraction'],flat_rectangles=report['flat_rectangles'])
            fingerprint=dict(bounds=layout['bounds'],plateaus=regions,ramps=ramps)
            layout['generation']['layout_sha256']=hashlib.sha256(json.dumps(canonical(fingerprint),sort_keys=True).encode()).hexdigest()
            return canonical(layout)
        except ValueError as error:failures[str(error)]+=1
    raise RuntimeError(f'Unable to generate seed {seed} in {settings["maximum_attempts"]} attempts: {dict(failures)}')
