"""Clearance-aware paths on the actual compact terrain; renderer-independent."""
import heapq,math
from collections import deque
import numpy as np
from terrain.layout import height
from terrain.validation import raster,clearance_mask,validate


def grid_path(z,safe,start,goal,step,grade):
    queue=[(0.,0.,start)];cost={start:0.};parent={start:None};h,w=z.shape
    while queue:
        _,g,u=heapq.heappop(queue)
        if g>cost[u]+1e-9:continue
        if u==goal:
            path=[u]
            while parent[path[-1]] is not None:path.append(parent[path[-1]])
            return path[::-1]
        y,x=u
        for dy,dx in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            v=(y+dy,x+dx);yy,xx=v
            if not (0<=yy<h and 0<=xx<w and safe[v]):continue
            length=math.hypot(dx,dy)*step;dz=abs(z[v]-z[u])
            if dz>length*grade+1e-7:continue
            if dx and dy and (not safe[y,xx] or not safe[yy,x] or abs(z[y,xx]-z[u])>step*grade+1e-7 or abs(z[yy,x]-z[u])>step*grade+1e-7):continue
            new=g+math.hypot(length,dz)
            if new>=cost.get(v,float('inf')):continue
            parent[v]=u;cost[v]=new;heuristic=math.hypot(xx-goal[1],yy-goal[0])*step
            heapq.heappush(queue,(new+heuristic,new,v))
    return None


def point_index(xy,bounds,step,shape):
    return (min(shape[0]-1,max(0,int((xy[1]-bounds[2])/step))),min(shape[1]-1,max(0,int((xy[0]-bounds[0])/step))))


def shortcut(points,layout,safe,step,grade):
    # Preserve the clearance mask along every shortcut, including at ramp corners.
    def valid(a,b):
        n=max(2,math.ceil(np.linalg.norm(b-a)/.10));delta=b-a;ts=list(np.linspace(0,1,n+1))
        # Visit every cell the line crosses, even a tiny corner between samples.
        for axis,origin in [(0,layout['bounds'][0]),(1,layout['bounds'][2])]:
            if abs(delta[axis])<1e-12:continue
            lo,hi=sorted([a[axis],b[axis]])
            for k in range(math.floor((lo-origin)/step)+1,math.floor((hi-origin)/step)+1):
                t=(origin+k*step-a[axis])/delta[axis]
                if 0<t<1:ts.append(t)
        ts=np.unique(ts);ts=np.sort(np.r_[ts,(ts[:-1]+ts[1:])/2]);ps=a+ts[:,None]*delta;zs=[]
        for p in ps:
            if not safe[point_index(p,layout['bounds'],step,safe.shape)]:return False
            zs.append(height(layout,*p))
        return bool(np.all(np.abs(np.diff(zs))<=np.linalg.norm(b-a)*np.diff(ts)*grade+1e-6))
    out=[points[0]];i=0
    while i<len(points)-1:
        j=len(points)-1
        while j>i+1 and not valid(points[i],points[j]):j-=1
        out.append(points[j]);i=j
    return np.array(out)


def resample_path(points,layout,frames,phase=0.,amplitude=.26):
    dense=[]
    for a,b in zip(points,points[1:]):
        dense.extend(np.linspace(a,b,max(2,math.ceil(np.linalg.norm(b-a)/.08)),endpoint=False))
    dense.append(points[-1]);xy=np.array(dense);z=np.array([height(layout,*p) for p in xy]);xyz=np.c_[xy,z]
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))];s=np.linspace(0,1,frames)
    # A monotone time warp changes speed without reversing progress.
    u=s-amplitude*(np.sin(2*np.pi*s+phase)-math.sin(phase))/(2*np.pi)
    sampled=np.column_stack([np.interp(u*distance[-1],distance,xyz[:,i]) for i in range(3)])
    sampled[:,2]=[height(layout,*p) for p in sampled[:,:2]]
    return sampled,float(distance[-1])


def build_plan(layout,theme,frames=192,fps=24):
    report=validate(layout,raise_on_failure=True);settings=layout['generation']['settings'];step=settings['walk_grid_step'];xs,ys,z,_,_=raster(layout)
    safe=clearance_mask(z,step,settings['maximum_grade'],.70)
    rects=report['flat_rectangles'];centres=[]
    for r in rects:
        a,b,c,d=r['bounds'];p=np.array([(a+b)/2,(c+d)/2]);idx=point_index(p,layout['bounds'],step,z.shape)
        if safe[idx]:centres.append((idx,r['height']))
    candidates=[]
    for i,(a,za) in enumerate(centres):
        for b,zb in centres[i+1:]:
            if abs(za-zb)<1.:continue
            route=grid_path(z,safe,a,b,step,settings['maximum_grade'])
            if not route:continue
            points=shortcut(np.array([[xs[x],ys[y]] for y,x in route]),layout,safe,step,settings['maximum_grade'])
            sampled,length=resample_path(points,layout,frames)
            if not 7<=length<=25 or min(np.ptp(sampled[:,:2],axis=0))<1.25:continue
            span=float(np.ptp(sampled[:,2]));candidates.append((span,length,points,sampled))
    if len(candidates)<2:raise ValueError('Not enough distinct terrain routes')
    candidates.sort(key=lambda p:(-p[0],abs(p[1]-15)))
    variants=[]
    for candidate in candidates:
        for reverse in [True,False]:
            points=candidate[2][::-1] if reverse else candidate[2]
            sampled,length=resample_path(points,layout,frames,phase=1.3,amplitude=.32)
            variants.append((candidate,points,sampled,length))
    best=None
    for a in candidates:
        for b,points,second,length in variants:
            separation=np.linalg.norm(a[3]-second,axis=1).min()
            if separation<1.8:continue
            score=min(a[0],b[0])*10+min(separation,4)-.15*(abs(a[1]-15)+abs(length-15))
            if best is None or score>best[0]:best=(score,a,points,second,length,float(separation))
    if best is None:raise ValueError('No collision-free pair of terrain trajectories')
    _,a,bpoints,b,length,separation=best;positions={'person':a[3],'ground':b};routes={'person':a[2].tolist(),'ground':bpoints.tolist()}
    t=np.linspace(0,1,frames);peak=max(layout['levels']);air=np.c_[4.8*np.sin(1.2*np.pi*t+.3),4.1*np.sin(1.8*np.pi*t+1.),peak+3.2+1.5*np.sin(2*np.pi*t-.4)]
    positions['air']=air;duration=frames/fps
    return dict(theme=theme['id'],seed=layout['generation']['seed'],frames=frames,fps=fps,duration_seconds=duration,
        ground_routes=routes,positions={k:v.tolist() for k,v in positions.items()},route_lengths_m={'person':a[1],'ground':length},
        minimum_ground_actor_separation_m=separation,ground_navigation_clearance_radius_m=.70,
        vertical_spans_m={k:float(np.ptp(v[:,2])) for k,v in positions.items()},
        timing='monotone sinusoidal speed warp; no physics integration',camera=dict(start_degrees=-132,sweep_degrees=68))


def heading_series(positions):
    p=np.asarray(positions,float);delta=np.empty_like(p)
    for i in range(len(p)):delta[i]=p[min(i+3,len(p)-1)]-p[max(i-3,0)]
    return np.unwrap(np.arctan2(delta[:,1],delta[:,0]))
