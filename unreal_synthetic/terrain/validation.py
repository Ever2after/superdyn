"""Pure geometry checks for randomly partitioned, ramp-connected ground."""
from collections import deque
import numpy as np
from terrain.surfaces import ramp_blend
from terrain.layout import ramp_checks


def raster(layout,step=None):
    step=step or layout['generation']['settings']['walk_grid_step'];x0,x1,y0,y1=layout['bounds']
    xs=x0+(np.arange(round((x1-x0)/step))+.5)*step;ys=y0+(np.arange(round((y1-y0)/step))+.5)*step
    x,y=np.meshgrid(xs,ys);z=np.zeros_like(x);regions=np.full(x.shape,-1,int);ramp_ids=np.full(x.shape,-1,int);coverage=np.zeros(x.shape,np.int16)
    for i,p in enumerate(layout['plateaus']):
        a,b,c,d=p['bounds'];mask=(x>=a)&(x<b)&(y>=c)&(y<d);z[mask]=p['height'];regions[mask]=i;coverage[mask]+=1
    if (regions<0).any():raise ValueError('Partition leaves uncovered ground')
    if (coverage>1).any():raise ValueError('Partition overlaps regions')
    for i,p in enumerate(layout['ramps']):
        a,b,c,d=p['bounds'];mask=(x>=a)&(x<b)&(y>=c)&(y<d)
        u=(x-a)/(b-a) if p['axis']=='x' else (y-c)/(d-c)
        z=np.where(mask,p['z0']+(p['z1']-p['z0'])*ramp_blend(u),z);ramp_ids[mask]=i
    return xs,ys,z,regions,ramp_ids


def largest_flat_rectangle(mask,step,x0,y0,minimum_side):
    histogram=np.zeros(mask.shape[1],int);best=None;minimum_cells=int(np.ceil(minimum_side/step-1e-9))
    for row,values in enumerate(mask):
        histogram=np.where(values,histogram+1,0);stack=[]
        for col in range(len(histogram)+1):
            h=histogram[col] if col<len(histogram) else 0;left=col
            while stack and stack[-1][1]>h:
                start,height=stack.pop();width=col-start;left=start
                if height>=minimum_cells and width>=minimum_cells:
                    area=height*width*step**2
                    if best is None or area>best['area_m2']:
                        best=dict(bounds=[x0+start*step,x0+col*step,y0+(row-height+1)*step,y0+(row+1)*step],area_m2=float(area))
            if not stack or stack[-1][1]<h:stack.append((left,int(h)))
    return best


def reachable(z,step,maximum_grade,allowed=None,start=(0,0)):
    h,w=z.shape;seen=np.zeros((h,w),bool);queue=deque([start]);seen[start]=True;limit=step*maximum_grade+2e-7
    while queue:
        y,x=queue.popleft()
        for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=yy<h and 0<=xx<w and not seen[yy,xx] and (allowed is None or allowed[yy,xx]) and abs(z[yy,xx]-z[y,x])<=limit:
                seen[yy,xx]=True;queue.append((yy,xx))
    return seen


def clearance_mask(z,step,maximum_grade,radius):
    """Conservative disc erosion around cliffs and the outside ground boundary."""
    hazard=np.zeros(z.shape,bool);limit=step*maximum_grade+2e-7
    cliffs=np.abs(np.diff(z,axis=0))>limit;hazard[:-1]|=cliffs;hazard[1:]|=cliffs
    cliffs=np.abs(np.diff(z,axis=1))>limit;hazard[:,:-1]|=cliffs;hazard[:,1:]|=cliffs
    hazard[[0,-1],:]=True;hazard[:,[0,-1]]=True;blocked=hazard.copy();h,w=z.shape;n=int(np.ceil(radius/step))
    for dy in range(-n,n+1):
        for dx in range(-n,n+1):
            if np.hypot(dx,dy)*step>radius+1e-9:continue
            blocked[max(0,dy):min(h,h+dy),max(0,dx):min(w,w+dx)] |= hazard[max(0,-dy):min(h,h-dy),max(0,-dx):min(w,w-dx)]
    return ~blocked


def validate(layout,settings=None,*,raise_on_failure=False):
    settings=settings or layout['generation']['settings'];step=settings['walk_grid_step'];xs,ys,z,regions,ramp_ids=raster(layout,step)
    checks=[]
    def check(name,passed,evidence):checks.append(dict(name=name,passed=bool(passed),evidence=evidence))
    check('all_cells_have_ground',bool((regions>=0).all()),int(z.size))
    fraction=float(np.mean(ramp_ids<0));check('sufficient_flat_ground',fraction>=settings['minimum_flat_fraction'],fraction)
    if 'maximum_flat_fraction' in settings:
        check('sufficient_ramp_coverage',fraction<=settings['maximum_flat_fraction']+1e-9,dict(flat_fraction=fraction,ramp_fraction=1-fraction,maximum_flat_fraction=settings['maximum_flat_fraction']))
    rects=[];x0,_,y0,_=layout['bounds']
    for i,p in enumerate(layout['plateaus']):
        best=largest_flat_rectangle((regions==i)&(ramp_ids<0),step,x0,y0,settings['minimum_flat_rectangle_side'])
        rects.append(dict(region=p['id'],height=p['height'],**(best or dict(bounds=[],area_m2=0.))))
    check('every_region_has_broad_flat_rectangle',all(r['area_m2']>=settings['minimum_flat_rectangle_area'] for r in rects),rects)
    connections=ramp_checks(layout)
    check('ramp_ends_join_across_full_width',all(c['passed'] for c in connections),connections)
    check('ramp_grades_are_bounded',all(c['maximum_grade']<=settings['maximum_grade']+1e-9 for c in connections),[c['maximum_grade'] for c in connections])
    if all(c['passed'] for c in checks):
        seen=reachable(z,step,settings['maximum_grade']);coverage=float(seen.mean())
    else:coverage=0.
    check('entire_walkable_surface_is_connected',coverage==1.,dict(fraction=coverage,cells=int(z.size),step_m=step))
    radius=settings.get('navigation_clearance_radius',0.)
    if coverage==1. and radius>0:
        safe=clearance_mask(z,step,settings['maximum_grade'],radius);probes=[]
        for r in rects:
            a,b,c,d=r['bounds'];probes.append(((a+b)/2,(c+d)/2))
        for r in layout['ramps']:
            a,b,c,d=r['bounds'];probes.append(((a+b)/2,(c+d)/2))
        indexes=[(min(len(ys)-1,max(0,int((y-y0)/step))),min(len(xs)-1,max(0,int((x-x0)/step)))) for x,y in probes]
        if all(safe[point] for point in indexes):
            navigable=reachable(z,step,settings['maximum_grade'],safe,indexes[0]);connected=all(navigable[point] for point in indexes)
        else:connected=False
        check('region_centres_and_ramps_connect_with_clearance',connected,dict(radius_m=radius,probes=len(indexes),safe_area_fraction=float(safe.mean())))
    # Landing aprons must remain flat even after all other ramps are inserted.
    from terrain.layout import height
    errors=[];apron=settings['landing_length']
    for r in layout['ramps']:
        a,b,c,d=r['bounds']
        for t in np.linspace(.02,.98,5):
            if r['axis']=='x':pts=[(a-apron,c+(d-c)*t),(b+apron,c+(d-c)*t)]
            else:pts=[(a+(b-a)*t,c-apron),(a+(b-a)*t,d+apron)]
            try:errors.extend([abs(height(layout,*pts[0])-r['z0']),abs(height(layout,*pts[1])-r['z1'])])
            except ValueError:errors.append(1e6)
    check('full_width_flat_landing_aprons',max(errors,default=0)<1e-5,max(errors,default=0))
    check('multiple_height_levels',len(set(layout['levels']))>=settings['level_count_range'][0] and max(layout['levels'])>=settings['minimum_height_span'],layout['levels'])
    height_metrics={}
    rule=settings.get('height_assignment',{})
    if rule.get('mode')=='mixed_extrema':
        from terrain.sampling import adjacency
        from terrain.height_patterns import metrics
        height_metrics=metrics(layout['plateaus'],adjacency(layout['plateaus'],1e-9))
        check('multiple_separated_high_and_low_zones',
            height_metrics['global_high_components']>=rule['minimum_peak_components'] and height_metrics['global_low_components']>=rule['minimum_low_components'],height_metrics)
        check('extreme_zones_have_meaningful_area',min(height_metrics['high_area_fraction'],height_metrics['low_area_fraction'])>=rule['minimum_extreme_area_fraction']-1e-9,
            {k:height_metrics[k] for k in ('high_area_fraction','low_area_fraction')})
        check('height_distribution_is_not_a_dominant_plane',height_metrics['planar_trend_r2']<=rule['maximum_planar_trend_r2']+1e-9,height_metrics['planar_trend_r2'])
    report=dict(status='passed' if all(c['passed'] for c in checks) else 'failed',checks=checks,flat_fraction=fraction,flat_rectangles=rects,
                height_metrics=height_metrics,ramp_connections=connections,walkable_grid_cells=int(z.size),ground_area_m2=(layout['bounds'][1]-x0)*(layout['bounds'][3]-y0))
    if report['status']=='failed' and raise_on_failure:raise ValueError(', '.join(c['name'] for c in checks if not c['passed']))
    return report
