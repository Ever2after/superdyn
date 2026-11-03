"""A contiguous solid ground made from broad plateaus and connecting ramps."""
import json
from pathlib import Path
import numpy as np
from terrain import CONFIG_DIR
from terrain.surfaces import ramp_blend


def load():return json.loads((CONFIG_DIR/'terraced_draft.json').read_text())


def contains(bounds,x,y):
    x0,x1,y0,y1=bounds
    return x0<=x<x1 and y0<=y<y1


def patch_at(cfg,x,y):
    selected=dict(id='lower_plaza',height=0.)
    for patch in cfg['plateaus']+cfg['ramps']:
        if contains(patch['bounds'],x,y):selected=patch
    return selected


def patch_height(patch,x,y):
    if 'axis' not in patch:return float(patch['height'])
    x0,x1,y0,y1=patch['bounds'];u=(x-x0)/(x1-x0) if patch['axis']=='x' else (y-y0)/(y1-y0)
    return float(patch['z0']+(patch['z1']-patch['z0'])*ramp_blend(u))


def height(cfg,x,y):
    x0,x1,y0,y1=cfg['bounds'];step=cfg['grid_step']
    if not (x0<=x<=x1 and y0<=y<=y1):raise ValueError('Outside draft ground')
    i=min(round((x1-x0)/step)-1,max(0,int(np.floor((x-x0)/step))))
    j=min(round((y1-y0)/step)-1,max(0,int(np.floor((y-y0)/step))))
    a=x0+i*step;b=y0+j*step;p=patch_at(cfg,a+step/2,b+step/2)
    u=(x-a)/step;v=(y-b)/step
    z00=patch_height(p,a,b);z10=patch_height(p,a+step,b);z11=patch_height(p,a+step,b+step);z01=patch_height(p,a,b+step)
    return float(z00*(1-u)+z10*(u-v)+z11*v if u>=v else z00*(1-v)+z11*u+z01*(v-u))


def terrain_mesh(cfg):
    """Only exposed top/step faces and bottom: no boxes floating over a base."""
    x0,x1,y0,y1=cfg['bounds'];step=cfg['grid_step'];nx=round((x1-x0)/step);ny=round((y1-y0)/step)
    cells={};counts={};vertices=[];faces=[];roles=[];index={};bottom=cfg['bottom_z']
    def vertex(p):
        key=tuple(round(float(v),9) for v in p)
        if key not in index:index[key]=len(vertices);vertices.append(key)
        return index[key]
    def face(points,role):
        ids=[vertex(p) for p in points];ids=[p for k,p in enumerate(ids) if k==0 or p!=ids[k-1]]
        if len(ids)>2 and ids[0]==ids[-1]:ids.pop()
        if len(set(ids))>=3:faces.append(ids);roles.append(role)
    def wall(a,b,za,zb,na,nb):
        da=za-na;db=zb-nb
        if max(abs(da),abs(db))<1e-8:return
        if da*db < -1e-15:
            t=da/(da-db);c=np.array(a)+(np.array(b)-a)*t;z=za+(zb-za)*t
            wall(a,c,za,z,na,z);wall(c,b,z,zb,z,nb);return
        if da+db>=0:face([(*a,na),(*b,nb),(*b,zb),(*a,za)],'wall')
        else:face([(*a,za),(*b,zb),(*b,nb),(*a,na)],'wall')
    for i in range(nx):
        for j in range(ny):
            x=x0+i*step;y=y0+j*step;p=patch_at(cfg,x+step/2,y+step/2)
            z=[patch_height(p,x,y),patch_height(p,x+step,y),patch_height(p,x+step,y+step),patch_height(p,x,y+step)]
            cells[i,j]=z;counts[p['id']]=counts.get(p['id'],0)+1
            a=(x,y,z[0]);b=(x+step,y,z[1]);c=(x+step,y+step,z[2]);d=(x,y+step,z[3])
            face([a,b,c],'ramp' if 'axis' in p else 'flat');face([a,c,d],'ramp' if 'axis' in p else 'flat')
            face([(x,y,bottom),(x,y+step,bottom),(x+step,y+step,bottom),(x+step,y,bottom)],'bottom')
    for (i,j),z in cells.items():
        x=x0+i*step;y=y0+j*step
        if i+1<nx:n=cells[i+1,j];wall((x+step,y),(x+step,y+step),z[1],z[2],n[0],n[3])
        else:wall((x+step,y),(x+step,y+step),z[1],z[2],bottom,bottom)
        if j+1<ny:n=cells[i,j+1];wall((x+step,y+step),(x,y+step),z[2],z[3],n[1],n[0])
        else:wall((x+step,y+step),(x,y+step),z[2],z[3],bottom,bottom)
        if i==0:wall((x,y+step),(x,y),z[3],z[0],bottom,bottom)
        if j==0:wall((x,y),(x+step,y),z[0],z[1],bottom,bottom)
    ramps={p['id'] for p in cfg['ramps']};flat=sum(n for key,n in counts.items() if key not in ramps)*step**2
    return vertices,faces,roles,dict(area_m2=(x1-x0)*(y1-y0),flat_area_m2=flat,flat_fraction=flat/((x1-x0)*(y1-y0)),patch_area_m2={key:n*step**2 for key,n in counts.items()})


def ramp_checks(cfg):
    checks=[]
    for r in cfg['ramps']:
        x0,x1,y0,y1=r['bounds'];length=x1-x0 if r['axis']=='x' else y1-y0
        a=np.array([x0,(y0+y1)/2]) if r['axis']=='x' else np.array([(x0+x1)/2,y0])
        b=np.array([x1,(y0+y1)/2]) if r['axis']=='x' else np.array([(x0+x1)/2,y1])
        direction=(b-a)/np.linalg.norm(b-a);epsilon=1e-5
        before=a-epsilon*direction;after=b+epsilon*direction
        errors=[abs(height(cfg,*before)-r['z0']),abs(height(cfg,*after)-r['z1'])]
        for lateral in np.linspace(.01,.99,7):
            if r['axis']=='x':points=[(x0-epsilon,y0+(y1-y0)*lateral),(x1+epsilon,y0+(y1-y0)*lateral)]
            else:points=[(x0+(x1-x0)*lateral,y0-epsilon),(x0+(x1-x0)*lateral,y1+epsilon)]
            errors.extend([abs(height(cfg,*points[0])-r['z0']),abs(height(cfg,*points[1])-r['z1'])])
        checks.append(dict(id=r['id'],start=a.tolist(),end=b.tolist(),z0=r['z0'],z1=r['z1'],join_error_m=max(errors),
                           maximum_grade=abs(r['z1']-r['z0'])/(length*(1-.16)),passed=max(errors)<1e-5))
    return checks
