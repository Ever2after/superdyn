"""Plan A/B-only dressing and terrain-following first-person routes."""
import heapq,json,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/ab_walkthroughs'
class Ground:
 def __init__(self):
  data=json.loads((ROOT/'inputs/ab_walkthroughs/height_grid.json').read_text());self.h={(x,y):z for x,y,z in data['heights']}
 def z(self,x,y):
  x/=12.5;y/=12.5;i=math.floor(x);j=math.floor(y);u=x-i;v=y-j
  vals=[self.h.get((i,j)),self.h.get((i+1,j)),self.h.get((i,j+1)),self.h.get((i+1,j+1))]
  if any(a is None for a in vals):return self.h.get((round(x),round(y)),0)
  return (1-u)*(1-v)*vals[0]+u*(1-v)*vals[1]+(1-u)*v*vals[2]+u*v*vals[3]

ANCHORS=[(500,900),(650,500),(700,-250),(500,-780),(700,-250),(300,250),(100,100),(100,-400),(0,-650),(-800,-650),(-900,-400),(-900,200),(-900,650),(-50,650),(650,650),(500,900)]
def obstacle(record):
 if record.get('overhead'):return None
 x,y=record['anchor_cm'];a,b=record['half_xy_cm']
 if record['category']=='tree':a=b=38
 return (x,y,a,b)

def route_points(ground,anchors,objects=()):
 boxes=[b for r in objects if (b:=obstacle(r))]
 safe_cache={}
 def free(n):
  x,y=n[0]*25,n[1]*25
  if not (abs(x)<=1075 and abs(y)<=925):return False
  if n not in safe_cache:
   z=ground.z(x,y);safe_cache[n]=all(abs(ground.z(x+dx,y+dy)-z)<35 for dx in [-25,0,25] for dy in [-25,0,25])
  return safe_cache[n] and not any(abs(x-a)<c+42 and abs(y-b)<d+42 for a,b,c,d in boxes)
 def nearest(p):
  n=(round(p[0]/25),round(p[1]/25));candidates=[(n[0]+a,n[1]+b) for a in range(-12,13) for b in range(-12,13)]
  return min((q for q in candidates if free(q)),key=lambda q:math.dist(q,n))
 def segment(a,b):
  a,b=nearest(a),nearest(b);queue=[(0,0,a)];prev={};cost={a:0}
  while queue:
   _,g,n=heapq.heappop(queue)
   if n==b:
    path=[n]
    while path[-1]!=a:path.append(prev[path[-1]])
    return [(x*25,y*25) for x,y in path[::-1]]
   if g>cost[n]+1e-8:continue
   for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
    q=(n[0]+dx,n[1]+dy)
    if not free(q) or (dx and dy and (not free((n[0]+dx,n[1])) or not free((n[0],n[1]+dy)))):continue
    dist=25*math.hypot(dx,dy);dz=abs(ground.z(q[0]*25,q[1]*25)-ground.z(n[0]*25,n[1]*25))
    if dz/dist>.65:continue
    if dx and dy:
     z=ground.z(n[0]*25,n[1]*25);zq=ground.z(q[0]*25,q[1]*25)
     if any(abs(ground.z(t[0]*25,t[1]*25)-z)>20 or abs(ground.z(t[0]*25,t[1]*25)-zq)>20 for t in [(n[0]+dx,n[1]),(n[0],n[1]+dy)]):continue
    new=g+dist*(1+.35*dz/dist)
    if new<cost.get(q,1e30):cost[q]=new;prev[q]=n;heapq.heappush(queue,(new+25*math.dist(q,b),new,q))
  raise RuntimeError('No walkable route between '+str([a,b]))
 points=[]
 for a,b in zip(anchors,anchors[1:]):points+=segment(a,b)[0 if not points else 1:]
 return points

def specifications():
 """Load stable inputs independently of historical review outputs."""
 specs = json.loads((ROOT/'inputs/ab_walkthroughs/placements.json').read_text())['specs']
 if any(spec['source_pack'] == 'unfinished_building' for spec in specs):
  raise ValueError('C-grade source is excluded')
 return specs

def initial_request():
 OUT.mkdir(parents=True, exist_ok=True)
 g=Ground();reserved=route_points(g,ANCHORS)
 r=dict(ready=True,action='build_scene',output=str(OUT),map_path='/Game/SuperDyn/ABWalkthroughs/'+('' if OUT.name == 'ab_walkthroughs' else 'Experiments/'+OUT.name+'/')+'Maps/PopulatedYard',specs=specifications(),reserved_path_cm=[list(p) for p in reserved],eye_height_cm=165)
 (OUT/'request.json').write_text(json.dumps(r,indent=2)+'\n');return r

def make_routes(manifest):
 g=Ground();objects=manifest['objects']
 plans=[('01_east_to_west','East yard, upper terrace and west structures',ANCHORS[:13]),
 ('02_west_to_east','West structures, north passage and vehicle bays',list(reversed(ANCHORS[1:14]))),
 ('03_full_loop','Full terrain circuit',ANCHORS)]
 cases=[]
 for case,title,anchors in plans:
  points=np.array(route_points(g,anchors,objects),dtype=float)
  # Round grid corners inside the reserved passage.
  for _ in range(2):points=np.vstack([points[0],np.array([[.75*a+.25*b,.25*a+.75*b] for a,b in zip(points,points[1:])]).reshape(-1,2),points[-1]])
  lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))];distance=float(lengths[-1]);n=960
  sample=np.linspace(0,distance,n);xy=np.column_stack([np.interp(sample,lengths,points[:,k]) for k in [0,1]])
  look=np.minimum(sample+180,distance);future=np.column_stack([np.interp(look,lengths,points[:,k]) for k in [0,1]])
  yaw=np.unwrap(np.arctan2(future[:,1]-xy[:,1],future[:,0]-xy[:,0]));yaw[-4:]=yaw[-5]
  kernel=np.exp(-.5*(np.arange(-72,73)/24)**2);kernel/=kernel.sum();yaw=np.convolve(np.pad(yaw,(72,72),mode="edge"),kernel,mode="valid")
  poses=[];z=[g.z(*p) for p in xy]
  for i,(x,y) in enumerate(xy):
   ahead=g.z(*future[i]);d=max(60,math.dist(xy[i],future[i]));pitch=max(-18,min(18,math.degrees(math.atan2(ahead-z[i],d))))-3
   angle=math.degrees(yaw[i])+17*math.sin(2*math.pi*i/(24*10))
   poses.append(dict(location_cm=[float(x),float(y),z[i]+165],rotation_rpy_deg=[0,pitch,angle],scale=[1,1,1]))
  packet=dict(case_id=case,frames=n,fps=24,width=1280,height=720,camera=poses,title=title,horizontal_fov_degrees=80)
  folder=OUT/case;folder.mkdir(exist_ok=True);(folder/'route.json').write_text(json.dumps(dict(packet=packet,path_length_cm=distance,average_speed_cm_s=distance/40,eye_height_cm=165),indent=2)+'\n')
  cases.append(dict(id=case,title=title,output=str(folder),sequence_path=manifest['map_path'].rsplit('/Maps/',1)[0]+'/Sequences/LS_'+case))
 r=dict(ready=True,action='prepare_preview',output=str(OUT),map_path=manifest['map_path'],cases=cases)
 (OUT/'request.json').write_text(json.dumps(r,indent=2)+'\n')
 draw_plan(objects,cases,g)
 return cases

def draw_plan(objects,cases,g):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(12,10));xs=np.arange(-1100,1101,25);ys=np.arange(-950,951,25);a=[[g.z(x,y) for x in xs] for y in ys];ax.imshow(a,origin='lower',extent=[-1112,1112,-962,962],cmap='terrain',alpha=.4)
 colors={'vehicle':'#d04d31','human':'#0d75c2','animal':'#9c4bd2','structure':'#383838','tree':'#258844','roof':'#383838'}
 for o in objects:
  if o.get('overhead'):continue
  x,y=o['anchor_cm'];hx,hy=o['half_xy_cm'];ax.add_patch(plt.Rectangle((x-hx,y-hy),2*hx,2*hy,facecolor=colors.get(o['category'],'#a37b34'),alpha=.65))
 for case,color in zip(cases,['#d52728','#2463d2','#e69d00']):
  poses=json.loads((Path(case['output'])/'route.json').read_text())['packet']['camera'];xy=np.array([p['location_cm'][:2] for p in poses]);ax.plot(xy[:,0],xy[:,1],color=color,lw=2,label=case['id'])
 ax.legend();ax.set(xlabel='X (cm)',ylabel='Y (cm)',title='A/B assets and first-person routes');ax.set_aspect('equal');fig.tight_layout();fig.savefig(OUT/'layout_routes.png',dpi=130);plt.close(fig)

if __name__=='__main__':initial_request()
