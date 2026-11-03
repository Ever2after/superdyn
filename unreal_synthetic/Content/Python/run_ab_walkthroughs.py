"""Populate the enclosed terrain with A/B assets and render eye-level tours."""
import json,math,os,time,traceback
from pathlib import Path
import unreal
from superdyn_ue.io_utils import save_json,load_scene
from superdyn_ue.scene_builder import add_transform_track,transform,initialize_terrain_collision
from superdyn_ue.official_assets import bounds_points
from superdyn_ue.enclosure_audit import validate as validate_enclosure
from superdyn_ue.render_queue import add_job

ROOT=Path(__file__).resolve().parents[2]
REQUEST_PATH=Path(os.environ['SUPERDYN_UE_REQUEST'])
HANDLE=None;STAMP=None;EXECUTOR=None;FAILED=False;QUEUE_JOBS=[]
GRID=json.loads((ROOT/'inputs/ab_walkthroughs/height_grid.json').read_text())
HEIGHTS={(x,y):z for x,y,z in GRID['heights']}
ACTORS={};SKELETAL=[]

def configure(component):
 component.set_mobility(unreal.ComponentMobility.MOVABLE)
 component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
 component.set_editor_property('render_custom_depth',True)
 component.set_editor_property('custom_depth_stencil_value',10)

def height(x,y):
 x/=12.5;y/=12.5;i=math.floor(x);j=math.floor(y);u=x-i;v=y-j
 q=[HEIGHTS.get((i,j)),HEIGHTS.get((i+1,j)),HEIGHTS.get((i,j+1)),HEIGHTS.get((i+1,j+1))]
 if any(z is None for z in q):return HEIGHTS.get((round(x),round(y)),0)
 return (1-u)*(1-v)*q[0]+u*(1-v)*q[1]+(1-u)*v*q[2]+u*v*q[3]

def status(out,stage,**kw):save_json(Path(out)/'native_status.json',dict(stage=stage,**kw))
def mesh_box(mesh,scale,yaw):
 points=bounds_points(mesh.get_imported_bounds() if isinstance(mesh,unreal.SkeletalMesh) else mesh.get_bounds())
 lo=[min(getattr(p,a) for p in points) for a in ['x','y','z']];hi=[max(getattr(p,a) for p in points) for a in ['x','y','z']]
 c,s=abs(math.cos(math.radians(yaw))),abs(math.sin(math.radians(yaw)))
 half=[((hi[0]-lo[0])*c+(hi[1]-lo[1])*s)*scale/2,((hi[0]-lo[0])*s+(hi[1]-lo[1])*c)*scale/2]
 return lo,hi,half

def fit(spec,half,placed,reserved):
 original=spec['anchor_cm'];radius=spec.get('search_cm',300);candidates=[(0,0)]+sorted([(x,y) for x in range(-radius,radius+1,25) for y in range(-radius,radius+1,25) if x*x+y*y<=radius*radius],key=lambda p:p[0]**2+p[1]**2)
 hx,hy=half
 if spec['category']=='tree':hx=hy=38
 if spec.get('overhead'):return original,height(*original)
 for dx,dy in candidates:
  x,y=original[0]+dx,original[1]+dy
  if abs(x)+hx>1110 or abs(y)+hy>960:continue
  if any(abs(x-a)<hx+45 and abs(y-b)<hy+45 for a,b in reserved):continue
  conflict=False
  for p in placed:
   if p.get('overhead') or (spec.get('structure_group') and spec.get('structure_group')==p.get('structure_group')):continue
   px,py=p['anchor_cm'];a,b=p['half_xy_cm']
   if p['category']=='tree':a=b=38
   if abs(x-px)<hx+a+20 and abs(y-py)<hy+b+20:conflict=True;break
  if conflict:continue
  heights=[height(x+ox,y+oy) for ox in [-hx,0,hx] for oy in [-hy,0,hy]]
  tolerance=18 if spec['category'] in ['human','animal','tree'] else 8
  if max(heights)-min(heights)>tolerance:continue
  return [x,y],max(heights)
 raise RuntimeError('No flat, clear placement for '+spec['id'])

def build_scene(request):
 out=Path(request['output']);levels=unreal.get_editor_subsystem(unreal.LevelEditorSubsystem);editor=unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
 base='/Game/SuperDyn/Themes/industrial';target=request['map_path']
 unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
 if unreal.EditorAssetLibrary.does_asset_exist(target):ok=levels.load_level(target)
 else:ok=levels.new_level_from_template(target,base+'/Maps/03_low_orbit')
 if not ok:raise RuntimeError('Cannot create populated terrain')
 keep={'SD_terrain','SD_Backdrop','SD_Sun','SD_SkyLight','SD_Camera'}
 for a in editor.get_all_level_actors():
  if a.get_actor_label().startswith('SD_') and a.get_actor_label() not in keep and not a.get_actor_label().startswith('SD_Theme_'):editor.destroy_actor(a)
 by={a.get_actor_label():a for a in editor.get_all_level_actors()};terrain=by['SD_terrain'];initialize_terrain_collision({'terrain':terrain})
 original=unreal.load_asset(base+'/Sequences/LS_03_low_orbit');settings=json.loads(unreal.EditorAssetLibrary.get_metadata_tag(original,'SuperDynThemeSettings'))
 enclosure=validate_enclosure(dict(theme_settings=settings,packet=load_scene(ROOT/'inputs/03_low_orbit/scene.json.gz')),editor)
 records=[];SKELETAL.clear();ACTORS.clear()
 for index,spec in enumerate(request['specs']):
  status(out,'placing_assets',index=index,total=len(request['specs']),asset=spec['id'])
  if spec['source_pack']=='unfinished_building' or spec['mesh'].startswith('/Game/Scene_UnfinishedBuilding/'):raise RuntimeError('C-grade source forbidden')
  mesh=unreal.load_asset(spec['mesh'])
  if not isinstance(mesh,(unreal.StaticMesh,unreal.SkeletalMesh)):raise RuntimeError('Mesh unavailable: '+spec['mesh'])
  skeletal=isinstance(mesh,unreal.SkeletalMesh)
  lo,hi,_=mesh_box(mesh,1,spec['yaw_deg']);dims=[b-a for a,b in zip(lo,hi)]
  divisor=(hi[2] if skeletal else dims[2]) if spec.get('height_size') else max(dims)
  scale=spec['size_cm']/divisor;lo,hi,half=mesh_box(mesh,scale,spec['yaw_deg'])
  xy,ground=fit(spec,half,records,request['reserved_path_cm'])
  a=editor.spawn_actor_from_class(unreal.SkeletalMeshActor if skeletal else unreal.StaticMeshActor,unreal.Vector());a.set_actor_label('SD_AB_'+spec['id']);comp=a.get_component_by_class(unreal.SkeletalMeshComponent if skeletal else unreal.StaticMeshComponent);configure(comp)
  if skeletal:
   comp.set_skeletal_mesh_asset(mesh);comp.set_simulate_physics(False);comp.set_update_animation_in_editor(True);comp.set_editor_property('visibility_based_anim_tick_option',unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES)
   if spec.get('animation'):
    clip=unreal.load_asset(spec['animation'])
    if not isinstance(clip,unreal.AnimSequence) or clip.get_editor_property('skeleton')!=mesh.get_editor_property('skeleton'):raise RuntimeError('Idle skeleton mismatch: '+spec['id'])
    path=target.rsplit('/Maps/',1)[0]+'/Motion/'+spec['id'];local=unreal.load_asset(path)
    if local is None:local=unreal.EditorAssetLibrary.duplicate_asset(spec['animation'],path)
    local.set_editor_property('enable_root_motion',True);local.set_editor_property('force_root_lock',True);local.set_editor_property('root_motion_root_lock',unreal.RootMotionRootLock.ANIM_FIRST_FRAME);unreal.EditorAssetLibrary.save_loaded_asset(local)
    comp.set_animation_mode(unreal.AnimationMode.ANIMATION_SINGLE_NODE);comp.play_animation(local,True)
    SKELETAL.append((a,comp,local,dict(spec,anchor_cm=xy),ground))
  else:
   comp.set_static_mesh(mesh);unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem).get_number_verts(mesh,0)
   comp.set_collision_profile_name('BlockAll');comp.set_collision_enabled(unreal.CollisionEnabled.QUERY_ONLY)
   try:comp.set_editor_property('forced_lod_model',2 if spec['category']=='vehicle' else 0)
   except Exception:pass
  angle=math.radians(spec['yaw_deg']);cx=(lo[0]+hi[0])/2*scale;cy=(lo[1]+hi[1])/2*scale
  offset=[0,0] if skeletal else [cx*math.cos(angle)-cy*math.sin(angle),cx*math.sin(angle)+cy*math.cos(angle)]
  z=ground+spec.get('raise_cm',0)+(2 if skeletal else -lo[2]*scale+.5)
  pose=dict(location_cm=[xy[0]-offset[0],xy[1]-offset[1],z],rotation_rpy_deg=[0,0,spec['yaw_deg']],scale=[scale]*3)
  a.set_actor_transform(transform(pose),False,True)
  if any(comp.get_material(i) is None for i in range(comp.get_num_materials())):raise RuntimeError('Missing material: '+spec['id'])
  record=dict(spec,anchor_cm=xy,requested_anchor_cm=spec['anchor_cm'],ground_z_cm=ground,half_xy_cm=half,scale=scale,actor=a.get_actor_label(),transform=pose,source_materials=[comp.get_material(i).get_path_name() for i in range(comp.get_num_materials())])
  records.append(record);ACTORS[spec['id']]=a
  if spec.get('double_face'):
   back=editor.spawn_actor_from_class(unreal.StaticMeshActor,unreal.Vector());back.set_actor_label('SD_AB_'+spec['id']+'_back');bc=back.get_component_by_class(unreal.StaticMeshComponent);configure(bc);bc.set_static_mesh(mesh)
   reverse=spec['yaw_deg']+180;r=math.radians(reverse);backpose=dict(pose,rotation_rpy_deg=[0,0,reverse],location_cm=[xy[0]-(cx*math.cos(r)-cy*math.sin(r)),xy[1]-(cx*math.sin(r)+cy*math.cos(r)),z]);back.set_actor_transform(transform(backpose),False,True);record['backface_actor']=back.get_actor_label()
 camera=by['SD_Camera'];cc=camera.get_cine_camera_component();film=cc.get_editor_property('filmback');film.set_editor_property('sensor_width',36);film.set_editor_property('sensor_height',20.25);cc.set_editor_property('filmback',film);cc.set_editor_property('current_focal_length',36/(2*math.tan(math.radians(40))))
 focus=cc.get_editor_property('focus_settings');focus.set_editor_property('focus_method',unreal.CameraFocusMethod.DISABLE);cc.set_editor_property('focus_settings',focus)
 levels.save_current_level()
 manifest=dict(status='scene_built',map_path=target,objects=records,logical_objects=len(records),source_packs=sorted({r['source_pack'] for r in records}),excluded=['unfinished_building','city_park_environment:full_map','modular_rural_cabins:full_map'],enclosure_validation=enclosure,eye_height_cm=165,scope='A/B meshes only; original surrounding walls and HDRI; no source sample maps instantiated')
 save_json(out/'scene_manifest.json',manifest);status(out,'scene_built',objects=len(records),source_packs=len(manifest['source_packs']))

def set_frame(frame):
 p=unreal.MovieSceneSequencePlaybackParams(frame=unreal.FrameTime(unreal.FrameNumber(frame)),position_type=unreal.MovieScenePositionType.FRAME,update_method=unreal.UpdatePositionMethod.JUMP)
 unreal.LevelSequenceEditorBlueprintLibrary.set_global_position(p,unreal.MovieSceneTimeUnit.DISPLAY_RATE);unreal.LevelSequenceEditorBlueprintLibrary.force_update()

def prepare(request):
 levels=unreal.get_editor_subsystem(unreal.LevelEditorSubsystem);editor=unreal.get_editor_subsystem(unreal.EditorActorSubsystem);out=Path(request['output']);camera=next(a for a in editor.get_all_level_actors() if a.get_actor_label()=='SD_Camera');world=unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world();ignored=[a for a in editor.get_all_level_actors() if a.get_actor_label()!='SD_terrain'];manifest=json.loads((out/'scene_manifest.json').read_text());foot_records=[]
 def native_ground(x,y):
  hit=unreal.SystemLibrary.line_trace_single(world,unreal.Vector(x,y,1000),unreal.Vector(x,y,-200),unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,True,ignored,unreal.DrawDebugTrace.NONE,True)
  if hit is None:raise RuntimeError('Native ground ray missed '+str([x,y]))
  try:return hit.get_editor_property('location').z
  except Exception:return hit.to_tuple()[4].z
 # Use the actual collision surface for placement and every eye-height key.
 by_label={a.get_actor_label():a for a in editor.get_all_level_actors()}
 for record in manifest['objects']:
  x,y=record['anchor_cm'];actual=native_ground(x,y);delta=actual-record['ground_z_cm'];actor=by_label[record['actor']];loc=actor.get_actor_location();loc.z+=delta;actor.set_actor_location(loc,False,True)
  if record.get('backface_actor'):
   back=by_label[record['backface_actor']];loc=back.get_actor_location();loc.z+=delta;back.set_actor_location(loc,False,True)
  record['ground_z_cm']=actual;record['native_ground_adjustment_cm']=delta
 SKELETAL[:]=[(a,c,clip,spec,native_ground(*spec['anchor_cm'])) for a,c,clip,spec,z in SKELETAL]
 manifest['native_ground_aligned']=True;save_json(out/'scene_manifest.json',manifest)
 for index,case in enumerate(request['cases']):
  status(out,'building_routes',route=case['id']);route=json.loads((Path(case['output'])/'route.json').read_text());packet=route['packet']
  correction=0
  for pose in packet['camera']:
   x,y,z=pose['location_cm'];actual=native_ground(x,y)+165;correction=max(correction,abs(z-actual));pose['location_cm'][2]=actual
  for f,pose in enumerate(packet['camera']):
   future=packet['camera'][min(f+36,packet['frames']-1)]['location_cm'];here=pose['location_cm'];d=max(60,math.dist(here[:2],future[:2]));pose['rotation_rpy_deg'][1]=max(-18,min(18,math.degrees(math.atan2(future[2]-here[2],d))))-3
  max_vertical=max(abs(b['location_cm'][2]-a['location_cm'][2]) for a,b in zip(packet['camera'],packet['camera'][1:]))
  if max_vertical>12:raise RuntimeError('Camera route crosses a cliff: '+str([case['id'],max_vertical]))
  route['max_vertical_step_cm']=max_vertical;route['native_ground_correction_max_cm']=correction;save_json(Path(case['output'])/'route.json',route)
  path,name=case['sequence_path'].rsplit('/',1);unreal.EditorAssetLibrary.make_directory(path)
  if unreal.EditorAssetLibrary.does_asset_exist(case['sequence_path']):unreal.EditorAssetLibrary.delete_asset(case['sequence_path'])
  seq=unreal.AssetToolsHelpers.get_asset_tools().create_asset(name,path,unreal.LevelSequence,unreal.LevelSequenceFactoryNew());seq.set_display_rate(unreal.FrameRate(packet['fps'],1));seq.set_tick_resolution_directly(unreal.FrameRate(packet['fps']*1000,1));seq.set_playback_start(0);seq.set_playback_end(packet['frames']);binding=add_transform_track(seq,camera,packet['camera']);cut=seq.add_track(unreal.MovieSceneCameraCutTrack).add_section();cut.set_range(0,packet['frames']);bid=unreal.MovieSceneObjectBindingID();bid.set_editor_property('guid',binding.get_id());cut.set_camera_binding_id(bid)
  for actor,comp,clip,spec,ground in SKELETAL:
   section=seq.add_possessable(actor).add_track(unreal.MovieSceneSkeletalAnimationTrack).add_section();section.set_range(0,packet['frames']);params=section.get_editor_property('params');params.set_editor_property('animation',clip);params.set_editor_property('force_custom_mode',True);params.set_editor_property('skip_anim_notifiers',True);section.set_editor_property('params',params)
  unreal.LevelSequenceEditorBlueprintLibrary.open_level_sequence(seq);set_frame(0)
  if index==0:
   for actor,comp,clip,spec,ground in SKELETAL:
    names=[str(comp.get_bone_name(i)) for i in range(comp.get_num_bones())];feet=[n for n in names if any(s in n.lower() for s in ['foot','ball','toe','hoof','paw']) and not n.lower().startswith('ik_')]
    if feet:
     low=min(comp.get_socket_location(n).z for n in feet);loc=actor.get_actor_location();delta=ground+3-low
     if abs(delta)>100:raise RuntimeError('Invalid posed foot offset '+spec['id'])
     loc.z+=delta;actor.set_actor_location(loc,False,True);foot_records.append(dict(actor=actor.get_actor_label(),foot_bones=feet,root_adjustment_cm=delta))
  error=0;support=0;misses=[]
  for frame in list(range(0,packet['frames'],24))+[packet['frames']-1]:
   pose=packet['camera'][frame];x,y,z=pose['location_cm'];hit=unreal.SystemLibrary.line_trace_single(world,unreal.Vector(x,y,1000),unreal.Vector(x,y,-200),unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,True,ignored,unreal.DrawDebugTrace.NONE,True)
   if hit is None:misses.append(frame);continue
   try:p=hit.get_editor_property('location')
   except Exception:p=hit.to_tuple()[4]
   support=max(support,abs(z-p.z-165))
  for f in [round((packet['frames']-1)*t/4) for t in range(5)]:
   set_frame(f);a=camera.get_actor_location();error=max(error,math.dist([a.x,a.y,a.z],packet['camera'][f]['location_cm']))
  if misses or support>2 or error>.1:raise RuntimeError('Camera support/evaluation failed '+str([case['id'],misses,support,error]))
  unreal.EditorAssetLibrary.save_loaded_asset(seq);save_json(Path(case['output'])/'native_validation.json',dict(status='passed',engine=unreal.SystemLibrary.get_engine_version(),frames=packet['frames'],eye_height_cm=165,terrain_support_error_cm=support,camera_evaluation_error_cm=error,path_length_cm=route['path_length_cm'],average_speed_cm_s=route['average_speed_cm_s'],camera_probes=len(range(0,packet['frames'],24))+1,scope='Eye-level camera on actual terrain; navigation avoids actor footprints; no player hands/HUD'))
  save_json(Path(case['output'])/'demo_scene.json',dict(map_path=request['map_path'],sequence_path=case['sequence_path']))
 unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence();levels.save_current_level();save_json(out/'foot_alignment.json',foot_records)
 submit(request,True)

def errored(executor,pipeline,is_fatal,text):
 global FAILED
 FAILED=True;status(REQUEST_PATH.parent,'failed',error=str(text))
def finished(executor,success):
 passed=bool(success and not FAILED)
 for case,preview in QUEUE_JOBS:save_json(Path(case['output'])/('preview_status.json' if preview else 'render_status.json'),dict(status='frames_complete' if passed else 'failed',preview=preview,unreal_execution_verified=True))
 status(REQUEST_PATH.parent,'preview_complete' if passed and QUEUE_JOBS[0][1] else 'frames_complete' if passed else 'failed');unreal.SystemLibrary.quit_editor()

def submit(request,preview=False):
 global EXECUTOR,FAILED,QUEUE_JOBS
 subsystem=unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem);queue=subsystem.get_queue();queue.delete_all_jobs();FAILED=False;QUEUE_JOBS=[]
 for case in request['cases']:
  route=json.loads((Path(case['output'])/'route.json').read_text());packet=route['packet'];seq=unreal.load_asset(case['sequence_path']);folder=Path(case['output'])/('preview' if preview else '')
  job=add_job(queue,dict(packet=packet,map_path=request['map_path'],sequence=seq),folder,preview=preview,spatial_samples=1)
  aa=job.get_configuration().find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting);aa.set_editor_property('anti_aliasing_method',unreal.AntiAliasingMethod.AAM_TSR)
  QUEUE_JOBS.append((case,preview))
 EXECUTOR=unreal.MoviePipelinePIEExecutor();EXECUTOR.on_executor_errored_delegate.add_callable_unique(errored);EXECUTOR.on_executor_finished_delegate.add_callable_unique(finished);status(request['output'],'rendering_preview' if preview else 'rendering',cases=len(request['cases']));subsystem.render_queue_with_executor_instance(EXECUTOR)

def poll(delta):
 global STAMP
 if unreal.AssetRegistryHelpers.get_asset_registry().is_loading_assets():return
 stamp=REQUEST_PATH.stat().st_mtime_ns
 if stamp==STAMP:return
 STAMP=stamp;request=json.loads(REQUEST_PATH.read_text())
 if not request.get('ready'):return
 try:
  unreal.SystemLibrary.execute_console_command(None,'r.RDG.AsyncCompute 0')
  if request['action']=='build_scene':build_scene(request)
  elif request['action']=='prepare_preview':prepare(request)
  elif request['action']=='reload_preview':
   levels=unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
   if not levels.load_level(request['map_path']):raise RuntimeError('Cannot reopen populated terrain')
   editor=unreal.get_editor_subsystem(unreal.EditorActorSubsystem);by={a.get_actor_label():a for a in editor.get_all_level_actors()};initialize_terrain_collision({'terrain':by['SD_terrain']});manifest=json.loads((Path(request['output'])/'scene_manifest.json').read_text());SKELETAL.clear()
   for record in manifest['objects']:
    if record.get('animation'):
     actor=by[record['actor']];comp=actor.get_component_by_class(unreal.SkeletalMeshComponent);clip=unreal.load_asset(request['map_path'].rsplit('/Maps/',1)[0]+'/Motion/'+record['id']);SKELETAL.append((actor,comp,clip,record,record['ground_z_cm']))
   prepare(request)
  elif request['action']=='render':submit(request,False)
 except Exception as error:
  status(request['output'],'failed',error=str(error),traceback=traceback.format_exc());unreal.log_error(traceback.format_exc());unreal.SystemLibrary.quit_editor()

HANDLE=unreal.register_slate_post_tick_callback(poll)
