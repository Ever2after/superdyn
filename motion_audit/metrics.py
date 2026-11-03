"""Camera/scene disentanglement metrics. All cameras are OpenCV world-to-camera."""
import cv2
import numpy as np

THRESHOLDS = dict(camera_translation=0.05, camera_rotation_deg=10.,
                  dynamic_residual=0.03, dynamic_area=0.01, joint_duty=0.25)
AMPLITUDE_THRESHOLDS = dict(camera_translation=.2, camera_rotation_deg=40.,
                            dynamic_residual=.12, dynamic_area=.01, joint_duty=.25)

def camera_rates(extrinsics, times, depth_scale):
    rotations = extrinsics[:, :3, :3].astype(np.float64)
    centers = -np.einsum('tji,tj->ti', rotations, extrinsics[:, :3, 3])
    dt = np.diff(times)
    translation = np.linalg.norm(np.diff(centers, axis=0), axis=1) / depth_scale / dt
    relative = rotations[1:] @ rotations[:-1].transpose(0,2,1)
    angles = np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))
    return translation, angles/dt, centers

def rigid_flow(depth, k0, k1, e0, e1):
    h,w = depth.shape
    y,x = np.mgrid[:h,:w]
    pixels = np.stack([x,y,np.ones_like(x)],-1).astype(np.float64)
    camera = (pixels @ np.linalg.inv(k0).T) * depth[...,None]
    world = (camera - e0[:3,3]) @ e0[:3,:3]
    target = world @ e1[:3,:3].T + e1[:3,3]
    proj = target @ k1.T
    uv = proj[...,:2] / np.maximum(proj[...,2:],1e-9)
    flow = uv - pixels[...,:2]
    valid = (target[...,2]>0) & np.isfinite(flow).all(-1)
    return flow.astype(np.float32), valid, target[...,2]

def remap(array, flow):
    h,w = flow.shape[:2]
    y,x = np.mgrid[:h,:w]
    u=(x+flow[...,0]).astype(np.float32)
    v=(y+flow[...,1]).astype(np.float32)
    valid=(u>=0)&(u<w-1)&(v>=0)&(v<h-1)
    return cv2.remap(array.astype(np.float32),u,v,cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,borderValue=0),valid

def interval_statistics(depth, conf, extrinsics, intrinsics, masks, forward, backward, times,
                        *, pixel_quality_filter=True):
    """Returns per-interval statistics; invalid reconstruction is never 'low motion'."""
    t,h,w=depth.shape
    diagonal=np.hypot(h,w)
    fg=masks>0.5
    depth_good=np.isfinite(depth)&(depth>0)
    if pixel_quality_filter:
        for i in range(t):
            depth_good[i] &= conf[i]>=np.quantile(conf[i][np.isfinite(conf[i])],.2)
    background_depths=depth[depth_good&~fg]
    scale=float(np.median(background_depths)) if len(background_depths)>0 else float('nan')
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('No reliable static-background depth scale')
    trans,rot,centers=camera_rates(extrinsics,times,scale)
    rows=[]
    for i,dt in enumerate(np.diff(times)):
        f=forward[i];b=backward[i]
        b_at_f,inbounds=remap(b,f)
        fb=np.linalg.norm(f+b_at_f,axis=-1)
        correspondence=inbounds&np.isfinite(f).all(-1)
        if pixel_quality_filter:
            correspondence &= fb<=1.+.05*np.linalg.norm(f,axis=-1)
        conf_target,_=remap(depth_good[i+1].astype(np.float32),f)
        f_rigid,rigid_valid,z_rigid=rigid_flow(depth[i],intrinsics[i],intrinsics[i+1],extrinsics[i],extrinsics[i+1])
        valid=correspondence&depth_good[i]&(conf_target>.99)&rigid_valid
        residual=np.linalg.norm(f-f_rigid,axis=-1)/diagonal/dt
        static=valid&~cv2.dilate(fg[i].astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        dynamic=valid&fg[i]
        static_n=int(static.sum());dyn_n=int(dynamic.sum())
        static_error=float(np.median(residual[static])) if static_n else None
        foreground_n=int(fg[i].sum())
        dynamic_coverage=dyn_n/max(foreground_n,1)
        dynamic_observable=foreground_n<.01*h*w or (dyn_n>=20 and dynamic_coverage>=.2)
        duration=float(times[-1]-times[0])
        quality=(static_n>=max(200,.05*h*w)) and static_error is not None and static_error*duration<=.08 and dynamic_observable
        minimum_dynamic=max(20,.001*h*w) if pixel_quality_filter else 1
        d75=float(np.quantile(residual[dynamic],.75)) if dyn_n>=minimum_dynamic else 0.
        cpx=float(np.median(np.linalg.norm(f_rigid[static],axis=-1))/diagonal/dt) if static_n else None
        # Depth-based 3D motion is a secondary noisy diagnostic, not the primary filter.
        target_depth,_=remap(depth[i+1],f)
        y,x=np.mgrid[:h,:w]
        p=np.stack([x,y,np.ones_like(x)],-1)
        p1=p.copy().astype(float);p1[...,:2]+=f
        c0=(p@np.linalg.inv(intrinsics[i]).T)*depth[i,...,None]
        c1=(p1@np.linalg.inv(intrinsics[i+1]).T)*target_depth[...,None]
        w0=(c0-extrinsics[i,:3,3])@extrinsics[i,:3,:3]
        w1=(c1-extrinsics[i+1,:3,3])@extrinsics[i+1,:3,:3]
        d3=np.linalg.norm(w1-w0,axis=-1)/scale/dt
        row=dict(interval=i,dt=float(dt),camera_translation=float(trans[i]),
                 camera_rotation_deg=float(rot[i]),camera_image_motion=cpx,
                 dynamic_area=float(fg[i].mean()),dynamic_residual_p75=d75,
                 dynamic_3d_p75=float(np.quantile(d3[dynamic],.75)) if dyn_n>=minimum_dynamic else 0.,
                 static_residual_median=static_error,static_valid_fraction=static_n/(h*w),
                 static_residual_per_window=static_error*duration if static_error is not None else None,
                 dynamic_valid_fraction=dyn_n/(h*w),dynamic_correspondence_coverage=dynamic_coverage,
                 dynamic_observable=bool(dynamic_observable),
                 flow_valid_fraction=float(valid.mean()),quality_valid=bool(quality))
        rows.append(row)
    return rows,dict(depth_scale=scale,centers=centers.tolist(),
                     camera_path_depths=float((trans*np.diff(times)).sum()),
                     camera_rotation_path_deg=float((rot*np.diff(times)).sum()))

def summarize_intervals(rows, thresholds=None):
    th=THRESHOLDS | (thresholds or {})
    valid=[r for r in rows if r['quality_valid']]
    coverage=sum(r['dt'] for r in valid)/sum(r['dt'] for r in rows)
    result=dict(quality_coverage=coverage,valid_intervals=len(valid),total_intervals=len(rows),
                eligible=coverage>=.75-1e-9)
    if not valid:
        return result
    weights=np.array([r['dt'] for r in valid]);weights/=weights.sum()
    trans=np.array([r['camera_translation'] for r in valid])
    rot=np.array([r['camera_rotation_deg'] for r in valid])
    d=np.array([r['dynamic_residual_p75'] for r in valid])
    area=np.array([r['dynamic_area'] for r in valid])
    c=(trans>=th['camera_translation'])|(rot>=th['camera_rotation_deg'])
    dh=(d>=th['dynamic_residual'])&(area>=th['dynamic_area'])
    result.update(camera_translation_median=float(np.median(trans)),camera_rotation_deg_median=float(np.median(rot)),
                  dynamic_residual_median=float(np.median(d)),dynamic_area_mean=float(weights@area),
                  camera_high_duty=float(weights@c),dynamic_high_duty=float(weights@dh),
                  joint_high_duty=float(weights@(c&dh)),
                  translation_joint_high_duty=float(weights@((trans>=th['camera_translation'])&dh)),
                  rotation_joint_high_duty=float(weights@((rot>=th['camera_rotation_deg'])&dh)))
    result.update(camera_high=result['camera_high_duty']>=th['joint_duty']-1e-9,
                  dynamic_high=result['dynamic_high_duty']>=th['joint_duty']-1e-9,
                  joint_high=result['joint_high_duty']>=th['joint_duty']-1e-9)
    return result

def summarize_amplitudes(rows,thresholds=None):
    """Primary comparison: clock-invariant, window-equivalent displacement.

    Per-interval change divided by fractional window duration. The resulting
    unit is scene depths/window, degrees/window or image diagonals/window.
    It equals speed times window duration, retaining temporal overlap.
    """
    duration=sum(r['dt'] for r in rows)
    transformed=[r | {k:r[k]*duration for k in ['camera_translation','camera_rotation_deg','dynamic_residual_p75']} for r in rows]
    th=AMPLITUDE_THRESHOLDS | (thresholds or {})
    result=summarize_intervals(transformed,th)
    result['duration']=duration
    result['camera_translation_per_window_median']=result.pop('camera_translation_median',None)
    result['camera_rotation_deg_per_window_median']=result.pop('camera_rotation_deg_median',None)
    result['dynamic_residual_per_window_median']=result.pop('dynamic_residual_median',None)
    valid=[r for r in rows if r['quality_valid']]
    result['dynamic_residual_path_diagonals']=sum(r['dynamic_residual_p75']*r['dt'] for r in valid)
    result['camera_translation_path_depths_valid']=sum(r['camera_translation']*r['dt'] for r in valid)
    result['camera_rotation_path_deg_valid']=sum(r['camera_rotation_deg']*r['dt'] for r in valid)
    if valid:
        ct=result['camera_translation_path_depths_valid']>=th['camera_translation']
        cr=result['camera_rotation_path_deg_valid']>=th['camera_rotation_deg']
        dynamic=(result['dynamic_residual_path_diagonals']>=th['dynamic_residual'] and
                 result['dynamic_area_mean']>=th['dynamic_area'])
        result['camera_high']=bool(ct or cr)
        result['translation_high']=bool(ct)
        result['rotation_high']=bool(cr)
        result['dynamic_high']=bool(dynamic)
        result['joint_high']=bool((ct or cr) and dynamic and result['joint_high_duty']>=th['joint_duty']-1e-9)
        result['translation_joint_high']=bool(ct and dynamic and result['translation_joint_high_duty']>=th['joint_duty']-1e-9)
        result['rotation_joint_high']=bool(cr and dynamic and result['rotation_joint_high_duty']>=th['joint_duty']-1e-9)
    return result
