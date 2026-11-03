"""Closed perimeter walls with a level top above the highest terrain surface."""

import math


def _compress(points):
    result = []
    for point in points:
        result.append(point)
        while len(result) >= 3:
            a,b,c = result[-3:]
            cross = (b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])
            if abs(cross) > .01:
                break
            result.pop(-2)
    return result


def boundary_profiles(record):
    vertices = [v for section in record["sections"] for v in section["vertices_cm"]]
    low = [min(v[i] for v in vertices) for i in range(3)]
    high = [max(v[i] for v in vertices) for i in range(3)]
    profiles = []
    for name,axis,fixed,sign in [
        ("west",0,low[0],-1),("east",0,high[0],1),
        ("south",1,low[1],-1),("north",1,high[1],1),
    ]:
        heights = {}
        for point in vertices:
            if abs(point[axis]-fixed) < .02:
                coordinate = round(point[1-axis],4)
                heights[coordinate] = max(heights.get(coordinate,-math.inf),point[2])
        if len(heights) < 2:
            raise ValueError("Terrain has no usable rectangular edge: " + name)
        profiles.append(dict(side=name, axis=axis, fixed=fixed, outward_sign=sign,
                             floor_profile_cm=_compress(sorted(heights.items()))))
    return profiles,low,high


def ground_elevation_range(record):
    # The retaining mesh extends below the actual ground to close its underside.
    heights = [vertex[2] for section in record["sections"]
               for vertex,normal in zip(section["vertices_cm"],section["normals"])
               if normal[2] > .5]
    if not heights:
        raise ValueError("Terrain must contain upward-facing ground surfaces")
    return min(heights),max(heights)


def wall_records(record, clearance_above_highest_cm=100.0, thickness_cm=30.0):
    if clearance_above_highest_cm <= 0 or thickness_cm <= 0:
        raise ValueError("Wall clearance and thickness must be positive")
    profiles,low,high = boundary_profiles(record)
    ground_low,ground_high = ground_elevation_range(record)
    height_cm = ground_high-ground_low+clearance_above_highest_cm
    top_z = ground_low+height_cm
    result = []
    for profile in profiles:
        axis,fixed,sign = profile["axis"],profile["fixed"],profile["outward_sign"]
        points = profile["floor_profile_cm"]
        section = dict(material="boundary",vertices_cm=[],normals=[],uv0=[],triangles=[])
        def point(t,z,outside=False):
            xyz = [0.0,0.0,z]
            xyz[axis] = fixed + (sign*thickness_cm if outside else 0)
            xyz[1-axis] = t
            return xyz
        for (ta,_),(tb,_) in zip(points,points[1:]):
            bottom = low[2]-5.0
            corners = [point(ta,bottom),point(tb,bottom),point(tb,top_z),point(ta,top_z),
                       point(ta,bottom,True),point(tb,bottom,True),point(tb,top_z,True),point(ta,top_z,True)]
            center = [sum(v[i] for v in corners)/8 for i in range(3)]
            for indices in [[0,1,2,3],[4,7,6,5],[0,4,5,1],[3,2,6,7],[0,3,7,4],[1,5,6,2]]:
                face = [corners[i] for i in indices]
                a = [face[1][i]-face[0][i] for i in range(3)]
                b = [face[2][i]-face[0][i] for i in range(3)]
                n = [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
                mid = [sum(v[i] for v in face)/4-center[i] for i in range(3)]
                if sum(n[i]*mid[i] for i in range(3)) < 0:
                    face.reverse();n=[-x for x in n]
                length = math.sqrt(sum(x*x for x in n))
                if length < 1e-8:
                    continue
                n = [x/length for x in n]
                start = len(section["vertices_cm"])
                section["vertices_cm"].extend(face)
                section["normals"].extend([n]*4)
                drop = max(range(3),key=lambda i:abs(n[i]))
                uv_axes = [i for i in range(3) if i != drop]
                section["uv0"].extend([[v[uv_axes[0]]/100,v[uv_axes[1]]/100] for v in face])
                section["triangles"].extend([[start,start+1,start+2],[start,start+2,start+3]])
        result.append(dict(key="boundary_"+profile["side"],parent=None,sections=[section],
                           profile=profile,top_z_cm=top_z,height_above_lowest_ground_cm=height_cm,
                           clearance_above_highest_ground_cm=clearance_above_highest_cm,
                           ground_elevation_range_cm=[ground_low,ground_high],thickness_cm=thickness_cm,
                           bottom_z_cm=low[2]-5,terrain_bounds_cm=[low,high]))
    return result
