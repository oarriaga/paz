import pytest
import jax.numpy as jp
from paz.graphics.shapes.cylinder import intersect_canonical_cylinder, compute_canonical_normals_cylinder
from paz.graphics.shapes.cone import intersect_canonical_cone, compute_canonical_normals_cone
from paz.graphics.shapes.sphere import intersect_canonical_sphere
from paz.graphics.shapes.cube import intersect_canonical_cube
from paz.graphics.shapes.plane import intersect_canonical_plane
from paz.graphics.constants import EPSILON, FARAWAY

# --- Helper Function ---


def build_cylinder_depths(num_points, feature):
    depths = jp.full((4, num_points), FARAWAY)
    return depths.at[feature].set(0.0)


def compute_cylinder_wall_normals(points):
    return compute_canonical_normals_cylinder(points, build_cylinder_depths(len(points), 0))


def compute_cylinder_lower_cap_normals(points):
    return compute_canonical_normals_cylinder(points, build_cylinder_depths(len(points), 2))

def check_no_self_intersection(intersect_fn, name, points, directions):
    hit_mask, depths, depth = intersect_fn(points, directions)
    is_hit = hit_mask[0]
    ray_depth = depth[0, 0]
    
    print(f"[{name}] Hit: {is_hit}, Depth: {ray_depth}")
    
    # Debug info
    _hit_mask, _depths, _depth = intersect_fn(points, directions)
    _depths_A = _depths[0,0]
    _depths_B = _depths[1,0]
    print(f"[{name}] Raw depths: A={_depths_A}, B={_depths_B}")
    print(f"[{name}] EPSILON: {EPSILON}")

    # Acne is defined as a hit where the ray_depth is very small (<= EPSILON)
    # So we assert that IF it's a hit, THEN the ray_depth MUST be > EPSILON.
    assert not (is_hit and ray_depth <= EPSILON), f"[{name}] Acne detected! Ray hit itself at depth {ray_depth}"

def check_reflection_acne(intersect_fn, normal_fn, name, point, incident_dir):
    normals = normal_fn(point)
    N = normals[0]
    I = incident_dir[0]
    dot = jp.dot(I, N)
    R = I - 2 * dot * N
    R = R / jp.linalg.norm(R)
    
    print(f"[{name}] Point: {point[0]}")
    print(f"[{name}] Normal: {N}")
    print(f"[{name}] Incident: {I}")
    print(f"[{name}] Reflected: {R}")
    
    if jp.dot(R, N) < 0:
        print(f"[{name}] WARNING: Reflected ray points INSIDE!")

    # Apply Renderer Offset Logic
    dot_val = jp.dot(R, N)
    offset_dir = jp.sign(dot_val) * N
    origin_shifted = point + offset_dir * EPSILON

    origins_in = jp.array([origin_shifted[0]])
    dirs_in = jp.array([R])
    
    hit_mask, depths, depth = intersect_fn(origins_in, dirs_in)
    is_hit = hit_mask[0]
    ray_depth = depth[0, 0]
    
    print(f"[{name}] Hit: {is_hit}, Depth: {ray_depth}")
    
    if is_hit:
        assert ray_depth > 0.1, f"[{name}] Acne! Reflected ray hit surface at t={ray_depth}"
    else:
        print(f"[{name}] No self-intersection. Good.")

def check_corner_escape(intersect_fn, normal_fn, name, point, ray_dir):
    normals = normal_fn(point)
    N = normals[0]
    print(f"[{name}] Point: {point[0]}")
    print(f"[{name}] Computed Normal: {N}")
    
    dot_val = jp.dot(ray_dir[0], N)
    offset = jp.sign(dot_val) * N * EPSILON
    
    new_origin = point + offset
    print(f"[{name}] New Origin: {new_origin[0]}")
    
    hit_mask, depths, depth = intersect_fn(new_origin, ray_dir)
    is_hit = hit_mask[0]
    ray_depth = depth[0, 0]
    print(f"[{name}] Hit: {is_hit}, Depth: {ray_depth}")
    
    if is_hit:
        assert ray_depth > 0.1, f"[{name}] Acne detected at corner! Depth: {ray_depth}"

# --- Basic Geometry Acne Tests ---

def test_cylinder_body_acne():
    points = jp.array([[1.0, 0.0, 0.0]])
    directions = jp.array([[1.0, 0.0, 0.0]])
    check_no_self_intersection(intersect_canonical_cylinder, "Cylinder Body", points, directions)

def test_cylinder_cap_top_acne():
    points = jp.array([[0.0, 1.0, 0.0]])
    directions = jp.array([[0.0, 1.0, 0.0]])
    check_no_self_intersection(intersect_canonical_cylinder, "Cylinder Top Cap", points, directions)

def test_cylinder_cap_bottom_acne():
    points = jp.array([[0.0, -1.0, 0.0]])
    directions = jp.array([[0.0, -1.0, 0.0]])
    check_no_self_intersection(intersect_canonical_cylinder, "Cylinder Bottom Cap", points, directions)

def test_cylinder_junction_top_acne():
    points = jp.array([[1.0, 1.0, 0.0]])
    directions = jp.array([[1.0, 1.0, 0.0]]) 
    check_no_self_intersection(intersect_canonical_cylinder, "Cylinder Junction Top", points, directions)

def test_cone_body_acne():
    points = jp.array([[0.5, -0.5, 0.0]])
    directions = jp.array([[1.0, 1.0, 0.0]])
    check_no_self_intersection(intersect_canonical_cone, "Cone Body", points, directions)

def test_cone_cap_bottom_acne():
    points = jp.array([[0.0, -1.0, 0.0]])
    directions = jp.array([[0.0, -1.0, 0.0]])
    check_no_self_intersection(intersect_canonical_cone, "Cone Bottom Cap", points, directions)

def test_cone_junction_acne():
    points = jp.array([[1.0, -1.0, 0.0]])
    directions = jp.array([[1.0, -1.0, 0.0]])
    check_no_self_intersection(intersect_canonical_cone, "Cone Junction", points, directions)

def test_sphere_acne():
    points = jp.array([[1.0, 0.0, 0.0]])
    directions = jp.array([[1.0, 0.0, 0.0]])
    check_no_self_intersection(intersect_canonical_sphere, "Sphere Body", points, directions)

def test_sphere_acne_precision():
    points = jp.array([[1.0 - 1e-5, 0.0, 0.0]])
    directions = jp.array([[1.0, 0.0, 0.0]])
    check_no_self_intersection(intersect_canonical_sphere, "Sphere Precision", points, directions)

def test_sphere_grazing_inward_acne():
    points = jp.array([[1.0, 0.0, 0.0]])
    directions = jp.array([[-1e-4, 1.0, 0.0]])
    check_no_self_intersection(intersect_canonical_sphere, "Sphere Grazing Inward", points, directions)

def test_cube_precision():
    points = jp.array([[1.0 - 1e-5, 0.0, 0.0]])
    directions = jp.array([[1.0, 0.0, 0.0]])
    check_no_self_intersection(intersect_canonical_cube, "Cube Precision", points, directions)

def test_cylinder_precision():
    points = jp.array([[1.0 - 1e-5, 0.0, 0.0]])
    directions = jp.array([[1.0, 0.0, 0.0]])
    check_no_self_intersection(intersect_canonical_cylinder, "Cylinder Precision", points, directions)

def test_cylinder_grazing_inward():
    points = jp.array([[1.0, 0.0, 0.0]])
    directions = jp.array([[-1e-4, 1.0, 0.0]])
    check_no_self_intersection(intersect_canonical_cylinder, "Cylinder Grazing Inward", points, directions)

def test_sphere_grazing_acne():
    points = jp.array([[1.0, 0.0, 0.0]])
    directions = jp.array([[0.0, 1.0, 0.0]]) 
    check_no_self_intersection(intersect_canonical_sphere, "Sphere Grazing", points, directions)

def test_cube_acne():
    points = jp.array([[1.0, 0.0, 0.0]])
    directions = jp.array([[1.0, 0.0, 0.0]])
    check_no_self_intersection(intersect_canonical_cube, "Cube Face", points, directions)

def test_plane_acne():
    points = jp.array([[0.0, 0.0, 0.0]])
    directions = jp.array([[0.0, 1.0, 0.0]])
    check_no_self_intersection(intersect_canonical_plane, "Plane", points, directions)

# --- Reflection/Bounce Acne Tests ---

def test_cylinder_grazing_bounce_acne():
    error = 1e-5
    point = jp.array([[1.0 - error, 0.0, 0.0]])
    R = jp.array([[0.005, 1.0, 0.0]])
    R = R / jp.linalg.norm(R)
    
    normal = jp.array([[1.0, 0.0, 0.0]])
    dot = jp.sum(R * normal, axis=-1, keepdims=True)
    offset = jp.sign(dot) * normal * EPSILON
    new_origin = point + offset
    
    hit_mask, depths, depth = intersect_canonical_cylinder(new_origin, R)
    is_hit = hit_mask[0]
    ray_depth = depth[0, 0]
    
    print(f"[CylinderBounce] Hit: {is_hit}, Depth: {ray_depth}")
    if is_hit:
        assert ray_depth > 1e-1, f"Acne detected! Ray failed to escape surface. Depth: {ray_depth}"

def test_cone_grazing_bounce_acne():
    error = 1e-5
    point = jp.array([[0.5 - error, -0.5, 0.0]])
    R = jp.array([[0.71, -0.70, 0.0]])
    R = R / jp.linalg.norm(R)
    
    inv_sqrt2 = 1.0 / jp.sqrt(2.0)
    normal = jp.array([[inv_sqrt2, inv_sqrt2, 0.0]])
    dot = jp.sum(R * normal, axis=-1, keepdims=True)
    offset = jp.sign(dot) * normal * EPSILON
    new_origin = point + offset
    
    hit_mask, depths, depth = intersect_canonical_cone(new_origin, R)
    is_hit = hit_mask[0]
    ray_depth = depth[0, 0]
    
    print(f"[ConeBounce] Hit: {is_hit}, Depth: {ray_depth}")
    if is_hit:
        assert ray_depth > 1e-1, f"Acne detected on Cone! Depth: {ray_depth}"

def test_cylinder_grazing_acne():
    point = jp.array([[1.0 - 1e-5, 0.0, 0.0]])
    incident = jp.array([[-1.0, 0.1, 0.0]])
    incident = incident / jp.linalg.norm(incident)
    check_reflection_acne(intersect_canonical_cylinder, compute_cylinder_wall_normals, "CylinderGrazing", point, incident)

def test_cone_grazing_acne():
    point = jp.array([[0.5 - 1e-5, -0.5, 0.0]])
    incident = jp.array([[-1.0, 0.0, 0.0]])
    incident = incident / jp.linalg.norm(incident)
    check_reflection_acne(intersect_canonical_cone, compute_canonical_normals_cone, "ConeGrazing", point, incident)

def test_cylinder_grazing_shallow_acne():
    point = jp.array([[1.0 - 1e-5, 0.0, 0.0]])
    incident = jp.array([[-0.01, 1.0, 0.0]])
    incident = incident / jp.linalg.norm(incident)
    check_reflection_acne(intersect_canonical_cylinder, compute_cylinder_wall_normals, "CylinderShallow", point, incident)

# --- Corner/Rim Acne Tests ---

def test_cylinder_bottom_corner_acne():
    point = jp.array([[0.9995, -1.0, 0.0]])
    D = jp.array([[0.707, -0.707, 0.0]])
    check_corner_escape(intersect_canonical_cylinder, compute_cylinder_lower_cap_normals, "CylinderBottomCorner", point, D)

def test_cone_base_corner_acne():
    point = jp.array([[0.9995, -1.0, 0.0]])
    D = jp.array([[0.707, -0.707, 0.0]])
    check_corner_escape(intersect_canonical_cone, compute_canonical_normals_cone, "ConeBaseCorner", point, D)

# --- Normal Investigation Tests (Correctness of Fixes) ---

def test_cylinder_body_near_cap_normal():
    point = jp.array([[1.0, 1.0 - 1e-4, 0.0]])
    normals = compute_canonical_normals_cylinder(point, build_cylinder_depths(len(point), 0))
    N = normals[0]
    print(f"Cyl Normal Near Cap: {N}")
    assert jp.abs(N[0] - 1.0) < 1e-2, f"Normal X should be 1.0, got {N[0]}"
    assert jp.abs(N[1] - 0.0) < 1e-2, f"Normal Y should be 0.0, got {N[1]}"

def test_cone_body_near_cap_normal():
    point = jp.array([[0.9999, -0.9999, 0.0]])
    normals = compute_canonical_normals_cone(point)
    N = normals[0]
    print(f"Cone Normal Near Cap: {N}")
    assert jp.abs(N[0] - 0.7071) < 1e-2, f"Normal X should be 0.7071, got {N[0]}"
    assert jp.abs(N[1] - 0.7071) < 1e-2, f"Normal Y should be 0.7071, got {N[1]}"

# --- Cone Center/Tip Tests ---

def test_cone_tip_normal_singularity():
    point = jp.array([[0.0, 0.0, 0.0]])
    normals = compute_canonical_normals_cone(point)
    N = normals[0]
    norm_val = jp.linalg.norm(N)
    assert norm_val > 0.5, f"Tip normal is zero! {N}"
    assert not jp.any(jp.isnan(N)), "Tip normal is NaN!"

def test_cone_axis_normal_stability():
    point = jp.array([[0.1, -0.1, 0.0]])
    normals = compute_canonical_normals_cone(point)
    N = normals[0]
    print(f"Shell Normal: {N}")
    assert jp.abs(jp.linalg.norm(N) - 1.0) < 1e-3, f"Normal not normalized! {jp.linalg.norm(N)}"
    assert N[0] > 0.5
    assert N[1] > 0.5
