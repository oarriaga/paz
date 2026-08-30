import jax.numpy as jp

from paz.graphics.constants import EPSILON, FARAWAY
from paz.graphics.shapes.cone import compute_canonical_normals_cone
from paz.graphics.shapes.cone import intersect_canonical_cone


def compute_shell_normals(points):
    depths = jp.full((4, len(points)), FARAWAY)
    return compute_canonical_normals_cone(points, depths.at[0].set(0.0))


def compute_lower_cap_normals(points):
    depths = jp.full((4, len(points)), FARAWAY)
    return compute_canonical_normals_cone(points, depths.at[2].set(0.0))


def compute_upper_tip_normals(points):
    depths = jp.full((4, len(points)), FARAWAY)
    return compute_canonical_normals_cone(points, depths.at[3].set(0.0))


def assert_avoids_self_hit(origins, directions):
    hit_mask, _, depth = intersect_canonical_cone(origins, directions)
    assert not bool(hit_mask[0]) or float(depth[0, 0]) > EPSILON


def assert_bounce_escapes(origins, directions):
    hit_mask, _, depth = intersect_canonical_cone(origins, directions)
    assert not bool(hit_mask[0]) or float(depth[0, 0]) > 0.1


def assert_reflection_escapes(compute_normals, point, incident):
    normal = compute_normals(point)[0]
    reflected = incident[0] - 2.0 * jp.dot(incident[0], normal) * normal
    reflected = reflected / jp.linalg.norm(reflected)
    offset = jp.sign(jp.dot(reflected, normal)) * normal * EPSILON
    directions = jp.expand_dims(reflected, 0)
    assert_bounce_escapes(point + offset, directions)


def assert_corner_escapes(compute_normals, point, directions):
    normal = compute_normals(point)[0]
    offset = jp.sign(jp.dot(directions[0], normal)) * normal * EPSILON
    assert_bounce_escapes(point + offset, directions)


def test_intersection_hit_wall():
    origins = jp.array([[0.0, -0.5, -5.0]])
    directions = jp.array([[0.0, 0.0, 1.0]])
    hit_mask, _, depth = intersect_canonical_cone(origins, directions)
    assert hit_mask[0]
    assert jp.allclose(depth[0], 4.5)


def test_intersection_hit_cap():
    origins = jp.array([[0.0, 5.0, 0.0]])
    directions = jp.array([[0.0, -1.0, 0.0]])
    hit_mask, _, depth = intersect_canonical_cone(origins, directions)
    assert hit_mask[0]
    assert jp.allclose(depth[0], 6.0)


def test_normal_on_cap():
    point = jp.array([[0.0, 0.0, 0.0]])
    depths = jp.array([[FARAWAY], [FARAWAY], [FARAWAY], [0.0]])
    expected_normal = jp.array([0.0, 1.0, 0.0])
    actual_normal = compute_canonical_normals_cone(point, depths)
    assert jp.allclose(actual_normal, expected_normal)


def test_normal_on_wall():
    point = jp.array([[0.3, -0.5, 0.4]])
    depths = jp.array([[0.0], [FARAWAY], [FARAWAY], [FARAWAY]])
    expected_normal = jp.array([0.3, 0.5, 0.4])
    actual_normal = compute_canonical_normals_cone(point, depths)
    expected_direction = expected_normal / jp.linalg.norm(expected_normal)
    actual_direction = actual_normal / jp.linalg.norm(actual_normal)
    assert jp.allclose(actual_direction, expected_direction)


def test_shell_normal_near_cap():
    point = jp.array([[0.9999, -0.9999, 0.0]])
    normal = compute_shell_normals(point)[0]
    assert jp.abs(normal[0] - 0.7071) < 1e-2
    assert jp.abs(normal[1] - 0.7071) < 1e-2


def test_shell_normal_near_axis_stays_normalized():
    point = jp.array([[0.1, -0.1, 0.0]])
    normal = compute_shell_normals(point)[0]
    assert jp.abs(jp.linalg.norm(normal) - 1.0) < 1e-3
    assert normal[0] > 0.5
    assert normal[1] > 0.5


def test_tip_normal_stays_finite():
    point = jp.array([[0.0, 0.0, 0.0]])
    normal = compute_upper_tip_normals(point)[0]
    assert jp.linalg.norm(normal) > 0.5
    assert not jp.any(jp.isnan(normal))


def test_surface_ray_avoids_self_hit():
    origins = jp.array([[0.5, -0.5, 0.0]])
    directions = jp.array([[1.0, 1.0, 0.0]])
    assert_avoids_self_hit(origins, directions)


def test_bottom_cap_ray_avoids_self_hit():
    origins = jp.array([[0.0, -1.0, 0.0]])
    directions = jp.array([[0.0, -1.0, 0.0]])
    assert_avoids_self_hit(origins, directions)


def test_junction_ray_avoids_self_hit():
    origins = jp.array([[1.0, -1.0, 0.0]])
    directions = jp.array([[1.0, -1.0, 0.0]])
    assert_avoids_self_hit(origins, directions)


def test_grazing_bounce_escapes_surface():
    point = jp.array([[0.5 - 1e-5, -0.5, 0.0]])
    reflected = jp.array([[0.71, -0.70, 0.0]])
    reflected = reflected / jp.linalg.norm(reflected)
    inverse_root_two = 1.0 / jp.sqrt(2.0)
    normal = jp.array([[inverse_root_two, inverse_root_two, 0.0]])
    dot = jp.sum(reflected * normal, axis=-1, keepdims=True)
    origins = point + jp.sign(dot) * normal * EPSILON
    assert_bounce_escapes(origins, reflected)


def test_reflected_grazing_ray_escapes_shell():
    point = jp.array([[0.5 - 1e-5, -0.5, 0.0]])
    incident = jp.array([[-1.0, 0.0, 0.0]])
    incident = incident / jp.linalg.norm(incident)
    assert_reflection_escapes(compute_shell_normals, point, incident)


def test_base_corner_ray_escapes_cap():
    point = jp.array([[0.9995, -1.0, 0.0]])
    directions = jp.array([[0.707, -0.707, 0.0]])
    assert_corner_escapes(compute_lower_cap_normals, point, directions)
