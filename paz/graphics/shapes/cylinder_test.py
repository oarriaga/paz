import jax
import jax.numpy as jp

import paz
from paz.graphics.constants import EPSILON, FARAWAY
from paz.graphics.shapes.cylinder import compute_canonical_normals_cylinder
from paz.graphics.shapes.cylinder import intersect_canonical_cylinder


def compute_wall_normals(points):
    depths = jp.full((4, len(points)), FARAWAY)
    return compute_canonical_normals_cylinder(points, depths.at[0].set(0.0))


def compute_lower_cap_normals(points):
    depths = jp.full((4, len(points)), FARAWAY)
    return compute_canonical_normals_cylinder(points, depths.at[2].set(0.0))


def assert_avoids_self_hit(origins, directions):
    hit_mask, _, depth = intersect_canonical_cylinder(origins, directions)
    assert not bool(hit_mask[0]) or float(depth[0, 0]) > EPSILON


def assert_bounce_escapes(origins, directions):
    hit_mask, _, depth = intersect_canonical_cylinder(origins, directions)
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
    origins = jp.array([[0.0, 0.5, -5.0]])
    directions = jp.array([[0.0, 0.0, 1.0]])
    hit_mask, _, depth = intersect_canonical_cylinder(origins, directions)
    assert hit_mask[0]
    assert jp.allclose(depth[0], 4.0)


def test_intersection_hit_top_cap():
    origins = jp.array([[0.5, 5.0, 0.0]])
    directions = jp.array([[0.0, -1.0, 0.0]])
    hit_mask, _, depth = intersect_canonical_cylinder(origins, directions)
    assert hit_mask[0]
    assert jp.allclose(depth[0], 4.0)


def test_normal_on_top_cap():
    point = jp.array([[0.5, 1.0, 0.5]])
    depths = jp.array([[FARAWAY], [FARAWAY], [FARAWAY], [0.0]])
    expected_normal = jp.array([0.0, 1.0, 0.0])
    actual_normal = compute_canonical_normals_cylinder(point, depths)
    assert jp.allclose(actual_normal, expected_normal)


def test_normal_on_wall():
    point = jp.array([[1.0, 0.5, 0.0]])
    depths = jp.array([[0.0], [FARAWAY], [FARAWAY], [FARAWAY]])
    expected_normal = jp.array([1.0, 0.0, 0.0])
    actual_normal = compute_canonical_normals_cylinder(point, depths)
    assert jp.allclose(actual_normal, expected_normal)


def test_wall_normal_near_cap():
    point = jp.array([[1.0, 1.0 - 1e-4, 0.0]])
    normal = compute_wall_normals(point)[0]
    assert jp.abs(normal[0] - 1.0) < 1e-2
    assert jp.abs(normal[1]) < 1e-2


def test_rotated_intersection_keeps_wall_normal():
    with jax.default_device(jax.devices("cpu")[0]):
        translation = paz.SE3.translation(jp.array([0.0, 0.0, 1.0]))
        scaling = paz.SE3.scaling(jp.array([1.3, 1.3, 0.2]))
        transform = translation @ scaling @ paz.SE3.rotation_x(jp.pi / 2)
        shape = paz.graphics.Cylinder(transform)
        seam_point = jp.array([[1.0, 1.0 - 1e-4, 0.0]])
        world_point = paz.algebra.transform_points(transform, seam_point)
        ray_origin = world_point + jp.array([[2.0, 0.0, 0.0]])
        ray_direction = jp.array([[-1.0, 0.0, 0.0]])
        args = shape, ray_origin, ray_direction
        intersection = paz.graphics.shapes.intersect(*args)
        hit_mask, _, hit_point, world_normals, _ = intersection
        assert hit_mask[0]
        assert jp.allclose(hit_point, world_point, atol=1e-5)
        assert world_normals[0, 0] > 0.95
        assert jp.abs(world_normals[0, 2]) < 0.1


def test_rotated_oblique_cap_hit_keeps_cap_normal():
    with jax.default_device(jax.devices("cpu")[0]):
        translation = paz.SE3.translation(jp.array([0.0, 0.0, 1.05]))
        scaling = paz.SE3.scaling(jp.array([1.3, 1.3, 0.2]))
        transform = translation @ scaling @ paz.SE3.rotation_x(jp.pi / 2)
        shape = paz.graphics.Cylinder(transform)
        eye = jp.array([5.656854, 4.0, 5.656854])
        target = jp.array([0.0, 0.0, 1.0])
        up = jp.array([0.0, 1.0, 0.0])
        pose = paz.SE3.view_transform(eye, target, up)
        rays = paz.graphics.camera.build_rays((128, 128), jp.pi / 3.0, pose)
        ray_origin = rays[0][6456:6457]
        ray_direction = rays[1][6456:6457]
        args = shape, ray_origin, ray_direction
        intersection = paz.graphics.shapes.intersect(*args)
        hit_mask, _, hit_point, world_normals, _ = intersection
        inverse_transpose = jp.linalg.inv(transform).T
        cap_normal = jp.array([[0.0, 1.0, 0.0]])
        expected = paz.algebra.transform_points(inverse_transpose, cap_normal)
        expected_normal = paz.algebra.normalize(expected)
        assert hit_mask[0]
        assert jp.allclose(hit_point[0, 2], 1.25, atol=1e-5)
        assert jp.allclose(world_normals, expected_normal, atol=1e-5)


def test_surface_ray_avoids_self_hit():
    origins = jp.array([[1.0, 0.0, 0.0]])
    directions = jp.array([[1.0, 0.0, 0.0]])
    assert_avoids_self_hit(origins, directions)


def test_inset_surface_ray_avoids_self_hit():
    origins = jp.array([[1.0 - 1e-5, 0.0, 0.0]])
    directions = jp.array([[1.0, 0.0, 0.0]])
    assert_avoids_self_hit(origins, directions)


def test_top_cap_ray_avoids_self_hit():
    origins = jp.array([[0.0, 1.0, 0.0]])
    directions = jp.array([[0.0, 1.0, 0.0]])
    assert_avoids_self_hit(origins, directions)


def test_bottom_cap_ray_avoids_self_hit():
    origins = jp.array([[0.0, -1.0, 0.0]])
    directions = jp.array([[0.0, -1.0, 0.0]])
    assert_avoids_self_hit(origins, directions)


def test_top_junction_ray_avoids_self_hit():
    origins = jp.array([[1.0, 1.0, 0.0]])
    directions = jp.array([[1.0, 1.0, 0.0]])
    assert_avoids_self_hit(origins, directions)


def test_grazing_inward_ray_avoids_self_hit():
    origins = jp.array([[1.0, 0.0, 0.0]])
    directions = jp.array([[-1e-4, 1.0, 0.0]])
    assert_avoids_self_hit(origins, directions)


def test_grazing_bounce_escapes_surface():
    point = jp.array([[1.0 - 1e-5, 0.0, 0.0]])
    reflected = jp.array([[0.005, 1.0, 0.0]])
    reflected = reflected / jp.linalg.norm(reflected)
    normal = jp.array([[1.0, 0.0, 0.0]])
    dot = jp.sum(reflected * normal, axis=-1, keepdims=True)
    origins = point + jp.sign(dot) * normal * EPSILON
    assert_bounce_escapes(origins, reflected)


def test_reflected_grazing_ray_escapes_wall():
    point = jp.array([[1.0 - 1e-5, 0.0, 0.0]])
    incident = jp.array([[-1.0, 0.1, 0.0]])
    incident = incident / jp.linalg.norm(incident)
    assert_reflection_escapes(compute_wall_normals, point, incident)


def test_reflected_shallow_ray_escapes_wall():
    point = jp.array([[1.0 - 1e-5, 0.0, 0.0]])
    incident = jp.array([[-0.01, 1.0, 0.0]])
    incident = incident / jp.linalg.norm(incident)
    assert_reflection_escapes(compute_wall_normals, point, incident)


def test_bottom_corner_ray_escapes_cap():
    point = jp.array([[0.9995, -1.0, 0.0]])
    directions = jp.array([[0.707, -0.707, 0.0]])
    assert_corner_escapes(compute_lower_cap_normals, point, directions)
