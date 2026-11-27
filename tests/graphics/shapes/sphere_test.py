import jax.numpy as jp

from paz.graphics.constants import EPSILON, FARAWAY
from paz.graphics.shapes.sphere import compute_canonical_normals_sphere
from paz.graphics.shapes.sphere import intersect_canonical_sphere


def assert_avoids_self_hit(origins, directions):
    hit_mask, _, depth = intersect_canonical_sphere(origins, directions)
    assert not bool(hit_mask[0]) or float(depth[0, 0]) > EPSILON


def test_intersection_hit():
    origins = jp.array([[0.0, 0.0, -5.0]])
    directions = jp.array([[0.0, 0.0, 1.0]])
    hit_mask, _, depth = intersect_canonical_sphere(origins, directions)
    assert hit_mask[0]
    assert jp.allclose(depth[0], 4.0)


def test_intersection_miss():
    origins = jp.array([[0.0, 2.0, -5.0]])
    directions = jp.array([[0.0, 0.0, 1.0]])
    hit_mask, _, depth = intersect_canonical_sphere(origins, directions)
    assert not hit_mask[0]
    assert jp.allclose(depth[0], FARAWAY)


def test_normal_at_pole():
    point = jp.array([[0.0, 1.0, 0.0]])
    expected_normal = jp.array([0.0, 1.0, 0.0])
    actual_normal = compute_canonical_normals_sphere(point)
    assert jp.allclose(actual_normal, expected_normal)


def test_normal_at_equator():
    point = jp.array([[1.0, 0.0, 0.0]])
    expected_normal = jp.array([1.0, 0.0, 0.0])
    actual_normal = compute_canonical_normals_sphere(point)
    assert jp.allclose(actual_normal, expected_normal)


def test_surface_ray_avoids_self_hit():
    origins = jp.array([[1.0, 0.0, 0.0]])
    directions = jp.array([[1.0, 0.0, 0.0]])
    assert_avoids_self_hit(origins, directions)


def test_inset_surface_ray_avoids_self_hit():
    origins = jp.array([[1.0 - 1e-5, 0.0, 0.0]])
    directions = jp.array([[1.0, 0.0, 0.0]])
    assert_avoids_self_hit(origins, directions)


def test_grazing_ray_avoids_self_hit():
    origins = jp.array([[1.0, 0.0, 0.0]])
    directions = jp.array([[0.0, 1.0, 0.0]])
    assert_avoids_self_hit(origins, directions)


def test_grazing_inward_ray_avoids_self_hit():
    origins = jp.array([[1.0, 0.0, 0.0]])
    directions = jp.array([[-1e-4, 1.0, 0.0]])
    assert_avoids_self_hit(origins, directions)
