import jax.numpy as jp

from paz.graphics.constants import EPSILON
from paz.graphics.shapes.plane import compute_canonical_normals_plane
from paz.graphics.shapes.plane import intersect_canonical_plane


def test_intersection_hit():
    origins = jp.array([[0.0, 10.0, 0.0]])
    directions = jp.array([[0.0, -1.0, 0.0]])
    hit_mask, _, depth = intersect_canonical_plane(origins, directions)
    assert hit_mask[0]
    assert jp.allclose(depth[0], 10.0)


def test_intersection_miss_parallel():
    origins = jp.array([[0.0, 10.0, 0.0]])
    directions = jp.array([[1.0, 0.0, 0.0]])
    hit_mask, _, _ = intersect_canonical_plane(origins, directions)
    assert not hit_mask[0]


def test_normal():
    point = jp.array([[10.0, 0.0, -5.0]])
    expected_normal = jp.array([0.0, 1.0, 0.0])
    actual_normal = compute_canonical_normals_plane(point)
    assert jp.allclose(actual_normal, expected_normal)


def test_surface_ray_avoids_self_hit():
    origins = jp.array([[0.0, 0.0, 0.0]])
    directions = jp.array([[0.0, 1.0, 0.0]])
    hit_mask, _, depth = intersect_canonical_plane(origins, directions)
    assert not bool(hit_mask[0]) or float(depth[0, 0]) > EPSILON
