import jax.numpy as jp

import paz
from paz.graphics.constants import EPSILON
from paz.graphics.shapes.cube import compute_canonical_normals_cube
from paz.graphics.shapes.cube import intersect_canonical_cube


def assert_avoids_self_hit(origins, directions):
    hit_mask, _, depth = intersect_canonical_cube(origins, directions)
    assert not bool(hit_mask[0]) or float(depth[0, 0]) > EPSILON


def test_intersection_hit():
    origins = jp.array([[0.0, 0.0, -5.0]])
    directions = jp.array([[0.0, 0.0, 1.0]])
    hit_mask, _, depth = intersect_canonical_cube(origins, directions)
    assert hit_mask[0]
    assert jp.allclose(depth[0], 4.0)


def test_intersection_miss():
    origins = jp.array([[0.0, 2.0, -5.0]])
    directions = jp.array([[0.0, 0.0, 1.0]])
    hit_mask, _, _ = intersect_canonical_cube(origins, directions)
    assert not hit_mask[0]


def test_normal_on_positive_x_face():
    point = jp.array([[1.0, 0.5, -0.5]])
    expected_normal = jp.array([1.0, 0.0, 0.0])
    actual_normal = compute_canonical_normals_cube(point)
    assert jp.allclose(actual_normal, expected_normal)


def test_normal_on_negative_y_face():
    point = jp.array([[0.2, -1.0, -0.8]])
    expected_normal = jp.array([0.0, -1.0, 0.0])
    actual_normal = compute_canonical_normals_cube(point)
    assert jp.allclose(actual_normal, expected_normal)


def test_surface_ray_avoids_self_hit():
    origins = jp.array([[1.0, 0.0, 0.0]])
    directions = jp.array([[1.0, 0.0, 0.0]])
    assert_avoids_self_hit(origins, directions)


def test_inset_ray_hits_exit_face():
    origins = jp.array([[1.0 - 1e-3, 0.0, 0.0]])
    directions = jp.array([[1.0, 0.0, 0.0]])
    hit_mask, _, depth = intersect_canonical_cube(origins, directions)
    assert hit_mask[0]
    assert jp.allclose(depth[0], 1e-3, atol=1e-6)


def test_intersection_from_inside_hits_exit_face():
    origins = jp.array([[0.0, 0.0, 0.0]])
    directions = jp.array([[0.0, 0.0, -1.0]])
    hit_mask, _, depth = intersect_canonical_cube(origins, directions)
    assert hit_mask[0]
    assert jp.allclose(depth[0], 1.0)


def test_oblique_intersection_from_inside_hits_exit_face():
    origins = jp.array([[0.2, 0.1, 0.0]])
    directions = jp.array([[1.0, 0.3, -0.2]])
    hit_mask, _, depth = intersect_canonical_cube(origins, directions)
    assert hit_mask[0]
    assert jp.allclose(depth[0], 0.8)


def test_intersection_behind_origin_misses():
    origins = jp.array([[0.0, 0.0, 5.0]])
    directions = jp.array([[0.0, 0.0, 1.0]])
    hit_mask, _, _ = intersect_canonical_cube(origins, directions)
    assert not hit_mask[0]


def test_inside_hit_normal_faces_ray_origin():
    origins = jp.array([[0.0, 0.0, 0.0]])
    directions = jp.array([[0.0, 0.0, -1.0]])
    args = paz.graphics.Cube(), origins, directions
    hit_mask, _, _, normals, _, _ = paz.graphics.shapes.intersect(*args)
    assert hit_mask[0]
    assert jp.allclose(normals[0], jp.array([0.0, 0.0, 1.0]), atol=1e-4)
