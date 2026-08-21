import jax.numpy as jp

from paz.graphics.constants import FARAWAY
from paz.graphics.mesh.intersect import intersect_canonical_mesh


def make_triangle():
    vertices = jp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    faces = jp.array([[0, 2, 1]])
    return vertices, faces


def test_intersect_canonical_mesh_hit():
    vertices, faces = make_triangle()
    origins = jp.array([[0.25, 0.25, -1.0]])
    directions = jp.array([[0.0, 0.0, 1.0]])
    args = vertices, faces, origins, directions
    hit_mask, depth, _, _ = intersect_canonical_mesh(*args)
    assert hit_mask[0, 0]
    assert jp.allclose(depth[0, 0], 1.0, atol=1e-5)


def test_intersect_canonical_mesh_miss():
    vertices, faces = make_triangle()
    origins = jp.array([[5.0, 5.0, -1.0]])
    directions = jp.array([[0.0, 0.0, 1.0]])
    args = vertices, faces, origins, directions
    hit_mask, depth, _, _ = intersect_canonical_mesh(*args)
    assert not hit_mask[0, 0]
    assert jp.allclose(depth[0, 0], FARAWAY)


def test_intersect_canonical_mesh_miss_returns_faraway():
    vertices, faces = make_triangle()
    origins = jp.array([[-1.0, -1.0, -1.0]])
    directions = jp.array([[0.0, 0.0, 1.0]])
    args = vertices, faces, origins, directions
    _, depth, _, _ = intersect_canonical_mesh(*args)
    assert depth[0, 0] >= FARAWAY - 1.0


def test_intersect_canonical_mesh_rejects_negative_depth():
    vertices, faces = make_triangle()
    origins = jp.array([[0.25, 0.25, 1.0]])
    directions = jp.array([[0.0, 0.0, 1.0]])
    args = vertices, faces, origins, directions
    hit_mask, depth, _, _ = intersect_canonical_mesh(*args)
    assert not hit_mask[0, 0]
    assert jp.allclose(depth[0, 0], FARAWAY)


def test_intersect_canonical_mesh_rejects_parallel_ray():
    vertices, faces = make_triangle()
    origins = jp.array([[0.25, 0.25, -1.0]])
    directions = jp.array([[1.0, 0.0, 0.0]])
    args = vertices, faces, origins, directions
    hit_mask, depth, _, _ = intersect_canonical_mesh(*args)
    assert not hit_mask[0, 0]
    assert jp.allclose(depth[0, 0], FARAWAY)
