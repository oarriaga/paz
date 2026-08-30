import jax.numpy as jp

from paz.graphics.mesh.geometry import build_edges
from paz.graphics.mesh.geometry import compute_canonical_normals
from paz.graphics.mesh.geometry import compute_position
from paz.graphics.mesh.geometry import extract_points
from paz.graphics.mesh.geometry import transform_points


def make_triangle():
    vertices = jp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    faces = jp.array([[0, 2, 1]])
    return vertices, faces


def test_extract_points():
    vertices, faces = make_triangle()
    A, B, C = extract_points(vertices, faces)
    assert jp.allclose(A[0], vertices[0])
    assert jp.allclose(B[0], vertices[2])
    assert jp.allclose(C[0], vertices[1])


def test_build_edges_shape():
    vertices, faces = make_triangle()
    edges_AC, edges_AB, points_A = build_edges(vertices, faces)
    assert edges_AC.shape == (1, 1, 3)
    assert edges_AB.shape == (1, 1, 3)
    assert points_A.shape == (1, 1, 3)


def test_compute_canonical_normals_direction():
    vertices, faces = make_triangle()
    shape_points = jp.zeros((1, 4, 3))
    normals = compute_canonical_normals(vertices, faces, shape_points)
    assert normals.shape == (1, 4, 3)
    assert jp.abs(normals[0, 0, 2]) > 0.9


def test_compute_canonical_normals_floor_points_up():
    half = 2.0
    vertices = jp.array(
        [
            [-half, 0.0, -half],
            [half, 0.0, -half],
            [half, 0.0, half],
            [-half, 0.0, half],
        ]
    )
    faces = jp.array([[0, 2, 1], [0, 3, 2]])
    shape_points = jp.zeros((2, 1, 3))
    normals = compute_canonical_normals(vertices, faces, shape_points)
    assert jp.all(normals[:, 0, 1] > 0.9)


def test_compute_position_shape():
    origins = jp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    directions = jp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    depths = jp.array([[[2.0], [3.0]], [[4.0], [5.0]]])
    positions = compute_position(origins, directions, depths)
    assert positions.shape == (2, 2, 3)


def test_transform_points_shape():
    points = jp.ones((3, 4, 3))
    affine = jp.eye(4)
    result = transform_points(affine, points)
    assert result.shape == (3, 4, 3)


def test_transform_points_identity():
    points = jp.array([[[1.0, 2.0, 3.0]]])
    affine = jp.eye(4)
    result = transform_points(affine, points)
    assert jp.allclose(result, points, atol=1e-5)
