import jax.numpy as jp

import paz

from paz.graphics.mesh.geometry import build_edges
from paz.graphics.mesh.geometry import compute_canonical_normals
from paz.graphics.mesh.geometry import compute_position
from paz.graphics.mesh.geometry import compute_vertex_normals
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


def test_build_edges_start_at_first_face_vertex():
    vertices, faces = make_triangle()
    edges_AC, edges_AB, points_A = build_edges(vertices, faces)
    assert jp.allclose(edges_AC, jp.array([[[1.0, 0.0, 0.0]]]))
    assert jp.allclose(edges_AB, jp.array([[[0.0, 1.0, 0.0]]]))
    assert jp.allclose(points_A, jp.zeros((1, 1, 3)))


def test_compute_canonical_normals_follow_face_winding():
    vertices, faces = make_triangle()
    shape_points = jp.zeros((1, 4, 3))
    normals = compute_canonical_normals(vertices, faces, shape_points)
    expected = jp.broadcast_to(jp.array([0.0, 0.0, -1.0]), (1, 4, 3))
    assert jp.allclose(normals, expected, atol=1e-4)


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


def test_compute_position_moves_each_ray_by_its_depth():
    origins = jp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    directions = jp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    depths = jp.array([[[2.0], [3.0]], [[4.0], [5.0]]])
    positions = compute_position(origins, directions, depths)
    first_face = [[0.0, 0.0, 2.0], [1.0, 0.0, 3.0]]
    second_face = [[0.0, 0.0, 4.0], [1.0, 0.0, 5.0]]
    assert jp.allclose(positions, jp.array([first_face, second_face]))


def test_transform_points_applies_translation():
    points = jp.array([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]])
    offset = jp.array([10.0, -2.0, 0.5])
    result = transform_points(paz.SE3.translation(offset), points)
    assert jp.allclose(result, points + offset, atol=1e-5)


def test_vertex_normals_of_a_sphere_point_outwards():
    vertices, faces, edges = paz.graphics.mesh.build_sphere(1.0, 2)
    normals = compute_vertex_normals(vertices, faces)
    radial = vertices / jp.linalg.norm(vertices, axis=-1, keepdims=True)
    assert jp.all(jp.sum(normals * radial, axis=-1) > 0.999)
