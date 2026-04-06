import jax.numpy as jp
import paz

from paz.backend.mesh import build_laplacian, compute_volume


def legacy_build_laplacian(vertices, faces):
    yzx = faces[:, [1, 2, 0]].flatten()
    zxy = faces[:, [2, 0, 1]].flatten()
    yzx_zxy = jp.concatenate((yzx, zxy))
    zxy_yzx = jp.concatenate((zxy, yzx))
    pairs = jp.stack([yzx_zxy, zxy_yzx], axis=0)
    pairs = jp.unique(pairs, axis=1)
    ones = jp.ones(pairs.shape[1])
    diag_args = pairs[0]
    diag_pairs = jp.stack((diag_args, diag_args), axis=0)
    idx = jp.concatenate((pairs, diag_pairs), axis=1)
    values = jp.concatenate((-ones, ones))
    laplacian = jp.zeros((len(vertices), len(vertices)))
    for row, col, value in zip(idx[0], idx[1], values):
        prev = laplacian[row, col]
        laplacian = laplacian.at[row, col].set(jp.add(prev, value))
    return laplacian


def test_build_laplacian_matches_legacy():
    face_sets = [
        (3, jp.array([[0, 1, 2]], dtype=jp.int32)),
        (4, jp.array([[0, 1, 2], [0, 2, 3]], dtype=jp.int32)),
        (5, jp.array([[0, 1, 2], [2, 1, 3], [2, 3, 4]], dtype=jp.int32)),
    ]
    for num_vertices, faces in face_sets:
        vertices = jp.zeros((num_vertices, 3))
        result = build_laplacian(vertices, faces)
        expected = legacy_build_laplacian(vertices, faces)
        assert jp.array_equal(result, expected)


def test_build_laplacian_single_triangle():
    vertices = jp.zeros((3, 3))
    faces = jp.array([[0, 1, 2]], dtype=jp.int32)
    result = build_laplacian(vertices, faces)
    row0 = [2.0, -1.0, -1.0]
    row1 = [-1.0, 2.0, -1.0]
    row2 = [-1.0, -1.0, 2.0]
    expected = jp.array((row0, row1, row2))
    assert jp.array_equal(result, expected)


def test_build_laplacian_has_expected_graph_properties():
    vertices = jp.zeros((4, 3))
    faces = jp.array([[0, 1, 2], [0, 2, 3]], dtype=jp.int32)
    laplacian = build_laplacian(vertices, faces)
    expected_degree = jp.array([3.0, 2.0, 3.0, 2.0])
    row_sums = jp.sum(laplacian, axis=1)
    diagonal = jp.diag(laplacian)
    assert jp.array_equal(laplacian, laplacian.T)
    assert jp.allclose(row_sums, jp.zeros_like(row_sums))
    assert jp.array_equal(diagonal, expected_degree)


def test_compute_volume_matches_cube_volume():
    vertices, faces, _ = paz.graphics.mesh.build_cube(2.0)
    result = compute_volume(vertices, faces)
    assert jp.allclose(result, 8.0)


def test_compute_volume_is_invariant_to_face_winding():
    vertices, faces, _ = paz.graphics.mesh.build_cube(2.0)
    reversed_faces = faces[:, ::-1]
    volume = compute_volume(vertices, faces)
    reversed_volume = compute_volume(vertices, reversed_faces)
    assert jp.allclose(volume, reversed_volume)


def test_compute_volume_keeps_sphere_volume_after_face_flip():
    vertices, faces, _ = paz.graphics.mesh.build_sphere(1.0, 2)
    reversed_faces = faces[:, ::-1]
    volume = compute_volume(vertices, faces)
    reversed_volume = compute_volume(vertices, reversed_faces)
    assert jp.allclose(volume, reversed_volume)


def test_paz_mesh_exports_compute_volume():
    vertices, faces, _ = paz.graphics.mesh.build_cube(2.0)
    result = paz.mesh.compute_volume(vertices, faces)
    assert jp.allclose(result, 8.0)
