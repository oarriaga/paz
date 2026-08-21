import jax.numpy as jp

from paz.graphics.mesh.builders import build_cube, build_sphere


def compute_face_centers_and_normals(vertices, faces):
    A = vertices[faces[:, 0]]
    B = vertices[faces[:, 1]]
    C = vertices[faces[:, 2]]
    centers = (A + B + C) / 3.0
    normals = jp.cross(B - A, C - A)
    return centers, normals


def test_build_cube():
    vertices, faces, edges = build_cube(1.0)
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert edges.shape[1] == 2
    assert len(vertices) == 8
    assert len(faces) == 12


def test_build_cube_faces_point_outward():
    vertices, faces, _ = build_cube(1.0)
    centers, normals = compute_face_centers_and_normals(vertices, faces)
    dots = jp.sum(centers * normals, axis=1)
    assert jp.all(dots > 0.0)


def test_build_sphere_faces_point_outward():
    vertices, faces, _ = build_sphere(1.0, 2)
    centers, normals = compute_face_centers_and_normals(vertices, faces)
    dots = jp.sum(centers * normals, axis=1)
    assert jp.all(dots > 0.0)
