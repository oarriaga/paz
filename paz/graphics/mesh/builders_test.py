import jax.numpy as jp

import paz
from paz.graphics.mesh.builders import build_cube, build_sphere, load_mesh


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


def write_quad(directory, header, texture_coordinates, faces):
    lines = header + ["v 0 0 0", "v 1 0 0", "v 1 1 0", "v 0 1 0"]
    path = directory / "quad.obj"
    path.write_text("\n".join(lines + texture_coordinates + faces))
    return path


def write_textured_quad(directory):
    top_row = [[0, 255, 0], [255, 255, 0]]
    bottom_row = [[0, 0, 0], [255, 0, 0]]
    texture = jp.array([top_row, bottom_row], dtype=jp.uint8)
    paz.image.write(directory / "texture.png", texture)
    material = "newmtl textured\nmap_Kd texture.png\n"
    (directory / "quad.mtl").write_text(material)
    header = ["mtllib quad.mtl", "usemtl textured"]
    texture_coordinates = ["vt 0.25 0.25", "vt 0.75 0.25"]
    texture_coordinates += ["vt 0.75 0.75", "vt 0.25 0.75"]
    faces = ["f 1/1 2/2 3/3", "f 1/1 3/3 4/4"]
    return write_quad(directory, header, texture_coordinates, faces)


def test_load_mesh_reads_texture_colors_at_vertices(tmp_path):
    vertices, faces, vertex_colors = load_mesh(write_textured_quad(tmp_path))
    assert faces.shape == (2, 3)
    assert jp.allclose(vertex_colors, vertices)


def test_load_mesh_reads_untextured_mesh(tmp_path):
    path = write_quad(tmp_path, [], [], ["f 1 2 3", "f 1 3 4"])
    vertices, faces, vertex_colors = load_mesh(path)
    assert vertices.shape == (4, 3)
    assert faces.shape == (2, 3)
    assert vertex_colors.shape == (4, 3)
    assert jp.all((vertex_colors >= 0.0) & (vertex_colors <= 1.0))
