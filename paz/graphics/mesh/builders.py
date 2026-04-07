import jax
import jax.numpy as jp

from .types import Mesh


def fill_bottom_with_last(x, total_size):
    if len(x) > total_size:
        raise ValueError("`x` length should be smaller than `total_size`")
    missing_size = total_size - len(x)
    last = x[-2:-1]
    repeated_last = jp.repeat(last, missing_size, axis=0)
    return jp.concatenate([x, repeated_last], axis=0)


def fill_mesh(mesh_to_fill, num_vertices, num_faces, num_edges):
    vertices = fill_bottom_with_last(mesh_to_fill.vertices, num_vertices)
    faces = fill_bottom_with_last(mesh_to_fill.faces, num_faces)
    edges = fill_bottom_with_last(mesh_to_fill.edges, num_edges)
    vertex_colors = mesh_to_fill.vertex_colors
    vertex_colors = fill_bottom_with_last(vertex_colors, num_vertices)
    vertex_uvs = mesh_to_fill.vertex_uvs
    if vertex_uvs is None:
        vertex_uvs = jp.zeros((len(mesh_to_fill.vertices), 2))
    vertex_uvs = fill_bottom_with_last(vertex_uvs, num_vertices)
    return Mesh(
        vertices,
        vertex_colors,
        mesh_to_fill.transform,
        mesh_to_fill.material,
        faces,
        edges,
        mesh_to_fill.pattern,
        vertex_uvs,
    )


def merge_meshes(*meshes):
    max_vertices = max(mesh.vertices.shape[0] for mesh in meshes)
    max_faces = max(mesh.faces.shape[0] for mesh in meshes)
    max_edges = max(mesh.edges.shape[0] for mesh in meshes)
    args = (max_vertices, max_faces, max_edges)
    filled = [fill_mesh(mesh, *args) for mesh in meshes]
    batched = jax.tree.map(lambda *args: jp.stack(args), *filled)
    mask = jp.ones(len(meshes), dtype=bool)
    return batched, mask


def build_cube(size=1.0):
    import trimesh
    import numpy as onp

    mesh = trimesh.creation.box(extents=[size, size, size])
    vertices = jp.array(mesh.vertices.view(onp.ndarray))
    faces = jp.array(mesh.faces.view(onp.ndarray))
    edges = jp.array(mesh.edges.view(onp.ndarray))
    return vertices, faces, edges


def build_sphere(radius=1.0, subdivisions=3):
    import trimesh
    import numpy as onp

    mesh = trimesh.creation.icosphere(subdivisions, radius)
    vertices = jp.array(mesh.vertices.view(onp.ndarray))
    faces = jp.array(mesh.faces.view(onp.ndarray))
    edges = jp.array(mesh.edges.view(onp.ndarray))
    return vertices, faces, edges


def load_mesh(filepath):
    import trimesh
    import numpy as onp

    mesh = trimesh.load(filepath)
    vertices = jp.array(mesh.vertices.view(onp.ndarray))
    faces = jp.array(mesh.faces.view(onp.ndarray))
    vertex_colors = mesh.visual.vertex_colors[:, :3]
    vertex_colors = jp.array(vertex_colors.view(onp.ndarray))
    vertex_colors = vertex_colors / 255.0
    return vertices, faces, vertex_colors
