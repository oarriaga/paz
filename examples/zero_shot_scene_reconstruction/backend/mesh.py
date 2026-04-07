import jax
import jax.numpy as jp
import numpy as np
import trimesh
import paz

from .scene import to_render_material


def build_plane_mesh(x_size=1.0, y_size=1.0, subdivisions=4):
    x_half = x_size / 2.0
    y_half = y_size / 2.0
    vertices = np.array(
        [
            [-x_half, 0.0, -y_half],
            [x_half, 0.0, -y_half],
            [x_half, 0.0, y_half],
            [-x_half, 0.0, y_half],
        ]
    )
    faces = np.array([[0, 2, 1], [0, 3, 2]])
    for _ in range(subdivisions):
        vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    vertices = jp.array(mesh.vertices.view(np.ndarray))
    faces = jp.array(mesh.faces.view(np.ndarray))
    edges = jp.array(mesh.edges.view(np.ndarray))
    return vertices, faces, edges


def build_object_meshes(divisions, materials, transforms):
    verts, faces, edges = paz.graphics.mesh.build_sphere(1.0, divisions)
    num_objects = len(transforms)
    num_verts = len(verts)
    all_verts, all_colors = [], []
    for arg in range(num_objects):
        all_verts.append(verts)
        color = materials.color[arg]
        all_colors.append(jp.repeat(color[None], num_verts, axis=0))
    batched_verts = jp.array(all_verts)
    batched_colors = jp.array(all_colors)
    batched_faces = jp.tile(faces[None], (num_objects, 1, 1))
    batched_edges = jp.tile(edges[None], (num_objects, 1, 1))
    batched_uvs = jp.zeros((num_objects, num_verts, 2))
    render_mats = _batch_materials(materials)
    mesh_fields = (batched_verts, batched_colors, transforms, render_mats)
    mesh_fields = (*mesh_fields, batched_faces, batched_edges)
    return paz.graphics.mesh.Mesh(*mesh_fields, vertex_uvs=batched_uvs)


def _batch_materials(materials):
    n = len(materials.color)
    phong = (materials.color, materials.ambient, materials.diffuse)
    extra = (materials.specular, materials.shininess)
    defaults = (jp.zeros(n), jp.zeros(n), jp.ones(n))
    return paz.graphics.Material(*phong, *extra, *defaults)


def build_floor(floor_material, size=4.0, divisions=2):
    verts, faces, edges = build_plane_mesh(size, size, divisions)
    num_verts = len(verts)
    render_mat = to_render_material(floor_material)
    color = jp.repeat(floor_material.color[None], num_verts, axis=0)
    fields = (verts, color, jp.eye(4), render_mat, faces, edges)
    return paz.graphics.mesh.Mesh(*fields)


def append_mesh(batched, single):
    def _append(batch, element):
        return jp.concatenate([batch, element[None]], axis=0)

    return jax.tree.map(_append, batched, single)


def append_floor(meshes, floor):
    return append_mesh(meshes, _fill_floor(floor, meshes))


def _fill_floor(floor, batched_meshes):
    num_v = batched_meshes.vertices.shape[1]
    num_f = batched_meshes.faces.shape[1]
    num_e = batched_meshes.edges.shape[1]
    return paz.graphics.mesh.fill_mesh(floor, num_v, num_f, num_e)
