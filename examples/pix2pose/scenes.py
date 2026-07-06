import numpy as np
import jax
import jax.numpy as jp
import trimesh
from scipy.spatial import cKDTree
from keras.utils import get_file

import paz
from paz.graphics.mesh import Mesh, merge_meshes, render_coordinates
from paz.graphics.types import Material, PointLight
from paz.graphics.viewer import mesh_renderer

MESH_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.12/"
MESH_FILES = ["texture_map.png", "textured.mtl", "textured.obj"]


def download_power_drill():
    subdir = "paz/meshes/035_power_drill"
    for filename in MESH_FILES:
        path = get_file(filename, MESH_URL + filename, cache_subdir=subdir)
    return path


def build_face_edges(faces):
    pairs = [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    return jp.concatenate(pairs, axis=0)


def simplify_mesh(mesh, target_faces):
    if hasattr(mesh, "simplify_quadric_decimation"):
        return mesh.simplify_quadric_decimation(target_faces)
    return mesh.simplify_quadratic_decimation(target_faces)


def transfer_vertex_colors(source_vertices, source_colors, vertices):
    nearest = cKDTree(source_vertices).query(vertices)[1]
    return source_colors[nearest]


def load_textured_mesh(mesh_path, target_faces):
    mesh = trimesh.load(mesh_path, process=False)
    mesh.visual = mesh.visual.to_color()
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0
    if target_faces is None or target_faces >= len(mesh.faces):
        return vertices, np.asarray(mesh.faces), colors
    decimated = simplify_mesh(mesh, target_faces)
    new_vertices = np.asarray(decimated.vertices)
    colors = transfer_vertex_colors(vertices, colors, new_vertices)
    return new_vertices, np.asarray(decimated.faces), colors


def build_mesh(mesh_path, target_faces=20000):
    vertices, faces, colors = load_textured_mesh(mesh_path, target_faces)
    vertices = jp.asarray(vertices, "float32")
    faces = jp.asarray(faces, "int32")
    center = (jp.min(vertices, axis=0) + jp.max(vertices, axis=0)) / 2.0
    vertices = vertices - center
    edges = build_face_edges(faces)
    material = Material(jp.ones(3), 0.4, 0.8, 0.2, 32.0)
    colors = jp.asarray(colors, "float32")
    return Mesh(vertices, colors, paz.SE3.identity(), material, faces, edges)


def object_extents(mesh):
    return jp.max(mesh.vertices, axis=0) - jp.min(mesh.vertices, axis=0)


def build_image_renderer(mesh, size, distance, y_FOV, chunk_size, tiles):
    meshes, mask = merge_meshes(mesh)
    position = jp.array([1.5, 2.0, 2.5]) * distance
    lights = [PointLight(jp.ones(3) * 1.6, position)]
    H, W = size
    return mesh_renderer(meshes, mask, H, W, y_FOV, lights, chunk_size, tiles)


def build_coordinate_renderer(mesh, size, y_FOV, chunk_size):
    lower = jp.min(mesh.vertices, axis=0)
    upper = jp.max(mesh.vertices, axis=0)

    @jax.jit
    def render_nocs(pose):
        coordinates, hit = render_coordinates(size, y_FOV, pose, mesh, chunk_size)  # fmt: skip
        nocs = (coordinates - lower) / (upper - lower)
        nocs = nocs * hit[..., None]
        return nocs, hit.astype("float32")

    return render_nocs


def camera_pose(rotation, distance):
    origin = rotation @ jp.array([0.0, 0.0, distance])
    world_up = rotation @ jp.array([0.0, 1.0, 0.0])
    return paz.SE3.view_transform(origin, jp.zeros(3), world_up)


def sample_pose(key, distance_range):
    rotation_key, distance_key = jax.random.split(key)
    distance = jax.random.uniform(distance_key, (), minval=distance_range[0],
                                   maxval=distance_range[1])
    return camera_pose(paz.SO3.sample(rotation_key), distance)
