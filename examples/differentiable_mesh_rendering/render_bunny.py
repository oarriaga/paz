from pathlib import Path

import jax.numpy as jp
import paz
from paz.graphics.mesh import Mesh, load_mesh, merge_meshes
from paz.graphics.types import Material, PointLight
from paz.graphics.viewer import mesh_renderer, viewer

H, W = 512, 512
Y_FOV = jp.pi / 4.0


def normalize_vertices(vertices):
    bounds_min = jp.min(vertices, axis=0)
    bounds_max = jp.max(vertices, axis=0)
    center = (bounds_min + bounds_max) / 2.0
    max_extent = jp.max(bounds_max - bounds_min)
    return (vertices - center) / max_extent


def build_face_edges(faces):
    pairs = [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    return jp.concatenate(pairs, axis=0)


def make_bunny_mesh(path):
    vertices, faces, vertex_colors = load_mesh(path)
    vertices = normalize_vertices(vertices)
    edges = build_face_edges(faces)
    material = Material(jp.zeros(3), 0.15, 0.75, 0.25, 64.0)
    shift = paz.SE3.translation(jp.array([0.0, -0.12, 0.0]))
    # rotate = paz.SE3.rotation_y(jp.pi)
    # transform = shift @ rotate
    transform = shift
    args = (vertices, vertex_colors, transform, material, faces, edges)
    return Mesh(*args)


example_dir = Path(__file__).resolve().parent
mesh_path = example_dir / "dragon.obj"
mesh_path = "/home/dfki.uni-bremen.de/loarriagacamargo/Documents/Repositories/common-3d-test-models/data/nefertiti.obj"
mesh_path = example_dir / "bunny.obj"

camera_origin = jp.array([1.3, 0.55, -2.2])
camera_target = jp.array([0.0, -0.05, 0.0])
camera_up = jp.array([0.0, 1.0, 0.0])
camera_pose = paz.SE3.view_transform(camera_origin, camera_target, camera_up)

lights = [
    PointLight(jp.ones(3) * 1.4, camera_origin),
    PointLight(jp.array([0.6, 0.6, 0.7]), jp.array([-2.0, 3.0, 2.0])),
]

bunny = make_bunny_mesh(mesh_path)
meshes, mask = merge_meshes(bunny)

render_fn = mesh_renderer(meshes, mask, H, W, Y_FOV, lights, 1024 * 12, (4, 4))
viewer(render_fn, camera_pose, H=H, W=W)
