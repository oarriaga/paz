from pathlib import Path

import jax.numpy as jp
import paz
from paz.graphics import Scene
from paz.graphics.mesh import Mesh, load_mesh
from paz.graphics.types import Material, PointLight
from paz.graphics.viewer import scene_renderer, viewer

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
    shift = paz.SE3.translation(jp.array([0.0, 0.48, 0.0]))
    transform = shift
    args = (vertices, vertex_colors, transform, material, faces, edges)
    return Mesh(*args)


example_dir = Path(__file__).resolve().parent
mesh_path = example_dir / "dragon.obj"
mesh_root = "/home/dfki.uni-bremen.de/loarriagacamargo/Documents/Repositories"
mesh_path = f"{mesh_root}/common-3d-test-models/data/nefertiti.obj"
mesh_path = example_dir / "bunny.obj"

camera_origin = jp.array([1.3, 0.55, -2.2])
camera_target = jp.array([0.0, -0.05, 0.0])
camera_up = jp.array([0.0, 1.0, 0.0])
camera_pose = paz.SE3.view_transform(camera_origin, camera_target, camera_up)

lights = [
    PointLight(jp.array([0.6, 0.6, 0.7]), jp.array([-2.0, 3.0, 2.0])),
]

bunny = make_bunny_mesh(mesh_path)
plane = paz.graphics.Plane(material=paz.graphics.Material(jp.ones(3)))
scene = Scene([bunny, plane])

TILES = (4, 4)
tile_rays = (H * W) // (TILES[0] * TILES[1])
render_args = scene, H, W, Y_FOV, lights, True, tile_rays, TILES
render_fn = scene_renderer(*render_args)
viewer(render_fn, camera_pose, H=H, W=W)
