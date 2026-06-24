import numpy as np
import jax
import jax.numpy as jp

import paz
from paz.graphics.mesh import Mesh, load_mesh, build_cube, merge_meshes
from paz.graphics.types import Material, PointLight
from paz.graphics.viewer import mesh_renderer


def normalize_vertices(vertices):
    lower, upper = jp.min(vertices, axis=0), jp.max(vertices, axis=0)
    center = (lower + upper) / 2.0
    return (vertices - center) / jp.max(upper - lower)


def build_face_edges(faces):
    pairs = [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    return jp.concatenate(pairs, axis=0)


def colored_cube():
    vertices, faces, edges = build_cube(1.0)
    colors = (normalize_vertices(vertices) + 0.5)
    return vertices, faces, edges, jp.clip(colors, 0.0, 1.0)


def build_mesh(mesh_path=None):
    if mesh_path is None:
        vertices, faces, edges, colors = colored_cube()
    else:
        vertices, faces, colors = load_mesh(mesh_path)
        edges = build_face_edges(faces)
    vertices = normalize_vertices(vertices)
    material = Material(jp.zeros(3), 0.6, 0.0, 0.3, 32.0)
    transform = paz.SE3.identity()
    return Mesh(vertices, colors, transform, material, faces, edges)


def build_renderer(mesh, image_size, distance, y_FOV=jp.pi / 4.0):
    meshes, mask = merge_meshes(mesh)
    lights = [PointLight(jp.ones(3) * 1.6, jp.array([distance, distance, distance]))]  # fmt: skip
    H = W = image_size
    return mesh_renderer(meshes, mask, H, W, y_FOV, lights, 1024 * 8, (2, 2))


def camera_pose(rotation, distance):
    origin = rotation @ jp.array([0.0, 0.0, distance])
    world_up = rotation @ jp.array([0.0, 1.0, 0.0])
    return paz.SE3.view_transform(origin, jp.zeros(3), world_up)


def random_poses(key, num_views, distance):
    keys = jax.random.split(key, num_views)
    return [camera_pose(paz.SO3.sample(view_key), distance)
            for view_key in keys]


def grid_poses(theta_steps, phi_steps, distance):
    thetas = jp.linspace(0.2, jp.pi - 0.2, theta_steps)
    phis = jp.linspace(0.0, 2.0 * jp.pi, phi_steps, endpoint=False)
    poses = []
    for theta in thetas:
        for phi in phis:
            rotation = paz.SO3.rotation_y(phi) @ paz.SO3.rotation_x(theta)
            poses.append(camera_pose(rotation, distance))
    return poses


def render_views(render_fn, poses):
    views = [np.asarray(render_fn(pose)) for pose in poses]
    return np.stack(views).astype("uint8")
