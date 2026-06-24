import numpy as np
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


def view_pose(theta, phi, distance):
    x = distance * np.sin(theta) * np.cos(phi)
    y = distance * np.cos(theta)
    z = distance * np.sin(theta) * np.sin(phi)
    origin = jp.array([x, y, z])
    up = jp.array([0.0, 1.0, 0.0])
    return paz.SE3.view_transform(origin, jp.zeros(3), up)


def random_poses(num_views, distance, top_only, seed):
    rng = np.random.default_rng(seed)
    upper = (np.pi / 2.0) if top_only else np.pi
    thetas = rng.uniform(0.2, upper - 0.2, num_views)
    phis = rng.uniform(0.0, 2.0 * np.pi, num_views)
    return [view_pose(t, p, distance) for t, p in zip(thetas, phis)]


def grid_poses(theta_steps, phi_steps, distance, top_only):
    upper = (np.pi / 2.0) if top_only else np.pi
    thetas = np.linspace(0.2, upper - 0.2, theta_steps)
    phis = np.linspace(0.0, 2.0 * np.pi, phi_steps, endpoint=False)
    return [view_pose(t, p, distance) for t in thetas for p in phis]


def render_views(render_fn, poses):
    views = [np.asarray(render_fn(pose)) for pose in poses]
    return np.stack(views).astype("float32") / 255.0
