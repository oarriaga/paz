import numpy as np
import jax
import jax.numpy as jp

import paz
from paz.graphics import Scene
from paz.graphics.mesh import Mesh, load_mesh
from paz.graphics.types import Material, PointLight
from paz.graphics.viewer import scene_renderer

FACE_QUADS = [
    [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],
    [[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]],
    [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
    [[0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1]],
    [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    [[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
]
FACE_COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
               [1, 1, 0], [1, 0, 1], [0, 1, 1]]


def normalize_vertices(vertices):
    lower, upper = jp.min(vertices, axis=0), jp.max(vertices, axis=0)
    center = (lower + upper) / 2.0
    return (vertices - center) / jp.max(upper - lower)


def build_face_edges(faces):
    pairs = [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    return jp.concatenate(pairs, axis=0)


def colored_cube():
    vertices, faces, colors = [], [], []
    for face_arg, quad in enumerate(FACE_QUADS):
        corners = np.asarray(quad, "float32") - 0.5
        vertices.extend(corners.tolist())
        faces.extend(orient_outward(corners, 4 * face_arg))
        colors.extend([FACE_COLORS[face_arg]] * 4)
    vertices, faces = jp.asarray(vertices), jp.asarray(faces)
    return vertices, faces, cube_edges(), jp.asarray(colors)


def orient_outward(corners, base):
    triangles = [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    if np.dot(normal, corners.mean(axis=0)) < 0:
        triangles = [triangle[::-1] for triangle in triangles]
    return triangles


def cube_edges():
    edges = []
    for face_arg in range(len(FACE_QUADS)):
        ring = [4 * face_arg + offset for offset in range(4)]
        for corner in range(4):
            edges.append([ring[corner], ring[(corner + 1) % 4]])
    return jp.asarray(edges)


def build_mesh(mesh_path=None):
    if mesh_path is None:
        vertices, faces, edges, colors = colored_cube()
    else:
        vertices, faces, colors = load_mesh(mesh_path)
        edges = build_face_edges(faces)
    vertices = normalize_vertices(vertices)
    material = Material(jp.ones(3), 0.3, 0.8, 0.2, 32.0)
    transform = paz.SE3.identity()
    return Mesh(vertices, colors, transform, material, faces, edges)


def build_renderer(mesh, image_size, distance, y_FOV=jp.pi / 4.0):
    scene = Scene([mesh])
    position = jp.array([1.5, 2.0, 2.5]) * distance
    lights = [PointLight(jp.ones(3) * 1.6, position)]
    H = W = image_size
    args = scene, H, W, y_FOV, lights, False, 1024 * 8, (2, 2)
    return scene_renderer(*args)


def camera_pose(rotation, distance):
    origin = rotation @ jp.array([0.0, 0.0, distance])
    world_up = rotation @ jp.array([0.0, 1.0, 0.0])
    return paz.SE3.view_transform(origin, jp.zeros(3), world_up)


def random_poses(key, num_views, distance):
    poses = []
    for view_key in jax.random.split(key, num_views):
        poses.append(camera_pose(paz.SO3.sample(view_key), distance))
    return poses


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
