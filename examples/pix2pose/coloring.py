import os
import numpy as np
import trimesh
from pyrender import Mesh, Scene, Viewer
from pyrender.constants import RenderFlags


def normalize_min_max(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def load_obj(path):
    mesh = trimesh.load(path)
    return mesh


def extract_corners3D(vertices):
    point3D_min = np.min(vertices, axis=0)
    point3D_max = np.max(vertices, axis=0)
    return point3D_min, point3D_max


def compute_vertices_colors(vertices):
    corner3D_min, corner3D_max = extract_corners3D(vertices)
    normalized_colors = normalize_min_max(vertices, corner3D_min, corner3D_max)
    colors = (255 * normalized_colors).astype('uint8')
    return colors


def color_object(path):
    mesh = load_obj(path)
    colors = compute_vertices_colors(mesh.vertices)
    mesh.visual = mesh.visual.to_color()
    mesh.visual.vertex_colors = colors
    return mesh


if __name__ == "__main__":
    scene = Scene(bg_color=[0, 0, 0])
    root = os.path.expanduser('~')
    mesh_path = '.keras/paz/datasets/ycb_models/035_power_drill/textured.obj'
    path = os.path.join(root, mesh_path)
    mesh = color_object(path)
    mesh = Mesh.from_trimesh(mesh, smooth=False)
    mesh.primitives[0].material.metallicFactor = 0.0
    mesh.primitives[0].material.roughnessFactor = 1.0
    mesh.primitives[0].material.alphaMode = 'OPAQUE'
    scene.add(mesh)
    Viewer(scene, use_raymond_lighting=True, flags=RenderFlags.FLAT)
