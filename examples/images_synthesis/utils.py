import os
import numpy as np
import trimesh
from pyrender import RenderFlags, Mesh, Scene, Viewer
from paz.backend.boxes import extract_bounding_box_corners
from paz.backend.image import normalize_min_max


def as_mesh(scene_or_mesh, scale=None):
    if scale is None:
        scale = [1.0, 1.0, 1.0]
    scale = np.asarray(scale)
    if hasattr(scene_or_mesh, "bounds") and scene_or_mesh.bounds is None:
        return None
    if isinstance(scene_or_mesh, trimesh.Scene):
        dump = scene_or_mesh.dump()
        mesh = dump.sum()
    else:
        mesh = scene_or_mesh
    assert isinstance(mesh, trimesh.Trimesh), f"Can't convert {type(scene_or_mesh)} to trimesh.Trimesh!"
    return mesh


def load_obj(path):
    mesh = as_mesh(trimesh.load(path))
    return mesh


def color_object(path):
    mesh = load_obj(path)
    colors = compute_vertices_colors(mesh.vertices)
    mesh.visual = mesh.visual.to_color()
    mesh.visual.vertex_colors = colors
    mesh = Mesh.from_trimesh(mesh, smooth=False)
    mesh.primitives[0].material.metallicFactor = 0.0
    mesh.primitives[0].material.roughnessFactor = 1.0
    mesh.primitives[0].material.alphaMode = 'OPAQUE'
    return mesh


def quick_color_visualize():
    scene = Scene(bg_color=[0, 0, 0])
    root = os.path.expanduser('~')
    mesh_path = '.keras/paz/datasets/ycb_models/035_power_drill/textured.obj'
    path = os.path.join(root, mesh_path)
    mesh = color_object(path)
    scene.add(mesh)
    Viewer(scene, use_raymond_lighting=True, flags=RenderFlags.FLAT)
    # mesh_extents = np.array([0.184, 0.187, 0.052])


def compute_vertices_colors(vertices):
    corner3D_min, corner3D_max = extract_bounding_box_corners(vertices)
    normalized_colors = normalize_min_max(vertices, corner3D_min, corner3D_max)
    colors = (255 * normalized_colors).astype('uint8')
    return colors
