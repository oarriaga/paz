from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import trimesh
import random
from PIL import Image

from pyrender import PerspectiveCamera, OffscreenRenderer, DirectionalLight
from pyrender import Mesh, Scene, RenderFlags

from paz.backend.render import compute_modelview_matrices
from paz.backend.quaternion import quarternion_to_rotation_matrix, quarternion_to_rotation_matrix, quaternion_multiply


def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def color_star(mesh, symmetry_fold):
    vertices = mesh.vertices

    # Transform vertices in object coordinates to cylindric coordinates
    vertices_cylindric_coords = np.zeros_like(vertices)
    vertices_cylindric_coords[:, 0] = np.arctan2(vertices[:, 2], vertices[:, 0])
    vertices_cylindric_coords[:, 1] = vertices[:, 1]
    vertices_cylindric_coords[:, 2] = np.sqrt(vertices[:, 2]**2 + vertices[:, 0]**2)

    # Multiply by fold of symmetry
    vertices_cylindric_coords *= np.array([symmetry_fold, 1, 1])

    # Transform back to cartesian coordinates
    vertices_cartesian_coords = np.zeros_like(vertices)
    vertices_cartesian_coords[:, 0] = vertices_cylindric_coords[:, 2] * np.cos(vertices_cylindric_coords[:, 0])
    vertices_cartesian_coords[:, 1] = vertices_cylindric_coords[:, 1]
    vertices_cartesian_coords[:, 2] = vertices_cylindric_coords[:, 2] * np.sin(vertices_cylindric_coords[:, 0])

    x_min = vertices_cartesian_coords[:, 0].min()
    x_max = vertices_cartesian_coords[:, 0].max()
    y_min = vertices_cartesian_coords[:, 1].min()
    y_max = vertices_cartesian_coords[:, 1].max()
    z_min = vertices_cartesian_coords[:, 2].min()
    z_max = vertices_cartesian_coords[:, 2].max()

    # make vertices using RGB format
    vertices_x = 255 * normalize(vertices_cartesian_coords[:, 0:1], x_min, x_max)
    vertices_y = 255 * normalize(vertices_cartesian_coords[:, 1:2], y_min, y_max)
    vertices_z = 255 * normalize(vertices_cartesian_coords[:, 2:3], z_min, z_max)

    vertices_x = vertices_x.astype('uint8')
    vertices_y = vertices_y.astype('uint8')
    vertices_z = vertices_z.astype('uint8')
    colors = np.hstack([vertices_x, vertices_y, vertices_z])

    mesh.visual = mesh.visual.to_color()
    mesh.visual.vertex_colors = colors

    return mesh


if __name__ == "__main__":
    scene = Scene(bg_color=[0, 0, 0])
    y_fov = 3.14159/4.0
    distance = [0.3, 0.5]
    light_bounds = [0.5, 30]
    size = (128, 128)

    light = scene.add(DirectionalLight([1.0, 1.0, 1.0], 10))
    camera = scene.add(PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))
    camera_to_world, world_to_camera = compute_modelview_matrices(np.array([0., 0., .7]), [0.0, 0.0, 0.0], None, None)

    print("Camera to world: {}".format(camera_to_world))
    scene.set_pose(camera, camera_to_world)
    scene.set_pose(light, camera_to_world)

    #mesh, _ = color_object("/home/fabian/.keras/datasets/tless_obj/obj_000014.obj")
    mesh = trimesh.load("/home/fabian/.keras/datasets/custom_objects/simple_symmetry_object.obj")
    mesh = color_star(mesh, symmetry_fold=2)
    mesh = scene.add(Mesh.from_trimesh(mesh, smooth=False))

    angle = np.pi*1.4
    random_quaternion = trimesh.transformations.random_quaternion()
    random_quaternion = random_quaternion[[1, 2, 3, 0]]
    mesh.rotation = random_quaternion

    renderer = OffscreenRenderer(size[0], size[1])
    image_original, depth_original = renderer.render(scene, flags=RenderFlags.FLAT)

    plt.imshow(image_original)
    plt.show()