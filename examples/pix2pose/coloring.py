from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import trimesh

from pyrender import PerspectiveCamera, OffscreenRenderer, DirectionalLight
from pyrender import Mesh, Scene, RenderFlags

from paz.backend.render import compute_modelview_matrices
from paz.backend.quaternion import quarternion_to_rotation_matrix, quarternion_to_rotation_matrix, quaternion_multiply


def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def color_object(path):
    mesh = trimesh.load(path)
    vertices = mesh.vertices

    # Slice the object along one symmetry axis
    #print(vertices[:, 2] >= 0)
    vertices_selected = vertices

    x_min = vertices_selected[:, 0].min()
    x_max = vertices_selected[:, 0].max()
    y_min = vertices_selected[:, 1].min()
    y_max = vertices_selected[:, 1].max()
    z_min = vertices_selected[:, 2].min()
    z_max = vertices_selected[:, 2].max()

    # make vertices using RGB format
    vertices_x = 255 * normalize(vertices_selected[:, 0:1], x_min, x_max)
    vertices_y = 255 * normalize(vertices_selected[:, 1:2], y_min, y_max)
    vertices_z = 255 * normalize(vertices_selected[:, 2:3], z_min, z_max)

    vertices_x = vertices_x.astype('uint8')
    vertices_y = vertices_y.astype('uint8')
    vertices_z = vertices_z.astype('uint8')
    colors = np.hstack([vertices_x, vertices_y, vertices_z])

    mesh.visual = mesh.visual.to_color()
    mesh.visual.vertex_colors = colors

    angle = np.pi/5
    mesh_rotated = deepcopy(mesh)
    mesh_rotated.rotation = np.array([0, np.sin(angle / 2), 0, np.cos(angle / 2)])

    return mesh, mesh_rotated


def color_object_two_colors(path):
    mesh = trimesh.load(path)
    vertices = mesh.vertices

    # Slice the object along one symmetry axis
    # print(vertices[:, 2] >= 0)
    vertices_selected = vertices

    x_min = vertices_selected[:, 0].min()
    x_max = vertices_selected[:, 0].max()
    y_min = vertices_selected[:, 1].min()
    y_max = vertices_selected[:, 1].max()
    z_min = vertices_selected[:, 2].min()
    z_max = vertices_selected[:, 2].max()

    # make vertices using RGB format
    vertices_x = 255 * np.ones_like(vertices_selected[:, 0:1])
    vertices_y = 255 * normalize(vertices_selected[:, 1:2], y_min, y_max)
    vertices_z = 255 * normalize(vertices_selected[:, 2:3], z_min, z_max)

    vertices_x = vertices_x.astype('uint8')
    vertices_y = vertices_y.astype('uint8')
    vertices_z = vertices_z.astype('uint8')
    colors = np.hstack([vertices_x, vertices_y, vertices_z])

    mesh.visual = mesh.visual.to_color()
    mesh.visual.vertex_colors = colors

    angle = np.pi / 5
    mesh_rotated = deepcopy(mesh)
    mesh_rotated.rotation = np.array([0, np.sin(angle / 2), 0, np.cos(angle / 2)])

    return mesh, mesh_rotated


def calculate_canonical_pose_cylinder(mesh):
    # Calculate canonical pose for cylindrical object, idea taken from here: https://arxiv.org/abs/1908.07640
    angle = np.pi*5.9
    mesh.rotation = np.array([0, np.sin(angle / 2), 0, np.cos(angle / 2)])
    rotation_matrix = quarternion_to_rotation_matrix(mesh.rotation)
    print("Rotation matrix: {}".format(quarternion_to_rotation_matrix(mesh.rotation)))

    angle_backrotation = np.arctan2(rotation_matrix[0, 2]-rotation_matrix[2, 0], rotation_matrix[0, 0]+rotation_matrix[2, 2])
    angle_backrotation = np.round(angle_backrotation, 4)
    print("Angle backrotation: {}".format(angle_backrotation))
    quaternion = quaternion_multiply(np.array([0, -np.sin(angle_backrotation / 2), 0, np.cos(angle_backrotation / 2)]), mesh.rotation)
    mesh.rotation = quaternion


if __name__ == "__main__":
    scene = Scene(bg_color=[0, 0, 0])
    y_fov = 3.14159/4.0
    distance = [0.3, 0.5]
    light_bounds = [0.5, 30]
    size = (400, 400)

    light = scene.add(DirectionalLight([1.0, 1.0, 1.0], 10))
    camera = scene.add(PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))
    camera_to_world, world_to_camera = compute_modelview_matrices(np.array([.4, .4, .4]), [0.0, 0.0, 0.0], None, None)

    print("Camera to world: {}".format(camera_to_world))
    scene.set_pose(camera, camera_to_world)
    scene.set_pose(light, camera_to_world)

    #mesh = trimesh.load("/home/fabian/.keras/datasets/custom_objects/symmetry_z_2_object.obj")
    mesh, mesh_rotated = color_object("/home/fabian/.keras/datasets/custom_objects/cylinder_object.obj")

    mesh = scene.add(Mesh.from_trimesh(mesh, smooth=False))
    calculate_canonical_pose_cylinder(mesh)
    print("Mesh rotation: {}".format(mesh.rotation))
    #mesh_rotated = scene.add(Mesh.from_trimesh(mesh_rotated, smooth=False))
    #angle = np.pi
    #mesh_rotated.rotation = np.array([0, np.sin(angle / 2), 0, np.cos(angle / 2)])

    renderer = OffscreenRenderer(size[0], size[1])
    image_original, depth_original = renderer.render(scene, flags=RenderFlags.FLAT)

    plt.imshow(image_original)
    plt.show()