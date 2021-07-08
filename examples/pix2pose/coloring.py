from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import trimesh

from pyrender import PerspectiveCamera, OffscreenRenderer, DirectionalLight
from pyrender import Mesh, Scene, RenderFlags

from paz.backend.render import compute_modelview_matrices
from paz.backend.quaternion import quarternion_to_rotation_matrix


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

    print(len(vertices))
    print(len(np.unique(vertices, axis=0)))
    print(np.where((vertices_selected == vertices_selected[0]).all(axis=1)))

    # Rotate the object
    #angle = np.pi/10
    #colors_rotated = list()
    #for i in range(colors.shape[0]):
    #for vertex in mesh.vertices:
    #    print("Quaternion: {}".format(quarternion_to_rotation_matrix(np.array([0, np.sin(angle / 2), 0, np.cos(angle / 2)]))))
    #    colors_rotated.append(quarternion_to_rotation_matrix(np.array([0, np.sin(angle / 2), 0, np.cos(angle / 2)]))@vertex)

    #colors_rotated_array = np.asarray(colors_rotated)
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(colors_rotated_array[:, 0] / 255., colors_rotated_array[:, 1] / 255., colors_rotated_array[:, 2] / 255., c=colors_rotated_array[:] / 255.)
    #ax.scatter(colors[:10000, 0], colors[:10000, 1], colors[:10000, 2])
    #plt.show()

    #colors = np.concatenate((colors, colors))
    #colors = np.concatenate(((colors, np.expand_dims(np.ones(len(colors))*255, axis=-1))), axis=-1)

    mesh.visual = mesh.visual.to_color()
    mesh.visual.vertex_colors = colors
    #mesh.visual.vertex_colors[np.invert(vertices[:, 2] >= 0)] = colors[:2858]

    #trimesh.points.PointCloud(vertices_selected, colors=colors).export("./color_mesh.ply", file_type='ply')

    angle = np.pi/5
    mesh_rotated = deepcopy(mesh)
    mesh_rotated.rotation = np.array([0, np.sin(angle / 2), 0, np.cos(angle / 2)])

    return mesh, mesh_rotated


if __name__ == "__main__":
    scene = Scene(bg_color=[100, 100, 100])
    y_fov = 3.14159/4.0
    distance = [0.3, 0.5]
    light_bounds = [0.5, 30]
    size = (400, 400)

    light = scene.add(DirectionalLight([1.0, 1.0, 1.0], 10))
    camera = scene.add(PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))
    camera_to_world, world_to_camera = compute_modelview_matrices(np.array([.4, .4, .4]), [0.0, 0.0, 0.0], np.pi*2, 0.05)

    scene.set_pose(camera, camera_to_world)
    scene.set_pose(light, camera_to_world)

    #mesh = trimesh.load("/home/fabian/.keras/datasets/custom_objects/symmetry_z_2_object.obj")
    mesh, mesh_rotated = color_object("/home/fabian/.keras/datasets/custom_objects/symmetric_object_half.obj")

    scene.add(Mesh.from_trimesh(mesh, smooth=False))
    mesh_rotated = scene.add(Mesh.from_trimesh(mesh_rotated, smooth=False))
    angle = np.pi
    mesh_rotated.rotation = np.array([0, np.sin(angle / 2), 0, np.cos(angle / 2)])

    renderer = OffscreenRenderer(size[0], size[1])
    image_original, depth_original = renderer.render(scene, flags=RenderFlags.FLAT)

    plt.imshow(image_original)
    plt.show()