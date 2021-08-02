import numpy as np
import os
import sys
import time
from copy import deepcopy
import random
import matplotlib.pyplot as plt
from paz.backend.render import sample_uniformly, split_alpha_channel
from paz.backend.render import random_perturbation, sample_point_in_sphere
from paz.backend.render import compute_modelview_matrices
from paz.backend.quaternion import quarternion_to_rotation_matrix, rotation_vector_to_quaternion, quaternion_multiply, quaternion_to_euler
from pyrender import PerspectiveCamera, OffscreenRenderer, DirectionalLight, OrthographicCamera, IntrinsicsCamera
from pyrender import RenderFlags, Mesh, Scene, Material
import pyrender
import trimesh
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


#fig = plt.figure()
#ax = Axes3D(fig)

np.set_printoptions(threshold=sys.maxsize)


def calculate_canonical_pose_two_symmetries(mesh):
    # Calculate canonical pose for an object with 180° symmetry, idea taken from here: https://arxiv.org/abs/1908.07640
    rotation_matrices = [np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])]
    rotation_matrix_r = quarternion_to_rotation_matrix(mesh.rotation)

    norm_pairs = list()
    # Iterate over all rotation matrices
    for i, rotation_matrix_s in enumerate(rotation_matrices):
        matrix_norm = np.linalg.norm(np.dot(np.linalg.inv(rotation_matrix_s), rotation_matrix_r) - np.identity(3))
        norm_pairs.append((i, matrix_norm))

    # Only change the rotation if the choosen matrix is not the identity matrix
    min_norm_pair = min(norm_pairs, key=lambda t: t[1])

    if min_norm_pair[0] == 1:
        angle = np.pi
        mesh.rotation = quaternion_multiply(mesh.rotation, np.array([0, -np.sin(angle / 2), 0, np.cos(angle / 2)]))


def get_random_translation(translation_bounds=0.1):
    translation = np.array([np.random.uniform(-translation_bounds, translation_bounds),
                            np.random.uniform(-translation_bounds, translation_bounds),
                            np.random.uniform(-translation_bounds, translation_bounds)])
    return translation


class SingleView():
    """Render-ready scene composed of a single object and a single moving camera.

    # Arguments
        filepath: String containing the path to an OBJ file.
        viewport_size: List, specifying [H, W] of rendered image.
        y_fov: Float indicating the vertical field of view in radians.
        distance: List of floats indicating [max_distance, min_distance]
        light: List of floats indicating [max_light, min_light]
        top_only: Boolean. If True images are only take from the top.
        roll: Float, to sample [-roll, roll] rolls of the Z OpenGL camera axis.
        shift: Float, to sample [-shift, shift] to move in X, Y OpenGL axes.
    """
    def __init__(self, filepath, filepath_half_object, viewport_size=(128, 128), y_fov=3.14159 / 4.0,
                 distance=[0.3, 0.5], light_bounds=[0.5, 30], top_only=False,
                 roll=None, shift=None):
        self.distance, self.roll, self.shift = distance, roll, shift
        self.light_intensity, self.top_only = light_bounds, top_only
        self._build_scene(filepath, viewport_size, light_bounds, y_fov, colors=False)
        #self._build_scene_custom_coloring(filepath, filepath_half_object, viewport_size, light_bounds, y_fov)
        self.RGBA = RenderFlags.RGBA
        self.epsilon = 0.01
        self.viewport_size = viewport_size
        self.renderer = OffscreenRenderer(self.viewport_size[0], self.viewport_size[1])
        self.object_rotations = list()
        self.camera_translations = list()

    def _build_scene(self, path, size, light, y_fov, colors=True, rotation_matrix=np.eye(4), translation=np.zeros(3)):
        self.object_rotations = list()
        self.camera_translations = list()
        # Create two scenes: one for the colored objcet one for the error object
        # In the second scene we do not need a light because we use flat rendering

        loaded_trimesh = trimesh.load(path)
        self.scene_original = Scene(bg_color=[0, 0, 0, 0])
        light_original = self.scene_original.add(DirectionalLight([1.0, 1.0, 1.0], np.mean(light)))
        self.camera_original = self.scene_original.add(PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))
        #self.color_mesh_uniform(loaded_trimesh, np.array([255, 0, 0]))
        self.mesh_original = self.scene_original.add(Mesh.from_trimesh(loaded_trimesh, smooth=True))
        self.world_origin = self.mesh_original.mesh.centroid
        self.camera_translation = np.array([0., 0., 0.69])
        camera_to_world, world_to_camera = compute_modelview_matrices(self.camera_translation, self.world_origin, self.roll, self.shift)
        light_original.light.intensity = 5.0
        self.scene_original.set_pose(self.camera_original, camera_to_world)
        self.scene_original.set_pose(light_original, camera_to_world)
        print("World to camera: {}".format(world_to_camera))

        loaded_trimesh = trimesh.load(path)
        self.scene_color = Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0, 1.0])
        self.camera_color = self.scene_color.add(PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))
        # Encode the 3D locations in colors
        self.color_mesh(loaded_trimesh)
        self.mesh_color = self.scene_color.add(Mesh.from_trimesh(loaded_trimesh, smooth=False))
        self.world_origin = self.mesh_color.mesh.centroid
        self.scene_color.set_pose(self.camera_color, camera_to_world)
        self.camera_to_world = camera_to_world
        self.world_to_camera = world_to_camera

    def _build_scene_custom_coloring(self, path_full_object, path_half_object, size, light, y_fov, colors=True, rotation_matrix=np.eye(4), translation=np.zeros(3)):
        self.object_rotations = list()
        # Create two scenes: one for the colored objcet one for the error object
        # In the second scene we do not need a light because we use flat rendering

        loaded_trimesh = trimesh.load(path_full_object)
        self.scene_original = Scene(bg_color=[0, 0, 0, 0])
        light_original = self.scene_original.add(DirectionalLight([1.0, 1.0, 1.0], np.mean(light)))
        camera = self.scene_original.add(PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))
        #self.color_mesh(loaded_trimesh)
        self.mesh_original = self.scene_original.add(Mesh.from_trimesh(loaded_trimesh, smooth=True))
        self.world_origin = self.mesh_original.mesh.centroid
        camera_to_world, world_to_camera = compute_modelview_matrices(np.array([0., 0., 0.4]), self.world_origin, self.roll, self.shift)
        light_original.light.intensity = 5.0
        self.scene_original.set_pose(camera, camera_to_world)
        self.scene_original.set_pose(light_original, camera_to_world)

        loaded_trimesh = trimesh.load(path_half_object)
        self.scene_color = Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0, 1.0])
        camera = self.scene_color.add(PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))
        # Encode the 3D locations in colors
        self.color_mesh(loaded_trimesh)
        self.mesh_color01 = self.scene_color.add(Mesh.from_trimesh(loaded_trimesh, smooth=False))
        self.mesh_color02 = self.scene_color.add(Mesh.from_trimesh(loaded_trimesh, smooth=False))
        self.world_origin = self.mesh_color01.mesh.centroid
        self.scene_color.set_pose(camera, camera_to_world)

        self.world_to_camera = world_to_camera

    def render(self):
        self.renderer = OffscreenRenderer(self.viewport_size[0], self.viewport_size[1])
        camera_translation_z = random.uniform(0.5, 0.8)
        #camera_translation_z = 0.69
        # Random rotation and translation for the object
        rotation = trimesh.transformations.random_quaternion()

        translation = get_random_translation(0.05)

        # Random translation of the camera
        camera_translation = np.array([0., 0., camera_translation_z])
        camera_to_world, world_to_camera = compute_modelview_matrices(camera_translation, self.world_origin, self.roll, self.shift)
        self.scene_original.set_pose(self.camera_original, camera_to_world)
        self.scene_color.set_pose(self.camera_color, camera_to_world)
        self.camera_to_world = camera_to_world
        self.world_to_camera = world_to_camera

        self.camera_translations.append(camera_to_world[:3, 3] + translation)
        print("Camera translation: {}".format(camera_to_world[:3, 3] + translation))

        rotation_inverse = -rotation
        rotation_inverse[-1] = -rotation_inverse[-1]
        self.mesh_original.rotation = rotation
        self.mesh_color.rotation = rotation
        self.mesh_original.translation = translation
        self.mesh_color.translation = translation


        self.rotation_object = rotation_inverse
        self.object_rotations.append(rotation_inverse)
        self.object_to_world = quarternion_to_rotation_matrix(rotation_inverse)

        image_original, depth_original = self.renderer.render(self.scene_original, flags=self.RGBA)
        image_original, alpha_original = split_alpha_channel(image_original)

        image_colors, _ = self.renderer.render(self.scene_color, flags=pyrender.constants.RenderFlags.FLAT)
        self.renderer.delete()

        return image_original, image_colors, alpha_original

    def render_no_ambiguities(self):
        self.renderer = OffscreenRenderer(self.viewport_size[0], self.viewport_size[1])
        camera_translation_z = random.uniform(0.3, 1.5)
        # Random rotation and translation for the object
        rotation = trimesh.transformations.random_quaternion()
        translation = get_random_translation(0.15)*(camera_translation_z + 0.7)

        # Random translation of the camera
        camera_translation = np.array([0., 0., camera_translation_z])
        camera_to_world, world_to_camera = compute_modelview_matrices(camera_translation, self.world_origin, self.roll, self.shift)
        self.scene_original.set_pose(self.camera_original, camera_to_world)
        self.scene_color.set_pose(self.camera_color, camera_to_world)
        self.camera_to_world = camera_to_world
        self.world_to_camera = world_to_camera

        rotation_inverse = -rotation
        rotation_inverse[-1] = -rotation_inverse[-1]
        self.mesh_original.rotation = rotation
        self.mesh_color.rotation = rotation
        self.mesh_original.translation = translation
        self.mesh_color.translation = translation

        self.rotation_object = rotation_inverse
        self.object_to_world = quarternion_to_rotation_matrix(rotation_inverse)

        image_original, depth_original = self.renderer.render(self.scene_original, flags=self.RGBA)
        image_original, alpha_original = split_alpha_channel(image_original)

        calculate_canonical_pose_two_symmetries(self.mesh_color)
        image_colors, _ = self.renderer.render(self.scene_color, flags=pyrender.constants.RenderFlags.FLAT)
        self.renderer.delete()

        return image_original, image_colors, alpha_original

    def render_custom_coloring(self):
        rotation = trimesh.transformations.random_quaternion()
        angle = np.random.uniform(low=0.0, high=2.0)*np.pi
        #angle = np.pi
        #rotation = np.array([np.sin(angle / 2), 0, 0, np.cos(angle / 2)])
        #rotation /= np.linalg.norm(rotation)
        #translation = get_random_translation()

        self.mesh_original.rotation = rotation
        print("True rotation: {}".format(rotation))
        print("True rotation01: {}".format(quarternion_to_rotation_matrix(rotation)))
        print("True euler01: {}".format(trimesh.transformations.euler_from_matrix(quarternion_to_rotation_matrix(rotation))))

        self.rotation_inverse = -rotation
        self.rotation_inverse[-1] = -self.rotation_inverse[-1]

        self.mesh_color01.rotation = rotation
        self.mesh_color02.rotation = quaternion_multiply(rotation, np.array([0, np.sin(np.pi/2), 0, np.cos(np.pi/2)]))
        print("True rotation02: {}".format(quarternion_to_rotation_matrix(quaternion_multiply(rotation, np.array([0, np.sin(np.pi/2), 0, np.cos(np.pi/2)])))))
        print("True euler02: {}".format(trimesh.transformations.euler_from_matrix(quarternion_to_rotation_matrix(quaternion_multiply(rotation, np.array([0, np.sin(np.pi/2), 0, np.cos(np.pi/2)]))))))
        #self.mesh_original.translation = translation
        #self.mesh_color.translation = translation

        image_original, depth_original = self.renderer.render(self.scene_original, flags=self.RGBA)
        image_original, alpha_original = split_alpha_channel(image_original)

        image_colors, _ = self.renderer.render(self.scene_color, flags=pyrender.constants.RenderFlags.FLAT)

        return image_original, image_colors, alpha_original

    def normalize(self, x, x_min, x_max):
        return (x-x_min)/(x_max-x_min)

    def color_mesh(self, mesh):
        """ color the mesh
        # Arguments
            mesh: obj mesh
        # Returns
            mesh: colored obj mesh
        """
        x_min = mesh.vertices[:, 0].min()
        x_max = mesh.vertices[:, 0].max()
        y_min = mesh.vertices[:, 1].min()
        y_max = mesh.vertices[:, 1].max()
        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()

        # make vertices using RGB format
        vertices_x = 255 * self.normalize(mesh.vertices[:, 0:1], x_min, x_max)
        vertices_y = 255 * self.normalize(mesh.vertices[:, 1:2], y_min, y_max)
        vertices_z = 255 * self.normalize(mesh.vertices[:, 2:3], z_min, z_max)

        vertices_x = vertices_x.astype('uint8')
        vertices_y = vertices_y.astype('uint8')
        vertices_z = vertices_z.astype('uint8')
        colors = np.hstack([vertices_x, vertices_y, vertices_z])

        mesh.visual = mesh.visual.to_color()
        mesh.visual.vertex_colors = colors

        return mesh

    def color_mesh_uniform(self, mesh, color):
        vertices = mesh.vertices
        colors = np.tile(color, (len(vertices), 1))

        mesh.visual = mesh.visual.to_color()
        mesh.visual.vertex_colors = colors

        return mesh

    def color_mesh_uniform_vertices(self, mesh, color):
        vertices = mesh.vertices
        colors = np.tile(color, (len(vertices), 1))

        mesh.visual = mesh.visual.to_color()
        mesh.visual.vertex_colors = colors

        return vertices, colors


def render_images(save_path, object_path, num_images=1000):
    view = SingleView(filepath=object_path)

    for i in tqdm(range(num_images)):
        image_original, image_colors, alpha_original = view.render()

        plt.imshow(image_original)
        plt.show()

        plt.imshow(image_colors)
        plt.show()

        #np.save(os.path.join(save_path, "image_original/image_original_{}.npy".format(str(i).zfill(7))), image_original)
        #np.save(os.path.join(save_path, "image_colors/image_colors_{}.npy".format(str(i).zfill(7))), image_colors)
        #np.save(os.path.join(save_path, "alpha_original/alpha_original_{}.npy".format(str(i).zfill(7))), alpha_original)
        #np.save(os.path.join(save_path, "image_colors_no_ambiguities/image_colors_no_ambiguities_{}.npy".format(str(i).zfill(7))), image_colors_no_ambiguities)


def render_images_custom_coloring(save_path, object_path, object_path_half, num_images=1000):
    view = SingleView(filepath=object_path, filepath_half_object=object_path_half)

    for i in tqdm(range(num_images)):
        image_original, image_colors, alpha_original = view.render_custom_coloring()
        image_colors = deepcopy(image_colors)
        # make the edges close to the end of the object black
        image_colors[(image_colors[:, :, 0] < 25)] = np.array([0., 0., 0.])

        np.save(os.path.join(save_path, "image_original/image_original_{}.npy".format(str(i).zfill(7))), image_original)
        np.save(os.path.join(save_path, "image_colors/image_colors_{}.npy".format(str(i).zfill(7))), image_colors)
        np.save(os.path.join(save_path, "alpha_original/alpha_original_{}.npy".format(str(i).zfill(7))), alpha_original)
        np.save(os.path.join(save_path, "image_colors_no_ambiguities/image_colors_no_ambiguities_{}.npy".format(str(i).zfill(7))), image_colors_no_ambiguities)


if __name__ == "__main__":
    """
    num_samples = 5
    list_images = list()
    #view = SingleView(filepath="/home/fabian/.keras/datasets/036_wood_block/textured_edited.obj")
    view = SingleView(filepath="/home/fabian/.keras/datasets/tless_obj/obj_000014.obj")

    for _ in range(num_samples):
        image_original, image_colors, alpha_original = view.render()
        list_images.append((image_original, image_colors))

    fig = plt.figure(constrained_layout=False)
    spec = gridspec.GridSpec(ncols=num_samples, nrows=2, figure=fig)
    for i in range(num_samples):
        sub_fig = fig.add_subplot(spec[0, i])
        sub_fig.imshow(list_images[i][0])

        sub_fig = fig.add_subplot(spec[1, i])
        sub_fig.imshow(list_images[i][1])

    plt.show()
    """
    render_images("/media/fabian/Data/Masterarbeit/renders/simple_symmetric_object", "/home/fabian/.keras/datasets/tless_obj/obj_000005.obj", num_images=20)
    #render_images_custom_coloring("/media/fabian/Data/Masterarbeit/renders/simple_symmetric_object", "/home/fabian/.keras/datasets/custom_objects/simple_symmetry_object.obj", "/home/fabian/.keras/datasets/custom_objects/symmetric_object_half.obj", num_images=2)
