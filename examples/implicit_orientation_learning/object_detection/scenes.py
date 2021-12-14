import numpy as np
import pyrender.constants
from tqdm import tqdm
import matplotlib.pyplot as plt
from paz.backend.render import sample_uniformly, split_alpha_channel
from paz.backend.render import random_perturbation, sample_point_in_sphere
from paz.backend.render import compute_modelview_matrices
from pyrender import PerspectiveCamera, OffscreenRenderer, DirectionalLight, SpotLight
from pyrender import RenderFlags, Mesh, Scene
import trimesh


def quaternion_to_rotation_matrix(q):
    """Transforms quarternion into rotation vector
    # Arguments
        q: quarternion, Numpy array of shape ``[4]``
    # Returns
        Numpy array representing a rotation vector having a shape ``[3]``.
    """
    rotation_matrix = np.array([[1 - 2*(q[1]**2 + q[2]**2), 2*(q[0]*q[1] - q[3]*q[2]), 2*(q[3]*q[1] + q[0]*q[2])],
                                [2*(q[0]*q[1] + q[3]*q[2]), 1 - 2*(q[0]**2 + q[2]**2), 2*(q[1]*q[2] - q[3]*q[0])],
                                [2*(q[0]*q[2] - q[3]*q[1]), 2*(q[3]*q[0] + q[1]*q[2]), 1 - 2*(q[0]**2 + q[1]**2)]])

    return np.squeeze(rotation_matrix)

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
    def __init__(self, filepath, viewport_size=(128, 128), y_fov=3.14159 / 4.0,
                 distance=[0.5, 0.8], light=[0.5, 30], object_translation_bound=0.1, top_only=False,
                 roll=None, shift=None):
        self.distance, self.roll, self.shift = distance, roll, shift
        self.light_intensity, self.top_only = light, top_only
        self._build_scene(filepath, viewport_size, light, y_fov)
        self.renderer = OffscreenRenderer(viewport_size[0], viewport_size[1])
        self.RGBA = RenderFlags.RGBA
        self.epsilon = 0.01
        self.object_translation_bound = object_translation_bound

    def _build_scene(self, path, size, light, y_fov):
        self.scene = Scene(bg_color=[0, 0, 0, 0])
        self.light = self.scene.add(DirectionalLight([1.0, 1.0, 1.0], np.mean(light)))
        self.camera = self.scene.add(PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))

        loaded_mesh = trimesh.load(path)
        #self.color_mesh_uniform(loaded_mesh)
        self.mesh = self.scene.add(Mesh.from_trimesh(loaded_mesh, smooth=True))

        self.world_origin = self.mesh.mesh.centroid

    def _sample_parameters(self):
        camera_distance = sample_uniformly(self.distance)

        object_rotation = trimesh.transformations.random_quaternion()
        object_rotation = np.array([object_rotation[1], object_rotation[2], object_rotation[3], object_rotation[0]])

        object_translation = np.array([np.random.uniform(-self.object_translation_bound, self.object_translation_bound),
                                     np.random.uniform(-self.object_translation_bound, self.object_translation_bound),
                                     np.random.uniform(-self.object_translation_bound, self.object_translation_bound)])

        light_intensity = sample_uniformly(self.light_intensity)
        return camera_distance, object_rotation, object_translation, light_intensity

    def render(self):
        camera_distance, object_rotation, object_translation, intensity = self._sample_parameters()
        camera_origin = np.array([0., 0., camera_distance])

        camera_to_world, world_to_camera = compute_modelview_matrices(camera_origin, self.world_origin, None, None)

        self.mesh.rotation = object_rotation
        self.mesh.translation = object_translation

        self.light.light.intensity = intensity
        self.scene.set_pose(self.camera, camera_to_world)
        self.scene.set_pose(self.light, camera_to_world)

        camera_to_world = camera_to_world.flatten()
        world_to_camera = world_to_camera.flatten()
        matrices = np.vstack([world_to_camera, camera_to_world])

        image, depth = self.renderer.render(self.scene, flags=pyrender.constants.RenderFlags.RGBA)
        image, alpha = split_alpha_channel(image)

        object_mask = np.argwhere(np.sum(image, axis=-1) != 0)
        object_bounding_box = np.array([object_mask[:, 1].min(), object_mask[:, 0].min(), object_mask[:, 1].max(), object_mask[:, 0].max()])

        return image, alpha, object_bounding_box, object_translation, matrices

    def color_mesh_uniform(self, trimesh):
        colors = np.tile(np.array([255, 0, 0]), (len(trimesh.vertices), 1))

        trimesh.visual = trimesh.visual.to_color()
        trimesh.visual.vertex_colors = colors

        return trimesh