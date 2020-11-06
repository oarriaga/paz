import numpy as np
from paz.backend.render import sample_uniformly, split_alpha_channel
from paz.backend.render import random_perturbation, sample_point_in_sphere
from paz.backend.render import compute_modelview_matrices
from pyrender import PerspectiveCamera, OffscreenRenderer, DirectionalLight
from pyrender import RenderFlags, Mesh, Scene
import trimesh


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
                 distance=[0.3, 0.5], light=[0.5, 30], top_only=False,
                 roll=None, shift=None):
        self.distance, self.roll, self.shift = distance, roll, shift
        self.light_intensity, self.top_only = light, top_only
        self._build_scene(filepath, viewport_size, light, y_fov)
        self.renderer = OffscreenRenderer(viewport_size[0], viewport_size[1])
        self.RGBA = RenderFlags.RGBA
        self.epsilon = 0.01

    def _build_scene(self, path, size, light, y_fov):
        self.scene = Scene(bg_color=[0, 0, 0, 0])
        self.light = self.scene.add(
            DirectionalLight([1.0, 1.0, 1.0], np.mean(light)))
        self.camera = self.scene.add(
            PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))
        self.mesh = self.scene.add(
            Mesh.from_trimesh(trimesh.load(path), smooth=True))
        self.world_origin = self.mesh.mesh.centroid

    def _sample_parameters(self):
        distance = sample_uniformly(self.distance)
        camera_origin = sample_point_in_sphere(distance, self.top_only)
        camera_origin = random_perturbation(camera_origin, self.epsilon)
        light_intensity = sample_uniformly(self.light_intensity)
        return camera_origin, light_intensity

    def render(self):
        camera_origin, intensity = self._sample_parameters()
        camera_to_world, world_to_camera = compute_modelview_matrices(
            camera_origin, self.world_origin, self.roll, self.shift)
        self.light.light.intensity = intensity
        self.scene.set_pose(self.camera, camera_to_world)
        self.scene.set_pose(self.light, camera_to_world)
        image, depth = self.renderer.render(self.scene, flags=self.RGBA)
        image, alpha = split_alpha_channel(image)
        return image, alpha


class DictionaryView():
    """Render-ready scene composed of a single object and a single moving camera.

    # Arguments
        filepath: String containing the path to an OBJ file.
        viewport_size: List, specifying [H, W] of rendered image.
        y_fov: Float indicating the vertical field of view in radians.
        distance: List of floats indicating [max_distance, min_distance]
        top_only: Boolean. If True images are only take from the top.
        light: List of floats indicating [max_light, min_light]
    """
    def __init__(self, filepath, viewport_size=(128, 128),
                 y_fov=3.14159 / 4., distance=0.30, top_only=False,
                 light=5.0, theta_steps=10, phi_steps=10):
        self.scene = Scene(bg_color=[0, 0, 0])
        self.camera = self.scene.add(PerspectiveCamera(
            y_fov, aspectRatio=np.divide(*viewport_size)))
        self.mesh = self.scene.add(Mesh.from_trimesh(
            trimesh.load(filepath), smooth=True))
        self.world_origin = self.mesh.mesh.centroid
        self.light = self.scene.add(DirectionalLight([1.0, 1.0, 1.0], light))
        self.distance = distance
        # 0.1 values are to avoid gimbal lock
        theta_max = np.pi / 2.0 if top_only else np.pi
        self.thetas = np.linspace(0.1, theta_max - 0.1, theta_steps)
        self.phis = np.linspace(0.1, 2 * np.pi - 0.1, phi_steps)
        self.renderer = OffscreenRenderer(*viewport_size)
        self.RGBA = RenderFlags.RGBA

    def render(self):
        dictionary_data = []
        for theta_arg, theta in enumerate(self.thetas):
            for phi_arg, phi in enumerate(self.phis):
                x = self.distance * np.sin(theta) * np.cos(phi)
                y = self.distance * np.sin(theta) * np.sin(phi)
                z = self.distance * np.cos(theta)
                matrices = compute_modelview_matrices(
                    np.array([x, z, y]), self.world_origin)
                camera_to_world, world_to_camera = matrices
                self.scene.set_pose(self.camera, camera_to_world)
                self.scene.set_pose(self.light, camera_to_world)
                camera_to_world = camera_to_world.flatten()
                world_to_camera = world_to_camera.flatten()
                image, depth = self.renderer.render(
                    self.scene, flags=self.RGBA)
                image, alpha = split_alpha_channel(image)
                matrices = np.vstack([world_to_camera, camera_to_world])
                sample = {'image': image,
                          'alpha': alpha,
                          'depth': depth, 'matrices': matrices}
                dictionary_data.append(sample)
        return dictionary_data
