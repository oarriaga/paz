import numpy as np
from paz.backend.render import sample_uniformly
from paz.backend.render import scale_translation
from paz.backend.render import random_perturbation
from paz.backend.render import split_alpha_channel
from paz.backend.render import sample_point_in_sphere
from paz.backend.render import compute_modelview_matrices

from pyrender import PerspectiveCamera, OffscreenRenderer, DirectionalLight
from pyrender import RenderFlags, Mesh, Scene
import trimesh


class DualView():
    """Scene that renders a single object from two different locations.

    # Arguments
        OBJ_filepath: String containing the path to an OBJ file.
        viewport_size: List, specifying [H, W] of rendered image.
        y_fov: Float indicating the vertical field of view in radians.
        distance: Float. Max distance from the camera to the origin.
        light: Integer representing the light intensity.
        top_only: Boolean. If True images are only take from the top.
        scale: Float, factor to apply to translation vector.
        roll: Float, to sample [-roll, roll] rolls of the Z OpenGL camera axis.
        shift: Float, to sample [-shift, shift] to move in X, Y OpenGL axes.
    """

    def __init__(self, filepath, viewport_size=(128, 128), y_fov=3.14159 / 4.0,
                 distance=0.3, light=5.0, top_only=True, scale=10.0, roll=None,
                 shift=None):

        self._build_scene(filepath, viewport_size, light, y_fov)
        self.distance, self.roll = distance, roll
        self.top_only, self.shift, self.scale = top_only, shift, scale
        self.renderer = OffscreenRenderer(*viewport_size)
        self.RGBA = RenderFlags.RGBA
        self.epsilon = 0.01

    def _build_scene(self, path, size, light, y_fov):
        self.scene = Scene(bg_color=[0, 0, 0, 0])
        self.light = self.scene.add(DirectionalLight([1.0, 1.0, 1.0], light))
        self.camera = self.scene.add(
            PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))
        self.mesh = self.scene.add(
            Mesh.from_trimesh(trimesh.load(path), smooth=True))
        self.world_origin = self.mesh.mesh.centroid

    def _sample_camera_origin(self):
        distance = sample_uniformly(self.distance)
        camera_origin = sample_point_in_sphere(distance, self.top_only)
        camera_origin = random_perturbation(camera_origin, self.epsilon)
        return camera_origin

    def _change_scene(self):
        camera_origin = self._sample_camera_origin()
        camera_to_world, world_to_camera = compute_modelview_matrices(
            camera_origin, self.world_origin, self.roll, self.shift)
        self.scene.set_pose(self.camera, camera_to_world)
        self.scene.set_pose(self.light, camera_to_world)
        return camera_to_world, world_to_camera

    def render(self):
        A_to_world, world_to_A = self._change_scene()
        image_A, depth_A = self.renderer.render(self.scene, flags=self.RGBA)
        B_to_world, world_to_B = self._change_scene()
        image_B, depth_B = self.renderer.render(self.scene, flags=self.RGBA)
        image_A, alpha_A = split_alpha_channel(image_A)
        image_B, alpha_B = split_alpha_channel(image_B)
        world_to_A = scale_translation(world_to_A, self.scale)
        world_to_B = scale_translation(world_to_B, self.scale)
        A_to_world = scale_translation(A_to_world, self.scale)
        B_to_world = scale_translation(B_to_world, self.scale)
        matrices = np.vstack([world_to_A.flatten(), world_to_B.flatten(),
                              A_to_world.flatten(), B_to_world.flatten()])
        return {'image_A': image_A, 'alpha_A': alpha_A, 'depth_A': depth_A,
                'image_B': image_B, 'alpha_B': alpha_B, 'depth_B': depth_B,
                'matrices': matrices}
