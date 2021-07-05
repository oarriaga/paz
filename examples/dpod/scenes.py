import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, filepath_original, filepath_uv_mapping, num_objects, viewport_size=(128, 128), y_fov=3.14159 / 4.0,
                 distance=[0.7, 0.9], light=[0.5, 30], top_only=False,
                 roll=None, shift=None, num_color_classes=256):
        self.distance, self.roll, self.shift = distance, roll, shift
        self.light_intensity, self.top_only = light, top_only
        self._build_scene(filepath_original, filepath_uv_mapping, viewport_size, light, y_fov)
        self.renderer = OffscreenRenderer(viewport_size[1], viewport_size[0])
        self.epsilon = 0.01
        self.viewport_size = viewport_size
        self.num_objects = num_objects
        self.num_color_classes = num_color_classes

    def _build_scene(self, filepath_original, filepath_uv_mapping, size, light, y_fov):
        print("y_fov: {}".format(y_fov))
        # Scene with the normal object
        self.scene_original = Scene(bg_color=[0, 0, 0, 0], ambient_light=[255, 255, 255])
        self.camera_original = self.scene_original.add(PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))
        self.mesh_original = self.scene_original.add(Mesh.from_trimesh(trimesh.load(filepath_original), smooth=True))

        # Scene with the UV mapped object
        self.scene_uv_mapping = Scene(bg_color=[0, 0, 0, 0], ambient_light=[255, 255, 255])
        self.camera_uv_mapping = self.scene_uv_mapping.add(PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))
        self.mesh_uv_mapping = self.scene_uv_mapping.add(Mesh.from_trimesh(trimesh.load(filepath_uv_mapping), smooth=True))

        self.world_origin = self.mesh_original.mesh.centroid

    def _sample_parameters(self):
        distance = sample_uniformly(self.distance)
        camera_origin = sample_point_in_sphere(distance, self.top_only)
        camera_origin = random_perturbation(camera_origin, self.epsilon)
        light_intensity = sample_uniformly(self.light_intensity)
        return camera_origin, light_intensity

    def render(self):
        camera_origin, intensity = self._sample_parameters()
        camera_to_world, world_to_camera = compute_modelview_matrices(camera_origin, self.world_origin, self.roll, self.shift)

        self.scene_original.set_pose(self.camera_original, camera_to_world)
        self.scene_uv_mapping.set_pose(self.camera_uv_mapping, camera_to_world)

        image_original, _ = self.renderer.render(self.scene_original, flags=RenderFlags.RGBA)
        image_original, alpha_mask = split_alpha_channel(image_original)
        image_uv_mapping, _ = self.renderer.render(self.scene_uv_mapping, flags=RenderFlags.FLAT)

        # Generate ID mask
        image_id_mask = np.max(np.ceil(image_original/255.), axis=-1)
        id_mask = np.zeros((self.viewport_size[0], self.viewport_size[1], self.num_objects+1))

        id_mask[:, :, 0] = image_id_mask
        id_mask_background = np.ones((self.viewport_size[0], self.viewport_size[1])) - np.sum(id_mask, axis=-1)
        id_mask[:, :, -1] = id_mask_background

        # Generate u and v map
        u_map = np.zeros((self.viewport_size[0], self.viewport_size[1], self.num_color_classes))
        v_map = np.zeros((self.viewport_size[0], self.viewport_size[1], self.num_color_classes))

        for i in range(self.viewport_size[0]):
            for j in range(self.viewport_size[1]):
                if image_uv_mapping[i, j, 0] != 0:
                    u_map[i, j, image_uv_mapping[i, j, 0]] = 1

                if image_uv_mapping[i, j, 1] != 0:
                    u_map[i, j, image_uv_mapping[i, j, 1]] = 1

        return image_original, alpha_mask, image_uv_mapping, u_map, v_map, id_mask


if __name__ == "__main__":
    singleView = SingleView(filepath_original="/home/fabian/.keras/datasets/036_wood_block/textured_edited.obj",
                            filepath_uv_mapping="/home/fabian/.keras/datasets/036_wood_block/textured_edited_uv_mapping.obj",
                            viewport_size=(320, 320), num_objects=1)
    image_original, alpha_mask, image_uv_mapping, u_map, v_map, id_mask = singleView.render()

    plt.imshow(image_original)
    plt.show()