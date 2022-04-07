import numpy as np
from paz.backend.render import sample_uniformly, split_alpha_channel
from paz.backend.render import (
    sample_point_in_sphere, random_perturbation, compute_modelview_matrices)
from pyrender import (PerspectiveCamera, OffscreenRenderer, DirectionalLight,
                      RenderFlags, Mesh, Scene)
import trimesh
from .utils import color_object


class PixelMaskRenderer():
    """Render-ready scene composed of a single object and a single moving camera.

    # Arguments
        path_OBJ: String containing the path to an OBJ file.
        viewport_size: List, specifying [H, W] of rendered image.
        y_fov: Float indicating the vertical field of view in radians.
        distance: List of floats indicating [max_distance, min_distance]
        light: List of floats indicating [max_light, min_light]
        top_only: Boolean. If True images are only take from the top.
        roll: Float, to sample [-roll, roll] rolls of the Z OpenGL camera axis.
        shift: Float, to sample [-shift, shift] to move in X, Y OpenGL axes.
    """
    def __init__(self, path_OBJ, viewport_size=(128, 128), y_fov=3.14159 / 4.0,
                 distance=[0.3, 0.5], light=[0.5, 30], top_only=False,
                 roll=None, shift=None):
        self.distance, self.roll, self.shift = distance, roll, shift
        self.light_intensity, self.top_only = light, top_only
        self._build_scene(path_OBJ, viewport_size, light, y_fov)
        self.renderer = OffscreenRenderer(viewport_size[0], viewport_size[1])
        self.flags_RGBA = RenderFlags.RGBA
        self.flags_FLAT = RenderFlags.RGBA | RenderFlags.FLAT
        self.epsilon = 0.01

    def _build_scene(self, path, size, light, y_fov):
        self.scene = Scene(bg_color=[0, 0, 0, 0])
        self.light = self.scene.add(
            DirectionalLight([1.0, 1.0, 1.0], np.mean(light)))
        self.camera = self.scene.add(
            PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))
        self.pixel_mesh = self.scene.add(color_object(path))
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
        self.pixel_mesh.mesh.is_visible = False
        image, depth = self.renderer.render(self.scene, self.flags_RGBA)
        self.pixel_mesh.mesh.is_visible = True
        image, alpha = split_alpha_channel(image)
        self.mesh.mesh.is_visible = False
        RGB_mask, _ = self.renderer.render(self.scene, self.flags_FLAT)
        self.mesh.mesh.is_visible = True
        return image, alpha, RGB_mask


if __name__ == "__main__":
    import os
    from paz.backend.image import show_image, resize_image
    # -------------------------------------------------------------
    # Generic parameters
    # -------------------------------------------------------------
    root_path = os.path.expanduser('~')
    num_occlusions = 1
    image_shape = (128, 128, 3)
    viewport_size = image_shape[:2]
    y_fov = 3.14159 / 4.0
    light = [1.0, 30]
    top_only = False
    roll = 3.14159
    shift = 0.05

    # ------------------------------------------------------------
    # Training scene for hammer
    # ------------------------------------------------------------
    OBJ_name = '.keras/paz/datasets/ycb_models/048_hammer/textured.obj'
    # OBJ_name = '.keras/paz/datasets/ycb_models/037_scissors/textured.obj'
    # OBJ_name = '.keras/paz/datasets/ycb_models/'
    # '051_large_clamp/textured.obj'
    # OBJ_name = '/home/octavio/.keras/paz/datasets/'
    # 'new_052_large_clamp/textured.obj'
    path_OBJ = os.path.join(root_path, OBJ_name)
    distance = [0.30, 0.35]

    renderer = PixelMaskRenderer(path_OBJ, viewport_size, y_fov, distance,
                                 light, top_only, roll, shift)
    for arg in range(100):
        image, alpha, RGBA_mask = renderer.render()
        image = np.concatenate([image, RGBA_mask[..., 0:3]], axis=1)
        H, W = image.shape[:2]
        image = resize_image(image, (W * 3, H * 3))
        show_image(image)
    # inference
    """
    from paz.backend.camera import Camera
    from paz.pipelines.pose import RGBMaskToPose6D
    from paz.models.segmentation import UNET_VGG16
    camera = Camera()
    camera.intrinsics_from_HFOV(image_shape=(128, 128))
    # from meters to milimiters
    object_sizes = renderer.mesh.mesh.extents * 100
    model = UNET_VGG16(3, image_shape, freeze_backbone=True)
    # model.load_weights('experiments/UNET-VGG16_RUN_00_04-04-2022_12-29-44/model_weights.hdf5')
    model.load_weights('experiments/UNET-VGG16_RUN_00_06-04-2022_11-20-18/model_weights.hdf5')
    estimate_pose = RGBMaskToPose6D(model, object_sizes, camera, draw=True)

    image, alpha, RGBA_mask = renderer.render()
    image = np.copy(image)  # TODO: renderer outputs unwritable numpy arrays
    show_image(image)
    results = estimate_pose(image)
    show_image(results['image'])
    """
