import numpy as np
from paz.backend.render import sample_uniformly, split_alpha_channel
from paz.backend.render import (
    sample_point_in_sphere, random_perturbation, compute_modelview_matrices)
from paz.backend.image import write_image
from paz.pipelines import RandomizeRenderedImage
from utils import color_object, as_mesh
from pyrender import (PerspectiveCamera, OffscreenRenderer, DirectionalLight,
                      RenderFlags, Mesh, Scene)
import trimesh
import os
import glob


class PixelMaskRenderer():
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
            Mesh.from_trimesh(as_mesh(trimesh.load(path)), smooth=True))
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
    # -------------------------------------------------------------
    # State paths
    # -------------------------------------------------------------
    root_path = os.path.expanduser('~/')
    obj_name = 'models/textured.obj'
    obj_path = os.path.join(root_path + obj_name)
    dataset_path = os.path.join(root_path + 'training_images')
    images_path = os.path.join(dataset_path, 'images/')
    masks_path = os.path.join(dataset_path, "masks/")
    annotation_path = os.path.join(images_path + "annotation.txt")
    background_path = os.path.join(root_path, 'background/*.png')
    background_list = glob.glob(background_path)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    print("------------")
    print("Root path: " + root_path)
    print("OBJ path: " + obj_path)
    print("Dataset path: " + dataset_path)
    print("Image path: " + images_path)
    print("Annotation path: " + annotation_path)
    print("Background path: " + background_path)
    print("------------")
    # -------------------------------------------------------------
    # Generic parameters
    # -------------------------------------------------------------
    num_occlusions = 1
    max_radius_scale = 0.5
    image_shape = (300, 300, 3)
    viewport_size = image_shape[:2]
    y_fov = 3.14159 / 4.0
    light = [0.5, 15]
    top_only = True
    roll = 3.14159
    shift = 0.05
    distance = [0.5, 1]

    renderer = PixelMaskRenderer(obj_path, viewport_size, y_fov, distance,
                                 light, top_only, roll, shift)
    randomize_image = RandomizeRenderedImage(background_list, num_occlusions,
                                             max_radius_scale)

    with open(annotation_path, "w") as file:
        for image_arg in range(50):
            image, alpha, mask = renderer.render()
            RGB_mask = mask[..., 0:3]
            image_filename = 'image_%03d.png' % image_arg
            mask_filename = 'mask_%03d.png' % image_arg
            H, W = image.shape[:2]
            # extract the 2D bounding box
            y, x = np.nonzero(image[..., 2])
            x_min = np.min(x)
            y_min = np.min(y)
            x_max = np.max(x)
            y_max = np.max(y)
            row = "%s,%i,%i,coyote,%i,%i,%i,%i" % (image_filename, H, W, x_min, y_min, x_max, y_max)
            file.write(row + "\n")
            image_filename = os.path.join(images_path, image_filename)
            mask_filename = os.path.join(masks_path, mask_filename)
            image = randomize_image(image, alpha)
            write_image(image_filename, image)
            write_image(mask_filename, RGB_mask)
