import numpy as np
from paz.backend.render import sample_uniformly, split_alpha_channel
from pyrender import (PerspectiveCamera, OffscreenRenderer, DirectionalLight,
                      RenderFlags, Mesh, Scene)
import trimesh

from backend import to_affine_matrix
from backend import sample_affine_transform
from backend import calculate_canonical_rotation
from backend import compute_vertices_colors


def load_obj(path):
    mesh = trimesh.load(path)
    return mesh


def color_object(path):
    mesh = load_obj(path)
    colors = compute_vertices_colors(mesh.vertices)
    mesh.visual = mesh.visual.to_color()
    mesh.visual.vertex_colors = colors
    mesh = Mesh.from_trimesh(mesh, smooth=False)
    mesh.primitives[0].material.metallicFactor = 0.0
    mesh.primitives[0].material.roughnessFactor = 1.0
    mesh.primitives[0].material.alphaMode = 'OPAQUE'
    return mesh


class CanonicalScene():
    def __init__(self, path_OBJ, camera_pose, min_corner, max_corner,
                 symmetric_transforms, viewport_size=(128, 128),
                 y_fov=3.14159 / 4.0, light_intensity=[0.5, 30]):
        self.light_intensity = light_intensity
        self.symmetric_transforms = symmetric_transforms
        self.min_corner, self.max_corner = min_corner, max_corner
        self.scene = Scene(bg_color=[0, 0, 0, 0])
        self.light = self._build_light(light_intensity, camera_pose)
        self.camera = self._build_camera(y_fov, viewport_size, camera_pose)
        self.pixel_mesh = self.scene.add(color_object(path_OBJ))
        self.mesh = self.scene.add(
            Mesh.from_trimesh(trimesh.load(path_OBJ), smooth=True))

        self.renderer = OffscreenRenderer(viewport_size[0], viewport_size[1])

        self.flags_RGBA = RenderFlags.RGBA
        self.flags_FLAT = RenderFlags.RGBA | RenderFlags.FLAT

    def _build_light(self, light, pose):
        directional_light = DirectionalLight([1.0, 1.0, 1.0], np.mean(light))
        directional_light = self.scene.add(directional_light, pose=pose)
        return directional_light

    def _build_camera(self, y_fov, viewport_size, pose):
        aspect_ratio = np.divide(*viewport_size)
        camera = PerspectiveCamera(y_fov, aspectRatio=aspect_ratio)
        camera = self.scene.add(camera, pose=pose)
        return camera

    def _sample_parameters(self, min_corner, max_corner):
        mesh_transform = sample_affine_transform(min_corner, max_corner)
        light_intensity = sample_uniformly(self.light_intensity)
        return mesh_transform, light_intensity

    def render(self):
        mesh_transform, light_intensity = self._sample_parameters(
            self.min_corner, self.max_corner)
        mesh_rotation = mesh_transform[0:3, 0:3]
        canonical_rotation = calculate_canonical_rotation(
            mesh_rotation, self.symmetric_transforms)
        # mesh_rotation[0:3, 0:3] = canonical_rotation
        canonical_rotation = np.dot(mesh_rotation, canonical_rotation)
        mesh_rotation[0:3, 0:3] = canonical_rotation
        self.scene.set_pose(self.mesh, mesh_transform)
        self.scene.set_pose(self.pixel_mesh, mesh_transform)
        self.light.light.intensity = light_intensity

        self.pixel_mesh.mesh.is_visible = False
        image, depth = self.renderer.render(self.scene, self.flags_RGBA)
        self.pixel_mesh.mesh.is_visible = True
        image, alpha = split_alpha_channel(image)
        self.mesh.mesh.is_visible = False
        RGB_mask, _ = self.renderer.render(self.scene, self.flags_FLAT)
        self.mesh.mesh.is_visible = True
        return image, alpha, RGB_mask

    def render_symmetries(self):
        images, alphas, RGB_masks = [], [], []
        for rotation in self.symmetric_transforms:
            symmetric_transform = to_affine_matrix(rotation, np.zeros(3))
            self.scene.set_pose(self.mesh, symmetric_transform)
            self.scene.set_pose(self.pixel_mesh, symmetric_transform)
            self.pixel_mesh.mesh.is_visible = False
            image, depth = self.renderer.render(self.scene, self.flags_RGBA)
            self.pixel_mesh.mesh.is_visible = True
            image, alpha = split_alpha_channel(image)
            self.mesh.mesh.is_visible = False
            RGB_mask, _ = self.renderer.render(self.scene, self.flags_FLAT)
            self.mesh.mesh.is_visible = True
            images.append(image)
            alphas.append(alpha)
            RGB_masks.append(RGB_mask[..., 0:3])
        images = np.concatenate(images, axis=1)
        RGB_masks = np.concatenate(RGB_masks, axis=1)
        images = np.concatenate([images, RGB_masks], axis=0)
        return images


if __name__ == "__main__":
    import os
    from paz.backend.image import show_image
    from backend import build_rotation_matrix_x
    from backend import build_rotation_matrix_z

    # generic parameters
    root_path = os.path.expanduser('~')
    num_occlusions = 1
    image_shape = (128, 128, 3)
    viewport_size = image_shape[:2]
    y_fov = 3.14159 / 4.0
    light = [1.0, 30]

    # model = UNET_VGG16(3, image_shape, freeze_backbone=True)

    # solar panel parameters
    OBJ_name = 'single_solar_panel_02.obj'
    path_OBJ = os.path.join(root_path, OBJ_name)
    angles = np.linspace(0, 2 * np.pi, 7)[:6]
    symmetries = np.array([build_rotation_matrix_z(angle) for angle in angles])
    camera_rotation = build_rotation_matrix_x(np.pi)
    translation = np.array([0.0, 0.0, -1.0])
    camera_pose = to_affine_matrix(camera_rotation, translation)
    min_corner = [0.0, 0.0, -0.4]
    max_corner = [0.0, 0.0, +0.0]
    # model.load_weights('weights/UNET-VGG_solar_panel_canonical_13.hdf5')
    renderer = CanonicalScene(path_OBJ, camera_pose, min_corner,
                              max_corner, symmetries)
    renderer.scene.ambient_light = [1.0, 1.0, 1.0]
    image = renderer.render_symmetries()
    show_image(image)

    """
    # large clamp parameters
    # REMEMBER TO CHANGE THE Ns coefficient to values between [0, 1] in
    # textured.mtl. For example change 96.07 to .967
    OBJ_name = '.keras/paz/datasets/ycb_models/051_large_clamp/textured.obj'
    path_OBJ = os.path.join(root_path, OBJ_name)
    translation = np.array([0.0, 0.0, 0.25])
    camera_pose, y = compute_modelview_matrices(translation, np.zeros((3)))
    align_z = build_rotation_matrix_z(np.pi / 20)
    camera_pose[:3, :3] = np.matmul(align_z, camera_pose[:3, :3])
    min_corner = [-0.05, -0.02, -0.05]
    max_corner = [+0.05, +0.02, +0.01]
    # model.load_weights('weights/UNET-VGG_large_clamp_canonical_10.hdf5')

    angles = [0.0, np.pi]
    symmetries = np.array([build_rotation_matrix_y(angle) for angle in angles])
    renderer = CanonicalScene(path_OBJ, camera_pose, min_corner,
                              max_corner, symmetries)
    renderer.scene.ambient_light = [1.0, 1.0, 1.0]
    image = renderer.render_symmetries()
    show_image(image)
    """
    """
    # -------------------------------------------------------------
    # Training scene for hammer
    # --------------------------------------------------------------
    OBJ_name = '.keras/paz/datasets/ycb_models/048_hammer/textured.obj'
    path_OBJ = os.path.join(root_path, OBJ_name)
    distance = [0.5, 0.6]
    top_only = False
    roll = 3.14159
    shift = 0.05
    renderer = PixelMaskRenderer(
        path_OBJ, viewport_size, y_fov, distance, light, top_only, roll, shift)
    for arg in range(100):
        image, alpha, RGBA_mask = renderer.render()
        image = np.concatenate([image, RGBA_mask[..., 0:3]], axis=1)
        show_image(image)
    """
    """
    translation = np.array([0.0, 0.0, 0.50])
    camera_pose, y = compute_modelview_matrices(translation, np.zeros((3)))
    align_z = build_rotation_matrix_z(np.pi / 8)
    camera_pose[:3, :3] = np.matmul(align_z, camera_pose[:3, :3])
    min_corner = [-0.05, -0.02, -0.05]
    max_corner = [+0.05, +0.02, +0.01]

    symmetries, angles = [], [0.0, np.pi]
    for angle in angles:
        symmetry = build_rotation_matrix_y(angle)
        symmetries.append(symmetry)
    symmetries = np.array(symmetries)

    renderer = CanonicalScene(path_OBJ, camera_pose, min_corner,
                              max_corner, symmetries)
    renderer.scene.ambient_light = [1.0, 1.0, 1.0]
    image = renderer.render_symmetries()
    show_image(image)
    """

    """
    show_image(image)
    for arg in range(0):
        image, alpha, RGB_mask = renderer.render()
        show_image(RGB_mask[:, :, 0:3])

    background_wildcard = '.keras/paz/datasets/voc-backgrounds/*.png'
    background_wildcard = os.path.join(root_path, background_wildcard)
    image_paths = glob.glob(background_wildcard)

    H, W, num_channels = image_shape = (128, 128, 3)
    inputs_to_shape = {'input_1': [H, W, num_channels]}
    labels_to_shape = {'masks': [H, W, 4]}
    processor = DomainRandomization(
        renderer, image_shape, image_paths, inputs_to_shape,
        labels_to_shape, num_occlusions)

    for arg in range(100):
        sample = processor()
        image = sample['inputs']['input_1']
        image = (image * 255.0).astype('uint8')
        RGB_mask = sample['labels']['masks']
        # image, alpha, RGB_mask = renderer.render()
        RGB_mask_true = (RGB_mask[:, :, 0:3] * 255.0).astype('uint8')
        RGB_mask_pred = model.predict(np.expand_dims(image / 255.0, 0))
        RGB_mask_pred = np.squeeze(RGB_mask_pred * 255.0, 0)
        # error = np.square(RGB_mask_true - RGB_mask_pred)
        # error = RGB_mask_pred - RGB_mask
        RGB_mask_pred = RGB_mask_pred.astype('uint8')
        print(image.dtype, RGB_mask_pred.dtype, RGB_mask_true.dtype)
        # images = np.concatenate(
            [image, RGB_mask_pred, RGB_mask_true], axis=1)
        images = np.concatenate([image, RGB_mask_pred], axis=1)
        show_image(images)
    """
