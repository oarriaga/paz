import numpy as np
from backend import build_rotation_matrix_y
from paz.backend.render import sample_uniformly, split_alpha_channel
from pyrender import (PerspectiveCamera, OffscreenRenderer, DirectionalLight,
                      RenderFlags, Mesh, Scene)
import trimesh
from coloring import color_object
from backend import quaternion_to_rotation_matrix
from backend import to_affine_matrix


def sample_uniform(min_value, max_value):
    """Samples values inside segment [min_value, max_value)

    # Arguments
        segment_limits: List (2) containing min and max segment values.

    # Returns
        Float inside segment [min_value, max_value]
    """
    if min_value > max_value:
        raise ValueError('First value must be lower than second value')
    value = np.random.uniform(min_value, max_value)
    return value


def sample_inside_box3D(min_W, min_H, min_D, max_W, max_H, max_D):
    """ Samples points inside a 3D box defined by the
        width, height and depth limits.
                    ________
                   /       /|
                  /       / |
                 /       /  |
                /_______/   /
         |      |       |  /   /
       height   |       | / depth
         |      |_______|/   /

                --widht--

    # Arguments
        width_limits: List (2) with [min_value_width, max_value_width].
        height_limits: List (2) with [min_value_height, max_value_height].
        depth_limits: List (2) with [min_value_depth, max_value_depth].

    # Returns
        Array (3) of point inside the 3D box.
    """
    W = sample_uniform(min_W, max_W)
    H = sample_uniform(min_H, max_H)
    D = sample_uniform(min_D, max_D)
    box_point3D = np.array([W, H, D])
    return box_point3D


def sample_random_rotation_matrix2():
    """Samples SO3 in rotation matrix form.

    # Return
        Array (3, 3).

    # References
        [Lost in my terminal](http://blog.lostinmyterminal.com/python/2015/05/
            12/random-rotation-matrix.html)
        [real-time rendering](from http://www.realtimerendering.com/resources/
            GraphicsGems/gemsiii/rand_rotation.c)
    """
    theta = 2.0 * np.pi * np.random.uniform()
    phi = 2.0 * np.pi * np.random.uniform()
    z = 2.0 * np.random.uniform()
    # Compute a vector V used for distributing points over the sphere via the
    # reflection I - V Transpose(V).
    # This formulation of V will guarantee that if x[1] and x[2] are uniformly
    # distributed, the reflected points will be uniform on the sphere.
    # random_vector has length sqrt(2) to eliminate 2 in the Householder matrix
    r = np.sqrt(z)
    random_vector = np.array([np.sin(phi) * r,
                              np.cos(phi) * r,
                              np.sqrt(2.0 - z)])
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    R = np.array([[+cos_theta, +sin_theta, 0.0],
                  [-sin_theta, +cos_theta, 0.0],
                  [0.0, 0.0, 1.0]])
    random_rotation_matrix = (
        np.outer(random_vector, random_vector) - np.eye(3)).dot(R)
    return random_rotation_matrix


def sample_random_rotation_matrix():
    quaternion = np.random.rand(4)
    quaternion = quaternion / np.linalg.norm(quaternion)
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    return rotation_matrix


def sample_random_rotation_matrix3():
    epsilon = 0.1
    x_angle = np.random.uniform((-np.pi / 2.0) + epsilon, (np.pi / 2.0) - epsilon)
    y_angle = np.random.uniform((-np.pi / 2.0) + epsilon, (np.pi / 2.0) - epsilon)
    z_angle = np.random.uniform(np.pi, -np.pi)

    x_matrix = build_rotation_matrix_x(x_angle)
    y_matrix = build_rotation_matrix_y(y_angle)
    z_matrix = build_rotation_matrix_z(z_angle)

    rotation_matrix = np.dot(z_matrix, np.dot(y_matrix, x_matrix))
    return rotation_matrix


def sample_affine_transform(min_corner, max_corner):
    min_W, min_H, min_D = min_corner
    max_W, max_H, max_D = max_corner
    translation = sample_inside_box3D(min_W, min_H, min_D, max_W, max_H, max_D)
    rotation_matrix = sample_random_rotation_matrix3()
    affine_matrix = to_affine_matrix(rotation_matrix, translation)
    return affine_matrix


class CanonicalScene():
    def __init__(self, path_OBJ, camera_pose, min_corner, max_corner,
                 symmetric_transforms,
                 viewport_size=(128, 128), y_fov=3.14159 / 4.0,
                 light_intensity=[0.5, 30]):
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
        # print(images.shape)
        # print(RGB_masks.shape)
        images = np.concatenate([images, RGB_masks], axis=0)
        return images


def compute_norm_SO3(rotation_mesh, rotation):
    difference = np.dot(np.linalg.inv(rotation), rotation_mesh) - np.eye(3)
    distance = np.linalg.norm(difference, ord='fro')
    return distance


def calculate_canonical_rotation(rotation_mesh, rotations):
    norms = [compute_norm_SO3(rotation_mesh, R) for R in rotations]
    closest_rotation_arg = np.argmin(norms)
    # print(closest_rotation_arg)
    closest_rotation = rotations[closest_rotation_arg]
    canonical_rotation = np.linalg.inv(closest_rotation)
    return canonical_rotation


if __name__ == "__main__":
    import os
    from paz.backend.image import show_image
    from backend import build_rotation_matrix_z
    from backend import build_rotation_matrix_x
    # from backend import build_rotation_matrix_y
    path_OBJ = 'single_solar_panel_02.obj'
    root_path = os.path.expanduser('~')
    path_OBJ = os.path.join(root_path, path_OBJ)
    num_occlusions = 1
    image_shape = (128, 128, 3)
    viewport_size = image_shape[:2]
    y_fov = 3.14159 / 4.0
    distance = [1.0, 1.0]
    light = [1.0, 30]

    # min_corner = [-0.1, -0.1, -0.0]
    # max_corner = [+0.1, +0.1, +0.4]
    angles = np.linspace(0, 2 * np.pi, 7)[:6]
    symmetric_rotations = np.array(
        [build_rotation_matrix_z(angle) for angle in angles])
    min_corner = [0.0, 0.0, -0.4]
    max_corner = [0.0, 0.0, +0.0]
    # translation = np.array([0.0, 0.0, 1.0])
    translation = np.array([0.0, 0.0, 1.0])
    camera_rotation = np.eye(3)
    camera_rotation = build_rotation_matrix_x(np.pi)
    translation = np.array([0.0, 0.0, -1.0])
    camera_pose = to_affine_matrix(camera_rotation, translation)
    renderer = CanonicalScene(path_OBJ, camera_pose, min_corner,
                              max_corner, symmetric_rotations)
    # from pyrender import Viewer
    # Viewer(scene.scene)
    renderer.scene.ambient_light = [1.0, 1.0, 1.0]
    image = renderer.render_symmetries()
    show_image(image)
    for _ in range(0):
        image, alpha, RGB_mask = renderer.render()
        show_image(image)
        show_image(RGB_mask[:, :, 0:3])

    from pipelines import DomainRandomization
    from paz.abstract.sequence import GeneratingSequence
    from loss import WeightedReconstruction
    from paz.models import UNET_VGG16
    from tensorflow.keras.optimizers import Adam
    from metrics import mean_squared_error
    import glob

    background_wildcard = '.keras/paz/datasets/voc-backgrounds/*.png'
    background_wildcard = os.path.join(root_path, background_wildcard)
    image_paths = glob.glob(background_wildcard)

    H, W, num_channels = image_shape
    batch_size = 32
    steps_per_epoch = 1000
    beta = 3.0
    num_classes = 3
    learning_rate = 0.001
    max_num_epochs = 5

    inputs_to_shape = {'input_1': [H, W, num_channels]}
    labels_to_shape = {'masks': [H, W, 4]}

    processor = DomainRandomization(
        renderer, image_shape, image_paths, inputs_to_shape,
        labels_to_shape, num_occlusions)

    sequence = GeneratingSequence(processor, batch_size, steps_per_epoch)

    # build all symmetric rotations for solar pannel
    angles = np.linspace(0, 2 * np.pi, 7)[:6]
    rotations = np.array([build_rotation_matrix_z(angle) for angle in angles])

    # loss = WeightedSymmetricReconstruction(rotations, beta)
    loss = WeightedReconstruction(beta)

    model = UNET_VGG16(num_classes, image_shape, freeze_backbone=True)
    optimizer = Adam(learning_rate)

    model.compile(optimizer, loss, mean_squared_error)

    model.fit(
        sequence,
        epochs=max_num_epochs,
        verbose=1,
        workers=0)
    model.save_weights('UNET-VGG_solar_panel_canonical.hdf5')

