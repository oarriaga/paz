import numpy as np


def sample_point_in_full_sphere(distance=1.0):
    """Get a point of the top of the unit sphere.

    # Arguments
        distance: Float, indicating distance to origin.

    # Returns
        sphere_point: List of spatial coordinates of a sphere.
    """
    if distance <= 0:
        raise ValueError('distance should be bigger than 0')
    sphere_point = np.random.uniform(-1, 1, size=3)
    return (distance * sphere_point) / np.linalg.norm(sphere_point)


def sample_point_in_top_sphere(distance=1.0):
    """Get a point of the top of the unit sphere.

    # Arguments
        distance: Float, indicating distance to origin.

    # Returns
        sphere_point: List of spatial coordinates of a sphere.
    """
    if distance <= 0:
        raise ValueError('distance should be bigger than 0')
    sphere_point = sample_point_in_full_sphere(distance)
    if sphere_point[1] < 0:
        sphere_point[1] = sphere_point[1] * -1
    return sphere_point


def sample_point_in_sphere(distance, top_only=False):
    """ Samples random points from a sphere

    # Arguments
        distance: Float, indicating distance to origin.

    # Returns:
        List of spatial coordinates of a sphere.

    """
    if distance <= 0:
        raise ValueError('distance should be bigger than 0')
    if top_only:
        sphere_point = sample_point_in_top_sphere(distance)
    else:
        sphere_point = sample_point_in_full_sphere(distance)
    return sphere_point


def random_perturbation(localization, shift):
    """Adds noise to 'localization' vector coordinates.

    # Arguments
        localization: List of 3 floats.
        shift: Float indicating a uniform distribution [-shift, shift].

    # Returns
        perturbed localization
    """
    perturbation = np.random.uniform(-shift, shift, size=3)
    return localization + perturbation


def random_translation(localization, shift):
    """Adds noise to 'localization' vector coordinates.

    # Arguments
        localization: List of 3 floats.
        shift: Float indicating a uniform distribution [-shift, shift].
    # Returns
        perturbed localization
    """
    perturbation = np.zeros((3))
    perturbation[:2] = np.random.uniform(-shift, shift, size=2)
    return localization + perturbation


def get_look_at_transform(camera_position, target_position):
    """Make transformation from target position to camera position
    with orientation looking at the target position.

    # Arguments
        camera_position: Numpy-array of length 3. Camera position.
        target_position: Numpy-array of length 3. Target position.
    """
    camera_direction = camera_position - target_position
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    world_up = np.array([0.0, 1.0, 0.0])
    camera_right = np.cross(world_up, camera_direction)
    camera_right = camera_right / np.linalg.norm(camera_right)
    camera_up = np.cross(camera_direction, camera_right)
    camera_up = camera_up / np.linalg.norm(camera_up)
    rotation_transform = np.zeros((4, 4))
    rotation_transform[0, :3] = camera_right
    rotation_transform[1, :3] = camera_up
    rotation_transform[2, :3] = camera_direction
    rotation_transform[-1, -1] = 1
    translation_transform = np.eye(4)
    translation_transform[:3, -1] = - camera_position
    look_at_transform = np.matmul(rotation_transform, translation_transform)
    return look_at_transform


def compute_modelview_matrices(camera_origin, world_origin,
                               roll=None, translate=None):
    """Compute model-view matrices from camera to origin and origin to camera.

    # Arguments
        camera_origin: Numpy-array of length 3 determining the camera origin
        world_origin: Numpy-array of length 3 determining the world origin
        roll: `None` or float. If `None` no roll is performed. If float
        value should be between [0, 2*pi)

    # Returns
        Transformation from camera to world and world to camera.
    """
    world_to_camera = get_look_at_transform(camera_origin, world_origin)
    if roll is not None:
        world_to_camera = roll_camera(world_to_camera, roll)
    if translate is not None:
        world_to_camera = translate_camera(world_to_camera, translate)
    camera_to_world = np.linalg.inv(world_to_camera)
    return camera_to_world, world_to_camera


def roll_camera(world_to_camera, angle):
    """ Roll camera coordinate system.

    # Arguments:
        world_to_camera: Numpy array containing the affine transformation.
        max_roll: 'None' or float. If None, the camera is not rolled.
            If float it should be a value between [0, 2*pi)
    """
    angle = np.random.uniform(-angle, angle)
    z_rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.],
         [np.sin(angle), +np.cos(angle), 0.],
         [0., 0., 1.]])
    world_to_camera[:3, :3] = np.matmul(z_rotation, world_to_camera[:3, :3])
    return world_to_camera


def translate_camera(world_to_camera, translation):
    """ Translate camera coordinate system in its XY plane.

    # Arguments:
        world_to_camera: Numpy array containing the affine transformation.
        translation: List or array with two inputs.
    """
    translation = np.random.uniform(-translation, translation, 2)
    translation_transform = np.array(
        [[1.0, 0.0, 0.0, translation[0]],
         [0.0, 1.0, 0.0, translation[1]],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]])
    world_to_camera = np.matmul(translation_transform, world_to_camera)
    return world_to_camera


def scale_translation(matrix, scale=10.0):
    """ Changes the scale of the translation vector.
    Used for changing the regression problem to a bigger scale.

    # Arguments:
        matrix: Numpy array of shape [4, 4]
        scale: Float used to multiple all the translation component.

    # Returns:
        Numpy array of shape [4, 4]
    """
    matrix[:3, -1] = matrix[:3, -1] * 10.
    return matrix


def sample_uniformly(value):
    """ Samples from a uniform distribution.

    # Arguments
        values: List or float. If list it must have [min_value, max_value].

    # Returns
        Float
    """
    if isinstance(value, list):
        value = np.random.uniform(value[0], value[1])
    return value


def split_alpha_channel(image):
    """ Splits alpha channel from an RGBD image.

    # Arguments
        image: Numpy array of shape [H, W, 4]

    # Returns
        List of two numpy arrays of shape [H, W, 3] and [H, W]
    """
    image_shape = image.shape
    if len(image_shape) != 3:
        raise ValueError('Invalid image shape')
    if image_shape[-1] != 4:
        raise ValueError('Invalid number of channels')
    return image[..., :3], image[..., 3:4]
