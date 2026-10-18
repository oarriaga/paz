import jax.numpy as jp
import paz


def add_ones(points2D):
    ones = jp.ones((len(points2D), 1))
    points2D = jp.concatenate([points2D, ones], axis=-1)
    return points2D


def transform(points2D, affine_transform):
    original_dimension = points2D.shape[-1]
    points2D = add_ones(points2D)
    points2D = jp.matmul(affine_transform, points2D.T).T
    return points2D[:, :original_dimension]


def get_number(keypoints):
    return keypoints.shape[0]


def split(keypoints, keepdims=True, axis=1):
    """Split keypoints into components."""
    coordinates = jp.split(keypoints, 2, axis=axis)
    if not keepdims:
        coordinates = tuple(jp.squeeze(x, axis=-1) for x in coordinates)
    return coordinates


def merge(x, y):
    return jp.concatenate([x, y], axis=1)


def shift_to_box_origin(points, box):
    x_min, y_min, _, _ = box
    return points + jp.array([x_min, y_min])


def normalize_keypoints2D(points2D, height, width):
    image_shape = jp.array([width, height])
    return (2.0 * points2D / image_shape) - 1.0


def denormalize_keypoints2D(points2D, height, width):
    image_shape = jp.array([width, height])
    return (points2D + 1.0) / 2.0 * image_shape


def rotate_point2D(point2D, rotation_angle):
    angle = jp.pi * rotation_angle / 180.0
    sin_angle, cos_angle = jp.sin(angle), jp.cos(angle)
    x = point2D[0] * cos_angle - point2D[1] * sin_angle
    y = point2D[0] * sin_angle + point2D[1] * cos_angle
    return jp.array([x, y])


def rotate_keypoints2D(keypoints, angle, center):
    cos_angle, sin_angle = jp.cos(angle), jp.sin(angle)
    rotation = jp.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    return (keypoints - center) @ rotation.T + center


def flip_keypoints_left_right(keypoints, width):
    x, y = jp.split(keypoints, 2, axis=1)
    return jp.concatenate([width - x, y], axis=1)


def translate_keypoints(keypoints, translation):
    return keypoints + translation


def uv_to_vu(keypoints):
    return keypoints[:, ::-1]


def denormalize(keypoints, H, W):
    """Transform nomralized points2D to image UV coordinates i.e.
        [-1, 1] -> [U, V]. UV have maximum values of [W, H] respectively.

    Normalized Coordinates         Image plane

              (y)                   (0,0)-------->  (U)
                                      |        (W,0)
         (0,1) ^                      |
               |            |-->      |
               |                      v (H, 0)
               |
         (0,0) o -----> (x)          (V)
                    (1,0)

    # Arguments
        points2D: Numpy array of shape (num_keypoints, 2).
        height: Int. Height of the image
        width: Int. Width of the image

    # Returns
        Numpy array of shape (num_keypoints, 2).
    """

    keypoints = jp.clip(keypoints, -1.0, 1.0)
    keypoints = keypoints + 1.0
    keypoints = keypoints / 2.0
    x, y = split(keypoints)
    x = (W - 1.0) * x
    y = (H - 1.0) * (1.0 - y)
    denormalized_keypoints = merge(x, y)
    denormalized_keypoints = jp.round(denormalized_keypoints)
    return paz.cast(denormalized_keypoints, "int32")
