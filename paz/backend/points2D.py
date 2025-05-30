import jax.numpy as jp


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


def denormalize(points2D, H, W):
    """Transform nomralized points2D to image UV coordinates i.e.
        [-1, 1] -> [U, V]. UV have maximum values of [W, H] respectively.

             Image plane

           (0,0)-------->  (U)
             |
             |
             |
             v

            (V)

    # Arguments
        points2D: Numpy array of shape (num_keypoints, 2).
        height: Int. Height of the image
        width: Int. Width of the image

    # Returns
        Numpy array of shape (num_keypoints, 2).
    """
    image_shape = jp.array([W, H])
    points2D = points2D + 1.0  # [-1, 1], [-1, 1] -> [2, 0], [0, 2]
    points2D = points2D / 2.0  # [2 , 0], [0 , 2] -> [1, 0], [0, 1]
    points2D = points2D * image_shape  # [1 , 0], [0 , 1] -> [W, 0], [0, H]
    return points2D.astype("int32")
