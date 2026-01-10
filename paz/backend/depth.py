import jax.numpy as jp
import numpy as np
import cv2


def write(depth, filepath, scale=1e4):
    # depth image should be in meters
    depth = np.array(depth)
    depth = (scale * depth).astype("uint16")
    cv2.imwrite(filepath, depth)


def load(filepath, scale=1e4):
    depth = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
    depth = np.expand_dims(depth, axis=2)
    depth = depth / scale
    return np.array(depth)


def crop(image, box):
    x_min, y_min, x_max, y_max = box
    return image[y_min:y_max, x_min:x_max, :]


def to_RGB(depth):
    min_depth = jp.min(depth)
    max_depth = jp.max(depth)
    normalized_depth = depth - min_depth / (max_depth - min_depth)
    return (255 * normalized_depth).astype("uint8")


def compute_gradient_norm(depth):
    gradient_x = jp.abs(jp.diff(depth, axis=1, prepend=depth[:, :1]))
    gradient_y = jp.abs(jp.diff(depth, axis=0, prepend=depth[:1, :]))
    return jp.sqrt(gradient_x**2 + gradient_y**2)
