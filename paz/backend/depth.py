import jax
import jax.numpy as jp
import numpy as np
import paz
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


def bound(depth, max_depth):
    depth_2d = depth[..., 0] if depth.ndim == 3 else depth
    is_positive = depth_2d > 0
    is_close_by = depth_2d < max_depth
    mask = jp.logical_and(is_positive, is_close_by)
    return jp.where(mask, depth_2d, 0.0)


def compute_sobel_edges(depth, max_depth):
    valid_depth = bound(depth, max_depth)
    gradients = paz.image.apply_sobel(valid_depth)
    return paz.image.compute_channel_norm(gradients)


def compute_gradient_norm(depth):
    gradient_x = jp.abs(jp.diff(depth, axis=1, prepend=depth[:, :1]))
    gradient_y = jp.abs(jp.diff(depth, axis=0, prepend=depth[:1, :]))
    return jp.sqrt(gradient_x**2 + gradient_y**2)


def to_soft_mask(depth, min_depth, max_depth):
    inner = ((-depth) + max_depth) / (max_depth - min_depth + 1e-8)
    inner = inner - 0.5
    moved = jp.where(depth > 1e-5, inner, -1000.0)
    scale = jp.log((1 - 1e-3) / 1e-3)
    return jax.nn.sigmoid(scale * moved)
