import jax.numpy as jp


def to_box(mask, mask_value):
    height, width = mask.shape
    x_indices = jp.arange(width)
    x_indices = jp.broadcast_to(x_indices, (height, width))
    y_indices = jp.arange(height)
    y_indices = jp.broadcast_to(y_indices[:, None], (height, width))
    mask_valid = mask == mask_value
    x_min = jp.min(jp.where(mask_valid, x_indices, width))
    x_max = jp.max(jp.where(mask_valid, x_indices, -1))
    y_min = jp.min(jp.where(mask_valid, y_indices, height))
    y_max = jp.max(jp.where(mask_valid, y_indices, -1))
    return jp.array([x_min, y_min, x_max, y_max])
