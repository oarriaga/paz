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


def prepare(mask):
    mask = jp.array(mask).astype(bool)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def to_rgb(image, mask):
    image = jp.array(image)
    if jp.issubdtype(image.dtype, jp.integer):
        image = image.astype(jp.float32) / 255.0
    else:
        image = image.astype(jp.float32)
    mask = jp.expand_dims(prepare(mask), axis=-1)
    mask_size = jp.maximum(mask.sum(), 1.0)
    return (image * mask).sum(axis=(0, 1)) / mask_size


def to_RGB(image, mask):
    color = jp.clip(to_rgb(image, mask), 0.0, 1.0)
    return (255.0 * color).astype(jp.uint8)
