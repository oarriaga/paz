import jax.numpy as jp


def to_box(mask, mask_value):
    """Computes bounding box from mask image.

    # Arguments
        mask: Array mask corresponding to raw image.
        mask_value: Int, pixel gray value of foreground in mask image.

    # Returns:
        box: List containing box coordinates.
    """
    masked = jp.where(mask == mask_value)
    mask_x, mask_y = masked[1], masked[0]
    x_min, y_min = jp.min(mask_x), jp.min(mask_y)
    x_max, y_max = jp.max(mask_x), jp.max(mask_y)
    return jp.array([x_min, y_min, x_max, y_max])
