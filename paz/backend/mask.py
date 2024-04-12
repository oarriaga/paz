import numpy as np


def mask_to_box(mask, mask_value):
    """Computes bounding box from mask image.

    # Arguments
        mask: Array mask corresponding to raw image.
        mask_value: Int, pixel gray value of foreground in mask image.

    # Returns:
        box: List containing box coordinates.
    """
    masked = np.where(mask == mask_value)
    mask_x, mask_y = masked[1], masked[0]
    if mask_x.size <= 0 or mask_y.size <= 0:
        box = [0, 0, 0, 0]
    else:
        x_min, y_min = np.min(mask_x), np.min(mask_y)
        x_max, y_max = np.max(mask_x), np.max(mask_y)
        box = [x_min, y_min, x_max, y_max]
    return box
