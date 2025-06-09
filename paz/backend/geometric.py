import jax
import jax.numpy as jp
import paz


def do_crop(key, probability):
    random_value = jax.random.uniform(key, shape=())
    return probability >= random_value


def mask_boxes_and_labels(boxes, labels, crop_region):
    """Masks boxes and labels with crop region."""
    x_min_crop, y_min_crop, x_max_crop, y_max_crop = crop_region
    x_min, y_min, x_max, y_max = paz.boxes.split(boxes)

    x_min_valid = x_min < x_max_crop
    y_min_valid = y_min < y_max_crop
    x_max_valid = x_max > x_min_crop
    y_max_valid = y_max > y_min_crop

    mask = x_min_valid & y_min_valid & x_max_valid & y_max_valid

    return boxes[mask], labels[mask]


def adjust_boxes(boxes, crop_region):
    """Adjusts box coordinates relative to the crop region's top-left corner."""
    top_left = jp.maximum(boxes[:, :2], crop_region[:2]) - crop_region[:2]
    bottom_right = jp.minimum(boxes[:, 2:], crop_region[2:]) - crop_region[:2]
    return jp.concatenate([top_left, bottom_right], axis=1)


def adjust_boxes_and_labels(boxes, labels, crop_region):
    """Filters and adjusts boxes to fit within crop region, maintaining labels."""
    valid_boxes, valid_labels = mask_boxes_and_labels(
        boxes, labels, crop_region
    )
    adjusted_boxes = adjust_boxes(valid_boxes, crop_region)
    return jp.hstack([adjusted_boxes, valid_labels])


def compute_crop_limits(size):
    """Computes minimum and maximum allowable crop dimensions."""
    min_dim = jp.maximum(1, jp.floor(0.3 * size)).astype(jp.int32)
    return min_dim, size


def generate_crop_dimensions(key, W, H):
    """Generates valid crop dimensions respecting aspect ratio constraints."""
    min_w, max_w = compute_crop_limits(W)
    min_h, max_h = compute_crop_limits(H)
    return (
        jax.random.randint(key, (), min_w, max_w, dtype=jp.int32),
        jax.random.randint(key, (), min_h, max_h, dtype=jp.int32),
    )


def build_crop_region(key, width, height, orig_width, orig_height):
    """Constructs crop region coordinates within original image boundaries.
    Args:
        key (random key): Random number generator key.
        width (int): Crop width.
        height (int): Crop height.
        orig_width (int): Original image width.
        orig_height (int): Original image height.
    Returns:
        array: Crop region array [x_start, y_start, x_end, y_end].
    """
    x_start = jax.random.randint(key, (), 0, orig_width - width, dtype=jp.int32)
    y_start = jax.random.randint(key, (), 0, orig_height - height, dtype=jp.int32)
    return jp.array([x_start, y_start, x_start + width, y_start + height])



