import jax.numpy as jp
import jax
import cv2
import paz


def split(boxes, keepdims=True, axis=1):
    """Split boxes into x_min, y_min, x_max, y_max components."""
    coordinates = jp.split(boxes, 4, axis=axis)
    if not keepdims:
        coordinates = tuple(
            jp.squeeze(column, axis=-1) for column in coordinates
        )

    return coordinates


def merge(coordinate_0, coordinate_1, coordinate_2, coordinate_3):
    coordinates = [coordinate_0, coordinate_1, coordinate_2, coordinate_3]
    return jp.concatenate(coordinates, axis=1)


def compute_areas(boxes, keepdims=True):
    x_min, y_min, x_max, y_max = split(boxes, keepdims=keepdims)
    W = x_max - x_min
    H = y_max - y_min
    return W * H


def join(boxes):
    return jp.concatenate(boxes, axis=0)


def build_invalid(shape=(1, 4), value=-1):
    return jp.full(shape, value)


def square(boxes):
    """Makes boxes square with sides equal to the longest original side.

    # Arguments
        box: Numpy array with shape `(4)` with point corner coordinates.

    # Returns
        returns: List of box coordinates ints.
    """
    x_min, y_min, x_max, y_max = split(boxes)
    center_x = (x_max + x_min) / 2.0
    center_y = (y_max + y_min) / 2.0
    boxes_W = x_max - x_min
    boxes_H = y_max - y_min
    x_min = jp.where(boxes_H >= boxes_W, center_x - (boxes_H / 2.0), x_min)
    x_max = jp.where(boxes_H >= boxes_W, center_x + (boxes_H / 2.0), x_max)
    y_min = jp.where(boxes_H >= boxes_W, y_min, center_y - (boxes_W / 2.0))
    y_max = jp.where(boxes_H >= boxes_W, y_max, center_y + (boxes_W / 2.0))
    return merge(x_min, y_min, x_max, y_max).astype(int)


def compute_sizes(boxes, keepdims=True):
    """Compute width and height from boxes in corner format."""
    x_min, y_min, x_max, y_max = split(boxes, keepdims)
    W = x_max - x_min
    H = y_max - y_min
    return H, W


def compute_centers(boxes):
    """Compute center coordinates of boxes."""
    x_min, y_min, x_max, y_max = split(boxes)
    center_x = (x_max + x_min) / 2.0
    center_y = (y_max + y_min) / 2.0
    return center_x, center_y


def to_center_form(boxes):
    """Convert bounding boxes from corner to center form.

    # Arguments:
        boxes (array): Boxes in corner format [x_min, y_min, x_max, y_max]

    # Returns:
        array: Boxes in center format [center_x, center_y, width, height]
    """

    center_x, center_y = compute_centers(boxes)
    H, W = compute_sizes(boxes)
    return jp.concatenate([center_x, center_y, W, H], axis=1)


def to_corner_form(boxes):
    """Convert bounding boxes from center to corner form.

    # Arguments:
        Boxes: Array of boxes in center format ``[center_x, center_y, W, H]``.

    # Returns:
        Boxes in corner format ``[x_min, y_min, x_max, y_max]``.
    """
    center_x, center_y, W, H = split(boxes)
    x_min = center_x - (W / 2.0)
    x_max = center_x + (W / 2.0)
    y_min = center_y - (H / 2.0)
    y_max = center_y + (H / 2.0)
    return jp.concatenate([x_min, y_min, x_max, y_max], axis=1)


def pad_batch(ragged_boxes, size, value=-1):
    """Pad boxes array to specified size with value.

    # Arguments:
        boxes (list): List of lists containing `(num_boxes, 4)` box coordinates.
        size (int): target size for padding.
        value (int): Value to pad with.

    # Returns:
        Padded boxes with shape `(size, 4)`.
    """
    return jp.array([pad(box, size, value) for box in ragged_boxes])


def pad(image_boxes, size, value=-1):
    image_boxes = image_boxes[:size]
    padding = ((0, size - len(image_boxes)), (0, 0))
    return jp.pad(image_boxes, padding, "constant", constant_values=value)


def from_selection(image, radius=5, color=(255, 0, 0), window_name="image"):
    """Manually select bounding boxes by double-clicking on an image."""
    points, boxes = [], []

    def order_xyxy(point_A, point_B):
        (x1, y1) = point_A
        (x2, y2) = point_B
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def take_last_two_points(points):
        point_A = points[-1]
        point_B = points[-2]
        return point_A, point_B

    def on_double_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            paz.draw.circle(image, (x, y), radius, color)
            points.append((x, y))
            if len(points) % 2 == 0:
                point_A, point_B = take_last_two_points(points)
                box = order_xyxy(point_A, point_B)
                paz.draw.box(image, box, color, radius)
                boxes.append(box)

    def key_is_pressed(key="q", time=20):
        return cv2.waitKey(time) & 0xFF == ord(key)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_double_click)
    while True:
        paz.image.show(image, window_name, False)
        if key_is_pressed("q"):
            break
    cv2.destroyWindow(window_name)
    return jp.array(boxes)


def compute_IOU(box_A, boxes_B):
    """Computes intersection over union between `box_A` and `boxes_B`.

    # Arguments
        box_A: JAX Numpy array with shape `(4,)` representing a single box
            in corner form (x_min, y_min, x_max, y_max).
        boxes_B: JAX Numpy array with shape `(num_boxes_b, 4)` representing
            multiple boxes in corner form.

    # Returns
        JAX Numpy array of shape `(num_boxes_b,)` containing IoUs.
    """
    xy_min_inter = jp.maximum(box_A[0:2], boxes_B[:, 0:2])
    xy_max_inter = jp.minimum(box_A[2:4], boxes_B[:, 2:4])
    inter_wh = jp.maximum(0.0, xy_max_inter - xy_min_inter)
    intersection_area = inter_wh[:, 0] * inter_wh[:, 1]
    area_a = (box_A[2] - box_A[0]) * (box_A[3] - box_A[1])
    areas_b = (boxes_B[:, 2] - boxes_B[:, 0]) * (boxes_B[:, 3] - boxes_B[:, 1])
    union_area = (area_a + areas_b) - intersection_area
    union_area = jp.maximum(union_area, 1e-8)
    iou = intersection_area / union_area
    return jp.clip(iou, 0.0, 1.0)


def compute_IOUs(boxes_A, boxes_B):
    """Computes intersection over union (IOU) between `boxes_A` and `boxes_B`.

    For each box (rows `boxes_A`) it computes the IOU to all `boxes_B`.

    # Arguments
        boxes_A: Numpy array with shape `(num_boxes_A, 4)` in corner form.
        boxes_B: Numpy array with shape `(num_boxes_B, 4)` in corner form.

    # Returns
        Numpy array of shape `(num_boxes_A, num_boxes_B)`.
    """
    xy_min = jp.maximum(boxes_A[:, None, 0:2], boxes_B[:, 0:2])
    xy_max = jp.minimum(boxes_A[:, None, 2:4], boxes_B[:, 2:4])
    intersection = jp.maximum(0.0, xy_max - xy_min)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    areas_A = (boxes_A[:, 2] - boxes_A[:, 0]) * (boxes_A[:, 3] - boxes_A[:, 1])
    areas_B = (boxes_B[:, 2] - boxes_B[:, 0]) * (boxes_B[:, 3] - boxes_B[:, 1])
    # broadcasting for outer sum i.e. a sum of all possible combinations
    union_area = (areas_A[:, jp.newaxis] + areas_B) - intersection_area
    union_area = jp.maximum(union_area, 1e-8)
    return jp.clip(intersection_area / union_area, 0.0, 1.0)


def xyxy_to_xywh(boxes):
    x_min, y_min, x_max, y_max = split(boxes)
    W = x_max - x_min
    H = y_max - y_min
    return merge(x_min, y_min, W, H)


def xywh_to_xyxy(boxes):
    x_min, y_min, W, H = split(boxes)
    x_max = x_min + W
    y_max = y_min + H
    boxes = merge(x_min, y_min, x_max, y_max)
    return boxes


def flip_left_right(boxes, W):
    """Flips box coordinates from left-to-right and vice-versa.

    # Arguments
        boxes: Numpy array of shape `(num_boxes, 4)`.

    # Returns
        Numpy array of shape `(num_boxes, 4)`.
    """
    x_min, y_min, x_max, y_max = split(boxes)
    return merge(x_max, y_min, x_min, y_max)


def append_class(boxes, class_arg):
    class_args = jp.full((len(boxes), 1), class_arg)
    return jp.hstack((boxes, class_args))


def sample(key, H, W, box_size, num_boxes=15):
    keys = jax.random.split(key)
    H_box, W_box = box_size
    x_min = jax.random.randint(keys[0], (num_boxes, 1), 0, W - W_box + 1)
    y_min = jax.random.randint(keys[1], (num_boxes, 1), 0, H - H_box + 1)
    x_max = x_min + W_box
    y_max = y_min + H_box
    return merge(x_min, y_min, x_max, y_max)


def sample_negatives(key, boxes, H, W, box_size, num_boxes, num_trials):
    negative_boxes = paz.boxes.sample(key, H, W, box_size, num_trials)
    ious = compute_IOUs(negative_boxes, boxes)
    mean_ious = jp.mean(ious, axis=1)
    # best_args = jp.argsort(mean_ious)[::-1]
    best_args = jp.argsort(mean_ious)
    best_args = best_args[:num_boxes]
    return negative_boxes[best_args]


def denormalize(boxes, H, W):
    return (boxes * jp.array([[W, H, W, H]])).astype(int)


def normalize(boxes, H, W):
    return boxes / jp.array([[W, H, W, H]])


def scale(boxes, scale_W, scale_H):
    """Scales the width and height of a bounding box (xywh format)."""
    x_center, y_center, W, H = split(to_center_form(boxes))
    new_W = scale_W * W
    new_H = scale_H * H
    boxes = merge(x_center, y_center, new_W, new_H)
    return to_corner_form(boxes)


def translate(boxes, x_offset, y_offset):
    """Translates the center of a bounding box (xywh format)."""
    x_center, y_center, W, H = split(xyxy_to_xywh(boxes))
    x_new_center = x_center + x_offset
    y_new_center = y_center + y_offset
    boxes = merge(x_new_center, y_new_center, W, H)
    return xywh_to_xyxy(boxes)


def clip(boxes, H, W):
    """Clips bounding box coordinates to image boundaries."""
    x_min, y_min, x_max, y_max = split(boxes)
    x_min_clipped = jp.clip(x_min, 0, W - 1)
    y_min_clipped = jp.clip(y_min, 0, H - 1)
    x_max_clipped = jp.clip(x_max, 0, W - 1)
    y_max_clipped = jp.clip(y_max, 0, H - 1)
    boxes = merge(x_min_clipped, y_min_clipped, x_max_clipped, y_max_clipped)
    return boxes


def jitter(key, boxes, H, W, scale_range, shift_range):
    """Applies random scaling and translation, then clips"""
    # this function jitters all boxes with the same translation and scale
    # shall we jitter all of the boxes?
    keys = jax.random.split(key, 4)
    scale_min, scale_max = scale_range
    shift_min, shift_max = shift_range
    scale_W = jax.random.uniform(keys[0], minval=scale_min, maxval=scale_max)
    scale_H = jax.random.uniform(keys[1], minval=scale_min, maxval=scale_max)
    x_offset = jax.random.randint(keys[2], (), shift_min, shift_max + 1)
    y_offset = jax.random.randint(keys[3], (), shift_min, shift_max + 1)
    boxes = translate(scale(boxes, scale_W, scale_H), x_offset, y_offset)
    return clip(boxes, H, W)


def sample_positives(key, boxes, H, W, num_samples, scale_range, shift_range):

    def select_random_box(key, boxes):
        arg = jax.random.randint(key, shape=(), minval=0, maxval=len(boxes))
        return jp.expand_dims(boxes[arg], 0)

    def apply(boxes, key):
        box = select_random_box(key, boxes)
        box = jitter(key, box, H, W, scale_range, shift_range)
        return boxes, jp.squeeze(box, axis=0)

    keys = jax.random.split(key, num_samples)
    _, jittered_boxes = jax.lax.scan(apply, boxes, keys)
    return jittered_boxes.astype(boxes.dtype)


def filter_in_image(boxes, H, W):
    """Filter boxes that are outside the image boundaries."""
    x_min, y_min, x_max, y_max = split(boxes, keepdims=False)
    valid_mask = jp.logical_and(
        jp.logical_and(x_min >= 0, y_min >= 0),
        jp.logical_and(x_max < W, y_max < H),
    )
    return boxes[valid_mask]


def crop_with_pad(boxes, image, box_H, box_W, pad_value=0):
    x_min, y_min, x_max, y_max = split(boxes, False)
    boxes_H, boxes_W = compute_sizes(boxes, False)
    delta_x = jp.arange(box_W)
    delta_y = jp.arange(box_H)

    H, W = paz.image.get_size(image)
    x_args = jp.expand_dims(x_min[:, None] + delta_x[None, :], axis=1)
    y_args = jp.expand_dims(y_min[:, None] + delta_y[None, :], axis=2)
    x_args_in_image_bounds = (x_args >= 0) & (x_args < W)
    y_args_in_image_bounds = (y_args >= 0) & (y_args < H)

    x_offset_in_box_bounds = delta_x[None, :] < boxes_W[:, None]
    y_offset_in_box_bounds = delta_y[None, :] < boxes_H[:, None]
    x_offset_in_box_bounds = x_offset_in_box_bounds[:, None, :]
    y_offset_in_box_bounds = y_offset_in_box_bounds[:, :, None]

    mask_x = x_offset_in_box_bounds & x_args_in_image_bounds
    mask_y = y_offset_in_box_bounds & y_args_in_image_bounds
    valid_mask = jp.expand_dims(mask_y & mask_x, axis=3)
    safe_x_args = jp.clip(x_args, 0, W - 1)
    safe_y_args = jp.clip(y_args, 0, H - 1)
    gathered_image = image[safe_y_args, safe_x_args, :]
    # pad = jp.full_like(gathered_image, pad_value, dtype=image.dtype)
    return jp.where(valid_mask, gathered_image, pad_value)


def match(boxes, prior_boxes, IOU_threshold=0.5):
    """Matches each prior box with a ground truth box (box from `boxes`).
    It then selects which matched box will be considered positive e.g. iou > .5
    and returns for each prior box a ground truth box that is either positive
    (with a class argument different than 0) or negative.

    # Arguments
        boxes: Numpy array of shape `(num_ground_truh_boxes, 4 + 1)`,
            where the first the first four coordinates correspond to
            box coordinates and the last coordinates is the class
            argument. This boxes should be the ground truth boxes.
        prior_boxes: Numpy array of shape `(num_prior_boxes, 4)`.
            where the four coordinates are in center form coordinates.
        iou_threshold: Float between [0, 1]. Intersection over union
            used to determine which box is considered a positive box.

    # Returns
        Array of shape `(num_prior_boxes, 4 + 1)`.
            where the first the first four coordinates correspond to point
            form box coordinates and the last coordinates is the class
            argument.
    """

    def mark_best_match(per_prior_best_IOU, per_box_best_prior_arg):
        # The prior boxes that are the best match for each box are marked.
        # They are marked by setting an IOU larger (2) than the maxium (1).
        # the best prior box match of box_0 is per_box_best_prior_arg[0]
        # the best prior box match of box_1 is per_box_best_prior_arg[1]
        # ...
        return per_prior_best_IOU.at[per_box_best_prior_arg].set(2.0)

    def select_for_each_prior_box_a_box(boxes, per_prior_best_box):
        # Each prior box is assigned a ground truth box.
        assigned_boxes = boxes[per_prior_best_box]
        return assigned_boxes

    def force_match(per_prior_best_box, per_box_best_prior):
        # Ensures that every ground truth box is matched with at least one prior
        # box. Specifically, the prior box with which it has the highest IoU.
        for box_arg, prior_arg in enumerate(per_box_best_prior):
            per_prior_best_box = per_prior_best_box.at[prior_arg].set(box_arg)
        return per_prior_best_box

    def label_negative_boxes(assigned_boxes, per_prior_best_IOU):
        is_low_IOU_match = per_prior_best_IOU < IOU_threshold
        class_args = assigned_boxes[:, 4]
        class_args = jp.where(is_low_IOU_match, 0.0, class_args)
        return assigned_boxes.at[:, 4].set(class_args)

    prior_boxes = to_corner_form(prior_boxes)
    IOUs = compute_IOUs(boxes, prior_boxes)  # (boxes, prior_boxes)
    per_box_best_prior = jp.argmax(IOUs, axis=1)  # (boxes,)
    per_prior_best_box = jp.argmax(IOUs, axis=0)  # (prior_boxes,)
    per_prior_best_IOU = jp.max(IOUs, axis=0)  # (prior_boxes,)
    per_prior_best_IOU = mark_best_match(per_prior_best_IOU, per_box_best_prior)
    assign_args = (per_prior_best_box, per_box_best_prior)
    per_prior_best_box = force_match(*assign_args)
    selected_boxes = select_for_each_prior_box_a_box(boxes, per_prior_best_box)
    selected_boxes = label_negative_boxes(selected_boxes, per_prior_best_IOU)
    return selected_boxes


def remove_invalid(boxes, value=-1):
    # is_invalid_row_mask = jp.all(boxes == value, axis=1)
    is_invalid_row_mask = jp.any(boxes < 0.0, axis=1)
    is_valid_row_mask = jp.logical_not(is_invalid_row_mask)
    valid_boxes = boxes[is_valid_row_mask]
    return valid_boxes


def resize(boxes, H, W, H_new, W_new):
    """Resize boxes to a new image size."""
    x_min, y_min, x_max, y_max = split(boxes)
    x_min = (x_min / W) * W_new
    y_min = (y_min / H) * H_new
    x_max = (x_max / W) * W_new
    y_max = (y_max / H) * H_new
    return merge(x_min, y_min, x_max, y_max).astype(dtype=boxes.dtype)


def set_size(boxes, H_box, W_box):
    x_center, y_center = paz.boxes.compute_centers(boxes)
    x_min = x_center - (W_box / 2.0)
    y_min = y_center - (H_box / 2.0)
    x_max = x_center + (W_box / 2.0)
    y_max = y_center + (H_box / 2.0)
    return merge(x_min, y_min, x_max, y_max).astype(dtype=boxes.dtype)
