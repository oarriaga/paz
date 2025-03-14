import jax.numpy as jnp
import jax
from jax import lax, jit


def to_center_form(boxes):
    """Transform from corner coordinates to center coordinates.

    # Arguments
        boxes: JAX array with shape `(num_boxes, 4)`.

    # Returns
        JAX array with shape `(num_boxes, 4)`.
    """
    boxes = jnp.asarray(boxes)
    x_min, y_min = boxes[:, 0:1], boxes[:, 1:2]
    x_max, y_max = boxes[:, 2:3], boxes[:, 3:4]
    center_x = (x_max + x_min) / 2.0
    center_y = (y_max + y_min) / 2.0
    W = x_max - x_min
    H = y_max - y_min
    return jnp.concatenate([center_x, center_y, W, H], axis=1)


def to_corner_form(boxes):
    """Transform from center coordinates to corner coordinates.

    # Arguments
        boxes: JAX array with shape `(num_boxes, 4)`.

    # Returns
        JAX array with shape `(num_boxes, 4)`.
    """
    boxes = jnp.asarray(boxes)
    center_x, center_y = boxes[:, 0:1], boxes[:, 1:2]
    W, H = boxes[:, 2:3], boxes[:, 3:4]
    x_min = center_x - (W / 2.0)
    x_max = center_x + (W / 2.0)
    y_min = center_y - (H / 2.0)
    y_max = center_y + (H / 2.0)
    return jnp.concatenate([x_min, y_min, x_max, y_max], axis=1)


def encode(matched, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.

    # Arguments
        matched: JAX array of shape `(num_priors, 4)` with boxes in
            point-form.
        priors: JAX array of shape `(num_priors, 4)` with boxes in
            center-form.
        variances: (list[float]) Variances of priorboxes

    # Returns
        encoded boxes: JAX array of shape `(num_priors, 4)`.
    """

    variances = jnp.array(variances)
    boxes = matched[:, :4]
    boxes = to_center_form(boxes)
    center_difference_x = boxes[:, 0:1] - priors[:, 0:1]
    encoded_center_x = center_difference_x / priors[:, 2:3]
    center_difference_y = boxes[:, 1:2] - priors[:, 1:2]
    encoded_center_y = center_difference_y / priors[:, 3:4]
    encoded_center_x = encoded_center_x / variances[0]
    encoded_center_y = encoded_center_y / variances[1]
    encoded_W = jnp.log((boxes[:, 2:3] / priors[:, 2:3]) + 1e-8)
    encoded_H = jnp.log((boxes[:, 3:4] / priors[:, 3:4]) + 1e-8)
    encoded_W = encoded_W / variances[2]
    encoded_H = encoded_H / variances[3]
    encoded_boxes = [encoded_center_x, encoded_center_y, encoded_W, encoded_H]
    return jnp.concatenate(encoded_boxes + [matched[:, 4:]], axis=1)


def decode(predictions, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    """Decode default boxes into the ground truth boxes

    # Arguments
        loc: JAX array of shape `(num_priors, 4)`.
        priors: JAX array of shape `(num_priors, 4)`.
        variances: List of two floats. Variances of prior boxes.

    # Returns
        decoded boxes: JAX array of shape `(num_priors, 4)`.
    """
    variances = jnp.array(variances)
    center_x = predictions[:, 0:1] * priors[:, 2:3] * variances[0]
    center_x = center_x + priors[:, 0:1]
    center_y = predictions[:, 1:2] * priors[:, 3:4] * variances[1]
    center_y = center_y + priors[:, 1:2]
    W = priors[:, 2:3] * jnp.exp(predictions[:, 2:3] * variances[2])
    H = priors[:, 3:4] * jnp.exp(predictions[:, 3:4] * variances[3])
    boxes = jnp.concatenate([center_x, center_y, W, H], axis=1)
    boxes = to_corner_form(boxes)
    return jnp.concatenate([boxes, predictions[:, 4:]], 1)


def compute_ious(boxes_A, boxes_B):
    """Calculates the intersection over union between `boxes_A` and `boxes_B`.
    For each box present in the rows of `boxes_A` it calculates
    the intersection over union with respect to all boxes in `boxes_B`.
    The variables `boxes_A` and `boxes_B` contain the corner coordinates
    of the left-top corner `(x_min, y_min)` and the right-bottom
    `(x_max, y_max)` corner.

    # Arguments
        boxes_A: JAX array with shape `(num_boxes_A, 4)`.
        boxes_B: JAX array with shape `(num_boxes_B, 4)`.

    # Returns
        JAX array of shape `(num_boxes_A, num_boxes_B)`.
    """
    xy_min = jnp.maximum(boxes_A[:, None, 0:2], boxes_B[:, 0:2])
    xy_max = jnp.minimum(boxes_A[:, None, 2:4], boxes_B[:, 2:4])
    intersection = jnp.maximum(0.0, xy_max - xy_min)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    areas_A = (boxes_A[:, 2] - boxes_A[:, 0]) * (boxes_A[:, 3] - boxes_A[:, 1])
    areas_B = (boxes_B[:, 2] - boxes_B[:, 0]) * (boxes_B[:, 3] - boxes_B[:, 1])
    union_area = (areas_A[:, jnp.newaxis] + areas_B) - intersection_area
    union_area = jnp.maximum(union_area, 1e-8)
    return jnp.clip(intersection_area / union_area, 0.0, 1.0)


def compute_max_matches(boxes, prior_boxes):
    iou_matrix = compute_ious(prior_boxes, boxes)
    per_prior_which_box_iou = jnp.max(iou_matrix, axis=1)
    per_prior_which_box_arg = jnp.argmax(iou_matrix, axis=1)
    return per_prior_which_box_iou, per_prior_which_box_arg


def get_matches_masks(boxes, prior_boxes, positive_iou=0.5, negative_iou=0.4):
    prior_boxes = to_corner_form(prior_boxes)
    max_matches = compute_max_matches(boxes, prior_boxes)
    per_prior_which_box_iou, per_prior_which_box_arg = max_matches

    positive_mask = jnp.greater_equal(per_prior_which_box_iou, positive_iou)
    negative_mask = jnp.less(per_prior_which_box_iou, negative_iou)
    not_ignoring_mask = jnp.logical_or(positive_mask, negative_mask)
    ignoring_mask = jnp.logical_not(not_ignoring_mask)

    return per_prior_which_box_arg, positive_mask, ignoring_mask


def mask_classes(boxes, positive_mask, ignoring_mask):
    class_indices = boxes[:, 4]
    negative_mask = jnp.not_equal(positive_mask, 1.0)
    class_indices = jnp.where(negative_mask, 0.0, class_indices)
    class_indices = jnp.expand_dims(class_indices, axis=-1)
    boxes = boxes.at[:, 4:5].set(class_indices)
    return boxes


def match(boxes, prior_boxes, positive_iou=0.5, negative_iou=0.0):
    """Matches each prior box with a ground truth box (box from `boxes`).
    It then selects which matched box will be considered positive e.g. iou > .5
    and returns for each prior box a ground truth box that is either positive
    (with a class argument different than 0) or negative.

    # Arguments
        boxes: JAX array of shape `(num_ground_truh_boxes, 4 + 1)`,
            where the first the first four coordinates correspond to
            box coordinates and the last coordinates is the class
            argument. This boxes should be the ground truth boxes.
        prior_boxes: JAX array of shape `(num_prior_boxes, 4)`.
            where the four coordinates are in center form coordinates.
        positive_iou: Float between [0, 1]. Intersection over union
            used to determine which box is considered a positive box.
        negative_iou: Float between [0, 1]. Intersection over union
            used to determine which box is considered a negative box.

    # Returns
        JAX array of shape `(num_prior_boxes, 4 + 1)`.
            where the first the first four coordinates correspond to point
            form box coordinates and the last coordinates is the class
            argument.
    """
    matches = get_matches_masks(boxes, prior_boxes, positive_iou, negative_iou)
    per_prior_box_which_box_arg, positive_mask, ignoring_mask = matches
    matched_boxes = jnp.take(boxes, per_prior_box_which_box_arg, axis=0)
    matched_boxes = mask_classes(matched_boxes, positive_mask, ignoring_mask)
    return matched_boxes


def _update_prior_box_mapping(i, curr_arg, per_box_which_prior_arg):
    """
    Update the mapping of the best matching prior box index for the current box.

    Args:
        i: int, the current index in the fori_loop.
        curr_arg: JAX array, stores the best matching prior box index for each box.
        per_box_which_prior_arg: JAX array, the index of the best prior box for each current box.

    Returns:
        Updated mapping of the best matching prior box index for each box.
    """
    best_prior_box_arg = per_box_which_prior_arg[i]
    curr_arg = curr_arg.at[best_prior_box_arg].set(i)
    return curr_arg


def match2(boxes, prior_boxes, iou_threshold=0.5):
    """
    Match the boxes with prior boxes based on IoU (Intersection over Union).

    Args:
        boxes: JAX array of current boxes.
        prior_boxes: JAX array of prior boxes.
        iou_threshold: float, IoU threshold to filter out non-matching boxes.

    Returns:
        JAX array with updated box mappings.
    """
    ious = compute_ious(boxes, to_corner_form(jnp.float32(prior_boxes)))
    per_prior_which_box_iou = jnp.max(ious, axis=0)
    per_prior_which_box_arg = jnp.argmax(ious, axis=0)
    per_box_which_prior_arg = jnp.argmax(ious, axis=1)

    # Set IoU for best matching prior for each box to a high value
    per_prior_which_box_iou = per_prior_which_box_iou.at[
        per_box_which_prior_arg
    ].set(2)

    # Iterate and update mapping using the updated function
    per_prior_which_box_arg = jax.lax.fori_loop(
        0,
        per_box_which_prior_arg.shape[0],
        lambda i, curr_arg: _update_prior_box_mapping(
            i, curr_arg, per_box_which_prior_arg
        ),
        per_prior_which_box_arg,
    )

    # Filter boxes based on IoU threshold
    matches = boxes[per_prior_which_box_arg]
    condition = per_prior_which_box_iou < iou_threshold
    indices = jnp.where(condition)[0]
    matches = matches.at[indices, 4].set(
        0
    )  # Set class column to 0 for non-matching boxes

    return matches


def compute_iou(box, boxes):
    """Calculates the intersection over union between 'box' and all 'boxes'.
    Both `box` and `boxes` are in corner coordinates.

    # Arguments
        box: JAX array with length at least of 4.
        boxes: JAX array with shape `(num_boxes, 4)`.

    # Returns
        JAX array of shape `(num_boxes, 1)`.
    """

    x_min_A, y_min_A, x_max_A, y_max_A = box[:4]
    x_min_B, y_min_B = boxes[:, 0], boxes[:, 1]
    x_max_B, y_max_B = boxes[:, 2], boxes[:, 3]
    # calculating the intersection
    inner_x_min = jnp.maximum(x_min_B, x_min_A)
    inner_y_min = jnp.maximum(y_min_B, y_min_A)
    inner_x_max = jnp.minimum(x_max_B, x_max_A)
    inner_y_max = jnp.minimum(y_max_B, y_max_A)
    inner_w = jnp.maximum((inner_x_max - inner_x_min), 0)
    inner_h = jnp.maximum((inner_y_max - inner_y_min), 0)
    intersection_area = inner_w * inner_h
    # calculating the union
    box_area_B = (x_max_B - x_min_B) * (y_max_B - y_min_B)
    box_area_A = (x_max_A - x_min_A) * (y_max_A - y_min_A)
    union_area = box_area_A + box_area_B - intersection_area
    intersection_over_union = intersection_area / union_area
    return intersection_over_union


def _nms_iteration_step(x_min, y_min, x_max, y_max, areas, iou_thresh):
    """Performs one iteration of NMS, suppressing overlapping boxes."""

    def step(state):
        i, curr_scores, sel_indices = state
        # Choose the index with the highest score
        best_idx = jnp.argmax(curr_scores).astype(jnp.int32)
        updated_indices = sel_indices.at[i].set(best_idx)

        # Calculate IoU with the best box
        best_xmin = x_min[best_idx]
        best_ymin = y_min[best_idx]
        best_xmax = x_max[best_idx]
        best_ymax = y_max[best_idx]

        inter_xmin = jnp.maximum(x_min, best_xmin)
        inter_ymin = jnp.maximum(y_min, best_ymin)
        inter_xmax = jnp.minimum(x_max, best_xmax)
        inter_ymax = jnp.minimum(y_max, best_ymax)

        inter_width = jnp.maximum(inter_xmax - inter_xmin, 0.0)
        inter_height = jnp.maximum(inter_ymax - inter_ymin, 0.0)
        intersection = inter_width * inter_height

        union = areas + areas[best_idx] - intersection
        iou = intersection / union

        # Suppress overlapping boxes and the selected box
        suppressed_scores = jnp.where(iou > iou_thresh, -jnp.inf, curr_scores)
        updated_scores = suppressed_scores.at[best_idx].set(-jnp.inf)

        return (i + 1, updated_scores, updated_indices)

    return step


def _nms_continuation_condition(top_k):
    """Determines whether to continue the NMS iterations."""

    def condition(state):
        i, curr_scores, _ = state
        continue_iteration = jnp.logical_and(
            i < top_k, jnp.max(curr_scores) > -jnp.inf
        )
        return continue_iteration

    return condition


def apply_non_max_suppression(boxes, scores, iou_thresh=0.45, top_k=200):
    """
    Applies Non-Maximum Suppression using fixed-iteration loop.

    Args:
        boxes: JAX array of shape (num_boxes, 4) [x_min, y_min, x_max, y_max]
        scores: JAX array of shape (num_boxes,) with confidence scores
        iou_thresh: IoU threshold for suppression
        top_k: Maximum number of boxes to select

    Returns:
        selected_indices: JAX array of selected box indices
        num_selected: Number of selected boxes
    """
    num_boxes = boxes.shape[0]
    if num_boxes == 0:
        return jnp.zeros(0, dtype=jnp.int32), 0

    # Precompute box geometry features
    x_min, y_min = boxes[:, 0], boxes[:, 1]
    x_max, y_max = boxes[:, 2], boxes[:, 3]
    areas = (x_max - x_min) * (y_max - y_min)

    # Initialize state
    selected_indices = jnp.zeros(top_k, dtype=jnp.int32)
    init_state = (0, scores, selected_indices)

    # Create configured helper functions
    step_fn = _nms_iteration_step(
        x_min, y_min, x_max, y_max, areas, iou_thresh
    )
    cond_fn = _nms_continuation_condition(top_k)

    # Run the fixed-iteration loop
    final_state = lax.while_loop(cond_fn, step_fn, init_state)
    num_selected, _, final_indices = final_state

    selected_indices = lax.dynamic_slice(
        final_indices, (0,), (jnp.minimum(num_selected, top_k),)
    )
    return selected_indices, num_selected


def update_output_for_class(
    class_arg,
    out,
    decoded_boxes,
    class_predictions,
    conf_thresh,
    nms_thresh,
    top_k,
):
    # Extract the relevant class confidence scores
    conf = class_predictions[:, class_arg]

    # Mask for confident detections
    conf_mask = conf >= conf_thresh

    # Extract boxes and scores based on the mask
    boxes = decoded_boxes[conf_mask]
    scores = conf[conf_mask]

    # Apply non-max suppression
    indices, count = apply_non_max_suppression(
        boxes, scores, nms_thresh, top_k
    )
    selected_indices = indices[:count]

    # Concatenate boxes and scores
    selections = jnp.concatenate(
        (boxes[selected_indices], scores[selected_indices, None]), axis=1
    )

    # Update the output array with the selected detections
    out = out.at[class_arg, :count, :].set(selections)

    return out


def nms_per_class(box_data, nms_thresh=0.45, conf_thresh=0.01, top_k=200):
    """Applies non-maximum-suppression per class.
    # Arguments
        box_data: JAX array of shape `(num_prior_boxes, 4 + num_classes)`.
        nms_thresh: Float. Non-maximum suppression threshold.
        conf_thresh: Float. Filter scores with a lower confidence value before
            performing non-maximum suppression.
        top_k: Integer. Maximum number of boxes per class outputted by NMS.

    Returns
        JAX array of shape `(num_classes, top_k, 5)`.
    """
    decoded_boxes, class_predictions = box_data[:, :4], box_data[:, 4:]
    num_classes = class_predictions.shape[1]
    output = jnp.zeros((num_classes, top_k, 5))

    # skip the background class (start counter in 1)
    for class_arg in range(1, num_classes):
        conf_mask = class_predictions[:, class_arg] >= conf_thresh
        scores = class_predictions[:, class_arg][conf_mask]
        if len(scores) == 0:
            continue
        boxes = decoded_boxes[conf_mask]
        indices, count = apply_non_max_suppression(
            boxes, scores, nms_thresh, top_k
        )
        scores = jnp.expand_dims(scores, -1)
        selected_indices = indices[:count]
        selections = jnp.concatenate(
            (boxes[selected_indices], scores[selected_indices]), axis=1
        )
        output = output.at[class_arg, :count, :].set(selections)
    return output


def to_one_hot(class_indices, num_classes):
    """Transform from class index to one-hot encoded vector.

    # Arguments
        class_indices: JAX array. One dimensional array specifying
            the index argument of the class for each sample.
        num_classes: Integer. Total number of classes.

    # Returns
        JAX array with shape `(num_samples, num_classes)`.
    """
    one_hot_vectors = jnp.zeros((len(class_indices), num_classes))
    indices = jnp.arange(len(class_indices)), class_indices
    one_hot_vectors = one_hot_vectors.at[indices].set(1.0)
    return one_hot_vectors


def make_box_square(box):
    """Makes box coordinates square with sides equal to the longest original side.

    # Arguments
        box: JAX array with shape `(4)` with point corner coordinates.

    # Returns
        returns: List of box coordinates ints.
    """
    # TODO add ``calculate_center`` ``calculate_side_dimensions`` functions.
    # Extract coordinates from the box
    x_min, y_min, x_max, y_max = box[:4]

    # Calculate the center and dimensions of the box
    center_x = (x_max + x_min) / 2.0
    center_y = (y_max + y_min) / 2.0

    width = x_max - x_min
    height = y_max - y_min

    half_box = height / 2.0

    x_min = jnp.where(height >= width, jnp.int32(center_x - half_box), x_min)
    x_max = jnp.where(height >= width, jnp.int32(center_x + half_box), x_max)

    y_min = jnp.where(width > height, jnp.int32(center_y - width / 2.0), y_min)
    y_max = jnp.where(width > height, jnp.int32(center_y + width / 2.0), y_max)

    return x_min, y_min, x_max, y_max


def offset(coordinates, offset_scales):
    """Apply offsets to box coordinates

    # Arguments
        coordinates: List of floats containing coordinates in point form.
        offset_scales: List of floats having x and y scales respectively.

    # Returns
        coordinates: List of floats containing coordinates in point form.
            i.e. [x_min, y_min, x_max, y_max].
    """
    x_min, y_min, x_max, y_max = coordinates
    x_offset_scale, y_offset_scale = offset_scales
    x_offset = (x_max - x_min) * x_offset_scale
    y_offset = (y_max - y_min) * y_offset_scale
    x_min = jnp.int32(x_min - x_offset)
    y_max = jnp.int32(y_max + x_offset)
    y_min = jnp.int32(y_min - y_offset)
    x_max = jnp.int32(x_max + y_offset)
    return (x_min, y_min, x_max, y_max)


def clip(coordinates, image_shape):
    """Clip box to valid image coordinates
    # Arguments
        coordinates: List of floats containing coordinates in point form
            i.e. [x_min, y_min, x_max, y_max].
        image_shape: List of two integers indicating height and width of image
            respectively.

    # Returns
        List of clipped coordinates.
    """
    height, width = image_shape[:2]
    x_min, y_min, x_max, y_max = coordinates

    x_min = jnp.maximum(x_min, 0)
    y_min = jnp.maximum(y_min, 0)
    x_max = jnp.minimum(x_max, width)
    y_max = jnp.minimum(y_max, height)

    return x_min, y_min, x_max, y_max


def denormalize_box(box, image_shape):
    """Scales corner box coordinates from normalized values to image dimensions.

    # Arguments
        box: JAX array containing corner box coordinates.
        image_shape: List of integers with (height, width).

    # Returns
        returns: box corner coordinates in image dimensions
    """
    x_min, y_min, x_max, y_max = box[:4]
    height, width = image_shape
    x_min = jnp.int32(x_min * width)
    y_min = jnp.int32(y_min * height)
    x_max = jnp.int32(x_max * width)
    y_max = jnp.int32(y_max * height)
    return (x_min, y_min, x_max, y_max)


def flip_left_right(boxes, width):
    """Flips box coordinates from left-to-right and vice-versa.
    # Arguments
        boxes: JAX array of shape `[num_boxes, 4]`.
    # Returns
        JAX array of shape `[num_boxes, 4]`.
    """
    new_x_min = width - boxes[:, 2]  # Flip x_max to x_min
    new_x_max = width - boxes[:, 0]  # Flip x_min to x_max
    flipped_boxes = jnp.stack(
        [
            new_x_min,  # New x_min
            boxes[:, 1],  # Original y_min
            new_x_max,  # New x_max
            boxes[:, 3],  # Original y_max
        ],
        axis=1,
    )
    return flipped_boxes


def to_image_coordinates(boxes, image):
    """Transforms normalized box coordinates into image coordinates.
    # Arguments
        image: JAX array.
        boxes: JAX array of shape `[num_boxes, N]` where N >= 4.
    # Returns
        JAX array of shape `[num_boxes, N]`.
    """
    height, width = image.shape[:2]
    image_boxes = jnp.array(boxes)

    image_boxes = image_boxes.at[:, 0].set(boxes[:, 0] * width)
    image_boxes = image_boxes.at[:, 2].set(boxes[:, 2] * width)
    image_boxes = image_boxes.at[:, 1].set(boxes[:, 1] * height)
    image_boxes = image_boxes.at[:, 3].set(boxes[:, 3] * height)

    return image_boxes


def to_normalized_coordinates(boxes, image):
    """Transforms coordinates in image dimensions to normalized coordinates.
    # Arguments
        image: JAX array.
        boxes: JAX array of shape `[num_boxes, N]` where N >= 4.
    # Returns
        JAX array of shape `[num_boxes, N]`.
    """
    height, width = image.shape[:2]
    normalized_boxes = jnp.array(boxes)

    normalized_boxes = normalized_boxes.at[:, 0].set(boxes[:, 0] / width)
    normalized_boxes = normalized_boxes.at[:, 2].set(boxes[:, 2] / width)
    normalized_boxes = normalized_boxes.at[:, 1].set(boxes[:, 1] / height)
    normalized_boxes = normalized_boxes.at[:, 3].set(boxes[:, 3] / height)

    return normalized_boxes
