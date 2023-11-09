import numpy as np


def to_center_form(boxes):
    """Transform from corner coordinates to center coordinates.

    # Arguments
        boxes: Numpy array with shape `(num_boxes, 4)`.

    # Returns
        Numpy array with shape `(num_boxes, 4)`.
    """
    x_min, y_min = boxes[:, 0:1], boxes[:, 1:2]
    x_max, y_max = boxes[:, 2:3], boxes[:, 3:4]
    center_x = (x_max + x_min) / 2.0
    center_y = (y_max + y_min) / 2.0
    W = x_max - x_min
    H = y_max - y_min
    return np.concatenate([center_x, center_y, W, H], axis=1)


def to_corner_form(boxes):
    """Transform from center coordinates to corner coordinates.

    # Arguments
        boxes: Numpy array with shape `(num_boxes, 4)`.

    # Returns
        Numpy array with shape `(num_boxes, 4)`.
    """
    center_x, center_y = boxes[:, 0:1], boxes[:, 1:2]
    W, H = boxes[:, 2:3], boxes[:, 3:4]
    x_min = center_x - (W / 2.0)
    x_max = center_x + (W / 2.0)
    y_min = center_y - (H / 2.0)
    y_max = center_y + (H / 2.0)
    return np.concatenate([x_min, y_min, x_max, y_max], axis=1)


def encode(matched, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    """Encode the variances from the priorbox layers into the ground truth
    boxes we have matched (based on jaccard overlap) with the prior boxes.

    # Arguments
        matched: Numpy array of shape `(num_priors, 4)` with boxes in
            point-form.
        priors: Numpy array of shape `(num_priors, 4)` with boxes in
            center-form.
        variances: (list[float]) Variances of priorboxes

    # Returns
        encoded boxes: Numpy array of shape `(num_priors, 4)`.
    """
    boxes = matched[:, :4]
    boxes = to_center_form(boxes)
    center_difference_x = boxes[:, 0:1] - priors[:, 0:1]
    encoded_center_x = center_difference_x / priors[:, 2:3]
    center_difference_y = boxes[:, 1:2] - priors[:, 1:2]
    encoded_center_y = center_difference_y / priors[:, 3:4]
    encoded_center_x = encoded_center_x / variances[0]
    encoded_center_y = encoded_center_y / variances[1]
    encoded_W = np.log((boxes[:, 2:3] / priors[:, 2:3]) + 1e-8)
    encoded_H = np.log((boxes[:, 3:4] / priors[:, 3:4]) + 1e-8)
    encoded_W = encoded_W / variances[2]
    encoded_H = encoded_H / variances[3]
    encoded_boxes = [encoded_center_x, encoded_center_y, encoded_W, encoded_H]
    return np.concatenate(encoded_boxes + [matched[:, 4:]], axis=1)


def decode(predictions, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    """Decode default boxes into the ground truth boxes

    # Arguments
        loc: Numpy array of shape `(num_priors, 4)`.
        priors: Numpy array of shape `(num_priors, 4)`.
        variances: List of two floats. Variances of prior boxes.

    # Returns
        decoded boxes: Numpy array of shape `(num_priors, 4)`.
    """
    center_x = predictions[:, 0:1] * priors[:, 2:3] * variances[0]
    center_x = center_x + priors[:, 0:1]
    center_y = predictions[:, 1:2] * priors[:, 3:4] * variances[1]
    center_y = center_y + priors[:, 1:2]
    W = priors[:, 2:3] * np.exp(predictions[:, 2:3] * variances[2])
    H = priors[:, 3:4] * np.exp(predictions[:, 3:4] * variances[3])
    boxes = np.concatenate([center_x, center_y, W, H], axis=1)
    boxes = to_corner_form(boxes)
    return np.concatenate([boxes, predictions[:, 4:]], 1)


def compute_ious(boxes_A, boxes_B):
    """Calculates the intersection over union between `boxes_A` and `boxes_B`.
    For each box present in the rows of `boxes_A` it calculates
    the intersection over union with respect to all boxes in `boxes_B`.
    The variables `boxes_A` and `boxes_B` contain the corner coordinates
    of the left-top corner `(x_min, y_min)` and the right-bottom
    `(x_max, y_max)` corner.

    # Arguments
        boxes_A: Numpy array with shape `(num_boxes_A, 4)`.
        boxes_B: Numpy array with shape `(num_boxes_B, 4)`.

    # Returns
        Numpy array of shape `(num_boxes_A, num_boxes_B)`.
    """
    xy_min = np.maximum(boxes_A[:, None, 0:2], boxes_B[:, 0:2])
    xy_max = np.minimum(boxes_A[:, None, 2:4], boxes_B[:, 2:4])
    intersection = np.maximum(0.0, xy_max - xy_min)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    areas_A = (boxes_A[:, 2] - boxes_A[:, 0]) * (boxes_A[:, 3] - boxes_A[:, 1])
    areas_B = (boxes_B[:, 2] - boxes_B[:, 0]) * (boxes_B[:, 3] - boxes_B[:, 1])
    # broadcasting for outer sum i.e. a sum of all possible combinations
    union_area = (areas_A[:, np.newaxis] + areas_B) - intersection_area
    union_area = np.maximum(union_area, 1e-8)
    return np.clip(intersection_area / union_area, 0.0, 1.0)


def compute_max_matches(boxes, prior_boxes):
    iou_matrix = compute_ious(prior_boxes, boxes)
    per_prior_which_box_iou = np.max(iou_matrix, axis=1)
    per_prior_which_box_arg = np.argmax(iou_matrix, axis=1)
    return per_prior_which_box_iou, per_prior_which_box_arg


def get_matches_masks(boxes, prior_boxes, positive_iou=0.5, negative_iou=0.4):
    prior_boxes = to_corner_form(prior_boxes)
    max_matches = compute_max_matches(boxes, prior_boxes)
    per_prior_which_box_iou, per_prior_which_box_arg = max_matches
    positive_mask = np.greater_equal(per_prior_which_box_iou, positive_iou)
    negative_mask = np.less(per_prior_which_box_iou, negative_iou)
    not_ignoring_mask = np.logical_or(positive_mask, negative_mask)
    # ignoring mask are all masks not positive or negative
    ignoring_mask = np.logical_not(not_ignoring_mask)
    return per_prior_which_box_arg, positive_mask, ignoring_mask


def mask_classes(boxes, positive_mask, ignoring_mask):
    class_indices = boxes[:, 4]
    negative_mask = np.not_equal(positive_mask, 1.0)
    class_indices = np.where(negative_mask, 0.0, class_indices)
    # ignoring_mask = np.equal(ignoring_mask, 1.0)
    # class_indices = np.where(ignoring_mask, -1.0, class_indices)
    class_indices = np.expand_dims(class_indices, axis=-1)
    boxes[:, 4:5] = class_indices
    return boxes


def match_beta(boxes, prior_boxes, positive_iou=0.5, negative_iou=0.0):
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
        positive_iou: Float between [0, 1]. Intersection over union
            used to determine which box is considered a positive box.
        negative_iou: Float between [0, 1]. Intersection over union
            used to determine which box is considered a negative box.

    # Returns
        numpy array of shape `(num_prior_boxes, 4 + 1)`.
            where the first the first four coordinates correspond to point
            form box coordinates and the last coordinates is the class
            argument.
    """
    matches = get_matches_masks(boxes, prior_boxes, positive_iou, negative_iou)
    per_prior_box_which_box_arg, positive_mask, ignoring_mask = matches
    matched_boxes = np.take(boxes, per_prior_box_which_box_arg, axis=0)
    matched_boxes = mask_classes(matched_boxes, positive_mask, ignoring_mask)
    return matched_boxes


def match(boxes, prior_boxes, iou_threshold=0.5):
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
        numpy array of shape `(num_prior_boxes, 4 + 1)`.
            where the first the first four coordinates correspond to point
            form box coordinates and the last coordinates is the class
            argument.
    """
    ious = compute_ious(boxes, to_corner_form(np.float32(prior_boxes)))
    per_prior_which_box_iou = np.max(ious, axis=0)
    per_prior_which_box_arg = np.argmax(ious, 0)

    #  overwriting per_prior_which_box_arg if they are the best prior box
    per_box_which_prior_arg = np.argmax(ious, 1)
    per_prior_which_box_iou[per_box_which_prior_arg] = 2
    for box_arg in range(len(per_box_which_prior_arg)):
        best_prior_box_arg = per_box_which_prior_arg[box_arg]
        per_prior_which_box_arg[best_prior_box_arg] = box_arg

    matches = boxes[per_prior_which_box_arg]
    matches[per_prior_which_box_iou < iou_threshold, 4] = 0
    return matches


def compute_iou(box, boxes):
    """Calculates the intersection over union between 'box' and all 'boxes'.
    Both `box` and `boxes` are in corner coordinates.

    # Arguments
        box: Numpy array with length at least of 4.
        boxes: Numpy array with shape `(num_boxes, 4)`.

    # Returns
        Numpy array of shape `(num_boxes, 1)`.
    """

    x_min_A, y_min_A, x_max_A, y_max_A = box[:4]
    x_min_B, y_min_B = boxes[:, 0], boxes[:, 1]
    x_max_B, y_max_B = boxes[:, 2], boxes[:, 3]
    # calculating the intersection
    inner_x_min = np.maximum(x_min_B, x_min_A)
    inner_y_min = np.maximum(y_min_B, y_min_A)
    inner_x_max = np.minimum(x_max_B, x_max_A)
    inner_y_max = np.minimum(y_max_B, y_max_A)
    inner_w = np.maximum((inner_x_max - inner_x_min), 0)
    inner_h = np.maximum((inner_y_max - inner_y_min), 0)
    intersection_area = inner_w * inner_h
    # calculating the union
    box_area_B = (x_max_B - x_min_B) * (y_max_B - y_min_B)
    box_area_A = (x_max_A - x_min_A) * (y_max_A - y_min_A)
    union_area = box_area_A + box_area_B - intersection_area
    intersection_over_union = intersection_area / union_area
    return intersection_over_union


def apply_non_max_suppression(boxes, scores, iou_thresh=.45, top_k=200):
    """Apply non maximum suppression.

    # Arguments
        boxes: Numpy array, box coordinates of shape `(num_boxes, 4)`
            where each columns corresponds to x_min, y_min, x_max, y_max.
        scores: Numpy array, of scores given for each box in `boxes`.
        iou_thresh: float, intersection over union threshold for removing
            boxes.
        top_k: int, number of maximum objects per class.

    # Returns
        selected_indices: Numpy array, selected indices of kept boxes.
        num_selected_boxes: int, number of selected boxes.
    """

    selected_indices = np.zeros(shape=len(scores))
    if boxes is None or len(boxes) == 0:
        return selected_indices
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2]
    y_max = boxes[:, 3]
    areas = (x_max - x_min) * (y_max - y_min)
    remaining_sorted_box_indices = np.argsort(scores)
    remaining_sorted_box_indices = remaining_sorted_box_indices[-top_k:]

    num_selected_boxes = 0
    while len(remaining_sorted_box_indices) > 0:
        best_score_args = remaining_sorted_box_indices[-1]
        selected_indices[num_selected_boxes] = best_score_args
        num_selected_boxes = num_selected_boxes + 1
        if len(remaining_sorted_box_indices) == 1:
            break

        remaining_sorted_box_indices = remaining_sorted_box_indices[:-1]

        best_x_min = x_min[best_score_args]
        best_y_min = y_min[best_score_args]
        best_x_max = x_max[best_score_args]
        best_y_max = y_max[best_score_args]

        remaining_x_min = x_min[remaining_sorted_box_indices]
        remaining_y_min = y_min[remaining_sorted_box_indices]
        remaining_x_max = x_max[remaining_sorted_box_indices]
        remaining_y_max = y_max[remaining_sorted_box_indices]

        inner_x_min = np.maximum(remaining_x_min, best_x_min)
        inner_y_min = np.maximum(remaining_y_min, best_y_min)
        inner_x_max = np.minimum(remaining_x_max, best_x_max)
        inner_y_max = np.minimum(remaining_y_max, best_y_max)

        inner_box_widths = inner_x_max - inner_x_min
        inner_box_heights = inner_y_max - inner_y_min

        inner_box_widths = np.maximum(inner_box_widths, 0.0)
        inner_box_heights = np.maximum(inner_box_heights, 0.0)

        intersections = inner_box_widths * inner_box_heights
        remaining_box_areas = areas[remaining_sorted_box_indices]
        best_area = areas[best_score_args]
        unions = remaining_box_areas + best_area - intersections
        intersec_over_union = intersections / unions
        intersec_over_union_mask = intersec_over_union <= iou_thresh
        remaining_sorted_box_indices = remaining_sorted_box_indices[
            intersec_over_union_mask]

    return selected_indices.astype(int), num_selected_boxes


def nms_per_class(box_data, nms_thresh=.45, epsilon=0.01, top_k=200):
    """Applies non maximum suppression per class.
    This function takes all the detections from the detector which
    consists of boxes and their corresponding class scores to which it
    applies non maximum suppression for every class independently and
    then combines the result.

    # Arguments
        box_data: Array of shape `(num_nms_boxes, 4 + num_classes)`
            containing the box coordinates as well as the predicted
            scores of all the classes for all non suppressed boxes.
        nms_thresh: Float, Non-maximum suppression threshold.
        epsilon: Float, Filter scores with a lower confidence
            value before performing non-maximum supression.
        top_k: Int, Maximum number of boxes per class outputted by nms.

    # Returns
        Tuple: Containing an array non suppressed boxes of shape
            `(num_nms_boxes, 4 + num_classes)` and an array
            of corresponding class labels of shape `(num_nms_boxes, )`.
    """
    decoded_boxes = box_data[:, :4]
    class_predictions = box_data[:, 4:]
    num_classes = class_predictions.shape[1]
    nms_boxes = np.array([], dtype=float).reshape(0, box_data.shape[1])
    class_labels = np.array([], dtype=int)
    args = (decoded_boxes, class_predictions, epsilon, nms_thresh, top_k)
    for class_arg in range(num_classes):
        nms_boxes, class_labels = _nms_per_class(
            nms_boxes, class_labels, class_arg, *args)
    return nms_boxes, class_labels


def _nms_per_class(nms_boxes, class_labels, class_arg, decoded_boxes,
                   class_predictions, epsilon, nms_thresh, top_k):
    """Applies non maximum suppression for a given class.
    This function takes all the detections that belong only to the given
    single class and applies non maximum suppression for that class
    alone and returns the resulting non suppressed boxes.

    # Arguments
        nms_boxes: Array of shape `(num_boxes, 4 + num_classes)`.
        class_labels: Array of shape `(num_boxes, )`.
        class_arg: Int, class index.
        decoded_boxes: Array of shape `(num_prior_boxes, 4)`
            containing the box coordinates of all the
            non suppressed boxes.
        class_predictions: Array of shape
            `(num_nms_boxes, num_classes)` containing the predicted
            scores of all the classes for all the non suppressed boxes.
        epsilon: Float, Filter scores with a lower confidence
            value before performing non-maximum supression.
        nms_thresh: Float, Non-maximum suppression threshold.
        top_k: Int, Maximum number of boxes per class outputted by nms.

    # Returns
        Tuple: Containing an array non suppressed boxes per class of
            shape `(num_nms_boxes_per_class, 4 + num_classes) and an
            array corresponding class labels of shape
            `(num_nms_boxes_per_class, )`.
    """
    scores, mask = pre_filter_nms(class_arg, class_predictions, epsilon)

    if len(scores) != 0:
        boxes = decoded_boxes[mask]
        selected = apply_non_max_suppression(boxes, scores, nms_thresh, top_k)
        indices, count = selected
        selected_indices = indices[:count]
        selected_boxes = boxes[selected_indices]
        selected_classes = class_predictions[mask][selected_indices]
        selections = np.concatenate((selected_boxes, selected_classes), axis=1)
        nms_boxes = np.concatenate((nms_boxes, selections), axis=0)
        class_label = np.repeat(class_arg, count)
        class_labels = np.append(class_labels, class_label)
    return nms_boxes, class_labels


def pre_filter_nms(class_arg, class_predictions, epsilon):
    """Applies score filtering.
    This function takes all the predicted scores of a given class and
    filters out all the predictions less than the given `epsilon` value.

    # Arguments
        class_arg: Int, class index.
        class_predictions: Array of shape
            `(num_nms_boxes, num_classes)` containing the predicted
            scores of all the classes for all the non suppressed boxes.
        epsilon: Float, threshold value for score filtering.

    # Returns
        Tuple: Containing an array filtered scores of shape
            `(num_pre_filtered_boxes, )` and an array filter mask of
            shape `(num_prior_boxes, )`.
    """
    mask = class_predictions[:, class_arg] >= epsilon
    scores = class_predictions[:, class_arg][mask]
    return scores, mask


def merge_nms_box_with_class(box_data, class_labels):
    """Merges box coordinates with their corresponding class
    defined by `class_labels` which is decided by best box geometry
    by non maximum suppression (and not by the best scoring class)
    into a single output.
    This function retains only the predicted score of the class to
    which the box belongs to and sets the scores of all the remaining
    classes to zero, thereby combining box and class information in a
    single variable.

    # Arguments
        box_data: Array of shape `(num_nms_boxes, 4 + num_classes)`
            containing the box coordinates as well as the predicted
            scores of all the classes for all non suppressed boxes.
        class_labels: Array of shape `(num_nms_boxes, )` that contains
            the indices of the class whose score is to be retained.

    # Returns
        boxes: Array of shape `(num_nms_boxes, 4 + num_classes)`,
            containing coordinates of non supressed boxes along with
            scores of the class to which the box belongs. The scores of
            the other classes are zeros.
    """
    decoded_boxes = box_data[:, :4]
    class_predictions = box_data[:, 4:]
    retained_class_score = suppress_other_class_scores(
        class_predictions, class_labels)
    box_data = np.concatenate((decoded_boxes, retained_class_score), axis=1)
    return box_data


def suppress_other_class_scores(class_predictions, class_labels):
    """Retains the score of class in `class_labels` and
    sets other class scores to zero.

    # Arguments
        class_predictions: Array of shape
            `(num_nms_boxes, num_classes)` containing the predicted
            scores of all the classes for all the non suppressed boxes.
        class_labels: Array of shape `(num_nms_boxes, )` that contains
            the indices of the class whose score is to be retained.

    # Returns
        retained_class_score: Array of shape
            `(num_nms_boxes, num_classes)` that consists of score at
            only those location specified by 'class_labels' and zero
            at other class locations.

    # Note
        This approach retains the scores of that class in
        `class_predictions` defined by `class_labels` by generating
        a boolean mask `score_suppress_mask` with elements True at the
        locations where the score in `class_predictions` is to be
        retained and False wherever the class score is to be suppressed.
        This approach of retaining/suppressing scores does not make use
        of for loop, if-else condition and direct value assignment
        to arrays.
    """
    num_nms_boxes, num_classes = class_predictions.shape
    class_indices = np.arange(num_classes)
    class_indices = np.expand_dims(class_indices, axis=0)
    class_indices = np.repeat(class_indices, num_nms_boxes, axis=0)
    class_labels = np.expand_dims(class_labels, axis=1)
    class_labels = np.repeat(class_labels, num_classes, axis=1)
    """
    The difference of class_indices and class_labels contains zero
    at those locations of the result where the score is to be retained
    whose boolean value is False while others being True. This
    difference obtained as a boolean array gives a negative mask which
    when inverted gives the score_suppress_mask.
    """
    negative_mask = np.array(class_indices - class_labels, dtype=bool)
    score_suppress_mask = np.logical_not(negative_mask)
    retained_class_score = np.multiply(class_predictions, score_suppress_mask)
    return retained_class_score


def to_one_hot(class_indices, num_classes):
    """ Transform from class index to one-hot encoded vector.

    # Arguments
        class_indices: Numpy array. One dimensional array specifying
            the index argument of the class for each sample.
        num_classes: Integer. Total number of classes.

    # Returns
        Numpy array with shape `(num_samples, num_classes)`.
    """
    one_hot_vectors = np.zeros((len(class_indices), num_classes))
    for vector_arg, class_args in enumerate(class_indices):
        one_hot_vectors[vector_arg, class_args] = 1.0
    return one_hot_vectors


def make_box_square(box):
    """Makes box coordinates square with sides equal to the longest
        original side.

    # Arguments
        box: Numpy array with shape `(4)` with point corner coordinates.

    # Returns
        returns: List of box coordinates ints.
    """
    # TODO add ``calculate_center`` ``calculate_side_dimensions`` functions.
    x_min, y_min, x_max, y_max = box[:4]
    center_x = (x_max + x_min) / 2.0
    center_y = (y_max + y_min) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    if height >= width:
        half_box = height / 2.0
        x_min = int(center_x - half_box)
        x_max = int(center_x + half_box)

    if width > height:
        half_box = width / 2.0
        y_min = int(center_y - half_box)
        y_max = int(center_y + half_box)

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
    x_min = int(x_min - x_offset)
    y_max = int(y_max + x_offset)
    y_min = int(y_min - y_offset)
    x_max = int(x_max + y_offset)
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
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max > width:
        x_max = width
    if y_max > height:
        y_max = height
    return x_min, y_min, x_max, y_max


def denormalize_box(box, image_shape):
    """Scales corner box coordinates from normalized values to image dimensions

    # Arguments
        box: Numpy array containing corner box coordinates.
        image_shape: List of integers with (height, width).

    # Returns
        returns: box corner coordinates in image dimensions
    """
    x_min, y_min, x_max, y_max = box[:4]
    height, width = image_shape
    x_min = int(x_min * width)
    y_min = int(y_min * height)
    x_max = int(x_max * width)
    y_max = int(y_max * height)
    return (x_min, y_min, x_max, y_max)


def flip_left_right(boxes, width):
    """Flips box coordinates from left-to-right and vice-versa.
    # Arguments
        boxes: Numpy array of shape `[num_boxes, 4]`.
    # Returns
        Numpy array of shape `[num_boxes, 4]`.
    """
    flipped_boxes = boxes.copy()
    flipped_boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    return flipped_boxes


def to_image_coordinates(boxes, image):
    """Transforms normalized box coordinates into image coordinates.
    # Arguments
        image: Numpy array.
        boxes: Numpy array of shape `[num_boxes, N]` where N >= 4.
    # Returns
        Numpy array of shape `[num_boxes, N]`.
    """
    height, width = image.shape[:2]
    image_boxes = boxes.copy()
    image_boxes[:, 0] = boxes[:, 0] * width
    image_boxes[:, 2] = boxes[:, 2] * width
    image_boxes[:, 1] = boxes[:, 1] * height
    image_boxes[:, 3] = boxes[:, 3] * height
    return image_boxes


def to_normalized_coordinates(boxes, image):
    """Transforms coordinates in image dimensions to normalized coordinates.
    # Arguments
        image: Numpy array.
        boxes: Numpy array of shape `[num_boxes, N]` where N >= 4.
    # Returns
        Numpy array of shape `[num_boxes, N]`.
    """
    height, width = image.shape[:2]
    normalized_boxes = boxes.copy()
    normalized_boxes[:, 0] = boxes[:, 0] / width
    normalized_boxes[:, 2] = boxes[:, 2] / width
    normalized_boxes[:, 1] = boxes[:, 1] / height
    normalized_boxes[:, 3] = boxes[:, 3] / height
    return normalized_boxes


def extract_bounding_box_corners(points3D):
    """Extracts the (x_min, y_min, z_min) and the (x_max, y_max, z_max)
        coordinates from an array of  points3D
    # Arguments
        points3D: Array (num_points, 3)

    # Returns
        Left-down-bottom corner (x_min, y_min, z_min) and right-up-top
            (x_max, y_max, z_max) corner.
    """
    XYZ_min = np.min(points3D, axis=0)
    XYZ_max = np.max(points3D, axis=0)
    return XYZ_min, XYZ_max


def filter_boxes(boxes, conf_thresh):
    """Filters given boxes based on scores.

    # Arguments
        boxes: Array of shape `(num_nms_boxes, 4 + num_classes)`.
        conf_thresh: Float, Filter boxes with a confidence value
            lower than this.

    # Returns
        confident_boxes: Array of shape
            `(num_filtered_boxes, 4 + num_classes)`.
    """
    class_predictions = boxes[:, 4:]
    class_scores = np.max(class_predictions, axis=1)
    confidence_mask = class_scores >= conf_thresh
    confident_boxes = boxes[confidence_mask]
    return confident_boxes


def scale_box(predictions, image_scales):
    """
    # Arguments
        predictions: Array of shape `(num_boxes, num_classes+N)`
            model predictions.
        image_scales: Array of shape `()`, scale value of boxes.

    # Returns
        predictions: Array of shape `(num_boxes, num_classes+N)`
            model predictions.
    """
    boxes = predictions[:, :4]
    scales = image_scales[np.newaxis][np.newaxis]
    boxes = boxes * scales
    predictions = np.concatenate([boxes, predictions[:, 4:]], 1)
    return predictions


def change_box_coordinates(outputs):
    """Converts box coordinates format from (y_min, x_min, y_max, x_max)
    to (x_min, y_min, x_max, y_max).

    # Arguments
        outputs: Tensor, model output.

    # Returns
        outputs: Array, Processed outputs by merging the features
            at all levels. Each row corresponds to box coordinate
            offsets and sigmoid of the class logits.
    """
    outputs = outputs[0]
    boxes, classes = outputs[:, :4], outputs[:, 4:]
    s1, s2, s3, s4 = np.hsplit(boxes, 4)
    boxes = np.concatenate([s2, s1, s4, s3], axis=1)
    boxes = boxes[np.newaxis]
    classes = classes[np.newaxis]
    outputs = np.concatenate([boxes, classes], axis=2)
    return outputs
