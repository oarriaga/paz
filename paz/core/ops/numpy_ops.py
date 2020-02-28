from __future__ import division

import numpy as np

from .opencv_ops import *
from ..messages import Box2D


def compute_iou(box, boxes):
    """Calculates the intersection over union between 'box' and all 'boxes'

    The variables 'box' and 'boxes' contain the corner coordinates
    of the left-top corner (x_min, y_min) and the right-bottom (x_max, y_max)
    corner.

    # Arguments
        box: Numpy array with length at least of 4.
        box_B: Numpy array with shape (num_boxes, 4)

    # Returns
        Numpy array of shape (num_boxes, 1)
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


def compute_ious(boxes_A, boxes_B):
    """Calculates the intersection over union between 'boxes_A' and 'boxes_B'

    For each box present in the rows of 'boxes_A' it calculates
    the intersection over union with respect to all boxes in 'boxes_B'.

    The variables 'boxes_A' and 'boxes_B' contain the corner coordinates
    of the left-top corner (x_min, y_min) and the right-bottom (x_max, y_max)
    corner.

    # Arguments
        boxes_A: Numpy array with shape (num_boxes_A, 4)
        boxes_B: Numpy array with shape (num_boxes_B, 4)

    # Returns
        Numpy array of shape (num_boxes_A, num_boxes_B)
    """
    return np.apply_along_axis(compute_iou, 1, boxes_A, boxes_B)


def to_point_form(boxes):
    """Transform from center coordinates to corner coordinates.

    # Arguments
        boxes: Numpy array with shape (num_boxes, 4)

    # Returns
        Numpy array with shape (num_boxes, 4).
    """
    center_x, center_y = boxes[:, 0], boxes[:, 1]
    width, height = boxes[:, 2], boxes[:, 3]
    x_min = center_x - (width / 2.)
    x_max = center_x + (width / 2.)
    y_min = center_y - (height / 2.)
    y_max = center_y + (height / 2.)
    return np.concatenate([x_min[:, None], y_min[:, None],
                           x_max[:, None], y_max[:, None]], axis=1)


def to_center_form(boxes):
    """Transform from corner coordinates to center coordinates.

    # Arguments
        boxes: Numpy array with shape (num_boxes, 4)

    # Returns
        Numpy array with shape (num_boxes, 4).
    """
    x_min, y_min = boxes[:, 0], boxes[:, 1]
    x_max, y_max = boxes[:, 2], boxes[:, 3]
    center_x = (x_max + x_min) / 2.
    center_y = (y_max + y_min) / 2.
    width = x_max - x_min
    height = y_max - y_min
    return np.concatenate([center_x[:, None], center_y[:, None],
                           width[:, None], height[:, None]], axis=1)


def denormalize_box(box, image_shape):
    """Scales corner box coordinates from normalized values to image dimensions.

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


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.

    # Arguments
        matched: Numpy array of shape (num_priors, 4) with boxes in point-form
        priors: Numpy array of shape (num_priors, 4) with boxes in center-form
        variances: (list[float]) Variances of priorboxes

    # Returns
        encoded boxes: Numpy array of shape (num_priors, 4)
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:4]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:4])
    # match wh / prior wh
    g_wh = (matched[:, 2:4] - matched[:, :2]) / priors[:, 2:4]
    g_wh = np.log(np.abs(g_wh) + 1e-4) / variances[1]
    # return target for smooth_l1_loss
    return np.concatenate([g_cxcy, g_wh, matched[:, 4:]], 1)  # [num_priors,4]


def decode(predictions, priors, variances):
    """Decode default boxes into the ground truth boxes

    # Arguments
        loc: Numpy array of shape (num_priors, 4)
        priors: Numpy array of shape (num_priors, 4)
        variances: (list[float]) Variances of priorboxes

    # Returns
        decoded boxes: Numpy array of shape (num_priors, 4)
    """

    boxes = np.concatenate((
        priors[:, :2] + predictions[:, :2] * variances[0] * priors[:, 2:4],
        priors[:, 2:4] * np.exp(predictions[:, 2:4] * variances[1])), 1)
    boxes[:, :2] = boxes[:, :2] - (boxes[:, 2:4] / 2)
    boxes[:, 2:4] = boxes[:, 2:4] + boxes[:, :2]
    return np.concatenate([boxes, predictions[:, 4:]], 1)
    return boxes


def reversed_argmax(array, axis):
    """Performs the function of torch.max().
    In case of multiple occurrences of the maximum values, the indices
    corresponding to the last occurrence are returned.

    # Arguments:
        array : Numpy array
        axis : int, argmax operation along this specified axis
    # Returns: index_array : Numpy array of ints
    """
    array_flip = np.flip(array, axis=axis)
    return array.shape[axis] - np.argmax(array_flip, axis=axis) - 1


def match(boxes, prior_boxes, iou_threshold=0.5):
    """Matches each prior box with a ground truth box (box from ``boxes``).
    It then selects which matched box will be considered positive e.g. iou > .5
    and returns for each prior box a ground truth box that is either positive
    (with a class argument different than 0) or negative.
    # Arguments
        boxes: Numpy array of shape (num_ground_truh_boxes, 4 + 1),
            where the first the first four coordinates correspond to point
            form box coordinates and the last coordinates is the class
            argument. This boxes should be the ground truth boxes.
        prior_boxes: Numpy array of shape (num_prior_boxes, 4).
            where the four coordinates are in center form coordinates.
        iou_threshold: Float between [0, 1]. Intersection over union
            used to determine which box is considered a positive box.
    # Returns
        numpy array of shape (num_prior_boxes, 4 + 1).
            where the first the first four coordinates correspond to point
            form box coordinates and the last coordinates is the class
            argument.
    """
    ious = compute_ious(boxes, to_point_form(np.float32(prior_boxes)))
    best_box_iou_per_prior_box = np.max(ious, axis=0)

    best_box_arg_per_prior_box = reversed_argmax(ious, 0)
    best_prior_box_arg_per_box = reversed_argmax(ious, 1)

    best_box_iou_per_prior_box[best_prior_box_arg_per_box] = 2
    # overwriting best_box_arg_per_prior_box if they are the best prior box
    for box_arg in range(len(best_prior_box_arg_per_box)):
        best_prior_box_arg = best_prior_box_arg_per_box[box_arg]
        best_box_arg_per_prior_box[best_prior_box_arg] = box_arg
    matches = boxes[best_box_arg_per_prior_box]
    # setting class value to 0 (background argument)
    matches[best_box_iou_per_prior_box < iou_threshold, 4] = 0
    return matches


def make_box_square(box, offset_scale=0.05):
    """Makes box coordinates square.

    # Arguments
        box: Numpy array with shape (4) with point corner coordinates.
        offset_scale: Float, scale of the addition applied box sizes.

    # Returns
        returns: Numpy array with shape (4).
    """

    x_min, y_min, x_max, y_max = box[:4]
    center_x = (x_max + x_min) / 2.
    center_y = (y_max + y_min) / 2.
    width = x_max - x_min
    height = y_max - y_min

    if height >= width:
        half_box = height / 2.
        x_min = center_x - half_box
        x_max = center_x + half_box
    if width > height:
        half_box = width / 2.
        y_min = center_y - half_box
        y_max = center_y + half_box

    box_side_lenght = (x_max + x_min) / 2.
    offset = offset_scale * box_side_lenght
    x_min = x_min - offset
    x_max = x_max + offset
    y_min = y_min - offset
    y_max = y_max + offset
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def filter_detections(detections, arg_to_class, conf_thresh=0.5):
    """Filters boxes outputted from function ``nms_per_class`` as ``Box2D``
        messages depending on their confidence threshold.
    # Arguments
        detections. Numpy array of shape (num_classes, num_boxes, 5)
    """
    num_classes = detections.shape[0]
    filtered_detections = []
    for class_arg in range(1, num_classes):
        class_detections = detections[class_arg, :]
        confidence_mask = np.squeeze(class_detections[:, -1] >= conf_thresh)
        confident_class_detections = class_detections[confidence_mask]
        if len(confident_class_detections) == 0:
            continue
        class_name = arg_to_class[class_arg]
        for confident_class_detection in confident_class_detections:
            coordinates = confident_class_detection[:4]
            score = confident_class_detection[4]
            detection = Box2D(coordinates, score, class_name)
            filtered_detections.append(detection)
    return filtered_detections


def apply_non_max_suppression(boxes, scores, iou_thresh=.45, top_k=200):
    """Apply non maximum suppression.

    # Arguments
        boxes: Numpy array, box coordinates of shape (num_boxes, 4)
            where each columns corresponds to x_min, y_min, x_max, y_max
        scores: Numpy array, of scores given for each box in 'boxes'
        iou_thresh : float, intersection over union threshold
            for removing boxes.
        top_k: int, number of maximum objects per class

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


def nms_per_class(box_data, nms_thresh=.45, conf_thresh=0.01, top_k=200):
    """Applies non-maximum-suppression per class.
    # Arguments
        box_data: Numpy array of shape (num_prior_boxes, 4 + num_classes)
        nsm_thresh: Float. Non-maximum suppression threshold.
        conf_thresh: Float. Filter scores with a lower confidence value before
            performing non-maximum supression.
        top_k: Integer. Maximum number of boxes per class outputted by nms.

    Returns
        Numpy array of shape (num_classes, top_k, 5)
    """
    decoded_boxes, class_predictions = box_data[:, :4], box_data[:, 4:]
    num_classes = class_predictions.shape[1]
    output = np.zeros((num_classes, top_k, 5))

    # skip the background class (start counter in 1)
    for class_arg in range(1, num_classes):
        conf_mask = class_predictions[:, class_arg] >= conf_thresh
        scores = class_predictions[:, class_arg][conf_mask]
        if len(scores) == 0:
            continue
        boxes = decoded_boxes[conf_mask]
        indices, count = apply_non_max_suppression(
            boxes, scores, nms_thresh, top_k)
        scores = np.expand_dims(scores, -1)
        selected_indices = indices[:count]
        selections = np.concatenate(
            (boxes[selected_indices], scores[selected_indices]), axis=1)
        output[class_arg, :count, :] = selections
    return output


def to_one_hot(class_indices, num_classes):
    """ Transform from class index to one-hot encoded vector.

    # Arguments
        class_indices: Numpy array. One dimensional array specifying
            the index argument of the class for each sample.
        num_classes: Integer. Total number of classes.

    # Returns
        Numpy array with shape (num_samples, num_classes).
    """
    one_hot_vectors = np.zeros((len(class_indices), num_classes))
    for vector_arg, class_args in enumerate(class_indices):
        one_hot_vectors[vector_arg, class_args] = 1.0
    return one_hot_vectors


def quaternion_to_rotation_matrix(quaternion):
    """Transforms quaternion to rotation matrix

    # Arguments
        quaternion: Numpy array of shape (4)

    # Returns
        rotation_matrix: Numpy array of shape (3, 3)
    """

    q_w, q_x, q_y, q_z = quaternion
    sqw, sqx, sqy, sqz = np.square(quaternion)
    norm = (sqx + sqy + sqz + sqw)
    rotation_matrix = np.zeros((3, 3))

    # division of square length if quaternion is not already normalized
    rotation_matrix[0, 0] = (+sqx - sqy - sqz + sqw) / norm
    rotation_matrix[1, 1] = (-sqx + sqy - sqz + sqw) / norm
    rotation_matrix[2, 2] = (-sqx - sqy + sqz + sqw) / norm

    tmp1 = q_x * q_y
    tmp2 = q_z * q_w
    rotation_matrix[1, 0] = 2.0 * (tmp1 + tmp2) / norm
    rotation_matrix[0, 1] = 2.0 * (tmp1 - tmp2) / norm

    tmp1 = q_x * q_z
    tmp2 = q_y * q_w
    rotation_matrix[2, 0] = 2.0 * (tmp1 - tmp2) / norm
    rotation_matrix[0, 2] = 2.0 * (tmp1 + tmp2) / norm
    tmp1 = q_y * q_z
    tmp2 = q_x * q_w
    rotation_matrix[2, 1] = 2.0 * (tmp1 + tmp2) / norm
    rotation_matrix[1, 2] = 2.0 * (tmp1 - tmp2) / norm
    return rotation_matrix


def rotation_matrix_to_quaternion(rotation_matrix):
    """Calculates normalized quaternion from rotation matrix

    If w is negative the quaternion gets flipped, so that all w are >= 0

    # Arguments
        rotation_matrix: Numpy array with shape (3, 3)

    # Returns
        quaternion: Numpy array of shape (4)
    """
    trace = np.trace(rotation_matrix)

    if trace > 0:
        S = np.sqrt(trace + 1) * 2
        q_w = 0.25 * S
        q_x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
        q_y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
        q_z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
        return np.asarray([q_w, q_x, q_y, q_z])

    elif ((rotation_matrix[0, 0] > rotation_matrix[1, 1]) and
          (rotation_matrix[0, 0] > rotation_matrix[2, 2])):

        S = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] -
                    rotation_matrix[2, 2]) * 2
        q_w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
        q_x = 0.25 * S
        q_y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
        q_z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S

    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:

        S = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] -
                    rotation_matrix[2, 2]) * 2
        q_w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
        q_x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
        q_y = 0.25 * S
        q_z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S

    else:
        S = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] -
                    rotation_matrix[1, 1]) * 2
        q_w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
        q_x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S
        q_y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S
        q_z = 0.25 * S

    if q_w >= 0:
        return np.asarray([q_w, q_x, q_y, q_z])
    else:
        return -1 * np.asarray([q_w, q_x, q_y, q_z])


def multiply_quaternions(quaternion1, quaternion0):
    """Performs quaternion multiplication.
    # Arguments:
        quaternion1: Numpy array of shape (4)
        quaternion0: Numpy array of shape (4)

    # Returns:
        Numpy array of shape (4)
    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array(
        [-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


def invert_quaternion(quaternion):
    """Takes the inverse of a non-normalized quaternion.
    # Arguments:
        quaternion: Numpy array of shape (4)

    # Returns:
        Numpy array of shape (4)
    """
    norm = np.linalg.norm(quaternion)
    quaternion[1:] = -1.0 * quaternion[1:]
    return quaternion / norm


def substract_mean(image_array, mean):
    """ Subtracts image with channel-wise values.
    # Arguments
        image_array: Numpy array with shape (height, width, 3)
        mean: Numpy array of 3 floats containing the values to be subtracted
            to the image on each corresponding channel.
    """
    image_array = image_array.astype(np.float32)
    image_array[:, :, 0] -= mean[0]
    image_array[:, :, 1] -= mean[1]
    image_array[:, :, 2] -= mean[2]
    return image_array


def make_mosaic(images, shape, border=0):
    """ Creates an image mosaic.
    # Arguments
        images: Numpy array of shape (num_images, height, width, 3)
        shape: List of two integers indicating the mosaic shape.
            Shape must satisfy: shape[0] * shape[1] == len(images).
        border: Integer indicating the border per image.
    # Returns
        A numpy array containing all images.
    """
    num_images = len(images)
    num_rows, num_cols = shape
    image_shape = images.shape[1:]
    num_channels = images.shape[-1]
    mosaic = np.ma.masked_all(
        (num_rows * image_shape[0] + (num_rows - 1) * border,
         num_cols * image_shape[1] + (num_cols - 1) * border, num_channels),
        dtype=np.float32)
    paddedh = image_shape[0] + border
    paddedw = image_shape[1] + border
    for image_arg in range(num_images):
        row = int(np.floor(image_arg / num_cols))
        col = image_arg % num_cols
        # image = np.squeeze(images[image_arg])
        image = images[image_arg]
        image_shape = image.shape
        mosaic[row * paddedh:row * paddedh + image_shape[0],
               col * paddedw:col * paddedw + image_shape[1], :] = image
    return mosaic


def denormalize_keypoints(keypoints, height, width):
    """Transform normalized keypoint coordinates into image coordinates
    # Arguments
        keypoints: Numpy array of shape (num_keypoints, 2)
        height: Int. Height of the image
        width: Int. Width of the image
    """
    for keypoint_arg, keypoint in enumerate(keypoints):
        x, y = keypoint[:2]
        # transform key-point coordinates to image coordinates
        x = (min(max(x, -1), 1) * width / 2 + width / 2) - 0.5
        # flip since the image coordinates for y are flipped
        y = height - 0.5 - (min(max(y, -1), 1) * height / 2 + height / 2)
        x, y = int(round(x)), int(round(y))
        keypoints[keypoint_arg][:2] = [x, y]
    return keypoints


def normalize_keypoints(keypoints, height, width):
    """Transform keypoints in image coordinates to normalized coordinates
        keypoints: Numpy array of shape (num_keypoints, 2)
        height: Int. Height of the image
        width: Int. Width of the image
    """
    normalized_keypoints = np.zeros_like(keypoints, dtype=np.float32)
    for keypoint_arg, keypoint in enumerate(keypoints):
        x, y = keypoint[:2]
        # transform key-point coordinates to image coordinates
        x = (((x + 0.5) - (width / 2.0)) / (width / 2))
        y = (((height - 0.5 - y) - (height / 2.0)) / (height / 2))
        normalized_keypoints[keypoint_arg][:2] = [x, y]
    return normalized_keypoints


def apply_offsets(coordinates, offset_scales):
    """Apply offsets to coordinates
    #Arguments
        coordinates: List of floats containing coordinates in point form.
        offset_scales: List of floats having x and y scales respectively.
    #Returns
        coordinates: List of floats containing coordinates in point form.
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


def evaluate_VOC(
        detector,
        dataset,
        class_dict,
        iou_thresh=0.5,
        use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.

    This function evaluates predicted bounding boxes obtained from a dataset
    which has :math:`N` images by using average precision for each class.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        dataset: List containing information of the images and their data
        from the Test dataset
        detector : Object for inference
        class_dict: Dictionary of class names and their id
        iou_thresh (float): A prediction is correct if its
            Intersection over
            Union with the ground truth is above this value.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    Returns:
        dict:

        The keys, value-types and the description of the values are listed
        below.

        * **ap** (*numpy.ndarray*): An array of average precisions. \
            The :math:`l`-th value corresponds to the average precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`prediction_labels` or :obj:`ground_truth_labels`,
            the corresponding \
            value is set to :obj:`numpy.nan`.
        * **map** (*float*): The average of Average Precisions over classes.

    """

    prediction_boxes, prediction_labels, prediction_scores = get_predictions(dataset, detector, class_dict)
    ground_truth_boxes, ground_truth_labels, ground_truth_difficulties = get_ground_truths(dataset)

    num_positives, score, match = calculate_scores_and_matches(
        prediction_boxes,
        prediction_labels,
        prediction_scores,
        ground_truth_boxes,
        ground_truth_labels,
        ground_truth_difficulties,
        iou_thresh=iou_thresh
    )

    precision, recall = calculate_precision_and_recall(
        num_positives,
        score,
        match
    )

    ap = calc_detection_voc_ap(precision, recall, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)}


def calculate_scores_and_matches(
        prediction_boxes, prediction_labels, prediction_scores,
        ground_truth_boxes, ground_truth_labels,
        ground_truth_difficulties=None,
        iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.

    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        prediction_boxes (iterable of numpy.ndarray): An iterable of :math:`N`
            sets of bounding boxes.
            Its index corresponds to an index for the base dataset.
            Each element of :obj:`prediction_boxes` is a set of coordinates
            of bounding boxes. This is an array whose shape is :math:`(R, 4)`,
            where :math:`R` corresponds
            to the number of bounding boxes, which may vary among boxes.
            The second axis corresponds to
            :math:`y_{min}, x_{min}, y_{max}, x_{max}` of a bounding box.
        prediction_labels (iterable of numpy.ndarray): An iterable of labels.
            Similar to :obj:`prediction_boxes`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        prediction_scores (iterable of numpy.ndarray):
            An iterable of confidence
            scores for predicted bounding boxes.
            Similar to :obj:`prediction_boxes`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        ground_truth_boxes (iterable of numpy.ndarray):
            An iterable of ground truth
            bounding boxes
            whose length is :math:`N`.
            An element of :obj:`ground_truth_boxes` is a
            bounding box whose shape is :math:`(R, 4)`.
            Note that the number of
            bounding boxes in each image does not need to be same
            as the number
            of corresponding predicted boxes.
        ground_truth_labels (iterable of numpy.ndarray):
            An iterable of ground truth
            labels which are organized similarly to :obj:`ground_truth_boxes`.
        ground_truth_difficulties (iterable of numpy.ndarray):
            An iterable of boolean
            arrays which is organized similarly to :obj:`ground_truth_boxes`.
            This tells whether the
            corresponding ground truth bounding box is difficult or not.
            By default, this is :obj:`None`. In that case, this function
            considers all bounding boxes to be not difficult.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value..

    Returns:
        tuple of three dictionaries:

        num_positives: Dictionary containing number of
                        positives of each class
        score: Dictionary containing matching scores of boxes
                    in each class
        match: Dictionary containing match/non-match info of boxes
                    in each class

    """

    classes_count = len(np.unique(np.concatenate(ground_truth_labels)))

    num_positives = {label_id: 0 for label_id in range(1, classes_count + 1)}
    score = {label_id: [] for label_id in range(1, classes_count + 1)}
    match = {label_id: [] for label_id in range(1, classes_count + 1)}

    for prediction_box, prediction_label, prediction_score, ground_truth_box, \
        ground_truth_label, ground_truth_difficult in \
            zip(
                prediction_boxes, prediction_labels, prediction_scores,
                ground_truth_boxes, ground_truth_labels,
                ground_truth_difficulties):

        if ground_truth_difficult is None:
            ground_truth_difficult = np.zeros(
                ground_truth_box.shape[0], dtype=bool
            )

        for class_name_arg in np.unique(
                np.concatenate((
                    prediction_label,
                    ground_truth_label)).astype(int)
        ):
            prediction_mask_class_arg = prediction_label == class_name_arg
            prediction_box_class_arg = prediction_box[
                prediction_mask_class_arg
            ]
            prediction_score_class_arg = prediction_score[
                prediction_mask_class_arg
            ]
            # sort by score
            order = prediction_score_class_arg.argsort()[::-1]
            prediction_box_class_arg = prediction_box_class_arg[order]
            prediction_score_class_arg = prediction_score_class_arg[order]

            ground_truth_mask_class_arg = ground_truth_label == class_name_arg
            ground_truth_box_class_arg = ground_truth_box[
                ground_truth_mask_class_arg
            ]
            ground_truth_difficult_class_arg = ground_truth_difficult[
                ground_truth_mask_class_arg
            ]

            num_positives[class_name_arg] = \
                num_positives[class_name_arg] + \
                np.logical_not(ground_truth_difficult_class_arg).sum()

            score[class_name_arg].extend(prediction_score_class_arg)

            if len(prediction_box_class_arg) == 0:
                continue
            if len(ground_truth_box_class_arg) == 0:
                match[class_name_arg].extend(
                    (0,) * prediction_box_class_arg.shape[0]
                )
                continue

            # VOC evaluation follows integer typed bounding boxes.
            prediction_box_class_arg = prediction_box_class_arg.copy()
            prediction_box_class_arg[:, 2:] = \
                prediction_box_class_arg[:, 2:] + 1
            ground_truth_box_class_arg = \
                ground_truth_box_class_arg.copy()
            ground_truth_box_class_arg[:, 2:] = \
                ground_truth_box_class_arg[:, 2:] + 1

            iou = compute_ious(
                prediction_box_class_arg,
                ground_truth_box_class_arg
            )
            ground_truth_args = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            ground_truth_args[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selection = np.zeros(
                ground_truth_box_class_arg.shape[0], dtype=bool
            )
            for ground_truth_arg in ground_truth_args:
                if ground_truth_arg >= 0:
                    if ground_truth_difficult_class_arg[
                        ground_truth_arg
                    ]:
                        match[class_name_arg].append(-1)
                    else:
                        if not selection[ground_truth_arg]:
                            match[class_name_arg].append(1)
                        else:
                            match[class_name_arg].append(0)
                    selection[ground_truth_arg] = True
                else:
                    match[class_name_arg].append(0)

    # Checking if all the lists containing the same length
    lists = [prediction_boxes, prediction_labels, prediction_scores,
             ground_truth_boxes, ground_truth_labels,
             ground_truth_difficulties]
    if len(set(map(len, lists))) not in (0, 1):
        raise ValueError('Length of input iterables need to be same')

    return num_positives, score, match


def calculate_precision_and_recall(num_positives, scores, matches):
    """

    Args:
        num_positives: Dictionary containing number of positives of each class
        scores: Dictionary containing matching scores of boxes in each class
        matches: Dictionary containing match/non-match info of
                boxes in each class

    Returns:
        tuple of two lists
        * :obj:`precision`:
         A list of arrays. :obj:`precision[class_name_arg]` is precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`prediction_labels` or :obj:`ground_truth_labels`,
            :obj:`precision[class_name_arg]` is \
            set to :obj:`None`.
        * :obj:`recall`: A list of arrays.
            :obj:`recall[class_name_arg]` is recall \
            for class :math:`l`. If class :math:`l` that is not marked as \
            difficult does not exist in \
            :obj:`ground_truth_labels`, :obj:`recall[class_name_arg]` is \
            set to :obj:`None`.

    """
    num_foreground_class = max(num_positives.keys()) + 1
    precision = [None] * num_foreground_class
    recall = [None] * num_foreground_class

    for positive_key_arg in num_positives.keys():
        score_positive_key_arg = np.array(scores[positive_key_arg])
        match_positive_key_arg = np.array(
            matches[positive_key_arg], dtype=np.int8
        )

        order = score_positive_key_arg.argsort()[::-1]
        match_positive_key_arg = match_positive_key_arg[order]

        true_positives = np.cumsum(match_positive_key_arg == 1)
        false_positives = np.cumsum(match_positive_key_arg == 0)

        # If an element of false_positives + true_positives is 0,
        # the corresponding element of precision[positive_key_arg] is nan.
        precision[positive_key_arg] = \
            true_positives / (false_positives + true_positives)
        # If num_positives[positive_key_arg] is 0,
        # recall[positive_key_arg] is None.
        if num_positives[positive_key_arg] > 0:
            recall[positive_key_arg] = \
                true_positives / num_positives[positive_key_arg]

    return precision, recall


def calc_detection_voc_ap(precision, recall, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.

    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        precision (list of numpy.array): A list of arrays.
            :obj:`precision[foreground_class_arg]` indicates
            precision for class :math:`l`.
            If :obj:`precision[foreground_class_arg]` is :obj:`None`,
            this function returns
            :obj:`numpy.nan` for class :math:`l`.
        recall (list of numpy.array): A list of arrays.
            :obj:`recall[foreground_class_arg]` indicates recall for class
            :math:`l`.
            If :obj:`recall[foreground_class_arg]` is :obj:`None`,
            this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`precision[foreground_class_arg]` or
        :obj:`recall[foreground_class_arg]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.

    """

    num_foreground_class = len(precision)
    ap = np.empty(num_foreground_class)
    for foreground_class_arg in range(num_foreground_class):
        if precision[foreground_class_arg] is None \
                or recall[foreground_class_arg] is None:
            ap[foreground_class_arg] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[foreground_class_arg] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(recall[foreground_class_arg] >= t) == 0:
                    p_interpolation = 0
                else:
                    p_interpolation = np.max(
                        np.nan_to_num(
                            precision[foreground_class_arg]
                        )[recall[foreground_class_arg] >= t]
                    )
                average_precision_class = ap[foreground_class_arg]
                average_precision_class = (average_precision_class +
                                           (p_interpolation / 11))
                ap[foreground_class_arg] = average_precision_class
        else:
            # correct AP calculation
            # first append sentinel values at the end
            average_precision = np.concatenate(([0], np.nan_to_num(
                precision[foreground_class_arg]), [0]))
            average_recall = np.concatenate(
                ([0], recall[foreground_class_arg], [1])
            )

            average_precision = np.maximum.accumulate(
                average_precision[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            recall_change_arg = np.where(
                average_recall[1:] != average_recall[:-1])[0]

            # and sum (\Delta recall) * precision
            ap[foreground_class_arg] = np.sum(
                (average_recall[recall_change_arg + 1] -
                 average_recall[recall_change_arg]) *
                average_precision[recall_change_arg + 1]
            )

    return ap


def get_predictions(dataset, detector, class_dict):
    """
    Arguments:
        dataset: List containing information of the images from the
        Test dataset
        detector : Object for inference
        class_dict: Dictionary of class names and their id

    Returns:
        predictions_boxes: List containing prediction boxes
        predictions_labels: List containing corresponding prediction labels
        predictions_scores: List containing corresponding prediction scores
    """
    boxes, labels, scores = [], [], []
    for image in dataset:
        frame = load_image(image['image'])
        results = detector({'image': frame})
        box, label, score = [], [], []
        for box2D in results['boxes2D']:
            score.append(box2D.score)
            box.append(list(box2D.coordinates))
            label.append(class_dict[box2D.class_name])
        boxes.append(np.array(box, dtype=np.float32))
        labels.append(np.array(label))
        scores.append(np.array(score, dtype=np.float32))
    return boxes, labels, scores


def get_ground_truths(dataset):
    """
    Arguments:
        dataset: List containing information of the images from the
        Test dataset

    Returns:
        ground_truth_boxes: List containing ground truth boxes
        ground_truth_labels: List containing corresponding ground truth labels
        ground_truth_difficults: List containing corresponding
        ground truth difficults
    """

    boxes, labels, difficults = [], [], []
    for image in dataset:
        boxes.append(np.array(image['boxes'][:, :4]))
        labels.append(np.array(image['boxes'][:, 4]))
        difficults.append(np.array(image['difficulties']))
    return boxes, labels, difficults
