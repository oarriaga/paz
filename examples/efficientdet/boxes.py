import numpy as np
from paz.abstract import Box2D


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


def nms_per_class(box_data, nms_thresh=.45, conf_thresh=0.01, top_k=200):
    """Applies non-maximum-suppression per class.
    # Arguments
        box_data: Numpy array of shape `(num_prior_boxes, 4 + num_classes)`.
        nsm_thresh: Float. Non-maximum suppression threshold.
        conf_thresh: Float. Filter scores with a lower confidence value before
            performing non-maximum supression.
        top_k: Integer. Maximum number of boxes per class outputted by nms.

    Returns
        Numpy array of shape `(num_classes, top_k, 5)`.
    """
    decoded_boxes, class_predictions = box_data[:, :4], box_data[:, 4:]
    num_classes = class_predictions.shape[1]
    output = np.array([], dtype=float).reshape(0, box_data.shape[1])
    for class_arg in range(num_classes):
        mask = class_predictions[:, class_arg] >= conf_thresh
        scores = class_predictions[:, class_arg][mask]
        if len(scores) == 0:
            continue
        boxes = decoded_boxes[mask]
        indices, count = apply_non_max_suppression(
            boxes, scores, nms_thresh, top_k)
        scores = np.expand_dims(scores, -1)
        selected_indices = indices[:count]
        classes = class_predictions[mask]
        selections = np.concatenate(
            (boxes[selected_indices],
             classes[selected_indices]), axis=1)
        output = np.concatenate((output, selections))
    return output


def filter_boxes(boxes, conf_thresh):
    max_class_score = np.max(boxes[:, 4:], axis=1)
    confidence_mask = max_class_score >= conf_thresh
    confident_class_detections = boxes[confidence_mask]
    return confident_class_detections


def BoxesToBoxes2D(boxes):
    boxes2D = []
    for box in boxes:
        boxes2D.append(Box2D(box[:4], 1.0, None))
    return boxes2D


def BoxesWithOneHotVectorsToBoxes2D(boxes, arg_to_class):
    boxes2D = []
    for box in boxes:
        score = np.max(box[4:])
        class_arg = np.argmax(box[4:])
        class_name = arg_to_class[class_arg]
        boxes2D.append(Box2D(box[:4], score, class_name))
    return boxes2D


def BoxesWithClassArgToBoxes2D(boxes):
    raise NotImplementedError
