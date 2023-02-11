import numpy as np
from paz.backend.boxes import apply_non_max_suppression


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
    """Filters given boxes based on scores.

    # Arguments
        boxes: Numpy array of shape `(num_boxes, 4 + num_classes)`.
        conf_thresh: Float. Filter boxes with a confidence value lower
            than this.

    Returns
        Numpy array of shape `(num_boxes, 4 + num_classes)`.
    """
    classes = boxes[:, 4:]
    output = np.array([], dtype=float).reshape(0, boxes.shape[1])
    for class_arg in range(classes.shape[1]):
        per_class_score = classes[:, class_arg]
        mask = per_class_score >= conf_thresh
        selected_boxes = boxes[mask]
        output = np.concatenate((output, selected_boxes), axis=0)
    return output
