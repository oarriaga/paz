import numpy as np
from paz.backend.boxes import compute_ious, to_corner_form


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
