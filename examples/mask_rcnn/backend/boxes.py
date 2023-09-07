import numpy as np

from paz.backend.boxes import to_center_form, compute_ious


def normalized_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (x1, y1, x2, y2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (x2, y2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (x1, y1, x2, y2)] in normalized coordinates.
    """
    H, W = shape
    scale = np.array([H-1, W-1, H-1, W-1])
    shift = np.array([0., 0., 1., 1.])
    return np.divide(boxes - shift, scale)


def denormalized_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (x1, y1, x2, y2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (x2, y2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (x1, y1, x2, y2)] in pixel coordinates.
    """
    H, W = shape
    scale = np.array([H-1, W-1, H-1, W-1])
    shift = np.array([0., 0., 1., 1.])
    return np.around(np.multiply(boxes, scale) + shift)


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
       is associated with a level of the pyramid, but each ratio is used in
       all levels of the pyramid.

    # Returns:
        anchors: [N, (x1, y1, x2, y2)]. All generated anchors in one array.
                 Sorted with the same order of the given scales.
    """
    anchors = []
    for scale, feature_shape, feature_stride in zip(scales, feature_shapes,
                                                    feature_strides):
        anchors.append(generate_anchors(scale, ratios, feature_shape,
                                        feature_stride, anchor_stride))
    return np.concatenate(anchors, axis=0)


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """Generates anchor boxes.

    # Arguments:
        scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
        ratios: 1D array of anchor ratios of width/height.
                Example: [0.5, 1, 2]
        shape: [height, width] spatial shape of the feature map over which
                to generate anchors.
        feature_stride: feature map stride relative to the image in pixels.
        anchor_stride: anchor stride on feature map. For example, if the
            value is 2 then generate anchors for every other feature map
            pixel.

    # Returns:
        anchor boxes: [no. of anchors, (y_min, x_min, y_max, x_max)].
    """
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    shifts_Y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_X = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_X, shifts_Y = np.meshgrid(shifts_X, shifts_Y)

    box_widths, box_center_X = np.meshgrid(widths, shifts_X)
    box_heights, box_center_Y = np.meshgrid(heights, shifts_Y)

    box_centers = np.stack([box_center_Y, box_center_X],
                           axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def compute_RPN_bounding_box(anchors, rpn_match, groundtruth_boxes,
                             anchor_iou_argmax,
                             std_dev=np.array([0.1, 0.1, 0.2, 0.2])):
    """
    For positive anchors, compute shift and scale needed to transform them
    to match the corresponding groundtruth boxes.

    # Arguments:
        groundtruth_boxes: [num_groundtruth_boxes, (y1, x1, y2, x2)].
        RPN_match: [N].
        anchors: [num_anchors, (x1, y1, x2, y2)].
        anchor_iou_argmax: [1].
        std_dev = [x1, y1, x2, y2].

    # Return:
        RPN bounding boxes: [max anchors per image,
                            (dx, dy, log(dw), log(dh))].
    """
    positive_anchors_index = np.where(rpn_match == 1)[0]
    groundtruth = groundtruth_boxes[anchor_iou_argmax]

    groundtruth = to_center_form(groundtruth[positive_anchors_index])
    positive_anchors = to_center_form(anchors[positive_anchors_index])

    RPN_box = encode_boxes(positive_anchors, groundtruth)
    RPN_box = RPN_box / std_dev
    RPN_box = np.stack(RPN_box)
    return RPN_box


def encode_boxes(boxes, ground_truth_boxes):
    """
    Compute the logarithmic box refinement that the RPN should predict.

    # Arguments:
       boxes:[dx, dy, log(dw), log(dh)]
       ground_truth_boxes:[dx, dy, log(dw), log(dh)]

    # Return:
     rpn_box = [dx, dy, log(dw), log(dh)]
    """
    RPN_box = [(ground_truth_boxes[:, 0] - boxes[:, 0]) / (boxes[:, 2]),
               (ground_truth_boxes[:, 1] - boxes[:, 1]) / (boxes[:, 3]),
               np.log(ground_truth_boxes[:, 2] / (boxes[:, 2])),
               np.log(ground_truth_boxes[:, 3] / (boxes[:, 3]))]
    return np.array(RPN_box).T


def compute_anchor_boxes_overlaps(anchors, groundtruth_class_ids,
                                  groundtruth_boxes, crowd_threshold=0.001):
    """Given the anchors and groundtruth boxes, compute overlaps by handling
    the crowds. A crowd box in COCO is a bounding box around several instances.
    Exclude them from training. A crowd box is given a negative class ID.

    # Arguments:
    anchors: [num_anchors, (x1, y1, x2, y2)]
    groundtruth_class_ids: [num_groundtruth_boxes] Integer class IDs.
    groundtruth_boxes: [num_groundtruth_boxes, (y1, x1, y2, x2)]

    # Returns:
    overlaps: [num_anchors, num_groundtruth_boxes]
    no_crowd_bools : [N] True/False values
    """
    crowd_instances = np.where(groundtruth_class_ids < 0)[0]
    if crowd_instances.shape[0] > 0:

        non_crowd_ix = np.where(groundtruth_class_ids > 0)[0]
        crowd_boxes = groundtruth_boxes[crowd_instances]

        groundtruth_boxes = groundtruth_boxes[non_crowd_ix]
        crowd_overlaps = compute_ious(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < crowd_threshold)
    else:
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)
    overlaps = compute_ious(anchors, groundtruth_boxes)
    return overlaps, no_crowd_bool, groundtruth_boxes


def compute_RPN_match(anchors, overlaps, no_crowd_bool, anchor_size=256):
    """
    Match anchors to groundtruth boxes
    If an anchor overlaps a groundtruth box with IoU >= 0.7 then it's positive.
    If an anchor overlaps a groundtruth box with IoU < 0.3 then it's negative.
    Neutral anchors are those that don't match the conditions above,
    and they don't influence the loss function.

    RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    NOTE: Positives should not be more than half the anchors

    # Arguments:
    anchors: [num_anchors, (x1, y1, x2, y2)]
    overlaps: [num_anchors, num_groundtruth_boxes]
    groundtruth_boxes: [num_groundtruth_boxes, (y1, x1, y2, x2)]

    # Return:
    rpn_match: [N]
    anchor_iou_argmax: [num_anchors, num_groundtruth_boxes]
    """
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]

    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    groundtruth_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0)
                                         )[:, 0]
    rpn_match[groundtruth_iou_argmax] = 1
    rpn_match[anchor_iou_max >= 0.7] = 1

    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (anchor_size // 2)
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (anchor_size - np.sum(rpn_match == 1))
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    return rpn_match, anchor_iou_argmax


def concatenate_RPN_values(RPN_bounding_box, RPN_match, anchors_shape,
                           RPN_box_shape=256):
    """
    Concatenate RPN match and RPN bounding box values to handle RPN losses
    directly to the model.

    # Arguments:
    RPN_bounding_box: [N, (x1, y1, x2, y2)]
    RPN_match: [anchor_shape, num_groundtruth_boxes]
    anchors_shape: [N, 4]

    # Return:
    RPN_values: [N]
    """
    zeros_array = np.zeros((anchors_shape, 3))
    RPN_bounding_box_padded = np.zeros((RPN_box_shape, 4))
    RPN_match = np.reshape(RPN_match, (-1, 1))
    RPN_match_padded = np.concatenate((RPN_match, zeros_array), axis=1)
    RPN_bounding_box_padded[:len(RPN_bounding_box)] = RPN_bounding_box
    RPN_values = np.concatenate((RPN_bounding_box_padded,
                                 RPN_match_padded))
    return RPN_values
