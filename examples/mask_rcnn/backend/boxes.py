import numpy as np

from paz.backend.boxes import to_center_form


def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1., w - 1., h - 1., w - 1.])
    shift = np.array([0., 0., 1., 1.])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0., 0., 1., 1.])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):  # TODO: modify the function to match the paz function format
    """Calculates IoU of the given box with the array of the given boxes.

    # Arguments:
        box: 1D vector [y_min, x_min, y_max, x_max]
        boxes: [boxes_count, (y_min, x_min, y_max, x_max)]
        box_area: float. the area of 'box'
        boxes_area: array of length boxes_count.

    # Returns:
        Intersection over union of given boxes
    """
    y_min = np.maximum(box[0], boxes[:, 0])
    y_max = np.minimum(box[2], boxes[:, 2])
    x_min = np.maximum(box[1], boxes[:, 1])
    x_max = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x_max - x_min, 0) * np.maximum(y_max - y_min, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes_A, boxes_B):  # TODO: modify the function to match the paz function format
    """Computes IoU overlaps between two sets of boxes.

    # Arguments:
        boxes_A, boxes_B: [N, (y_min, x_min, y_max, x_max)].
    """
    area1 = (boxes_A[:, 2] - boxes_A[:, 0]) * (boxes_A[:, 3] - boxes_A[:, 1])
    area2 = (boxes_B[:, 2] - boxes_B[:, 0]) * (boxes_B[:, 3] - boxes_B[:, 1])

    overlaps = np.zeros((boxes_A.shape[0], boxes_B.shape[0]))
    for i in range(overlaps.shape[1]):
        box_B = boxes_B[i]
        overlaps[:, i] = compute_iou(box_B, boxes_A, area2[i], area1)
    return overlaps


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
       is associated with a level of the pyramid, but each ratio is used in
       all levels of the pyramid.

    # Returns:
        anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array.
                 Sorted with the same order of the given scales.
    """
    anchors = []
    for level in range(len(scales)):
        anchors.append(generate_anchors(scales[level], ratios,
                                        feature_shapes[level],
                                        feature_strides[level], anchor_stride))
    return np.concatenate(anchors, axis=0)


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """Generates anchor boxes

    # Arguments:
        scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
        ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
        shape: [height, width] spatial shape of the feature map over which
                to generate anchors.
        feature_stride: feature map stride relative to the image in pixels.
        anchor_stride: anchor stride on feature map. For example, if the
            value is 2 then generate anchors for every other feature map pixel.
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

    box_centers = np.stack([box_center_Y, box_center_X], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def extract_boxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: box array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set box to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_rpn_bounding_box(groundtruth_boxes, rpn_match, anchors, anchor_iou_argmax):
    """
    For positive anchors, compute shift and scale needed to transform them
    to match the corresponding groundtruth boxes.

    # Return:
    RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    """
    rpn_box = np.zeros((256, 4))
    ids = np.where(rpn_match == 1)[0]
    ix = 0
    for i, a in zip(ids, anchors[ids]):

        groundtruth = groundtruth_boxes[anchor_iou_argmax[i]]

        groundtruth = box_to_center_format(groundtruth)
        a = box_to_center_format(a)
        rpn_box[ix] = normalize_log_refinement(a, groundtruth)

        rpn_box[ix] /= np.array([0.1, 0.1, 0.2, 0.2])
        ix += 1
    return rpn_box


def box_to_center_format(box):
    """
    Converts bounding box of format center, width and height.
    # Argument
     box: [xmin, ymin, xmax, ymax]
    # Return:
     box: [xcenter, ycenter, width, height]
    """
    H = box[2] - box[0]
    W = box[3] - box[1]
    center_y = box[0] + 0.5 * H
    center_x = box[1] + 0.5 * W
    return [center_y, center_x, H, W]


def normalize_log_refinement(box, ground_truth_box):
    """
    Compute the box refinement that the RPN should predict
    # Return:
     rpn_box = [dy, dx, log(dh), log(dw)]
    """
    # Compute the box refinement that the RPN should predict.

    rpn_box = [
        (ground_truth_box[0] - box[0]) / (box[2]),
        (ground_truth_box[1] - box[1]) / (box[3]),
        np.log(ground_truth_box[2] / (box[2])),
        np.log(ground_truth_box[3] / (box[3]))
    ]
    return rpn_box


def compute_anchor_boxes_overlaps(anchors, groundtruth_class_ids, groundtruth_boxes):
    """Given the anchors and groundtruth boxes, compute overlaps by handling the crowds.
    A crowd box in COCO is a bounding box around several instances. Exclude
    them from training. A crowd box is given a negative class ID.

    # Arguments:
    anchors: [num_anchors, (y1, x1, y2, x2)]
    groundtruth_class_ids: [num_groundtruth_boxes] Integer class IDs.
    groundtruth_boxes: [num_groundtruth_boxes, (y1, x1, y2, x2)]

    # Returns:
    overlaps: [num_anchors, num_groundtruth_boxes]
    no_crowd_bools : [N] True/False values
    """
    crowd_ix = np.where(groundtruth_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        non_crowd_ix = np.where(groundtruth_class_ids > 0)[0]
        crowd_boxes = groundtruth_boxes[crowd_ix]
        groundtruth_class_ids = groundtruth_class_ids[non_crowd_ix]
        groundtruth_boxes = groundtruth_boxes[non_crowd_ix]
        crowd_overlaps = compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    overlaps = compute_overlaps(anchors, groundtruth_boxes)
    return overlaps, no_crowd_bool, groundtruth_boxes


def compute_rpn_match(anchors, overlaps, no_crowd_bool):
    """
    Match anchors to groundtruth Boxes
    If an anchor overlaps a groundtruth box with IoU >= 0.7 then it's positive.
    If an anchor overlaps a groundtruth box with IoU < 0.3 then it's negative.
    Neutral anchors are those that don't match the conditions above,
    and they don't influence the loss function.
    However, don't keep any groundtruth box unmatched (rare, but happens). Instead,
    match it to the closest anchor (even if its max IoU is < 0.3)
    RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    1. Set negative anchors first. They get overwritten below if a groundtruth box is
    matched to them. Skip boxes in crowd areas.
    2. Set an anchor for each groundtruth box (regardless of IoU value).If multiple
    anchors have the same IoU match all of them.
    3. Set anchors with high overlap as positive.
    Subsample to balance positive and negative anchors
    Don't let positives be more than half the anchors

    # Return:
    rpn_match: [N]
    anchor_iou_argmax: [num_anchors, num_groundtruth_boxes]
    """
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    groundtruth_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
    rpn_match[groundtruth_iou_argmax] = 1
    rpn_match[anchor_iou_max >= 0.7] = 1

    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (256 // 2)
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (256 - np.sum(rpn_match == 1))
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    return rpn_match, anchor_iou_argmax
