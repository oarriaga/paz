import logging
import math
import numpy as np
import tensorflow as tf
import scipy
import skimage.transform
import urllib.request
import shutil
import warnings
import colorsys
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon
from distutils.version import LooseVersion

import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import ZeroPadding2D, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Dense, Activation
from tensorflow.keras.layers import TimeDistributed, Lambda, Reshape
from tensorflow.keras.layers import Input, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Add
from tensorflow.keras.models import Model
import cv2


def compute_iou(box, boxes, box_area, boxes_area):
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


def compute_overlaps(boxes_A, boxes_B):
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


def DataGenerator(dataset, backbone, image_shape, anchor_scales, batch_size, num_classes,
                  shuffle=True, augmentation=False):
    """An iterable that returns images and corresponding target class ids,
        bounding box deltas, and masks. It inherits from keras.utils.Sequence to avoid data redundancy
        when multiprocessing=True.
        dataset: The Dataset object to pick data from
        shuffle: If True, shuffles the samples before every epoch
        augmentation: Optional. From paz pipeline for augmentation.
        inputs list:
        - images: [batch, H, W, C]
        - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
        - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
        - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
        - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
        - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                    are those of the image unless use_mini_mask is True, in which
                    case they are defined in MINI_MASK_SHAPE.
        outputs list: Usually empty in regular training. But if detection_targets
            is True then the outputs list contains target class_ids, bbox deltas,
            and masks.
        """
    b = 0
    image_index = -1
    image_ids = np.array([i for i in range(len(dataset.load_data()))])
    backbone_shapes = compute_backbone_shapes(backbone, image_shape)
    anchors = generate_pyramid_anchors(anchor_scales, [0.5, 1, 2],
                                       backbone_shapes, [4, 8, 16, 32, 64], 1)

    while True:
        try:

            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)

            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, gt_class_ids, gt_boxes, gt_masks = load_image_gt(dataset, num_classes, image_id,
                                                                    augmentation=augmentation)

            rpn_match, rpn_bbox = build_rpn_targets(anchors, gt_class_ids, gt_boxes)

            if b == 0:
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, 256, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    [batch_size, image_shape[0], image_shape[1], image_shape[2]], dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, 100), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, 100, 4), dtype=np.int32)
                batch_gt_masks = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     100), dtype=bool)

                # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > 100:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), 100, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

                # Add to batch
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = normalize_image(image.astype(np.float32))
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks

            b += 1

            if b >= batch_size:

                inputs = [batch_images, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]

                zeros_array = np.zeros((batch_size, anchors.shape[0], 3))
                rpn_match_padded = np.concatenate((batch_rpn_match, zeros_array), axis=2)
                rpn_bbox_padded = np.concatenate((batch_rpn_bbox, zeros_array), axis=1)
                for i in range(batch_size):
                    c.append(np.concatenate((batch_rpn_bbox[i], rpn_match_padded[i])))
                batch_rpn = np.stack(c, axis=0)

                outputs = [batch_rpn_match, batch_rpn]
                yield inputs, outputs

                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise


def compute_backbone_shapes(backbone, image_shape):
    """Computes the width and height of each stage of the backbone network.
    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(backbone):
        return compute_backbone_shapes(image_shape)

    # Currently supports ResNet only
    assert backbone in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in [4, 8, 16, 32, 64]])


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


def normalize_image(images):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - np.array([123.7, 116.8, 103.9])


def load_image_gt(dataset, num_classes, image_id, augmentation=False,
                  use_mini_mask=False, mini_mask_shape=(28, 28)):
    """Load and return ground truth data for an image (image, mask, bounding boxes).
    augmentation: using paz library for augmentation

    # Returns:
    image: [height, width, 3]
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    data = dataset.load_data()
    image = data[image_id]['image']
    mask = data[image_id]['masks']
    class_ids = data[image_id]['box_data'][:, -1]

    # mask = resize_mask(mask, scale, padding, crop)
    image, window, scale, padding, crop = resize_image(
        image,
        min_dim=128,
        min_scale=0,
        max_dim=128,
        mode='square')
    mask = resize_mask(mask, scale, padding, crop)

    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]

    bbox = extract_bboxes(mask)

    if use_mini_mask:
        mask = minimize_mask(bbox.astype(np.int32), mask, mini_mask_shape)

    return image, np.array(class_ids), bbox, mask.astype(np.bool)


def build_rpn_targets(anchors, gt_class_ids, gt_boxes):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.
    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    # Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    overlaps, no_crowd_bool, gt_boxes = compute_anchor_boxes_overlaps(anchors, gt_class_ids, gt_boxes)

    rpn_match, anchor_iou_argmax = compute_rpn_match(anchors, overlaps, no_crowd_bool)

    rpn_bbox = compute_rpn_bbox(gt_boxes, rpn_match, anchors, anchor_iou_argmax)

    return rpn_match, rpn_bbox.astype(np.float32)


def compute_anchor_boxes_overlaps(anchors, gt_class_ids, gt_boxes):
    """Given the anchors and GT boxes, compute overlaps by handling the crowds.
    A crowd box in COCO is a bounding box around several instances. Exclude
    them from training. A crowd box is given a negative class ID.

    # Arguments:
    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    # Returns:
    overlaps: [num_anchors, num_gt_boxes]
    no_crowd_bools : [N] True/False values
    """
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        crowd_overlaps = compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    overlaps = compute_overlaps(anchors, gt_boxes)
    return overlaps, no_crowd_bool, gt_boxes


def compute_rpn_match(anchors, overlaps, no_crowd_bool):
    """
    Match anchors to GT Boxes
    If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    Neutral anchors are those that don't match the conditions above,
    and they don't influence the loss function.
    However, don't keep any GT box unmatched (rare, but happens). Instead,
    match it to the closest anchor (even if its max IoU is < 0.3)
    RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    1. Set negative anchors first. They get overwritten below if a GT box is
    matched to them. Skip boxes in crowd areas.
    2. Set an anchor for each GT box (regardless of IoU value).If multiple
    anchors have the same IoU match all of them.
    3. Set anchors with high overlap as positive.
    Subsample to balance positive and negative anchors
    Don't let positives be more than half the anchors

    # Return:
    rpn_match: [N]
    anchor_iou_argmax: [num_anchors, num_gt_boxes]
    """
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
    rpn_match[gt_iou_argmax] = 1
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


def compute_rpn_bbox(gt_boxes, rpn_match, anchors, anchor_iou_argmax):
    """
    For positive anchors, compute shift and scale needed to transform them
    to match the corresponding GT boxes.

    # Return:
    RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    """
    rpn_bbox = np.zeros((256, 4))
    ids = np.where(rpn_match == 1)[0]
    ix = 0
    for i, a in zip(ids, anchors[ids]):

        gt = gt_boxes[anchor_iou_argmax[i]]
        gt = box_to_center_format(gt)
        a = box_to_center_format(a)

        rpn_bbox[ix] = normalize_log_refinement(a, gt)
        rpn_bbox[ix] /= np.array([0.1, 0.1, 0.2, 0.2])
        ix += 1
    return rpn_bbox


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
    Compute the bbox refinement that the RPN should predict
    # Return:
     rpn_bbox = [dy, dx, log(dh), log(dw)]
    """
    # Compute the bbox refinement that the RPN should predict.

    rpn_bbox = [
        (ground_truth_box[0] - box[0]) / (box[2]),
        (ground_truth_box[1] - box[1]) / (box[3]),
        np.log(ground_truth_box[2] / (box[2])),
        np.log(ground_truth_box[3] / (box[3]))
    ]
    return rpn_bbox


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()
    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        y1, x1, y2, x2 = bbox[i, :4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape)
        mini_mask[:, :, i] = np.around(m).astype(bool)
    return mini_mask


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().
    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.
    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.
    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(np.uint8), window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.
    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
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
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)
