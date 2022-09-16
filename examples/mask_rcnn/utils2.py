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
from tensorflow.keras.utils import  Sequence
from tensorflow.keras.layers import ZeroPadding2D, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Dense, Activation
from tensorflow.keras.layers import TimeDistributed, Lambda, Reshape
from tensorflow.keras.layers import Input, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Add
from tensorflow.keras.models import Model

from pipeline import TrainingPipeline

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


def DataGenerator(dataset, config, shuffle=False, augmentation=False):
    """An iterable that returns images and corresponding target class ids,
        bounding box deltas, and masks. It inherits from keras.utils.Sequence to avoid data redundancy
        when multiprocessing=True.
        dataset: The Dataset object to pick data from
        config: The model config object
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
    error_count=0
    image_ids = np.array([i for i in range(len(dataset.load_data()))])
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                       config.RPN_ANCHOR_RATIOS,
                                       backbone_shapes,
                                       config.BACKBONE_STRIDES,
                                       config.RPN_ANCHOR_STRIDE)
    while True:
        try:

            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)

            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, gt_class_ids, gt_boxes, gt_masks = \
                         load_image_gt(dataset, config, image_id,
                          augmentation=augmentation)
            gt_class_ids = np.array(gt_class_ids)
            # Skip images that have no instances.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(anchors, gt_class_ids, gt_boxes, config)
            batch_size = config.BATCH_SIZE

            # Init batch arrays
            if b == 0:

                batch_rpn_match = np.zeros(
                    (batch_size, anchors.shape[0], 1), dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    (batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4), dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size, image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_masks = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = normalize_image(image.astype(np.float32), config)
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks

            b += 1

            if b>=batch_size:

                inputs = [batch_images, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                outputs = []

                yield inputs, outputs

                b=0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            logging.exception("Error processing image")
            error_count +=1
            if error_count > 5:
                raise



def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.
    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])


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


def normalize_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def load_image_gt(dataset, config, image_id, augmentation=True):
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
    class_ids = data[image_id]['box_data'][:,-1]

    bounding_box = data[image_id]['box_data']
    num_classes = config.NUM_CLASSES
    image_size = config.IMAGE_SHAPE[:2]

    if augmentation:
        augmentator = TrainingPipeline(bounding_box, num_classes=num_classes, size=image_size)
        sample = {'image': image, 'boxes': bounding_box, 'masks': mask}
        data = augmentator(sample)
        image= data['inputs']['image']
        bounding_box = data['labels']['boxes']
        mask = data['labels']['masks']
    return image, class_ids, bounding_box[:,:4], mask


def build_rpn_targets(anchors, gt_class_ids, gt_boxes, config):
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

    overlaps, no_crowd_bool = compute_anchor_boxes_overlaps(anchors, gt_class_ids, gt_boxes)

    rpn_match, anchor_iou_argmax = compute_rpn_match(anchors, overlaps, no_crowd_bool, config)

    rpn_bbox = compute_rpn_bbox(gt_boxes, rpn_match, anchors, config, anchor_iou_argmax)

    return rpn_match, rpn_bbox


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
    return overlaps, no_crowd_bool


def compute_rpn_match(anchors, overlaps, no_crowd_bool, config):
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
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    return rpn_match, anchor_iou_argmax


def compute_rpn_bbox(gt_boxes, rpn_match, anchors, config, anchor_iou_argmax):
    """
    For positive anchors, compute shift and scale needed to transform them
    to match the corresponding GT boxes.

    # Return:
    RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    """
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))
    ids = np.where(rpn_match == 1)[0]
    ix = 0
    for i, a in zip(ids, anchors[ids]):

        gt = gt_boxes[anchor_iou_argmax[i]]

        gt = box_to_center_format(gt)
        a = box_to_center_format(a)

        rpn_bbox[ix] = normalize_log_refinement(a, gt)
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
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
    return [center_x, center_y, W, H]


def normalize_log_refinement(box, ground_truth_box):
    """
    Compute the bbox refinement that the RPN should predict
    # Return:
     rpn_bbox = [dy, dx, log(dh), log(dw)]
    """
    ground_truth_box = box_to_center_format(ground_truth_box)
    box = box_to_center_format(box)

    # Compute the bbox refinement that the RPN should predict.
    rpn_bbox = [
        (ground_truth_box[1] - box[1]) / box[3],
        (ground_truth_box[0] - box[0]) / box[2],
        np.log(ground_truth_box[3] / box[3]),
        np.log(ground_truth_box[2] / box[2]),
    ]
    return rpn_bbox