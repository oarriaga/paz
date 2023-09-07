import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

from mask_rcnn.model.layer_utils import slice_batch


def trim_zeros(boxes):
    """Often boxes are represented with matrices of shape [N, 4] and
       are padded with zeros. This removes zero boxes.

    # Arguments:
        boxes: [N, 4] matrix of boxes.
        non_zeros: [N] a 1D boolean mask identifying the rows to keep.

    # Returns:
        boxes: [N, 4] matrix of boxes after removing zeros values.
        non_zeros: [N] a 1D boolean mask identifying the rows to keep after
        removing zero values.
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros)
    return boxes, non_zeros


def refine_bounding_box(box, prior_box):
    """Compute refinement needed to transform box to groundtruth_box.

    # Arguments:
        box: [N, (y_min, x_min, y_max, x_max)].
        prior_box: Ground-truth box [N, (y_min, x_min, y_max, x_max)].
    """
    box = tf.cast(box, tf.float32)
    prior_box = tf.cast(prior_box, tf.float32)
    x_box = box[:, 0]
    y_box = box[:, 1]

    H = box[:, 2] - x_box
    W = box[:, 3] - y_box
    center_y = x_box + (0.5 * H)
    center_x = y_box + (0.5 * W)

    prior_H = prior_box[:, 2] - prior_box[:, 0]
    prior_W = prior_box[:, 3] - prior_box[:, 1]
    prior_center_y = prior_box[:, 0] + (0.5 * prior_H)
    prior_center_x = prior_box[:, 1] + (0.5 * prior_W)

    dy = (prior_center_y - center_y) / H
    dx = (prior_center_x - center_x) / W
    dh = tf.math.log(prior_H / H)
    dw = tf.math.log(prior_W / W)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def pad_ROIs_value(positive_ROIs, negative_ROIs, ROI_class_ids, deltas,
                   masks, train_ROIs_per_image):
    """Used by Detection target layer in detection_target_graph.
    Zero pads prior deltas and masks to image size.

    # Arguments:
        num_positives: no of positive ROIs.
        num_negatives: no of negative ROIs.
        roi_priors : ROIs from ground truth.
        deltas : refinements to apply to priors.
        masks: proposal masks [batch, train_ROIs_per_image, height, width].

    # Return:
        ROIs: Normalized ground-truth boxes
                     [batch, max_ground_truth_instances, (x1, y1, x2, y2)].
        ROI_class_ids: [batch, train_ROIs_per_image]. Integer class IDs.
        ROI_deltas: [batch, train_ROIs_per_image, (dy, dx, log(dh), log(dw)].
        ROI_masks: [batch, train_ROIs_per_image, height, width].
    """
    ROIs, num_negatives, num_positives = pad_ROIs(positive_ROIs, negative_ROIs,
                                                  train_ROIs_per_image)
    num_samples = num_negatives + num_positives
    ROI_class_ids = tf.pad(ROI_class_ids, [(0, num_samples)])
    deltas = tf.pad(deltas, [(0, num_samples), (0, 0)])
    masks = tf.pad(masks, [[0, num_samples], (0, 0), (0, 0)])
    # masks_zero = tf.Variable(tf.zeros((128, 128, 4)))
    # for i in range(masks.shape[2]):
    #     masks_zero[:, :, i].assign(masks[:, :, i])
    return ROIs, ROI_class_ids, deltas, masks


def pad_ROIs(positive_ROIs, negative_ROIs, train_ROIs_per_image):
    """Zero pads ROIs deltas and masks to image size.

    # Arguments:
        positive_rois:  ROIs with IoU >= 0.5.
        negative_rois: ROIs with IoU <= 0.5.
        train_ROIs_per_image: Number of ROIs per image to
                              feed to classifier/mask heads.
    # Return:
        ROIs: Normalized ground-truth boxes
                     [batch, max_ground_truth_instances, (x1, y1, x2, y2)].
        num_positives: [N] no. of positive ROIs.
        num_negatives: [N] no. of negative ROIs.
    """
    ROIs = tf.concat([positive_ROIs, negative_ROIs], axis=0)
    num_negatives = tf.shape(negative_ROIs)[0]
    train_ROI = train_ROIs_per_image - tf.shape(ROIs)[0]
    num_positives = tf.maximum(train_ROI, 0)
    ROIs = tf.pad(ROIs, [(0, num_positives), (0, 0)])
    return ROIs, num_positives, num_negatives


def compute_largest_overlap(positive_overlaps):
    """Decorator function used to call an maximum positive overlap along
    axis 1.
     """
    def _compute_largest_overlap():
        return tf.argmax(positive_overlaps, axis=1)
    return _compute_largest_overlap


def get_empty_list():
    """Decorator function used to call an empty tensor list.
    """
    return tf.cast(tf.constant([]), tf.int64)


def compute_target_boxes(positive_overlaps, positive_ROIs, boxes,
                         bounding_box_standard_deviation):
    """Final computation of each target bounding boxes based on positive
    overlaps.

    # Arguments:
        positive_overlaps: [batch, N] value of overlaps containing instances.
        positive_ROIs: ROIs with IoU >=0.5.
        boxes: Normalized ground-truth boxes
                     [batch, max_ground_truth_instances, (x1, y1, x2, y2)].

    # Returns:
        ROI_boxes: Normalized ground-truth boxes
                     [batch, max_ground_truth_instances, (x1, y1, x2, y2)]
    """
    ROI_true_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=compute_largest_overlap(positive_overlaps),
        false_fn=get_empty_list)

    ROI_boxes = tf.gather(boxes, ROI_true_box_assignment)
    deltas = refine_bounding_box(positive_ROIs, ROI_boxes)
    deltas /= bounding_box_standard_deviation
    return deltas, ROI_boxes


def compute_target_class_ids(groundtruth_class_ids, positive_overlaps):
    """Final computation of each target class ids based on positive overlaps.

    # Arguments:
        groundtruth_class_ids: [batch, train_ROIs_per_image]. Integer
                               class IDs.
        positive_overlaps: [batch, N] value of overlaps containing instances.

    # Returns:
        ROI_class_ids: [batch, train_ROIs_per_image]. Integer class IDs.
    """
    ROI_true_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=compute_largest_overlap(positive_overlaps),
        false_fn=get_empty_list)
    ROI_class_ids = tf.gather(groundtruth_class_ids, ROI_true_box_assignment)
    return ROI_class_ids


def compute_target_masks(positive_ROIs, ROI_boxes, masks, positive_overlaps,
                         mask_shape, mini_mask):
    """Final computation of each target masks based on positive overlaps.

    # Arguments:
        positive_ROIs: ROIs with IoU >=0.5.
        ROI_priors: ROI from ground  truth.
        mask_shape : Shape of output mask.
        use_mini_mask: Resizes instance masks to a smaller size to reduce.
        positive_overlaps: [batch, N] value of overlaps containing instances.

    # Returns:
        masks: [batch, train_ROIs_per_image, height, width].
    """
    ROI_true_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=compute_largest_overlap(positive_overlaps),
        false_fn=get_empty_list)

    transposed_masks = tf.expand_dims(tf.transpose(masks, [2, 0, 1]), -1)
    ROI_masks = tf.gather(transposed_masks, ROI_true_box_assignment)

    if mini_mask:
        boxes = transform_ROI_coordinates(positive_ROIs, ROI_boxes)
    else:
        boxes = positive_ROIs

    box_ids = tf.range(0, tf.shape(ROI_masks)[0])
    ROI_masks_cast = tf.cast(ROI_masks, tf.float32)
    masks = tf.image.crop_and_resize(ROI_masks_cast, boxes, box_ids,
                                     mask_shape)
    masks = tf.squeeze(masks, axis=3)
    return tf.round(masks)


def transform_ROI_coordinates(boxes, ROI_boxes):
    """Converts ROI boxes to boxes values of same coordinates by retaining
    the shape. Applied when mini mask is set to true.

    # Arguments:
        boxes: Normalized ground-truth boxes
                     [batch, max_ground_truth_instances, (x1, y1, x2, y2)].
        ROI_boxes: Normalized ROI boxes
                     [batch, max_ground_truth_instances, (x1, y1, x2, y2)].

    # Return:
        boxes: Normalized boxes
                     [batch, max_ground_truth_instances, (x1, y1, x2, y2)].
    """
    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=1)
    ROI_y_min, ROI_x_min, ROI_y_max, ROI_x_max = tf.split(ROI_boxes, 4, axis=1)
    H = ROI_y_max - ROI_y_min
    W = ROI_x_max - ROI_x_min
    y_min = (y_min - ROI_y_min) / H
    x_min = (x_min - ROI_x_min) / W
    y_max = (y_max - ROI_y_min) / H
    x_max = (x_max - ROI_x_min) / W
    return tf.concat([y_min, x_min, y_max, x_max], 1)


def remove_zero_padding(proposals, class_ids, boxes, masks):
    """Removes zero boxes from proposals and groundtruth values.

    # Arguments:
        proposals: Normalized proposals
                   [batch, N, (y_min, x_min, y_max, x_max)].
        ground_truth : class_ids, boxes and masks.

    # Return:
        proposals:Normalized proposals
                   [batch, N, (y_min, x_min, y_max, x_max)].
        ground_truth: class_ids, boxes and masks.
    """
    proposals, _ = trim_zeros(proposals)
    boxes, non_zeros = trim_zeros(boxes)
    class_ids = tf.boolean_mask(class_ids, non_zeros)
    masks = tf.gather(masks, tf.where(non_zeros)[:, 0], axis=2)
    return proposals, class_ids, boxes, masks


def check_if_crowded(proposals, class_ids, boxes, masks):
    """Separates crowd instances from non crowd ones and removes the
    zero padding from groundtruth values.

    # Arguments:
        proposals: Normalized proposals
                   [batch, N, (y_min, x_min, y_max, x_max)].
        ground_truth: class_ids, boxes and masks.

    # Return:
        proposals:Normalized proposals
                   [batch, N, (y_min, x_min, y_max, x_max)].
        ground_truth: class_ids, boxes and masks.
    """
    proposals, class_ids, boxes, masks = remove_zero_padding(proposals,
                                                             class_ids,
                                                             boxes, masks)
    is_crowded = tf.where(class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(boxes, is_crowded)
    class_ids = tf.gather(class_ids, non_crowd_ix)
    boxes = tf.gather(boxes, non_crowd_ix)
    masks = tf.gather(masks, non_crowd_ix, axis=2)
    return proposals, class_ids, boxes, masks, crowd_boxes


def compute_ROI_overlaps(proposals, boxes, crowd_boxes,
                         train_ROIs_per_image, ROI_positive_ratio):
    """Computes IOU over the proposals and groundtruth values and segregates
    proposals as positive and negative.

    # Arguments:
        proposals: Normalized proposals
                   [batch, N, (y_min, x_min, y_max, x_max)].
        boxes: refined bounding boxes.
        crowd_boxes : bounding boxes of ground truth.
        overlaps: instances of overlaps.
        train_ROIs_per_image: Number of ROIs per image to feed to
                              classifier/mask heads
        ROI_positive_ratio: Percent of positive ROIs used to train
                            classifier/mask heads

    # Returns:
        positive_overlaps: [batch, N] value of overlaps containing instances
        positive_ROIs: [batch, N, (y_min, x_min, y_max, x_max)]
                       contain instances.
        negative_ROIs: [batch, N, (y_min, x_min, y_max, x_max)] don't
                       contain instances.
    """
    overlaps = compute_IOU(proposals, boxes)
    crowd_overlaps = compute_IOU(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    positive_indices, negative_indices = compute_indices(overlaps,
                                                         no_crowd_bool)
    return gather_ROIs(proposals, positive_indices, negative_indices,
                       train_ROIs_per_image, ROI_positive_ratio, overlaps)


def compute_indices(overlaps, no_crowd_bool):
    """Segregates the ROIs with iou >=0.5 as positive and rest as negative.

    # Arguments:
        overlaps: instances of overlaps.
        no_crowd_bool: overlaps which have less IOU values.

    # Returns:
        positive_indices: indices of positive overlaps.
        negative_indices: indices of negative overlaps.
    """
    ROI_iou_max = tf.reduce_max(overlaps, axis=1)
    positive_ROI_bool = (ROI_iou_max >= 0.5)
    positive_indices = tf.where(positive_ROI_bool)[:, 0]
    negative_indices = tf.where(tf.logical_and(
        ROI_iou_max < 0.5, no_crowd_bool))[:, 0]
    return positive_indices, negative_indices


def gather_ROIs(proposals, positive_indices, negative_indices,
                train_ROIs_per_image, ROI_positive_ratio, overlaps):
    """Computes positive, negative rois and positive_overlaps from the given
    proposals.

    # Arguments:
        proposals: ROIs.
        positive_indices: Indices of positive ROIs.
        negative_indices: Indices of negative ROIs.
        train_rois_per_image: Number of ROIs per image to feed to
                              classifier/mask heads.
        roi_positive_ratio: Percent of positive ROIs used to train
                            classifier/mask heads.

    # Returns:
        positive_overlaps: [batch, N] value of overlaps containing instances.
        positive_ROIs: [batch, N, (y_min, x_min, y_max, x_max)]
                       contain instances.
        negative_ROIs: [batch, N, (y_min, x_min, y_max, x_max)]
                       don't contain instances.
    """
    positive_count = int(train_ROIs_per_image * ROI_positive_ratio)
    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    negative_indices = get_negative_indices(positive_indices,
                                            ROI_positive_ratio,
                                            negative_indices)

    positive_ROIs = tf.gather(proposals, positive_indices)
    negative_ROIs = tf.gather(proposals, negative_indices)
    positive_overlaps = tf.gather(overlaps, positive_indices)
    return positive_overlaps, positive_ROIs, negative_ROIs


def get_negative_indices(positive_indices, ROI_positive_ratio,
                         negative_indices):
    """Computes negative indices of the proposals (ROIs) which do
    not contain any instances by maintaining the necessary ROI positive ratio,
    default is 0.33.

    # Arguments:
        positive_indices: Indices of positive ROIs
        ROI_positive_ratio: Percent of positive ROIs used to train
                            classifier/mask heads
        negative_indices: Indices of negative ROIs

    # Returns:
        negative_indices: List of overlaps between the boxes_a wrt to boxes_b
    """
    positive_count = tf.shape(positive_indices)[0]
    ratio = 1.0 / ROI_positive_ratio
    negative_count = tf.cast(ratio * tf.cast(positive_count, tf.float32),
                             tf.int32) - positive_count
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]

    return negative_indices


def compute_IOU(boxes_a, boxes_b):
    """Calculates intersection over union for given two inputs.

    # Arguments:
        boxes_a: Normalised proposals [batch, N, (y_min, x_min, y_max, x_max)]
        boxes_b: Refined bounding boxes [batch, max_ground_truth_instances,
                 (x1, y1, x2, y2)]

    # Returns:
        Overlaps: List of overlaps between the boxes_a wrt to boxes_b
    """
    box_a = tf.reshape(tf.tile(tf.expand_dims(boxes_a, 1), [1, 1,
                                                            tf.shape(boxes_b)
                                                            [0]]),
                       [-1, 4])
    box_b = tf.tile(boxes_b, [tf.shape(boxes_a)[0], 1])
    overlap, union = compute_overlap_union(box_a, box_b)
    iou = overlap / union
    overlaps = tf.reshape(iou, [tf.shape(boxes_a)[0], tf.shape(boxes_b)[0]])

    return overlaps


def compute_overlap_union(box_a, box_b):
    """Computes overlaps and unions for given two inputs.

    # Arguments:
        boxes_a : Normalised proposals [batch, N, (y_min, x_min, y_max, x_max)]
        boxes_b : Refined bounding boxes [batch, max_ground_truth_instances,
                  (x1, y1, x2, y2)]

    # Returns:
        Overlaps: List of overlaps between the boxes_a wrt to boxes_b
    """
    boxa_y_min, boxa_x_min, boxa_y_max, boxa_x_max = tf.split(box_a, 4, axis=1)
    boxb_y_min, boxb_x_min, boxb_y_max, boxb_x_max = tf.split(box_b, 4, axis=1)
    y_min = tf.maximum(boxa_y_min, boxb_y_min)
    x_min = tf.maximum(boxa_x_min, boxb_x_min)
    y_max = tf.minimum(boxa_y_max, boxb_y_max)
    x_max = tf.minimum(boxa_x_max, boxb_x_max)
    overlap = tf.maximum(x_max - x_min, 0) * tf.maximum(y_max - y_min, 0)

    box_a_area = (boxa_y_max - boxa_y_min) * (boxa_x_max - boxa_x_min)
    box_b_area = (boxb_y_max - boxb_y_min) * (boxb_x_max - boxb_x_min)
    union = box_a_area + box_b_area - overlap

    return overlap, union


def compute_targets_from_groundtruth_values(proposals, groundtruth_class_ids,
                                            groundtruth_boxes,
                                            groundtruth_masks,
                                            train_ROIs_per_image,
                                            ROI_positive_ratio, mask_shape,
                                            use_mini_mask,
                                            bounding_box_std_dev):
    """Used by Detection Target Layer and apply it in batches. Generates
    target box refinement, class_ids, and masks for proposals.

    # Arguments:
        proposals: Normalized proposals
                   [batch, N, (y_min, x_min, y_max, x_max)]
        prior_class_ids: [batch, max_ground_truth_instances] Integer class IDs
        prior_boxes: Normalized ground-truth boxes
                     [batch, max_ground_truth_instances, (x1, y1, x2, y2)]
        prior_masks: [batch, height, width, max_ground_truth_instances] of
                     Boolean type
        train_ROIs_per_image : Number of ROIs per image to feed to
                               classifier/mask heads
        ROI_positive_ratio : Percent of positive ROIs used to train
                             classifier/mask heads
        mask_shape : Shape of output mask
        use_mini_mask : Resizes instance masks to a smaller size to reduce
                        memory load.
        bounding_box_std_dev : Bounding box refinement standard deviation for
                               final detections

    # Returns:
        ROIs: [batch, train_ROIs_per_image, (y_min, x_min, y_max, x_max)]
        ROI_class_ids: [batch, train_ROIs_per_image]. Integer class IDs
        ROI_deltas: [batch, train_ROIs_per_image, (dy, dx, log(dh), log(dw)]
        ROI_masks: [batch, train_ROIs_per_image, height, width]
    """

    # Removes the padded zeros from the groundtruth
    # values and also separates the crowd from non crowd instances.

    # Check of assert is necessary
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="ROI_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    proposals, refined_class_ids, refined_boxes, refined_masks, \
        crowd_boxes = check_if_crowded(proposals, groundtruth_class_ids,
                                       groundtruth_boxes, groundtruth_masks)

    # Computes positive indices of proposals and positive, negative rois based
    # on the generated proposals
    positive_overlaps, positive_ROIs, negative_ROIs = \
        compute_ROI_overlaps(proposals, refined_boxes, crowd_boxes,
                             train_ROIs_per_image, ROI_positive_ratio)

    # Calculates the largest positive overlaps with the proposal and positive
    # ROI, then picks that
    # instances from groundtruth images for training alone
    # Calculates the top boxes, class_ids and masks based on largest
    # positive overlaps
    ROI_deltas, ROI_boxes = compute_target_boxes(positive_overlaps,
                                                 positive_ROIs,
                                                 refined_boxes,
                                                 bounding_box_std_dev)
    ROI_class_ids = compute_target_class_ids(refined_class_ids,
                                             positive_overlaps)

    ROI_masks = compute_target_masks(positive_ROIs, ROI_boxes, refined_masks,
                                     positive_overlaps, mask_shape,
                                     use_mini_mask)

    # Padding operation on the modified groundtruth values
    ROIs, ROI_class_ids, ROI_deltas, ROI_masks = \
        pad_ROIs_value(positive_ROIs, negative_ROIs, ROI_class_ids,
                       ROI_deltas, ROI_masks, train_ROIs_per_image)

    return ROIs, ROI_class_ids, ROI_deltas, ROI_masks


class DetectionTargetLayer(Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
       and masks for each.

    # Arguments:
        proposals: Normalized proposals
                   [batch, N, (y_min, x_min, y_max, x_max)]
        prior_class_ids: [batch, max_ground_truth_instances] Integer class IDs.
        prior_boxes: Normalized ground-truth boxes
                     [batch, max_ground_truth_instances, (x1, y1, x2, y2)]
        prior_masks: [batch, height, width, max_ground_truth_instances] of
                     Boolean type

    # Returns:
        ROIs: [batch, TRAIN_ROIS_PER_IMAGE, (y_min, x_min, y_max, x_max)]
        target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
        target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
        target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]

    # Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, images_per_gpu, mask_shape, train_ROIs_per_image,
                 ROI_positive_ratio, bounding_box_std_dev, use_mini_mask,
                 batch_size, **kwargs):
        super().__init__(**kwargs)
        self.images_per_gpu = images_per_gpu
        self.mask_shape = mask_shape
        self.train_ROIs_per_image = train_ROIs_per_image
        self.ROI_positive_ratio = ROI_positive_ratio
        self.bounding_box_std_dev = bounding_box_std_dev
        self.use_mini_mask = use_mini_mask
        self.batch_size = batch_size

    def call(self, inputs):
        proposals, prior_class_ids, prior_boxes, prior_masks = inputs
        names = ['ROIs', 'target_class_ids', 'target_bounding_box',
                 'target_mask']
        outputs = slice_batch([proposals, prior_class_ids, prior_boxes,
                               prior_masks],
                              [self.train_ROIs_per_image,
                               self.ROI_positive_ratio,
                               self.mask_shape, self.use_mini_mask,
                               tf.cast(self.bounding_box_std_dev,
                                       dtype=tf.float32)],
                              compute_targets_from_groundtruth_values,
                              self.images_per_gpu, names=names)
        return outputs
