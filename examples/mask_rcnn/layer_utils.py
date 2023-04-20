import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


def refine_bbox(box, prior_box):  # TODO: rename refine_box
    """Compute refinement needed to transform box to groundtruth_box.

    # Arguments:
        box: [N, (y_min, x_min, y_max, x_max)]
        prior_box: Ground-truth box [N, (y_min, x_min, y_max, x_max)]
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


def trim_zeros(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
       are padded with zeros. This removes zero boxes.

    # Arguments:
        boxes: [N, 4] matrix of boxes.
        non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def slice_batch(inputs, constants, function, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
       computation graph and then combines the results.

    # Arguments:
        inputs: list of tensors. All must have the same first dimension length
        graph_function: Function that returns a tensor that's part of a graph.
        batch_size: number of slices to divide the data into.
        names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    outputs = []
    for sample_arg in range(batch_size):

        input_slices = []
        for x in inputs:
            input_slice = x[sample_arg]
            input_slices.append(input_slice)
        for y in constants:
            input_slices.append(y)

        output_slice = function(*input_slices)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    results = []
    for output, name in zip(outputs, names):
        result = tf.stack(output, axis=0, name=name)
        results.append(result)

    if len(results) == 1:
        results = results[0]

    return results


def apply_box_delta(boxes, deltas):
    """Applies the given deltas to the given boxes.

    # Arguments:
        boxes: [N, (y_min, x_min, y_max, x_max)] boxes to update
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    H = boxes[:, 2] - boxes[:, 0]
    W = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + (0.5 * H)
    center_x = boxes[:, 1] + (0.5 * W)

    center_y = center_y + (deltas[:, 0] * H)
    center_x = center_x + (deltas[:, 1] * W)
    H = H * tf.exp(deltas[:, 2])
    W = W * tf.exp(deltas[:, 3])

    y_min = center_y - (0.5 * H)
    x_min = center_x - (0.5 * W)
    y_max = y_min + H
    x_max = x_min + W
    result = tf.stack([y_min, x_min, y_max, x_max], axis=1,
                      name='apply_box_deltas_out')    # TODO: remove names for tensors

    return result


def clip_boxes(boxes, window):
    """Clips boxes for given window size.

    # Arguments:
        boxes: [N, (y_min, x_min, y_max, x_max)]
        window: [4] in the form y_min, x_min, y_max, x_max
    """
    windows = tf.split(window, 4)
    window_y_min = windows[0]
    window_x_min = windows[1]
    window_y_max = windows[2]
    window_x_max = windows[3]

    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=1)
    y_min = tf.maximum(tf.minimum(y_min, window_y_max), window_y_min)
    x_min = tf.maximum(tf.minimum(x_min, window_x_max), window_x_min)
    y_max = tf.maximum(tf.minimum(y_max, window_y_max), window_y_min)
    x_max = tf.maximum(tf.minimum(x_max, window_x_max), window_x_min)

    clipped = tf.concat([y_min, x_min, y_max, x_max], axis=1,
                        name='clipped_boxes')
    clipped.set_shape((clipped.shape[0], 4))

    return clipped


def get_top_detections(scores, keep, nms_keep, detection_max_instances):
    """Select the the detection with highest score values and
    retain it. And conversion from sparse to dense(non crypted form).

    # Arguments:
        scores: [N] Predicted target class values
        keep: rois after supression
        nms_keep: rois after nms
        detection_max_instances: Max number of final detections
    """
    keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                tf.expand_dims(nms_keep, 0))

    keep = tf.sparse.to_dense(keep)[0]
    roi_count = detection_max_instances
    class_scores_keep = tf.gather(scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]

    return tf.gather(keep, top_ids)


def NMS_map(class_ids, scores, ROIs, keep, detection_max_instances,
            detection_nms_threshold, unique_class_id):
    """Mapping function used to greedily select a subset of
    bounding boxes in descending order of score. The output of
    is a set of integers indexing into the input collection of
    bounding boxes representing the selected boxes.

    # Arguments:
        box_data: contains class ids, scores and rois
        keep: rois after suppression
        unique_class_id : unique class ids [1x No. of classes]
        detection_max_instances: Max number of final detections
        detection_nms_threshold: Non-maximum suppression threshold for detection
    # Returns:
        class_keep: detected instances kept after nms
    """
    ids = tf.where(tf.equal(class_ids, unique_class_id))[:, 0]

    class_keep = tf.image.non_max_suppression(tf.gather(ROIs, ids), tf.gather(scores, ids),
                                              max_output_size=detection_max_instances,
                                              iou_threshold=detection_nms_threshold)
    class_keep = tf.gather(keep, tf.gather(ids, class_keep))
    gap = detection_max_instances - tf.shape(class_keep)[0]
    class_keep = tf.pad(class_keep, [(0, gap)], mode='CONSTANT', constant_values=-1)
    class_keep.set_shape([detection_max_instances])

    return class_keep


def NMS_map_call(pre_nms_class_ids, pre_nms_scores, pre_nms_ROIs, keeps, max_instances,
                 nms_threshold, unique_pre_nms_class_ids):
    """Decorator function used to call to NMS_map function.
    """
    def _NMS_map_call(class_id):
        return NMS_map(pre_nms_class_ids, pre_nms_scores, pre_nms_ROIs, keeps, max_instances,
                       nms_threshold, class_id)

    return tf.map_fn(_NMS_map_call, unique_pre_nms_class_ids, dtype=tf.int64)


def apply_NMS(class_ids, scores, refined_ROIs, keep,
              detection_max_instances, detection_nms_threshold):
    """Function used perform NMs on the detected instances per detected instances.

    # Arguments:
        class_ids  : [1x N] array values
        scores : probability scores for all classes
        refined_rois : rois after NMS
        keep : rois after suppression
        detection_max_instances: Max no. of instances the model can detect.
                                 Default value is 100
        detection_nms_threshold: Threshold value below which instances are ignored.
    # Return:
        nms_keep: Detected instances kept after performing nms.
    """
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(scores, keep)
    pre_nms_ROIs = tf.gather(refined_ROIs, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]
    keeps = keep

    nms_keep = NMS_map_call(pre_nms_class_ids, pre_nms_scores, pre_nms_ROIs, keeps,
                            detection_max_instances, detection_nms_threshold,
                            unique_pre_nms_class_ids)
    return merge_results(nms_keep)


def merge_results(nms_keep):
    """Used by top detection layer in apply_NMS to reshape keep values
    and remove negative indices if any.

    # Arguments:
        nms_keep : rois after nms
    """
    nms_keep = tf.reshape(nms_keep, [-1])

    return tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])


def filter_low_confidence(class_scores, keep, detection_min_confidence):
    """Function used to compute proposals with highest confidence
    and filter the lower ones based on the given threshold value.

    # Arguments:
        class_scores: probability scores for all classes
        keep : rois after supression
        detection_min_confidence: Minimum probability value to accept a detected instance
    # Return:
        keep: rois after thresholding
    """
    confidence = tf.where(class_scores >= detection_min_confidence)[:, 0]
    keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                tf.expand_dims(confidence, 0))

    return tf.sparse.to_dense(keep)[0]


def zero_pad_detections(detections, detection_max_instances):
    """Function used to match keep detection shape same
    as the max instance by zero padding.

    # Arguments:
        detections: num of detections
        detection_max_instances: Max number of final detections
    # Return:
        detections: num of detections after zero padding.
    """
    gap = detection_max_instances - tf.shape(detections)[0]

    return tf.pad(detections, [(0, gap), (0, 0)], 'CONSTANT')


def compute_delta_specific(probs, deltas):
    """Used by Detection Layer to compute the top class_ids,
    scores and deltas from the normalised proposals.

    # Arguments:
        probs: Normalized proposed classes
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    # Returns:
        class_ids: [N] Top class ids detected
        class_scores: [N] Scores of the top detected classes
        delta specific: [N, (dy, dx, log(dh), log(dw))]
    """
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    indices = tf.stack([tf.range(tf.shape(probs)[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    deltas_specific = tf.gather_nd(deltas, indices)

    return class_ids, class_scores, deltas_specific


def compute_refined_ROIs(ROIs, deltas, windows):
    """Used by Detection Layer to apply changes to bbox and
    clip the bboxes to specific window size.

    # Arguments:
        rois: Proposed regions form the  proposal layer
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        windows: [1x4] Default None
    # Returns:
        rois: Normalized proposals
    """
    refined_ROIs = apply_box_delta(ROIs, deltas)
    refined_ROIs = clip_boxes(refined_ROIs, windows)

    return refined_ROIs


def compute_keep(class_ids, class_scores, refined_ROIs, detection_min_confidence,
                 detection_max_instances, detection_nms_threshold):
    """Used by Detection Layer to keep proposed regions and
    class_ids after non-maximum suppresion.

    # Arguments:
        class_ids: predicted class ids
        class_scores: probability scores for all classes
        refined_rois: rois after NMS
        detection_min_confidence: Minimum probability value
                                  to accept a detected instance
        detection_max_instances: Max number of final detections
        detection_nms_threshold: Non-maximum suppression threshold for detection
    # Returns:
        keep: class idds after NMS
    """
    keep = tf.where(class_ids > 0)[:, 0]
    if detection_min_confidence:
        keep = filter_low_confidence(class_scores, keep, detection_min_confidence)

    nms_keep = apply_NMS(class_ids, class_scores, refined_ROIs, keep,
                         detection_max_instances, detection_nms_threshold)
    keep = get_top_detections(class_scores, keep, nms_keep, detection_max_instances)

    return keep


def refine_detections(ROIs, probs, deltas, bbox_std_dev, windows, detection_min_confidence,
                      detection_max_instances, detection_nms_threshold):
    """Used by Detection Layer to keep detections with high scores and confidence. Also remove
    the detections with low confidence and scores.

    # Arguments:
        rois: Proposed regions form the  proposal layer
        probs: Normalized proposed classes
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        bbox_std_dev : Bounding box refinement standard deviation for final detections
        windows:  [1x4] Default None
        detection_min_confidence: Minimum probability value to accept a detected instance,
                                 ROIs below this threshold are skipped
        detection_max_instances: Max number of final detections
        detection_nms_threshold: Non-maximum suppression threshold for detection
    # Return:
        detections: num of detections after zero padding.
    """
    class_ids, class_scores, deltas_specific = compute_delta_specific(probs, deltas)
    refined_ROIs = compute_refined_ROIs(ROIs, deltas_specific * bbox_std_dev, windows)
    keep = compute_keep(class_ids, class_scores, refined_ROIs, detection_min_confidence,
                        detection_max_instances, detection_nms_threshold)

    gather_refined_ROIs = tf.gather(refined_ROIs, keep)
    gather_class_ids = tf.cast(tf.gather(class_ids, keep), dtype=tf.float32)[..., tf.newaxis]
    gather_class_scores = tf.gather(class_scores, keep)[..., tf.newaxis]

    detections = tf.concat([gather_refined_ROIs, gather_class_ids, gather_class_scores],
                           axis=1)

    return zero_pad_detections(detections, detection_max_instances)


def trim_anchors_by_score(scores, deltas, anchors, images_per_gpu, pre_nms_limit):
    """Function used to select fixed number of anchors before nms.

    # Arguments:
        scores: [N] Predicted target class values
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        anchors: [batch, num_anchors, (y_min, x_min, y_max, x_max)] anchors
                 in normalized coordinates
        images_per_gpu: Number of images to train with on each GPU
        pre_nms_limit: type int, ROIs kept to keep
                       before non-maximum suppression
    # Returns:
        scores: [pre_nms_limit] Predicted target class values
        deltas: [pre_nms_limit, (dy, dx, log(dh), log(dw))] refinements to apply
        anchors: [batch, pre_nms_limit, (y_min, x_min, y_max, x_max)]
    """
    pre_nms_limit = tf.minimum(pre_nms_limit, tf.shape(anchors)[1])
    indices = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                          name='top_anchors').indices
    scores = slice_batch([scores, indices], [], tf.gather, images_per_gpu)
    deltas = slice_batch([deltas, indices], [], tf.gather, images_per_gpu)
    pre_nms_anchors = slice_batch([anchors, indices], [], tf.gather,
                                  images_per_gpu, names=['pre_nms_anchors'])

    return scores, deltas, pre_nms_anchors


def apply_box_deltas(pre_nms_anchors, deltas, images_per_gpu):
    """Function used to apply refinement to anchors based on
    delta values in slices.

    # Arguments:
        pre_nms_anchors: [N, (y_min, x_min, y_max, x_max)] boxes to update
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        images_per_gpu: Number of images to train with on each GPU
    # Returns:
        boxes: [N, (y_min, x_min, y_max, x_max)]
    """
    boxes = slice_batch([pre_nms_anchors, deltas], [], apply_box_delta,
                        images_per_gpu, names=['refined_anchors'])

    return boxes


def clip_image_boundaries(boxes, images_per_gpu):
    """Function used to clip boxes to the given window size
    in this case the normalised image boundaries.

    # Arguments:
        boxes: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        images_per_gpu: Number of images to train with on each GPU
    # Returns:
        boxes: [N, (dy, dx, log(dh), log(dw))]
    """
    window_size = np.array([0, 0, 1, 1], dtype=np.float32)
    boxes = slice_batch(boxes, [window_size], clip_boxes, images_per_gpu,
                        names=['refined_anchors_clipped'])

    return boxes


def compute_NMS(boxes, scores, proposal_count, nms_threshold):
    """Function used to compute non-max suppression on the image
    and refining the shape of the proposals.

    # Arguments:
        boxes: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        scores: [N] Predicted target class values
        proposal_count: Max number of proposals
        nms_threshold: Non-maximum suppression threshold for detection
    # Returns:
        proposals: Normalized proposals
                   [batch, N, (y_min, x_min, y_max, x_max)]
    """
    indices = tf.image.non_max_suppression(
        boxes, scores, proposal_count,
        nms_threshold, name='rpn_non_max_suppression')
    proposals = tf.gather(boxes, indices)
    proposals_shape = tf.shape(proposals)[0]
    padding = tf.maximum(proposal_count - proposals_shape, 0)
    proposals = tf.pad(proposals, [(0, padding), (0, 0)])

    return proposals


def pad_ROIs_value(positive_ROIs, negative_ROIs, ROI_class_ids, deltas,
                   masks, train_ROIs_per_image):
    """Used by Detection target layer in detection_target_graph.
    Zero pad prior deltas and masks to image size.

    # Arguments:
        num_positives: no of positive ROIs
        num_negatives: no of negative ROIs
        roi_priors : ROIs from ground truth
        deltas : refinements to apply to priors
        masks: proposal masks [batch, TRAIN_ROIS_PER_IMAGE, height, width]
    # Return:
        ROIs: Normalized ground-truth boxes
                     [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
        ROI_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs
        ROI_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
        ROI_masks: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
    """
    ROIs, num_negatives, num_positives = pad_ROIs(positive_ROIs, negative_ROIs,
                                                  train_ROIs_per_image)
    num_samples = num_negatives + num_positives
    ROI_class_ids = tf.pad(ROI_class_ids, [(0, num_samples)])
    deltas = tf.pad(deltas, [(0, num_samples), (0, 0)])
    masks = tf.pad(masks, [[0, num_samples], (0, 0), (0, 0)])
    return ROIs, ROI_class_ids, deltas, masks


def pad_ROIs(positive_ROIs, negative_ROIs, train_ROIs_per_image):
    """Zero pad ROIs deltas and masks to image size.

    # Arguments:
        positive_rois:  ROIs with IOU >= 0.5
        negative_rois: ROIs with IOU <= 0.5
        train_rois_per_image: Number of ROIs per image to
                              feed to classifier/mask heads
    # Return:
        ROIs: Normalized ground-truth boxes
                     [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
        num_positives: [N] no. of positive ROIs
        num_negatives: [N] no. of negative ROIs
    """
    ROIs = tf.concat([positive_ROIs, negative_ROIs], axis=0)
    num_negatives = tf.shape(negative_ROIs)[0]
    train_ROI = train_ROIs_per_image - tf.shape(ROIs)[0]
    num_positives = tf.maximum(train_ROI, 0)
    ROIs = tf.pad(ROIs, [(0, num_positives), (0, 0)])
    return ROIs, num_positives, num_negatives


def compute_largest_overlap(positive_overlaps):
    """Decorator function used to call an maximum positive overlap along axis 1.
     """

    def _compute_largest_overlap():
        return tf.argmax(positive_overlaps, axis=1)

    return _compute_largest_overlap


def get_empty_list():
    """Decorator function used to call an empty tensor list.
    """
    return tf.cast(tf.constant([]), tf.int64)


def compute_target_boxes(positive_overlaps, positive_ROIs, boxes,
                         bbox_standard_deviation):
    """Final computation of each target bounding boxes based on positive overlaps.

    # Arguments:
        positive_overlaps: [batch, N] value of overlaps containing instances
        positive_rois: ROIs with IOU >=0.5
        boxes: Normalized ground-truth boxes
                     [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    # Returns:
        ROI_boxes: Normalized ground-truth boxes
                     [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    """
    ROI_true_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=compute_largest_overlap(positive_overlaps), false_fn=get_empty_list)

    ROI_boxes = tf.gather(boxes, ROI_true_box_assignment)
    deltas = refine_bbox(positive_ROIs, ROI_boxes)
    deltas /= bbox_standard_deviation

    return deltas, ROI_boxes


def compute_target_class_ids(gt_class_ids, positive_overlaps):
    """Final computation of each target class ids based on positive overlaps.

    # Arguments:
        gt_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs
        positive_overlaps: [batch, N] value of overlaps containing instances
    # Returns:
        ROI_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs
    """
    ROI_true_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=compute_largest_overlap(positive_overlaps), false_fn=get_empty_list)
    ROI_class_ids = tf.gather(gt_class_ids, ROI_true_box_assignment)

    return ROI_class_ids


def compute_target_masks(positive_ROIs, ROI_boxes, masks, positive_overlaps, mask_shape,
                         mini_mask):
    """Final computation of each target masks based on positive overlaps.

    # Arguments:
        positive_rois: ROIs with IOU >=0.5
        roi_priors: ROI from ground  truth
        mask_shape : Shape of output mask
        use_mini_mask: Resizes instance masks to a smaller size to reduce
        positive_overlaps: [batch, N] value of overlaps containing instances
    # Returns:
        masks: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
    """
    ROI_true_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=compute_largest_overlap(positive_overlaps), false_fn=get_empty_list)

    transposed_masks = tf.expand_dims(tf.transpose(masks, [2, 0, 1]), -1)
    ROI_masks = tf.gather(transposed_masks, ROI_true_box_assignment)
    boxes = transform_ROI_coordinates(positive_ROIs, ROI_boxes) if \
        mini_mask else positive_ROIs

    box_ids = tf.range(0, tf.shape(ROI_masks)[0])
    ROI_masks_cast = tf.cast(ROI_masks, tf.float32)
    masks = tf.image.crop_and_resize(ROI_masks_cast, boxes, box_ids, mask_shape)
    masks = tf.squeeze(masks, axis=3)

    return tf.round(masks)


def transform_ROI_coordinates(boxes, ROI_boxes):
    """Function used to convert ROI boxes to boxes values of
    same coordinates by retaining the shape. Applied when mini
    mask is set to true.

    # Arguments:
        boxes: Normalized ground-truth boxes
                     [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
        roi_boxes: Normalized roi boxes
                     [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    # Return:
        boxes: Normalized boxes
                     [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    """
    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=1)
    ROI_y_min, ROI_x_min, ROI_y_max, ROI_x_max = tf.split(ROI_boxes, 4,
                                                          axis=1)
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
                   [batch, N, (y_min, x_min, y_max, x_max)]
        ground_truth : class_ids, boxes and masks
    # Return:
        proposals:Normalized proposals
                   [batch, N, (y_min, x_min, y_max, x_max)]
        ground_truth: class_ids, boxes and masks
    """
    proposals, _ = trim_zeros(proposals, name='trim_proposals')
    boxes, non_zeros = trim_zeros(boxes, name='trim_prior_boxes')
    class_ids = tf.boolean_mask(class_ids, non_zeros,
                                name='trim_prior_class_ids')
    masks = tf.gather(masks, tf.where(non_zeros)[:, 0], axis=2,
                      name='trim_prior_masks')

    return proposals, class_ids, boxes, masks


def check_if_crowded(proposals, class_ids, boxes, masks):
    """Function to separate crowd instances from non crowd ones
    and remove the zero padding from groundtruth values.

    # Arguments:
        proposals: Normalized proposals
                   [batch, N, (y_min, x_min, y_max, x_max)]
        ground_truth: class_ids, boxes and masks
    # Return:
        proposals:Normalized proposals
                   [batch, N, (y_min, x_min, y_max, x_max)]
        ground_truth: class_ids, boxes and masks
    """
    proposals, class_ids, boxes, masks = remove_zero_padding(proposals, class_ids,
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
    """Compute IOU over the proposals and groundtruth values and segregate proposals
    as positive and negative.

    # Arguments:
        proposals: Normalized proposals
                   [batch, N, (y_min, x_min, y_max, x_max)]
        boxes: refined bboxes
        crowd_boxes : bboxes of ground truth
        overlaps: instances of overlaps
        train_rois_per_image: Number of ROIs per image to feed to classifier/mask heads
        roi_positive_ratio: Percent of positive ROIs used to train classifier/mask heads
    # Returns:
        positive_overlaps: [batch, N] value of overlaps containing instances
        positive_ROIs: [batch, N, (y_min, x_min, y_max, x_max)] contain instances
        negative_ROIs: [batch, N, (y_min, x_min, y_max, x_max)] don't contain instances
    """
    overlaps = compute_IOU(proposals, boxes)
    crowd_overlaps = compute_IOU(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    positive_indices, negative_indices = compute_indices(overlaps, no_crowd_bool)

    return gather_ROIs(proposals, positive_indices, negative_indices,
                       train_ROIs_per_image, ROI_positive_ratio, overlaps)


def compute_indices(overlaps, no_crowd_bool):
    """Function used to segregate the ROIs with iou >=0.5 as positive and
    rest as negative.

    # Arguments:
        overlaps: instances of overlaps
        no_crowd_bool: overlaps which have less IOU values
    # Returns:
        positive_indices: indices of positive overlaps
        negative_indices: indices of negative overlaps
    """
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    negative_indices = tf.where(tf.logical_and(
        roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    return positive_indices, negative_indices


def gather_ROIs(proposals, positive_indices, negative_indices,
                train_ROIs_per_image, ROI_positive_ratio, overlaps):
    """Compute positive, negative rois and positive_overlaps
    from the given proposals.

    # Arguments:
        proposals: rois
        positive_indices: Indices of positive ROIs
        negative_indices: Indices of negative ROIs
        train_rois_per_image: Number of ROIs per image to feed to classifier/mask heads
        roi_positive_ratio: Percent of positive ROIs used to train classifier/mask heads
    # Returns:
        positive_overlaps: [batch, N] value of overlaps containing instances
        positive_ROIs: [batch, N, (y_min, x_min, y_max, x_max)] contain instances
        negative_ROIs: [batch, N, (y_min, x_min, y_max, x_max)] don't contain instances
    """
    positive_count = int(train_ROIs_per_image * ROI_positive_ratio)
    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    negative_indices = get_negative_indices(positive_indices, ROI_positive_ratio,
                                            negative_indices)

    positive_ROIs = tf.gather(proposals, positive_indices)
    negative_ROIs = tf.gather(proposals, negative_indices)
    positive_overlaps = tf.gather(overlaps, positive_indices)

    return positive_overlaps, positive_ROIs, negative_ROIs


def get_negative_indices(positive_indices, ROI_positive_ratio, negative_indices):
    """Compute negative indices of the proposals (ROIs) which do
    not contain any instances by maintaining the necessary ROI positive ratio,
    default is 0.33.

    # Arguments:
        positive_indices: Indices of positive ROIs
        roi_positive_ratio: Percent of positive ROIs used to train classifier/mask heads
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
    """Function used to calculate intersection over union for given two inputs.

    # Arguments:
        boxes_a: Normalised proposals [batch, N, (y_min, x_min, y_max, x_max)]
        boxes_b: Refined bboxes [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    # Returns:
        Overlaps: List of overlaps between the boxes_a wrt to boxes_b
    """
    box_a = tf.reshape(tf.tile(tf.expand_dims(boxes_a, 1), [1, 1, tf.shape(boxes_b)[0]]),
                       [-1, 4])
    box_b = tf.tile(boxes_b, [tf.shape(boxes_a)[0], 1])
    overlap, union = compute_overlap_union(box_a, box_b)
    iou = overlap / union
    overlaps = tf.reshape(iou, [tf.shape(boxes_a)[0], tf.shape(boxes_b)[0]])

    return overlaps


def compute_overlap_union(box_a, box_b):
    """Function used to compute overlaps and unions for given two inputs.

    # Arguments:
        boxes_a : Normalised proposals [batch, N, (y_min, x_min, y_max, x_max)]
        boxes_b : Refined bboxes [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
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


def compute_targets_from_groundtruth_values(proposals, gt_class_ids, gt_boxes, gt_masks,
                                            train_ROIs_per_image, ROI_positive_ratio, mask_shape,
                                            use_mini_mask, bbox_std_dev):
    """Used by Detection Target Layer and apply it in batches. Generates
    target box refinement, class_ids, and masks for proposals.

    # Arguments:
        proposals: Normalized proposals
                   [batch, N, (y_min, x_min, y_max, x_max)]
        prior_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
        prior_boxes: Normalized ground-truth boxes
                     [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
        prior_masks: [batch, height, width, MAX_GT_INSTANCES] of Boolean type
        train_rois_per_image : Number of ROIs per image to feed to classifier/mask heads
        roi_positive_ratio : Percent of positive ROIs used to train classifier/mask heads
        mask_shape : Shape of output mask
        use_mini_mask : Resizes instance masks to a smaller size to reduce
                        memory load.
        bbox_std_dev : Bounding box refinement standard deviation for final detections
    # Returns:
        ROIs: [batch, TRAIN_ROIS_PER_IMAGE, (y_min, x_min, y_max, x_max)]
        ROI_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs
        ROI_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
        ROI_masks: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
    """

    # Removes the padded zeros from the groundtruth
    # values and also separates the crowd from non crowd instances.
    # if tf.shape(proposals)[0] > 0:
    #     proposals = tf.identity(proposals)
    #     proposals, refined_class_ids, refined_boxes, refined_masks, \
    #         crowd_boxes = check_if_crowded(proposals, gt_class_ids,
    #                                        gt_boxes, gt_masks)
    # else:
    #     refined_class_ids, refined_boxes, refined_masks, \
    #       crowd_boxes = None, None, None, None

    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)
    
    proposals, refined_class_ids, refined_boxes, refined_masks, \
        crowd_boxes = check_if_crowded(proposals, gt_class_ids,
                                       gt_boxes, gt_masks)

    # Computes positive indices of proposals and positive, negative rois based on the generated proposals
    positive_overlaps, positive_ROIs, negative_ROIs = \
        compute_ROI_overlaps(proposals, refined_boxes, crowd_boxes,
                             train_ROIs_per_image, ROI_positive_ratio)

    # Calculates the largest positive overlaps with the proposal and positive ROI, then picks that
    # instances from groundtruth images for training alone
    # Calculates the top boxes, class_ids and masks based on largest positive overlaps
    ROI_deltas, ROI_boxes = compute_target_boxes(positive_overlaps, positive_ROIs,
                                                 refined_boxes, bbox_std_dev)
    ROI_class_ids = compute_target_class_ids(refined_class_ids, positive_overlaps)

    ROI_masks = compute_target_masks(positive_ROIs, ROI_boxes, refined_masks, positive_overlaps,
                                     mask_shape, use_mini_mask)

    # Padding operation on the modified groundtruth values
    ROIs, ROI_class_ids, ROI_deltas, ROI_masks = \
        pad_ROIs_value(positive_ROIs, negative_ROIs, ROI_class_ids,
                       ROI_deltas, ROI_masks, train_ROIs_per_image)

    return ROIs, ROI_class_ids, ROI_deltas, ROI_masks


def compute_ROI_level(boxes, image_shape):
    """Used by PyramidROIAlign to compute ROI levels and scaled area of the images.

    # Arguments:
        boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
                coordinates. Possibly padded with zeros if not enough
                boxes to fill the array
        image_shape: shape of image
    """
    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=2)
    H = y_max - y_min
    W = x_max - x_min
    scaled_area = compute_scaled_area(H, W, image_shape)
    roi_level = compute_max_ROI_level(scaled_area)

    return tf.squeeze(roi_level, 2)


def compute_max_ROI_level(scaled_area):
    """Used by compute_ROI_level to calculate the ROI level at each feature map.

    # Arguments:
        scaled_area: area of image scaled
    """
    roi_level = tf.experimental.numpy.log2(scaled_area)
    cast_roi_level = tf.cast(tf.round(roi_level), tf.int32)
    max_roi_level = tf.maximum(2, 4 + cast_roi_level)
    roi_level = tf.minimum(5, max_roi_level)
    return roi_level


def compute_scaled_area(H, W, image_shape):
    """Used by compute_ROI_level to obtain scaled area of the image.

    # Arguments:
        H: height of image
        W: width of iamge
        image_area: area of image
    """
    image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
    image_area_scaled = 224.0 / tf.sqrt(image_area)
    squared_area = tf.sqrt(H * W)
    scaled_area = squared_area / image_area_scaled

    return scaled_area


def apply_ROI_pooling(roi_level, boxes, feature_maps, pool_shape):
    """Used by PyramidROIAlign to pool all the feature maps and bbox at each level.

    # Arguments:
        roi_level
        boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
                coordinates. Possibly padded with zeros if not enough
                boxes to fill the array
        features_maps: List of feature maps from different levels
                      of the pyramid. Each is [batch, height, width, channels]
        pool_shape: [pool_height, pool_width] of the output pooled regions
    """
    pooled, box_to_level = [], []
    for index, level in enumerate(range(2, 6)):
        level_index = tf.where(tf.equal(roi_level, level))
        level_boxes = tf.gather_nd(boxes, level_index)
        box_indices = tf.cast(level_index[:, 0], tf.int32)
        box_to_level.append(level_index)

        level_boxes = tf.stop_gradient(level_boxes)
        box_indices = tf.stop_gradient(box_indices)
        crop_feature_maps = tf.image.crop_and_resize(feature_maps[index], level_boxes,
                                                     box_indices, pool_shape, method='bilinear')
        pooled.append(crop_feature_maps)
    pooled = tf.concat(pooled, axis=0)
    box_to_level = tf.concat(box_to_level, axis=0)

    return pooled, box_to_level


def rearrange_pooled_features(pooled, box_to_level, boxes):
    """Used by PyramidROIAlign to reshape the pooled features.

    # Arguments:
        pooled: [batch, num_boxes, pool_height, pool_width, channels].
                The width and height are those specific in the pool_shape in the layer
                constructor.
        box_to_level: bbox at each level
        boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
                coordinates. Possibly padded with zeros if not enough
                boxes to fill the array
    """
    sorting_tensor = (box_to_level[:, 0] * 100000) + box_to_level[:, 1]
    top_k_indices = tf.nn.top_k(sorting_tensor, k=tf.shape(
        box_to_level)[0]).indices[::-1]
    top_k_indices = tf.gather(box_to_level[:, 2], top_k_indices)
    pooled = tf.gather(pooled, top_k_indices)
    shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)

    return tf.reshape(pooled, shape)
