import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


def box_refinement(box, prior_box):
    """Compute refinement needed to transform box to prior_box

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
            input_slice=x[sample_arg]
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
    for output, name in zip(outputs,names):
        result = tf.stack(output, axis=0, name=name)
        results.append(result)

    if len(results) == 1:
        results = results[0]

    return results


def apply_box_deltas(boxes, deltas):
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
                      name='apply_box_deltas_out')
    return result


def clip_boxes(boxes, window):
    """Clips boxes for given window size

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

    #clipped.set_shape((clipped.shape[0], 4))     #TODO: Check impact during trianing
    return clipped


def get_top_detections(scores, keep, nms_keep, detection_max_instances):
    """Used by top detection layer in refine detection graph

    # Arguments:
        scores
        keep
        nms_keep
        detection_max_instances
    """

    keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                tf.expand_dims(nms_keep, 0))

    keep = tf.sparse.to_dense(keep)[0]     # Important step
    roi_count = detection_max_instances
    class_scores_keep = tf.gather(scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    return tf.gather(keep, top_ids)


def NMS_map(box_data, keep, unique_class_id,
            detection_max_instances, detection_nms_threshold):
    """Used by top detection layer in apply NMS

    # Arguments:
        pre_nms_elements
        keep
        unique_class_id
        detection_max_instances
        detection_nms_threshold
    """

    class_ids, scores, rois = box_data
    ids = tf.where(tf.equal(class_ids, unique_class_id))[:, 0]

    class_keep = tf.image.non_max_suppression(tf.gather(rois, ids), tf.gather(scores, ids),
                                              max_output_size=detection_max_instances,
                                              iou_threshold=detection_nms_threshold)
    class_keep = tf.gather(keep, tf.gather(ids, class_keep))
    gap = detection_max_instances - tf.shape(class_keep)[0]
    class_keep = tf.pad(class_keep, [(0, gap)], mode='CONSTANT', constant_values=-1)
    #class_keep.set_shape([detection_max_instances])    #TODO: Check impact during training
    return class_keep


def apply_NMS(class_ids, scores, refined_rois, keep,
              detection_max_instances, detection_nms_threshold):
    """Used by top detection layer in refine detection graph

    # Arguments:
        class_ids
        scores
        refined_rois
        keep
        detection_max_instances
        detection_nms_threshold
    """

    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    pre_nms_elements = [pre_nms_class_ids, pre_nms_scores, pre_nms_rois]

    nms_keep=[]
    for arg in [unique_pre_nms_class_ids]:
        nms_value = NMS_map(pre_nms_elements, keep, arg,
                            detection_max_instances, detection_nms_threshold)
        nms_keep.append(nms_value)
    nms_keep = tf.stack(nms_keep)
    nms_keep= tf.dtypes.cast(nms_keep, tf.int64)
    return merge_results(nms_keep)


def merge_results(nms_keep):
    """Used by top detection layer in apply_NMS

    # Arguments:
        nms_keep
    """

    nms_keep = tf.reshape(nms_keep, [-1])
    return tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])


def filter_low_confidence(class_scores, keep, detection_min_confidence):
    """Used by top detection layer in refine detection graph

    # Arguments:
        class_scores
        keep
        detection_min_confidence
    """

    confidence = tf.where(class_scores >= detection_min_confidence)[:, 0]
    keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                tf.expand_dims(confidence, 0))
    return tf.sparse.to_dense(keep)[0]    #Important step


def zero_pad_detections(detections, detection_max_instances):
    """Used by top detection layer in refine detection graph

    # Arguments:
        detections
        detection_max_instances
    """

    gap = detection_max_instances - tf.shape(detections)[0]
    return tf.pad(detections, [(0, gap), (0, 0)], 'CONSTANT')


def trim_by_score(scores, deltas, anchors, images_per_gpu, pre_nms_limit):
    """Used by proposal layer

    # Arguments:
        scores: [N] Predicted target class values
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        anchors: [batch, num_anchors, (y_min, x_min, y_max, x_max)] anchors
                 in normalized coordinates
        images_per_gpu: Number of images to train with on each GPU
        pre_nms_limit: type int, ROIs kept after tf.nn.top_k
                       and before non-maximum suppression
    """

    pre_nms_limit = tf.minimum(pre_nms_limit, tf.shape(anchors)[1])
    indices = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                          name='top_anchors').indices
    scores = slice_batch([scores, indices], [], tf.gather, images_per_gpu)
    deltas = slice_batch([deltas, indices], [], tf.gather, images_per_gpu)
    pre_nms_anchors = slice_batch([anchors, indices], [], tf.gather,
                                  images_per_gpu, names=['pre_nms_anchors'])
    return scores, deltas, pre_nms_anchors


def apply_box_delta(pre_nms_anchors, deltas, images_per_gpu):
    """Used by proposal layer

    # Arguments:
        pre_nms_anchors: [N, (y_min, x_min, y_max, x_max)] boxes to update
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        images_per_gpu: Number of images to train with on each GPU
    """

    boxes = slice_batch([pre_nms_anchors, deltas], [], apply_box_deltas,
                        images_per_gpu, names=['refined_anchors'])
    return boxes


def clip_image_boundaries(boxes, images_per_gpu):
    """Used by proposal layer

    # Arguments:
        boxes: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        images_per_gpu: Number of images to train with on each GPU
    """
    global window1
    window1 = np.array([0, 0, 1, 1], dtype=np.float32)
    boxes = slice_batch(boxes,[], clip_boxes_to_window_size, images_per_gpu,
                        names=['refined_anchors_clipped'])
    return boxes


def clip_boxes_to_window_size(x):
    """Used tp clip boundaries of images after proposals

        # Arguments:
            x: [N, (dy, dx, log(dh), log(dw))] proposed image
            window: Numpy array of size [1x4]
        """
    return clip_boxes(x,window1)


def pad_ROI_priors(num_positives, num_negatives, roi_priors,
                   deltas, masks):
    """Used by Detection target layer in detection_target_graph

    # Arguments:
        num_positives
        num_negatives
        roi_priors
        deltas
        masks
    """

    roi_class_ids, roi_boxes, _ = roi_priors
    num_samples = num_negatives + num_positives
    roi_boxes = tf.pad(roi_boxes, [(0, num_samples), (0, 0)])
    roi_class_ids = tf.pad(roi_class_ids, [(0, num_samples)])
    deltas = tf.pad(deltas, [(0, num_samples), (0, 0)])
    masks = tf.pad(masks, [[0, num_samples], (0, 0), (0, 0)])
    return roi_class_ids, deltas, masks


def pad_ROI(positive_rois, negative_rois, train_rois_per_image):
    """Used by Detection target layer in detection_target_graph

    # Arguments:
        positive_rois
        negative_rois
        train_rois_per_image
    """

    rois = tf.concat([positive_rois, negative_rois], axis=0)
    num_negatives = tf.shape(negative_rois)[0]
    train_roi = train_rois_per_image - tf.shape(rois)[0]
    num_positives = tf.maximum(train_roi, 0)
    rois = tf.pad(rois, [(0, num_positives), (0, 0)])
    return rois, num_positives, num_negatives


def update_priors(overlaps, positive_indices, positive_rois, priors,
                  bbox_standard_deviation):
    """Used by Detection target layer in detection_target_graph

        # Arguments:
            overlaps
            positive_indices
            positive_rois
            priors
            bbox_std_dev
        """

    class_ids, boxes, masks = priors
    global positive_overlaps
    positive_overlaps = tf.gather(overlaps, positive_indices)

    roi_true_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=compute_largest_overlap, false_fn=get_empty_list)

    roi_prior_boxes = tf.gather(boxes, roi_true_box_assignment)
    roi_prior_class_ids = tf.gather(class_ids, roi_true_box_assignment)
    deltas = box_refinement(positive_rois, roi_prior_boxes)
    deltas /= bbox_standard_deviation

    transposed_masks = tf.expand_dims(tf.transpose(masks, [2, 0, 1]), -1)
    roi_masks = tf.gather(transposed_masks, roi_true_box_assignment)
    return deltas, [roi_prior_class_ids, roi_prior_boxes, roi_masks]


def compute_largest_overlap():
    return tf.argmax(positive_overlaps, axis=1)


def get_empty_list():
    return tf.cast(tf.constant([]), tf.int64)


def compute_target_masks(positive_rois, roi_priors, mask_shape, mini_mask):
    """Used by Detection target layer in detection_target_graph

    # Arguments:
        positive_rois
        roi_priors
        mask_shape
        use_mini_mask
    """

    roi_class_ids, roi_boxes, roi_masks = roi_priors
    boxes = positive_rois
    if mini_mask:
        boxes = transform_ROI_coordinates(positive_rois, roi_boxes)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    roi_masks_cast = tf.cast(roi_masks, tf.float32)
    masks = tf.image.crop_and_resize(roi_masks_cast, boxes, box_ids, mask_shape)
    masks = tf.squeeze(masks, axis=3)
    return tf.round(masks)


def transform_ROI_coordinates(boxes, roi_boxes):
    """Used by Detection target layer in compute_target_masks

    # Arguments:
        boxes
        roi_boxes
    """

    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=1)
    roi_y_min, roi_x_min, roi_y_max, roi_x_max = tf.split(roi_boxes, 4,
                                                          axis=1)
    H = roi_y_max - roi_y_min
    W = roi_x_max - roi_x_min
    y_min = (y_min - roi_y_min) / H
    x_min = (x_min - roi_x_min) / W
    y_max = (y_max - roi_y_min) / H
    x_max = (x_max - roi_x_min) / W
    return tf.concat([y_min, x_min, y_max, x_max], 1)


def remove_zero_padding(proposals, class_ids, boxes, masks):
    """Used by Detection target layer in refine_instances

    # Arguments:
        proposals
        ground_truth
    """
    proposals, _ = trim_zeros(proposals, name='trim_proposals')
    boxes, non_zeros = trim_zeros(boxes, name='trim_prior_boxes')
    class_ids = tf.boolean_mask(class_ids, non_zeros,
                                name='trim_prior_class_ids')
    masks = tf.gather(masks, tf.where(non_zeros)[:, 0], axis=2,
                      name='trim_prior_masks')

    return proposals, class_ids, boxes, masks


def refine_instances(proposals, class_ids, boxes, masks):
    """Used by Detection target layer in detection_targets_graph

    # Arguments:
        proposals
        ground_truth
    """

    proposals, class_ids, boxes, masks = remove_zero_padding(proposals, class_ids,
                                                             boxes, masks)
    crowd_ix = tf.where(class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(boxes, crowd_ix)
    class_ids = tf.gather(class_ids, non_crowd_ix)
    boxes = tf.gather(boxes, non_crowd_ix)
    masks = tf.gather(masks, non_crowd_ix, axis=2)
    return [class_ids, boxes, masks], crowd_boxes


def compute_ROI_overlaps(proposals, boxes, crowd_boxes, overlaps,
                         train_rois_per_image, roi_positive_ratio):
    """Used by Detection target layer in detection_targets_graph

    # Arguments:
        proposals
        boxes
        crowd_boxes
        overlaps
        train_rois_per_image
        roi_positive_ratio
    """

    crowd_overlaps = compute_IOU(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    positive_indices, negative_indices = compute_positive_negative_indices(overlaps,
                                                                           no_crowd_bool)
    return gather_ROIs(proposals, positive_indices, negative_indices,
                       train_rois_per_image, roi_positive_ratio)


def compute_positive_negative_indices(overlaps, no_crowd_bool):
    """Used by Detection target layer in detection_targets_graph

        # Arguments:
            overlaps
            no_crowd_bool
        """
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    negative_indices = tf.where(tf.logical_and(
                                roi_iou_max < 0.5, no_crowd_bool))[:, 0]
    return positive_indices, negative_indices


def gather_ROIs(proposals, positive_indices, negative_indices,
                train_rois_per_image, roi_positive_ratio):
    """Used by Detection target layer in compute_ROI_overlaps

    # Arguments:
        proposals
        positive_indices
        negative_indices
        train_rois_per_image
        roi_positive_ratio
    """

    positive_count = int(train_rois_per_image * roi_positive_ratio)
    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    negative_indices = get_negative_indices(positive_indices, roi_positive_ratio, negative_indices)

    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)
    return positive_indices, positive_rois, negative_rois


def get_negative_indices(positive_indices, roi_positive_ratio, negative_indices):
    """Used by Detection target layer in gather_ROIs

        # Arguments:
            positive_indices
            roi_positive_ratio
            negative_indices
        """
    positive_count = tf.shape(positive_indices)[0]
    ratio = 1.0 / roi_positive_ratio
    negative_count = tf.cast(ratio * tf.cast(positive_count, tf.float32),
                             tf.int32) - positive_count
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    return negative_indices


def compute_IOU(boxes_a, boxes_b):
    """Used by Detection target layer in compute_ROI_overlaps

    # Arguments:
        boxes_a
        boxes_b
    """

    box_a = tf.reshape(tf.tile(tf.expand_dims(boxes_a, 1),[1, 1, tf.shape(boxes_b)[0]]),
                       [-1, 4])
    box_b = tf.tile(boxes_b, [tf.shape(boxes_a)[0], 1])
    overlap, union = compute_overlap_union(box_a,  box_b)
    iou = overlap / union
    overlaps = tf.reshape(iou,[tf.shape(boxes_a)[0], tf.shape(boxes_b)[0]])
    return overlaps


def compute_overlap_union(box_a, box_b):
    """Used by Detection target layer in compute_IOU

        # Arguments:
            boxes_a
            boxes_b
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


def compute_ROI_level(boxes, image_shape):
    """Used by PyramidROIAlign

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
    """Used by compute_ROI_level

           # Arguments:
               scaled_area: area of image scaled
           """
    roi_level = tf.experimental.numpy.log2(scaled_area)
    cast_roi_level = tf.cast(tf.round(roi_level), tf.int32)
    max_roi_level = tf.maximum(2, 4 + cast_roi_level)
    roi_level = tf.minimum(5, max_roi_level)
    return roi_level


def compute_scaled_area(H, W, image_shape):
    """Used by compute_ROI_level

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
    """Used by PyramidROIAlign

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
    """Used by PyramidROIAlign

    # Arguments:
        pooled: [batch, num_boxes, pool_height, pool_width, channels].
                The width and height are those specific in the pool_shape in the layer
                constructor.
        box_to_level
        boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
                coordinates. Possibly padded with zeros if not enough
                boxes to fill the array
    """
    sorting_tensor = (box_to_level[:, 0] * 100000) + box_to_level[:, 1]  #TODO: Big num? 10000, check ROI_level-->box_to_level
    top_k_indices = tf.nn.top_k(sorting_tensor, k=tf.shape(
        box_to_level)[0]).indices[::-1]
    top_k_indices = tf.gather(box_to_level[:, 2], top_k_indices)
    pooled = tf.gather(pooled, top_k_indices)
    shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
    return tf.reshape(pooled, shape)


def compute_NMS(boxes, scores, proposal_count, nms_threshold):
    """Used by Proposal Layer

        # Arguments:
            boxes
            scores
            proposal_count
            nms_threshold
        """

    indices = tf.image.non_max_suppression(
        boxes, scores, proposal_count,
        nms_threshold, name='rpn_non_max_suppression')
    proposals = tf.gather(boxes, indices)
    proposals_shape = tf.shape(proposals)[0]
    padding = tf.maximum(proposal_count - proposals_shape, 0)
    proposals = tf.pad(proposals, [(0, padding), (0, 0)])
    return proposals


def refine_detections(rois, probs, deltas, bbox_std_dev, windows, detection_min_confidence,
                            detection_max_instances, detection_nms_threshold):
    """Used by Detection Layer

            # Arguments:
                rois
                probs
                deltas
                bbox_std_dev
                windows
                detection_min_confidence
                detection_max_instances
                detection_nms_threshold
            """

    class_ids, class_scores, deltas_specific = compute_delta_specific(probs,deltas)

    refined_rois = compute_refined_rois(rois, deltas_specific * bbox_std_dev, windows)

    keep = compute_keep(class_ids, class_scores,refined_rois,detection_min_confidence,
                        detection_max_instances, detection_nms_threshold)
    gather_refined_rois = tf.gather(refined_rois, keep)
    gather_class_ids = tf.cast(tf.gather(class_ids, keep),dtype=tf.float32)[..., tf.newaxis]
    gather_class_scores = tf.gather(class_scores, keep)[..., tf.newaxis]

    detections = tf.concat([gather_refined_rois, gather_class_ids, gather_class_scores], axis=1)
    return zero_pad_detections(detections, detection_max_instances)


def compute_delta_specific(probs,deltas):
    """Used by Detection Layer"""
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    deltas_specific = tf.gather_nd(deltas, indices)
    return class_ids, class_scores, deltas_specific


def compute_refined_rois(rois, deltas, windows):
    """Used by Detection Layer

                    # Arguments:
                        rois
                        deltas
                        windows
                    """
    refined_rois = apply_box_deltas(rois, deltas)
    refined_rois = clip_boxes(refined_rois, windows)
    return refined_rois


def compute_keep(class_ids, class_scores,refined_rois, detection_min_confidence,
                 detection_max_instances, detection_nms_threshold):
    """Used by Detection Layer

                # Arguments:
                    class_ids
                    class_scores
                    refined_rois
                    detection_min_confidence
                    detection_max_instances
                    detection_nms_threshold
                """
    keep = tf.where(class_ids > 0)[:, 0]
    if detection_min_confidence:
        keep = filter_low_confidence(class_scores, keep, detection_min_confidence)

    nms_keep = apply_NMS(class_ids, class_scores, refined_rois, keep,
                         detection_max_instances, detection_nms_threshold)
    keep = get_top_detections(class_scores, keep, nms_keep, detection_max_instances)
    return keep


def detection_targets(proposals, prior_class_ids, prior_boxes,
                            prior_masks, train_rois_per_image, roi_positive_ratio,
                            mask_shape, use_mini_mask, bbox_std_dev):
    """Used by Detection Target Layer

                # Arguments:
                    proposals
                    prior_class_ids
                    prior_boxes,
                    prior_masks
                    train_rois_per_image
                    roi_positive_ratio
                    mask_shape
                    use_mini_mask
                    bbox_std_dev
                """
    refined_boxes, refined_priors, crowd_boxes = compute_refined_boxes(proposals, prior_class_ids,
                                                                       prior_boxes, prior_masks)
    overlaps = compute_IOU(proposals, refined_boxes)
    positive_indices, positive_rois, negative_rois = compute_ROI_overlaps(proposals, refined_boxes,
                                                                          crowd_boxes, overlaps,
                                                                          train_rois_per_image,
                                                                          roi_positive_ratio)
    deltas, roi_priors = update_priors(overlaps, positive_indices,positive_rois, refined_priors,
                                       bbox_std_dev)
    masks = compute_target_masks(positive_rois, roi_priors, mask_shape, use_mini_mask)
    rois, num_negatives, num_positives = pad_ROI(positive_rois, negative_rois, train_rois_per_image)
    roi_class_ids, deltas, masks = pad_ROI_priors(num_positives, num_negatives, roi_priors, deltas, masks)
    return rois, roi_class_ids, deltas, masks


def compute_refined_boxes(proposals, prior_class_ids, prior_boxes, prior_masks):
    """Used by Detection Target Layer

                    # Arguments:
                        proposals
                        prior_class_ids
                        prior_boxes,
                        prior_masks
                    """
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name='roi_assertion'),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)
    refined_priors, crowd_boxes = refine_instances(proposals, prior_class_ids, prior_boxes,
                                                   prior_masks)
    _, refined_boxes, _ = refined_priors
    return refined_boxes, refined_priors, crowd_boxes