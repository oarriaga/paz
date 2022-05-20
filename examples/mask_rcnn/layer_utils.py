import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


def box_refinement_graph(box, prior_box):
    """Compute refinement needed to transform box to prior_box

    # Arguments:
        box: [N, (y_min, x_min, y_max, x_max)]
        prior_box: Ground-truth box [N, (y_min, x_min, y_max, x_max)]
    """

    box = tf.cast(box, tf.float32)
    prior_box = tf.cast(prior_box, tf.float32)
    x_box= box[:,0]
    y_box= box[:,1]

    height = box[:, 2] - x_box
    width = box[:, 3] - y_box
    center_y = x_box + 0.5 * height
    center_x = y_box + 0.5 * width

    prior_height = prior_box[:, 2] - prior_box[:, 0]
    prior_width = prior_box[:, 3] - prior_box[:, 1]
    prior_center_y = prior_box[:, 0] + 0.5 * prior_height
    prior_center_x = prior_box[:, 1] + 0.5 * prior_width

    dy = (prior_center_y - center_y) / height
    dx = (prior_center_x - center_x) / width
    dh = tf.math.log(prior_height / height)
    dw = tf.math.log(prior_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def batch_slice(inputs, graph_function, batch_size, names=None):
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
    for sample in range(batch_size):  # i = sample_arg
        inputs_slice = [x[sample] for x in inputs]
        output_slice = graph_function(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)] #explicit for loop

    #for o, n in zip(outputs,names):
        #result = [tf.stack(o, axis=0, name=n)]

    if len(result) == 1:
        result = result[0]

    return result


def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
       are padded with zeros. This removes zero boxes.

    # Arguments:
        boxes: [N, 4] matrix of boxes.
        non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """

    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.

    # Arguments:
        boxes: [N, (y_min, x_min, y_max, x_max)] boxes to update
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """

    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width

    center_y = center_y + (deltas[:, 0] * height) #a= a + b
    center_x = center_x + (deltas[:, 1] * width)
    height = height * tf.exp(deltas[:, 2])
    width = width * tf.exp(deltas[:, 3])

    y_min = center_y - 0.5 * height
    x_min = center_x - 0.5 * width
    y_max = y_min + height
    x_max = x_min + width
    result = tf.stack([y_min, x_min, y_max, x_max], axis=1,
                      name='apply_box_deltas_out')
    return result


def clip_boxes_graph(boxes, window):
    """Clips boxes for given window size

    # Arguments:
        boxes: [N, (y_min, x_min, y_max, x_max)]
        window: [4] in the form y_min, x_min, y_max, x_max
    """

    window_y_min, window_x_min, window_y_max, window_x_max = \
        tf.split(window, 4)
    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=1)
    y_min = tf.maximum(tf.minimum(y_min, window_y_max), window_y_min)
    x_min = tf.maximum(tf.minimum(x_min, window_x_max), window_x_min)
    y_max = tf.maximum(tf.minimum(y_max, window_y_max), window_y_min)
    x_max = tf.maximum(tf.minimum(x_max, window_x_max), window_x_min)  #tf.clip

    clipped = tf.concat([y_min, x_min, y_max, x_max], axis=1,
                        name='clipped_boxes')
    clipped.set_shape((clipped.shape[0], 4))
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
    keep = tf.sparse.to_dense(keep)[0]     #avoid this
    roi_count = detection_max_instances
    class_scores_keep = tf.gather(scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    return tf.gather(keep, top_ids)


def NMS_map(pre_nms_elements, keep, unique_class_id,
            detection_max_instances, detection_nms_threshold):
    """Used by top detection layer in apply NMS

    # Arguments:
        pre_nms_elements
        keep
        unique_class_id
        detection_max_instances
        detection_nms_threshold
    """

    class_ids, scores, rois = pre_nms_elements
    ids = tf.where(tf.equal(class_ids, unique_class_id))[:, 0]

    class_keep = tf.image.non_max_suppression(tf.gather(rois, ids), tf.gather(scores, ids),
                                              max_output_size=detection_max_instances,
                                              iou_threshold=detection_nms_threshold)
    class_keep = tf.gather(keep, tf.gather(ids, class_keep))  #ixs = class_ids explicit variables
    gap = detection_max_instances - tf.shape(class_keep)[0]
    class_keep = tf.pad(class_keep, [(0, gap)], mode='CONSTANT', constant_values=-1)
    class_keep.set_shape([detection_max_instances])

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
    nms_keep = tf.map_fn(lambda x: NMS_map(pre_nms_elements, keep, x,
                                           detection_max_instances, detection_nms_threshold), #lambda remove
                         unique_pre_nms_class_ids, dtype=tf.int64)

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

    confidence = tf.where(
        class_scores >= detection_min_confidence)[:, 0]
    keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                tf.expand_dims(confidence, 0))
    return tf.sparse.to_dense(keep)[0]


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
        scores
        deltas
        anchors
        images_per_gpu
        pre_nms_limit
    """

    pre_nms_limit = tf.minimum(pre_nms_limit, tf.shape(anchors)[1])
    indices = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                          name='top_anchors').indices
    scores = batch_slice([scores, indices], tf.gather, images_per_gpu)
    deltas = batch_slice([deltas, indices], tf.gather, images_per_gpu)
    pre_nms_anchors = batch_slice([anchors, indices], tf.gather,
                                  images_per_gpu, names=['pre_nms_anchors'])
    return scores, deltas, pre_nms_anchors


def apply_box_delta(pre_nms_anchors, deltas, images_per_gpu):
    """Used by proposal layer

    # Arguments:
        pre_nms_anchors
        deltas
        images_per_gpu
    """

    boxes = batch_slice([pre_nms_anchors, deltas],
                        apply_box_deltas_graph, images_per_gpu,
                        names=['refined_anchors'])
    return boxes


def clip_image_boundaries(boxes, images_per_gpu):
    """Used by proposal layer

    # Arguments:
        boxes
        images_per_gpu
    """

    window = np.array([0, 0, 1, 1], dtype=np.float32)
    boxes = batch_slice(boxes, lambda x: clip_boxes_graph(x, window), #TODO: Remove lambda
                        images_per_gpu, names=['refined_anchors_clipped'])
    return boxes


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
    roi_boxes = tf.pad(roi_boxes,
                       [(0, num_samples), (0, 0)])
    roi_class_ids = tf.pad(roi_class_ids,
                           [(0, num_samples)])
    deltas = tf.pad(deltas, [(0, num_samples), (0, 0)])
    masks = tf.pad(masks, [[0, num_samples],
                  (0, 0), (0, 0)])
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


def update_priors(overlaps, positive_indices, positive_rois, priors, bbox_std_dev):
    """Used by Detection target layer in detection_target_graph

        # Arguments:
            overlaps
            positive_indices
            positive_rois
            priors
            bbox_std_dev
        """

    class_ids, boxes, masks = priors
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(positive_overlaps, axis=1), #TODO: remove lambda
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
    )
    roi_prior_boxes = tf.gather(boxes, roi_gt_box_assignment)
    roi_prior_class_ids = tf.gather(class_ids, roi_gt_box_assignment)
    deltas = box_refinement_graph(positive_rois, roi_prior_boxes)
    deltas /= bbox_std_dev
    transposed_masks = tf.expand_dims(tf.transpose(masks, [2, 0, 1]), -1)
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)
    return deltas, [roi_prior_class_ids, roi_prior_boxes, roi_masks]


def compute_target_masks(positive_rois, roi_priors, mask_shape, use_mini_mask):
    """Used by Detection target layer in detection_target_graph

    # Arguments:
        positive_rois
        roi_priors
        mask_shape
        use_mini_mask
    """

    roi_class_ids, roi_boxes, roi_masks = roi_priors
    boxes = positive_rois
    if use_mini_mask:
        boxes = transform_ROI_coordinates(positive_rois, roi_boxes)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                         box_ids, mask_shape)
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
    height = roi_y_max - roi_y_min
    width = roi_x_max - roi_x_min
    y_min = (y_min - roi_y_min) / height
    x_min = (x_min - roi_x_min) / width
    y_max = (y_max - roi_y_min) / height
    x_max = (x_max - roi_x_min) / width
    return tf.concat([y_min, x_min, y_max, x_max], 1)


def remove_zero_padding(proposals, ground_truth):
    """Used by Detection target layer in refine_instances

    # Arguments:
        proposals
        ground_truth
    """

    class_ids, boxes, masks = ground_truth
    proposals, _ = trim_zeros_graph(proposals, name='trim_proposals')
    boxes, non_zeros = trim_zeros_graph(boxes,
                                        name='trim_prior_boxes')
    class_ids = tf.boolean_mask(class_ids, non_zeros,
                                name='trim_prior_class_ids')
    masks = tf.gather(masks, tf.where(non_zeros)[:, 0], axis=2,
                        name='trim_prior_masks')

    return proposals, [class_ids, boxes, masks]


def refine_instances(proposals, ground_truth):
    """Used by Detection target layer in detection_targets_graph

    # Arguments:
        proposals
        ground_truth
    """

    proposals, ground_truth = remove_zero_padding(proposals,
                                                           ground_truth)
    class_ids, boxes, masks = ground_truth
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

    crowd_overlaps = compute_overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    negative_indices = tf.where(tf.logical_and(
                                roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    return gather_ROIs(proposals, positive_indices, negative_indices,
                       train_rois_per_image, roi_positive_ratio)


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
    positive_count = tf.shape(positive_indices)[0]

    ratio = 1.0 / roi_positive_ratio
    negative_count = tf.cast(ratio * tf.cast(positive_count, tf.float32),
                                 tf.int32) - positive_count
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]

    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)
    return positive_indices, positive_rois, negative_rois


def compute_overlaps_graph(boxes_a, boxes_b):
    """Used by Detection target layer in compute_ROI_overlaps

    # Arguments:
        boxes_a
        boxes_b
    """

    box_a = tf.reshape(tf.tile(tf.expand_dims(boxes_a, 1),
                               [1, 1, tf.shape(boxes_b)[0]]),
                       [-1, 4])
    box_b = tf.tile(boxes_b, [tf.shape(boxes_a)[0], 1])

    boxa_y_min, boxa_x_min, boxa_y_max, boxa_x_max = tf.split(box_a,
                                                              4, axis=1)
    boxb_y_min, boxb_x_min, boxb_y_max, boxb_x_max = tf.split(box_b,
                                                              4, axis=1)
    y_min = tf.maximum(boxa_y_min, boxb_y_min)
    x_min = tf.maximum(boxa_x_min, boxb_x_min)
    y_max = tf.minimum(boxa_y_max, boxb_y_max)
    x_max = tf.minimum(boxa_x_max, boxb_x_max)
    overlap = tf.maximum(x_max - x_min, 0) * tf.maximum(y_max - y_min, 0)

    box_a_area = (boxa_y_max - boxa_y_min) * (boxa_x_max - boxa_x_min)
    box_b_area = (boxb_y_max - boxb_y_min) * (boxb_x_max - boxb_x_min)
    union = box_a_area + box_b_area - overlap

    iou = overlap / union
    overlaps = tf.reshape(iou,
                          [tf.shape(boxes_a)[0], tf.shape(boxes_b)[0]])
    return overlaps


def compute_ROI_level(boxes, image_shape):
    """Used by PyramidROIAlign

    # Arguments:
        boxes
        image_shape
    """

    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=2)
    height = y_max - y_min
    width = x_max - x_min
    image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)

    roi_level = tf.experimental.numpy.log2(
        tf.sqrt(height * width) / (224.0 / tf.sqrt(image_area)))
    roi_level = tf.minimum(5, tf.maximum(
        2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
    return tf.squeeze(roi_level, 2)


def apply_ROI_pooling(roi_level, boxes, feature_maps, pool_shape):
    """Used by PyramidROIAlign

    # Arguments:
        roi_level
        boxes
        features_maps
        pool_shape
    """

    pooled, box_to_level = [], []
    for index, level in enumerate(range(2, 6)):
        level_index = tf.where(tf.equal(roi_level, level))
        level_boxes = tf.gather_nd(boxes, level_index)
        box_indices = tf.cast(level_index[:, 0], tf.int32)
        box_to_level.append(level_index)

        level_boxes = tf.stop_gradient(level_boxes)
        box_indices = tf.stop_gradient(box_indices)
        pooled.append(tf.image.crop_and_resize(
            feature_maps[index], level_boxes, box_indices, pool_shape,
            method='bilinear'))
    pooled = tf.concat(pooled, axis=0)
    box_to_level = tf.concat(box_to_level, axis=0)
    return pooled, box_to_level


def rearrange_pooled_features(pooled, box_to_level, boxes):
    """Used by PyramidROIAlign

    # Arguments:
        pooled
        box_to_level
        boxes
    """

    sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
    top_k_indices = tf.nn.top_k(sorting_tensor, k=tf.shape(
        box_to_level)[0]).indices[::-1]
    top_k_indices = tf.gather(box_to_level[:, 2], top_k_indices)
    pooled = tf.gather(pooled, top_k_indices)
    shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
    return tf.reshape(pooled, shape)


def compute_output_shape(input_shape, arg, layer='None'):
    """Used by to compute output shapes for different layers #keras output layer shape

    # Arguments:
        input_shape
        arg: arguments for layers
        layer : layer used
    """

    if layer == 'DetectionLayer':
        detection_max_instances = arg
        return (None, detection_max_instances, 6)
    elif layer == 'ProposalLayer':
        proposal_count = arg
        return (None, proposal_count, 4)
    elif layer == 'DetectionTargetLayer':
        train_rois_per_image, mask_shape = arg
        return [
            (None, train_rois_per_image, 4),  # ROIs
            (None, train_rois_per_image),  # class_ids
            (None, train_rois_per_image, 4),  # deltas
            (None, train_rois_per_image, mask_shape[0],
             mask_shape[1])  # masks
        ]
    elif layer == 'PyramidROIAlign':
        pool_shape = arg
        return input_shape[0][:2] + pool_shape + (input_shape[2][-1],)

    else:
        print("Invalid layer name")
        return 0