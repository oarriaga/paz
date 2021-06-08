import tensorflow as tf
import anchors


def pre_nms(params, cls_outputs, box_outputs, topk=True):
    eval_anchors = anchors.Anchors(params['min_level'],
                                   params['max_level'],
                                   params['num_scales'],
                                   params['aspect_ratios'],
                                   params['anchor_scale'],
                                   params['image_size'])
    cls_outputs, box_outputs = merge_class_box_level_outputs(
        params, cls_outputs, box_outputs
    )
    if topk:
        cls_outputs, box_outputs, classes, indices = topk_class_boxes(
            params, cls_outputs, box_outputs
        )
        anchor_boxes = tf.gather(eval_anchors.boxes, indices)
    else:
        anchor_boxes = eval_anchors.boxes
        classes = None
    boxes = anchors.decode_box_outputs(box_outputs, anchor_boxes)
    scores = tf.math.sigmoid(cls_outputs)
    return boxes, scores, classes


def merge_class_box_level_outputs(params, cls_outputs, box_outputs):
    cls_outputs_all, box_outputs_all = [], []
    batch_size = tf.shape(cls_outputs[0])[0]
    for level in range(0, params['max_level'] - params['min_level'] + 1):
        cls_outputs_all.append(tf.reshape(
            cls_outputs[level],
            [batch_size, -1, params['num_classes']]
        ))
        box_outputs_all.append(tf.reshape(
            box_outputs[level],
            [batch_size, -1, 4]
        ))
    return tf.concat(cls_outputs_all, 1), tf.concat(box_outputs_all, 1)


def topk_class_boxes(params, cls_outputs, box_outputs):
    batch_size = tf.shape(cls_outputs)[0]
    num_classes = params['num_classes']
    nms_config = params['nms_config']
    max_nms_inputs = nms_config.get('max_nms_inputs', 0)
    if max_nms_inputs > 0:
        cls_outputs_reshape = tf.reshape(cls_outputs, [batch_size, -1])
        _, cls_topk_indices = tf.math.top_k(cls_outputs_reshape,
                                            k=max_nms_inputs,
                                            sorted=False)
        indices = cls_topk_indices // num_classes
        classes = cls_topk_indices % num_classes
        cls_indices = tf.stack([indices, classes], axis=2)
        cls_outputs_topk = tf.gather_nd(cls_outputs,
                                        cls_indices,
                                        batch_dims=1)
        box_outputs_topk = tf.gather_nd(box_outputs,
                                        tf.expand_dims(indices, 2),
                                        batch_dims=1)
    else:
        clas_outputs_idx = tf.math.argmax(cls_outputs,
                                          axis=-1,
                                          output_type=tf.int32)
        num_anchors = tf.shape(cls_outputs)[1]
        classes = clas_outputs_idx
        indices = tf.tile(tf.expand_dims(tf.range(num_anchors),
                                         axis=0),
                          [batch_size, 1])
        cls_outputs_topk = tf.reduce_max(cls_outputs, -1)
        box_outputs_topk = box_outputs

    return cls_outputs_topk, box_outputs_topk, classes, indices


def nms(params, boxes, scores, classes, padded):
    nms_configs = params['nms_config']
    method = nms_configs['method']
    max_output_size = nms_configs['max_output_size']

    if method == 'hard' or not method:
        sigma = 0.0
        iou_thresh = nms_configs['iou_thresh'] or 0.5
        score_thresh = nms_configs['score_thresh'] or float('-inf')
    elif method == 'gaussian':
        sigma = nms_configs['sigma'] or 0.5
        iou_thresh = 1.0
        score_thresh = nms_configs['score_thresh'] or 0.001
    else:
        raise ValueError('Inference has invalid nms method {}'.format(method))
    nms_top_idx, nms_scores, nms_valid_lens = tf.raw_ops.NonMaxSuppressionV5(
        boxes=boxes,
        scores=scores,
        max_output_size=max_output_size,
        iou_threshold=iou_thresh,
        score_threshold=score_thresh,
        soft_nms_sigma=(sigma / 2),
        pad_to_max_output_size=padded)

    nms_boxes = tf.gather(boxes, nms_top_idx)
    nms_classes = tf.cast(
        tf.gather(classes, nms_top_idx) + 1, boxes.dtype)
    return nms_boxes, nms_scores, nms_classes, nms_valid_lens


def batch_map_fn(map_fn, inputs, *args):
    if isinstance(inputs[0], (list, tuple)):
        batch_size = len(inputs[0])
    else:
        batch_size = inputs[0].shape.as_list()[0]

    if not batch_size:
        # handle dynamic batch size:
        # tf.vectorized_map is faster than tf.map_fn.
        return tf.vectorized_map(map_fn, inputs, *args)

    outputs = []
    for i in range(batch_size):
        outputs.append(map_fn([x[i] for x in inputs]))

    return [tf.stack(y) for y in zip(*outputs)]


def clip_boxes(boxes, image_size):
    """Clip boxes to fit the image size."""
    image_size = (image_size, image_size) * 2
    return tf.clip_by_value(boxes, [0], image_size)


def per_class_nms(params, boxes, scores, classes, image_scales):
    def single_batch_fn(element):
        boxes_i, scores_i, classes_i = element[0], element[1], element[2]
        nms_boxes_cls, nms_scores_cls, nms_classes_cls = [], [], []
        nms_valid_len_cls = []
        for cid in range(params['num_classes']):
            indices = tf.where(tf.equal(classes_i, cid))
            if indices.shape[0] == 0:
                continue
            classes_cls = tf.gather_nd(classes_i, indices)
            boxes_cls = tf.gather_nd(boxes_i, indices)
            scores_cls = tf.gather_nd(scores_i, indices)

            (nms_boxes,
             nms_scores,
             nms_classes,
             nms_valid_len) = nms(params,
                                  boxes_cls,
                                  scores_cls,
                                  classes_cls,
                                  False)
            nms_boxes_cls.append(nms_boxes)
            nms_scores_cls.append(nms_scores)
            nms_classes_cls.append(nms_classes)
            nms_valid_len_cls.append(nms_valid_len)

        max_output_size = nms_config.get('max_output_size', 100)
        nms_boxes_cls = tf.pad(tf.concat(nms_boxes_cls, 0),
                               [[0, max_output_size],
                                [0, 0]])
        nms_scores_cls = tf.pad(tf.concat(nms_scores_cls, 0),
                                [[0, max_output_size]]
                                )
        nms_classes_cls = tf.pad(tf.concat(nms_classes_cls, 0),
                                 [[0, max_output_size]]
                                 )
        nms_valid_len_cls = tf.stack(nms_valid_len_cls)

        _, indices = tf.math.top_k(nms_scores_cls,
                                   k=max_output_size,
                                   sorted=True)
        return tuple((
            tf.gather(nms_boxes_cls, indices),
            tf.gather(nms_scores_cls, indices),
            tf.gather(nms_classes_cls, indices),
            tf.minimum(max_output_size, tf.reduce_sum(nms_valid_len_cls))
        ))

    nms_config = params['nms_config']
    nms_boxes, nms_scores, nms_classes, nms_valid_len = batch_map_fn(
        single_batch_fn, [boxes, scores, classes]
    )
    # nms_boxes = clip_boxes(nms_boxes, params['image_size'])
    if image_scales is not None:
        scales = tf.expand_dims(tf.expand_dims(image_scales, -1), -1)
        nms_boxes = nms_boxes * tf.cast(scales, nms_boxes.dtype)
    return nms_boxes, nms_scores, nms_classes, nms_valid_len


def generate_detections_from_nms_output(nms_boxes_bs,
                                        nms_classes_bs,
                                        nms_scores_bs,
                                        image_ids):
    image_ids_bs = tf.cast(tf.expand_dims(image_ids, -1),
                           nms_scores_bs.dtype)
    detections_bs = [
        image_ids_bs * tf.ones_like(nms_scores_bs),
        nms_boxes_bs[:, :, 1],
        nms_boxes_bs[:, :, 0],
        nms_boxes_bs[:, :, 3],
        nms_boxes_bs[:, :, 2],
        nms_scores_bs,
        nms_classes_bs
    ]
    return tf.stack(detections_bs, axis=-1, name='detections')


def postprocess_per_class(params, cls_outputs, box_outputs, image_scales):
    params['nms_config'] = {
        'method': 'gaussian',
        'iou_thresh': None,  # use the default value based on method.
        'score_thresh': 0.001,
        'sigma': None,
        'pyfunc': False,
        'max_nms_inputs': 0,
        'max_output_size': 100,
    }
    image_ids = list(range(tf.shape(cls_outputs[0])[0]))
    boxes, scores, classes = pre_nms(params, cls_outputs, box_outputs)
    (nms_boxes_bs,
     nms_scores_bs,
     nms_classes_bs,
     _) = per_class_nms(params,
                        boxes,
                        scores,
                        classes,
                        image_scales)
    return generate_detections_from_nms_output(nms_boxes_bs,
                                               nms_classes_bs,
                                               nms_scores_bs,
                                               image_ids)


def postprocess_global(params, cls_outputs, box_outputs, image_scales):

    """Post processing with global NMS.

    A fast but less accurate version of NMS.
    The idea is to treat the scores for
    different classes in a unified way, and
     perform NMS globally for all classes.

    Args:
      params: a dict of parameters.
      cls_outputs: a list of tensors for classes, each
      tensor denotes a level of logits with shape
      [N, H, W, num_class * num_anchors].
      box_outputs: a list of tensors for boxes, each tensor denotes a level of
      boxes with shape [N, H, W, 4 * num_anchors].
      Each box format is [y_min, x_min, y_max, x_man].
      image_scales: scaling factor or the final image and bounding boxes.

    Returns:
      A tuple of batch level (boxes, scores, classess, valid_len) after nms.
    """
    params['nms_config'] = {
        'method': 'hard',
        'iou_thresh': 0.5,  # use the default value based on method.
        'score_thresh': 0.001,
        'sigma': None,
        'pyfunc': False,
        'max_nms_inputs': 0,
        'max_output_size': 100,
    }
    image_ids = list(range(tf.shape(cls_outputs[0])[0]))
    boxes, scores, classes = pre_nms(params, cls_outputs, box_outputs)

    def single_batch_fn(element):
        return nms(params, element[0], element[1], element[2], True)

    nms_boxes, nms_scores, nms_classes, nms_valid_len = batch_map_fn(
        single_batch_fn, [boxes, scores, classes])
    nms_boxes = clip_boxes(nms_boxes, params['image_size'])
    if image_scales is not None:
        scales = tf.expand_dims(tf.expand_dims(image_scales, -1), -1)
        nms_boxes = nms_boxes * tf.cast(scales, nms_boxes.dtype)
    return generate_detections_from_nms_output(nms_boxes,
                                               nms_classes,
                                               nms_scores,
                                               image_ids)


def get_postprocessor(type):
    if type == 'global':
        return postprocess_global
    elif type == 'per_class':
        return postprocess_per_class
