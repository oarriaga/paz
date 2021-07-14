import tensorflow as tf
import anchors
import paz.processors as pr
from paz.abstract import SequentialProcessor
from paz.datasets.utils import get_class_names
from paz.backend.boxes import to_center_form
from utils import save_file


def merge_class_box_level_outputs(cls_outputs,
                                  box_outputs,
                                  num_levels,
                                  num_classes):
    cls_outputs_all, box_outputs_all = [], []
    batch_size = tf.shape(cls_outputs[0])[0]
    for level in range(0, num_levels):
        cls_outputs_all.append(tf.reshape(
            cls_outputs[level],
            [batch_size, -1, num_classes]
        ))
        box_outputs_all.append(tf.reshape(
            box_outputs[level],
            [batch_size, -1, 4]
        ))
    return tf.concat(cls_outputs_all, 1), tf.concat(box_outputs_all, 1)


def postprocess_paz(cls_outputs,
                    box_outputs,
                    image_scales,
                    min_level=3,
                    max_level=7,
                    num_scales=3,
                    aspect_ratios=[1.0, 2.0, 0.5],
                    anchor_scale=4,
                    image_size=512,
                    num_classes=90,
                    raw_images=None):
    coco = get_class_names('COCO')
    prior_anchors = anchors.Anchors(min_level,
                                    max_level,
                                    num_scales,
                                    aspect_ratios,
                                    anchor_scale,
                                    image_size)
    prior_boxes = prior_anchors.boxes
    prior_boxes = tf.expand_dims(prior_boxes, axis=0)
    s1, s2, s3, s4 = tf.split(prior_boxes, num_or_size_splits=4, axis=2)
    prior_boxes = tf.concat([s2, s1, s4, s3], axis=2)
    prior_boxes = prior_boxes[0]
    prior_boxes = to_center_form(prior_boxes)
    num_levels = max_level - min_level + 1
    cls_outputs, box_outputs = merge_class_box_level_outputs(
        cls_outputs, box_outputs, num_levels, num_classes
    )
    s1, s2, s3, s4 = tf.split(box_outputs, num_or_size_splits=4, axis=2)
    box_outputs = tf.concat([s2, s1, s4, s3], axis=2)
    cls_outputs = tf.sigmoid(cls_outputs)
    outputs = tf.concat([box_outputs, cls_outputs], axis=2)

    postprocessing = SequentialProcessor(
        [pr.Squeeze(axis=None),
         pr.DecodeBoxes(prior_boxes, variances=[1, 1, 1, 1]),
         pr.ScaleBox(image_scales),
         pr.NonMaximumSuppressionPerClass(0.4),
         pr.FilterBoxes(coco, 0.4),
         ])
    outputs = postprocessing(outputs)

    draw_boxes2D = pr.DrawBoxes2D(coco)
    image = draw_boxes2D(raw_images[0].numpy().astype('uint8'), outputs)
    save_file('paz_postprocess.jpg', image)
    return outputs
