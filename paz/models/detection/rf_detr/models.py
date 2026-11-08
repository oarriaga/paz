"""RF-DETR object detectors on a windowed DINOv2 backbone.

Every published variant shares the same DINOv2-small encoder (384 wide, 12
blocks, 6 heads, no register tokens), one fused ``P4`` feature level and a
256-wide decoder with 300 queries. They differ in patch size, input
resolution, window count, which blocks are tapped and attend globally, and
decoder depth.

``global_layers`` and ``out_layers`` are 0-based block indices. Upstream lists
them 1-based over the hidden states, where index 0 is the patch embedding, so
these are those numbers minus one; blocks past the last tapped one also attend
globally.

Detectors return ``(logits, boxes)``: per-query class logits and normalized
``(cx, cy, w, h)`` boxes. The trainable detectors instead return every
supervised stage stacked as ``(batch, stages, groups, queries, 4 + classes)``,
which is what ``paz.losses.detr`` scores.
"""
import math

import numpy as np
from keras import Model, ops

from paz.models.foundation.dinov2 import DINOv2SmallWindowedFeatures
from paz.models.detection.rf_detr import decoder, projector

NUM_QUERIES = 300
PROJECTOR_BLOCKS = 3
NANO = (384, 384, 3), 16, 2, (3, 6, 9), (2, 5, 8, 11), 2
SMALL = (512, 512, 3), 16, 2, (3, 6, 9), (2, 5, 8, 11), 3
MEDIUM = (576, 576, 3), 16, 2, (3, 6, 9), (2, 5, 8, 11), 4
BASE = (560, 560, 3), 14, 4, (2, 5, 8, 11), (1, 4, 7, 10), 3
LARGE = (704, 704, 3), 16, 2, (3, 6, 9), (2, 5, 8, 11), 4


def build_rf_detr(image_shape, patch_size, num_windows, global_layers,
                  out_layers, num_decoder_layers, num_classes, name):
    args = (image_shape, patch_size, num_windows, global_layers, out_layers,
            num_decoder_layers, num_classes, 1)
    backbone, stages = build_stages(*args)
    return Model(backbone.input, stages[-1], name=name)


def build_trainable_rf_detr(image_shape, patch_size, num_windows,
                            global_layers, out_layers, num_decoder_layers,
                            num_classes, num_groups, name):
    """Detector that reports every stage and query group to the loss."""
    args = (image_shape, patch_size, num_windows, global_layers, out_layers,
            num_decoder_layers, num_classes, num_groups)
    backbone, stages = build_stages(*args)
    outputs = stack_stages(stages, num_groups)
    return Model(backbone.input, outputs, name=name)


def build_stages(image_shape, patch_size, num_windows, global_layers,
                 out_layers, num_decoder_layers, num_classes, num_groups):
    """Backbone and one ``(logits, boxes)`` pair per supervised stage."""
    args = image_shape, patch_size, num_windows, global_layers, out_layers
    backbone = DINOv2SmallWindowedFeatures(*args)
    hidden_size, feedforward_size = 256, 2048
    num_self_heads, num_cross_heads, num_points = 8, 16, 2
    args = hidden_size, PROJECTOR_BLOCKS, "projector"
    features = [projector.build(backbone.outputs, *args)]
    args = num_self_heads, num_cross_heads, num_points, feedforward_size
    args = args + (num_classes, NUM_QUERIES, num_groups)
    stages = decoder.build_stages(features, num_decoder_layers, *args)
    return backbone, stages


def stack_stages(stages, num_groups):
    """Packs stages into ``(batch, stages, groups, queries, 4 + classes)``."""
    joined = [join_stage(*stage, num_groups) for stage in stages]
    return ops.stack(joined, axis=1)


def join_stage(logits, boxes, num_groups):
    """One stage as boxes then logits, with its query groups split apart."""
    joined = ops.concatenate([boxes, logits], axis=-1)
    num_queries = joined.shape[1] // num_groups
    return ops.reshape(joined, (-1, num_groups, num_queries, joined.shape[2]))


def build_detector(model):
    """Reads a trainable detector's last stage as ``(logits, boxes)``.

    Both models share every weight, so a detector built here reflects the
    training model as it trains.
    """
    stage = model.output[:, -1, 0]
    outputs = stage[..., 4:], stage[..., :4]
    return Model(model.input, outputs, name=f"{model.name}_detector")


def reset_class_heads(model, prior=0.01):
    """Biases fresh class heads towards predicting almost nothing.

    Both class heads are scored with a focal-style loss over every query and
    class, so a head that starts at even odds swamps the box terms. Upstream
    starts them at a prior of ``0.01``; a head reinitialized for a new class
    count needs the same treatment.
    """
    bias = -math.log((1.0 - prior) / prior)
    for layer in model.layers:
        if layer.name.startswith(("class_embed", "enc_class_embed")):
            weights = layer.get_weights()
            layer.set_weights([weights[0], np.full_like(weights[1], bias)])


def RFDETRNano(num_classes=91, name="rf_detr_nano"):
    return build_rf_detr(*NANO, num_classes, name)


def RFDETRSmall(num_classes=91, name="rf_detr_small"):
    return build_rf_detr(*SMALL, num_classes, name)


def RFDETRMedium(num_classes=91, name="rf_detr_medium"):
    return build_rf_detr(*MEDIUM, num_classes, name)


def RFDETRBase(num_classes=91, name="rf_detr_base"):
    return build_rf_detr(*BASE, num_classes, name)


def RFDETRLarge(num_classes=91, name="rf_detr_large"):
    return build_rf_detr(*LARGE, num_classes, name)


def TrainableRFDETRNano(num_classes=91, num_groups=1, name="rf_detr_nano"):
    return build_trainable_rf_detr(*NANO, num_classes, num_groups, name)


def TrainableRFDETRSmall(num_classes=91, num_groups=1, name="rf_detr_small"):
    return build_trainable_rf_detr(*SMALL, num_classes, num_groups, name)


def TrainableRFDETRMedium(num_classes=91, num_groups=1,
                          name="rf_detr_medium"):
    return build_trainable_rf_detr(*MEDIUM, num_classes, num_groups, name)


def TrainableRFDETRBase(num_classes=91, num_groups=1, name="rf_detr_base"):
    return build_trainable_rf_detr(*BASE, num_classes, num_groups, name)


def TrainableRFDETRLarge(num_classes=91, num_groups=1, name="rf_detr_large"):
    return build_trainable_rf_detr(*LARGE, num_classes, num_groups, name)
