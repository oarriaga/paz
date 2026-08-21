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

Models return ``(logits, boxes)``: per-query class logits and normalized
``(cx, cy, w, h)`` boxes.
"""
from keras import Model

from paz.models.foundation.dinov2 import DINOv2SmallWindowedFeatures
from paz.models.detection.rf_detr import decoder, projector

NUM_QUERIES = 300
PROJECTOR_BLOCKS = 3


def build_rf_detr(image_shape, patch_size, num_windows, global_layers,
                  out_layers, num_decoder_layers, num_classes, name):
    args = image_shape, patch_size, num_windows, global_layers, out_layers
    backbone = DINOv2SmallWindowedFeatures(*args)
    hidden_size, feedforward_size = 256, 2048
    num_self_heads, num_cross_heads, num_points = 8, 16, 2
    args = hidden_size, PROJECTOR_BLOCKS, "projector"
    features = [projector.build(backbone.outputs, *args)]
    args = num_self_heads, num_cross_heads, num_points, feedforward_size
    args = args + (num_classes, NUM_QUERIES)
    outputs = decoder.build(features, num_decoder_layers, *args)
    return Model(backbone.input, outputs, name=name)


def RFDETRNano(num_classes=91, name="rf_detr_nano"):
    args = (384, 384, 3), 16, 2, (3, 6, 9), (2, 5, 8, 11), 2
    return build_rf_detr(*args, num_classes, name)


def RFDETRSmall(num_classes=91, name="rf_detr_small"):
    args = (512, 512, 3), 16, 2, (3, 6, 9), (2, 5, 8, 11), 3
    return build_rf_detr(*args, num_classes, name)


def RFDETRMedium(num_classes=91, name="rf_detr_medium"):
    args = (576, 576, 3), 16, 2, (3, 6, 9), (2, 5, 8, 11), 4
    return build_rf_detr(*args, num_classes, name)


def RFDETRBase(num_classes=91, name="rf_detr_base"):
    args = (560, 560, 3), 14, 4, (2, 5, 8, 11), (1, 4, 7, 10), 3
    return build_rf_detr(*args, num_classes, name)


def RFDETRLarge(num_classes=91, name="rf_detr_large"):
    args = (704, 704, 3), 16, 2, (3, 6, 9), (2, 5, 8, 11), 4
    return build_rf_detr(*args, num_classes, name)
