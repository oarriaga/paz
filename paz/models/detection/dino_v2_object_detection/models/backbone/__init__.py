import keras
from keras import Input

from .dinov2 import DinoV2
from .projector import MultiScaleProjector
from .backbone import (
    Backbone,
    get_dinov2_lr_decay_rate,
    get_dinov2_weight_decay_rate,
)
from .position_encoding import (
    position_embedding_sine,
    build_position_encoding,
)

__all__ = [
    "DinoV2",
    "MultiScaleProjector",
    "Backbone",
    "get_dinov2_lr_decay_rate",
    "get_dinov2_weight_decay_rate",
    "position_embedding_sine",
    "build_position_encoding",
    "build_backbone",
]


def attach_position_encodings(features, position_embedding):
    positions = []
    for feature, feature_mask in features:
        positions.append(position_embedding(feature_mask, align_dim_orders=False))  # fmt: skip
    return positions


def build_backbone(encoder, window_block_indexes=None, drop_path=0.0, out_channels=256, out_feature_indexes=None, projector_scale=None, hidden_dim=256, position_embedding="sine", layer_norm=False, target_shape=(640, 640), rms_norm=False, load_dinov2_weights=True, patch_size=14, num_windows=4, positional_encoding_size=37):  # fmt: skip
    keys = ("name", "window_block_indexes", "drop_path", "out_channels", "out_feature_indexes", "projector_scale", "layer_norm", "target_shape", "rms_norm", "load_dinov2_weights", "patch_size", "num_windows", "positional_encoding_size")  # fmt: skip
    values = (encoder, window_block_indexes, drop_path, out_channels, out_feature_indexes, projector_scale, layer_norm, target_shape, rms_norm, load_dinov2_weights, patch_size, num_windows, positional_encoding_size)  # fmt: skip
    backbone = Backbone(**dict(zip(keys, values)))
    embedding = build_position_encoding(hidden_dim, position_embedding)
    height, width = target_shape
    images = Input((height, width, 3), name="images")
    mask = Input((height, width), dtype="bool", name="mask")
    # Backbone's single output is the list of [feature, mask] pairs; a
    # symbolic call returns it wrapped in a 1-tuple, so unwrap before use.
    features = backbone([images, mask])[0]
    positions = attach_position_encodings(features, embedding)
    return keras.Model([images, mask], (features, positions), name="joiner")
