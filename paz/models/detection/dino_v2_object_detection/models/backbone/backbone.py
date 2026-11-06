import keras
from keras import Input, ops

from .dinov2 import DinoV2, SIZE_TO_WIDTH
from .projector import MultiScaleProjector

__all__ = ["Backbone"]

LEVEL_TO_SCALE = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
NO_DECAY_TOKENS = ("gamma", "pos_embed", "rel_pos", "bias", "norm", "embeddings")  # fmt: skip
NAME_MESSAGE = "name should be dinov2, then either registers, windowed, both, or none, then the size"  # fmt: skip
SCALE_ORDER_MESSAGE = "only support projector scale P3/P4/P5/P6 in ascending order."  # fmt: skip


def parse_encoder_name(name):
    parts = name.split("_")
    assert parts[0] == "dinov2"
    use_registers = "registers" in parts
    for optional in ("registers", "windowed"):
        if optional in parts:
            parts.remove(optional)
    assert len(parts) == 2, NAME_MESSAGE
    return parts[-1], use_registers


def resolve_scale_factors(projector_scale):
    assert len(projector_scale) > 0
    assert sorted(projector_scale) == projector_scale, SCALE_ORDER_MESSAGE
    return [LEVEL_TO_SCALE[level] for level in projector_scale]


def build_backbone_encoder(size, use_registers, out_feature_indexes, window_block_indexes, target_shape, patch_size, num_windows, positional_encoding_size, drop_path):  # fmt: skip
    keys = ("size", "out_feature_indexes", "window_block_indexes", "shape", "use_registers", "patch_size", "num_windows", "positional_encoding_size", "drop_path_rate", "name")  # fmt: skip
    values = (size, out_feature_indexes, window_block_indexes, target_shape, use_registers, patch_size, num_windows, positional_encoding_size, drop_path, "encoder")  # fmt: skip
    return DinoV2(**dict(zip(keys, values)))


def build_backbone_projector(in_channels, out_channels, scale_factors, layer_norm, rms_norm):  # fmt: skip
    keys = ("in_channels", "out_channels", "scale_factors", "input_scales", "layer_norm", "rms_norm", "name")  # fmt: skip
    values = (in_channels, out_channels, scale_factors, [1.0] * len(in_channels), layer_norm, rms_norm, "projector")  # fmt: skip
    return MultiScaleProjector(**dict(zip(keys, values)))


# load_dinov2_weights is unused here but stays in the signature: main.py and
# the backbone tests pass it by keyword.
def Backbone(name, window_block_indexes=None, drop_path=0.0, out_channels=256, out_feature_indexes=None, projector_scale=None, layer_norm=False, target_shape=(640, 640), rms_norm=False, load_dinov2_weights=True, patch_size=14, num_windows=4, positional_encoding_size=37):  # fmt: skip
    size, use_registers = parse_encoder_name(name)
    scale_factors = resolve_scale_factors(projector_scale)
    in_channels = [SIZE_TO_WIDTH[size]] * len(out_feature_indexes)
    args = (size, use_registers, out_feature_indexes, window_block_indexes)
    shapes = (target_shape, patch_size, num_windows, positional_encoding_size)
    encoder = build_backbone_encoder(*args, *shapes, drop_path)
    projector_args = (in_channels, out_channels, scale_factors)
    projector = build_backbone_projector(*projector_args, layer_norm, rms_norm)
    height, width = target_shape
    images = Input((height, width, 3), name="images")
    mask = Input((height, width), dtype="bool", name="mask")
    features = projector(encoder(images))
    if not isinstance(features, (list, tuple)):
        features = [features]
    pairs = []
    for feature in features:
        pairs.append([feature, resize_mask_to_feature(mask, feature)])
    # Wrap in a 1-tuple so a single-scale model keeps its one output (the
    # list of [feature, mask] pairs) intact instead of collapsing it away.
    return keras.Model([images, mask], (pairs,), name="backbone")


def resize_mask_to_feature(mask, feature):
    size = (feature.shape[1], feature.shape[2])
    mask = ops.expand_dims(ops.cast(mask, "float32"), axis=-1)
    mask = ops.image.resize(mask, size, interpolation="nearest")
    return ops.cast(ops.squeeze(mask, axis=-1), "bool")


def read_dinov2_layer_id(name, num_layers):
    layer_id = num_layers + 1
    inside_layer = ".layer." in name and ".residual." not in name
    if name.startswith("backbone") and "embeddings" in name:
        layer_id = 0
    elif name.startswith("backbone") and inside_layer:
        layer_id = int(name[name.find(".layer.") :].split(".")[2]) + 1
    return layer_id


def get_dinov2_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    layer_id = read_dinov2_layer_id(name, num_layers)
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_dinov2_weight_decay_rate(name, weight_decay_rate=1.0):
    if any(token in name for token in NO_DECAY_TOKENS):
        weight_decay_rate = 0.0
    return weight_decay_rate
