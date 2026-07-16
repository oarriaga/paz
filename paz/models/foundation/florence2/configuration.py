from collections import namedtuple

VISION_FIELDS = ["image_size", "stage_dims", "stage_depths", "stage_heads",
                 "stage_groups", "patch_sizes", "patch_strides",
                 "patch_paddings", "patch_prenorms", "window_size",
                 "num_position_rows", "projection_dim"]
TEXT_FIELDS = ["vocabulary_size", "hidden_dim", "num_layers", "num_heads",
               "ffn_dim", "max_positions", "position_offset"]

VisionArgs = namedtuple("VisionArgs", VISION_FIELDS)
TextArgs = namedtuple("TextArgs", TEXT_FIELDS)

CONFIGS = {
    "florence2_large_flower": {
        "image_size": 112,
        "stage_dims": (256, 512, 1024, 2048),
        "stage_depths": (1, 1, 9, 1),
        "stage_heads": (8, 16, 32, 64),
        "stage_groups": (8, 16, 32, 64),
        "patch_sizes": (7, 3, 3, 3),
        "patch_strides": (4, 2, 2, 2),
        "patch_paddings": (3, 1, 1, 1),
        "patch_prenorms": (False, True, True, True),
        "window_size": 12,
        "num_position_rows": 50,
        "projection_dim": 1024,
        "vocabulary_size": 51290,
        "hidden_dim": 1024,
        "num_layers": 12,
        "num_heads": 16,
        "ffn_dim": 4096,
        "max_positions": 4098,
        "position_offset": 2,
    },
}


def to_vision_args(config):
    return VisionArgs(*[config[field] for field in VISION_FIELDS])


def to_text_args(config):
    return TextArgs(*[config[field] for field in TEXT_FIELDS])
