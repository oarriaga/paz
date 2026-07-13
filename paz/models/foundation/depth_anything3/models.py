from keras import Input, Model

from paz.models.foundation.depth_anything3.embeddings import build_view_embeddings
from paz.models.foundation.depth_anything3.backbone import build_da3_backbone

PATCH_SIZE = 14
MLP_RATIO = 4.0
LAYER_SCALE_INIT = 1.0
OUT_LAYERS = (5, 7, 9, 11)
ALT_START = 4


def build_da3_small_backbone(num_views, image_shape,
                             name="da3_small_backbone"):
    args = (num_views, image_shape, 384, 12, 6)
    return build_da3_backbone_model(*args, name)


def build_da3_backbone_model(num_views, image_shape, hidden_size, depth,
                             num_heads, name):
    images = Input((num_views, *image_shape), name="views")
    grid = grid_shape(image_shape, PATCH_SIZE)
    num_positions = grid[0] * grid[1] + 1
    tokens = build_view_embeddings(images, PATCH_SIZE, hidden_size,
                                   num_positions, num_views)
    features, camera_tokens = build_da3_backbone(
        tokens, num_views, grid, hidden_size, depth, num_heads, MLP_RATIO,
        LAYER_SCALE_INIT, OUT_LAYERS, ALT_START)
    return Model(images, features + camera_tokens, name=name)


def grid_shape(image_shape, patch_size):
    return image_shape[0] // patch_size, image_shape[1] // patch_size
