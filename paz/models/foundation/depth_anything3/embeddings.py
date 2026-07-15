from keras import ops

from paz.models.transformers.embeddings.token import LearnableTokens
from paz.models.foundation.dinov2 import embeddings as dinov2_embeddings
from paz.models.foundation.depth_anything3 import routing


def build_view_embeddings(images, patch_size, hidden_size, num_positions,
                          num_views):
    folded = routing.fold_view_images(images)
    args = folded, patch_size, hidden_size, num_positions, 0
    tokens = dinov2_embeddings.build(*args)
    return routing.restore_view_dimension(tokens, num_views)


def insert_camera_tokens(tokens, hidden_size):
    table = LearnableTokens(2, hidden_size, name="camera_token")(tokens[:, 0])
    reference_view = ops.expand_dims(table[:, 0:1], axis=2)
    other_views = broadcast_other_views(tokens[:, 1:], table[:, 1:2])
    camera = ops.concatenate([reference_view, other_views], axis=1)
    return ops.concatenate([camera, tokens[:, :, 1:, :]], axis=2)


def broadcast_other_views(view_slice, token):
    anchor = view_slice[:, :, :1, :]
    return ops.zeros_like(anchor) + ops.expand_dims(token, axis=2)
