from keras import ops
from keras.layers import Embedding, Lambda

from paz.models.transformers.attention import kernel
from paz.models.foundation.dinov2.embeddings import build_dinov2_embeddings
from paz.models.foundation.depth_anything3 import routing


def build_view_embeddings(images, patch_size, hidden_size, num_positions,
                          num_views):
    folded = routing.fold_view_images(images)
    tokens = build_dinov2_embeddings(folded, patch_size, hidden_size,
                                     num_positions, 0)
    return routing.restore_view_dimension(tokens, num_views)


def insert_camera_tokens(tokens, hidden_size):
    table = build_camera_table(tokens, hidden_size)
    reference = ops.expand_dims(table[:, 0:1], axis=2)
    source = broadcast_source(tokens[:, 1:], table[:, 1:2])
    camera = ops.concatenate([reference, source], axis=1)
    return ops.concatenate([camera, tokens[:, :, 1:, :]], axis=2)


def broadcast_source(view_slice, token):
    anchor = view_slice[:, :, :1, :]
    return ops.zeros_like(anchor) + ops.expand_dims(token, axis=2)


def build_camera_table(reference, hidden_size):
    indices = Lambda(batched_pair_indices, output_shape=(2,),
                     name="camera_token_indices")(reference)
    return Embedding(2, hidden_size, kernel(), name="camera_token")(indices)


def batched_pair_indices(reference):
    batch = ops.shape(reference)[0]
    base = ops.arange(2, dtype="int32")[None]
    return ops.broadcast_to(base, (batch, 2))
