"""Multimodal (text + vision) Gemma4 backbone.

Places image embeddings from the vision encoder into the token-embedding
sequence at the image-placeholder positions, then runs the text decoder stack.
Mirrors keras_hub Gemma4Backbone's interleaving: image embeddings are pre-scaled
by hidden_dim**-0.5 and the global sqrt(hidden_dim) scale is applied after
interleaving (so image positions keep their natural magnitude).
"""
from keras import ops
from keras.layers import Input, Lambda

from .model import (build_backbone_from_embedding, build_token_embedding)
from .vision import build_vision_encoder, num_patches

MULTIMODAL_NAME = "gemma4_multimodal_backbone"


def build_multimodal_backbone(config, vision_config, weights_path=None,
                              name=MULTIMODAL_NAME):
    token_embedding = build_token_embedding(
        config.vocabulary_size, config.hidden_dim, config.dtype)
    token_ids = Input((None,), dtype="int32", name="token_ids")
    padding_mask = Input((None,), dtype="int32", name="padding_mask")
    patch_dim = 3 * vision_config.patch_size ** 2
    pixel_values = Input(
        (num_patches(vision_config), patch_dim), name="pixel_values")
    pixel_position_ids = Input(
        (num_patches(vision_config), 2), dtype="int32",
        name="pixel_position_ids")
    vision_indices = Input((None,), dtype="int32", name="vision_indices")
    embedding = build_image_text_embedding(
        token_embedding(token_ids), pixel_values, pixel_position_ids,
        vision_indices, vision_config, config.hidden_dim)
    inputs = {"token_ids": token_ids, "padding_mask": padding_mask,
              "pixel_values": pixel_values,
              "pixel_position_ids": pixel_position_ids,
              "vision_indices": vision_indices}
    return build_backbone_from_embedding(
        embedding, token_ids, padding_mask, inputs, config, name, weights_path)


def build_image_text_embedding(text_embedding, pixel_values,
                               pixel_position_ids, vision_indices,
                               vision_config, hidden_dim):
    vision = build_vision_encoder(vision_config)
    images = vision({"pixel_values": pixel_values,
                     "pixel_position_ids": pixel_position_ids})
    images = images * ops.cast(float(hidden_dim) ** -0.5, images.dtype)
    images = ops.expand_dims(images, axis=1)
    return build_interleave(images, text_embedding, vision_indices)


def interleave_embeddings(image_embeddings, text_embeddings, vision_indices):
    batch, seq, dim = ops.shape(text_embeddings)
    num_images = ops.shape(image_embeddings)[1]
    num_patches = ops.shape(image_embeddings)[2]
    flat_text = ops.reshape(text_embeddings, (batch * seq, dim))
    offset = ops.expand_dims(ops.arange(batch, dtype="int32") * seq, axis=-1)
    valid = vision_indices[:, : num_images * num_patches]
    flat_image = ops.reshape(image_embeddings, (-1, dim))
    indices = ops.cast(ops.reshape(valid + offset, (-1, 1)), "int32")
    zeroth = ops.take(flat_text, ops.squeeze(offset, axis=-1), axis=0)
    updated = ops.scatter_update(flat_text, indices, flat_image)
    updated = ops.scatter_update(updated, offset, zeroth)
    return ops.reshape(updated, (batch, seq, dim))


def build_interleave(image_embeddings, text_embeddings, vision_indices):
    dim = text_embeddings.shape[-1]
    fn = lambda tensors: interleave_embeddings(*tensors)
    inputs = [image_embeddings, text_embeddings, vision_indices]
    return Lambda(fn, output_shape=(None, dim), name="interleave")(inputs)
