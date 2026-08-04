"""Multimodal (text + vision) Gemma4 backbone.

Places image embeddings from the vision encoder into the token-embedding
sequence at the image-placeholder positions, then runs the text decoder stack.
Mirrors keras_hub Gemma4Backbone's interleaving: image embeddings are pre-scaled
by hidden_dim**-0.5 and the global sqrt(hidden_dim) scale is applied after
interleaving (so image positions keep their natural magnitude).
"""
import keras
from keras import ops

from paz.models.foundation.gemma4.model import Gemma4Backbone
from paz.models.foundation.gemma4.vision import build_vision_encoder

MULTIMODAL_NAME = "gemma4_multimodal_backbone"


@keras.saving.register_keras_serializable(package="gemma4")
class Gemma4MultimodalBackbone(keras.Model):
    def __init__(self, config, vision_config, name=MULTIMODAL_NAME, **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = config
        self.backbone = Gemma4Backbone(config)
        self.vision_encoder = build_vision_encoder(vision_config)

    def call(self, inputs):
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]
        text = self.backbone.token_embedding(token_ids)
        images = self.encode_images(inputs)
        embedding = interleave_embeddings(
            images, text, inputs["vision_indices"])
        return self.backbone.forward_from_embedding(
            embedding, padding_mask, token_ids)

    def encode_images(self, inputs):
        images = self.vision_encoder({
            "pixel_values": inputs["pixel_values"],
            "pixel_position_ids": inputs["pixel_position_ids"]})
        scale = ops.cast(float(self.config.hidden_dim) ** -0.5, images.dtype)
        return ops.expand_dims(images * scale, axis=1)


def build_multimodal_backbone(config, vision_config):
    return Gemma4MultimodalBackbone(config, vision_config)


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
