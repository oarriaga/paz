"""Florence-2 multimodal encoder as consumed by FLOWER VLA.

Default parameter values are the Florence-2-large architecture used by
the FLOWER checkpoints. Two 112x112 views run through a shared DaViT
image encoder (17 tokens each), token ids (leading ``<Flow>`` prompt
token plus tokenized instruction) are embedded, and the concatenated
sequence goes through the BART encoder with BART learned positions
(offset 2) over the full merged sequence and an all-ones attention
mask, exactly as FLOWER does at inference.
"""
from keras import ops
from keras.layers import Embedding, Input, Lambda, LayerNormalization
from keras.models import Model

from paz.models.transformers.embeddings.absolute import embed_position
from paz.models.foundation.florence2.encoder import build_encoder
from paz.models.foundation.florence2.vision import ImageEncoder


def build(image_size=112, vocabulary_size=51290, hidden_dim=1024,
          num_layers=12, num_heads=16, ffn_dim=4096, max_positions=4098,
          position_offset=2, stage_dims=(256, 512, 1024, 2048),
          stage_depths=(1, 1, 9, 1), stage_heads=(8, 16, 32, 64),
          stage_groups=(8, 16, 32, 64), window_size=12):
    static_images = Input((image_size, image_size, 3), name="static_images")
    wrist_images = Input((image_size, image_size, 3), name="wrist_images")
    token_ids = Input((None,), dtype="int32", name="token_ids")
    encoder_args = (image_size, hidden_dim, stage_dims, stage_depths,
                    stage_heads, stage_groups, window_size)
    image_encoder = ImageEncoder(*encoder_args)
    static_tokens = image_encoder(static_images)
    wrist_tokens = image_encoder(wrist_images)
    embed = Embedding(vocabulary_size, hidden_dim, name="embed_tokens")
    text_tokens = embed(token_ids)
    tokens = (static_tokens, wrist_tokens, text_tokens)
    context = ops.concatenate(tokens, axis=1)
    x = add_positions(context, max_positions, position_offset)
    norm = LayerNormalization(epsilon=1e-5, name="layernorm_embedding")
    encoder_args = (norm(x), None, num_layers, num_heads, ffn_dim)
    context_tokens = build_encoder(*encoder_args)
    inputs = [static_images, wrist_images, token_ids]
    return Model(inputs, context_tokens, name="florence2_flower")


def add_positions(x, max_positions, position_offset):
    fn = build_shifted_positions(position_offset)
    kwargs = {"output_shape": position_shape,
              "name": "context_position_indices"}
    layer = Lambda(fn, **kwargs)
    positions = layer(x)
    embed_args = (x, max_positions, True, positions, "embed_positions")
    return x + embed_position(*embed_args)


def build_shifted_positions(offset):
    return lambda x: ops.arange(ops.shape(x)[-2], dtype="int32") + offset


def position_shape(shape):
    return (shape[-2],)
