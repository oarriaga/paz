"""Florence-2 multimodal encoder as consumed by FLOWER VLA.

Two 112x112 views run through a shared DaViT image encoder (17 tokens
each), token ids (leading ``<Flow>`` prompt token plus tokenized
instruction) are embedded, and the concatenated sequence goes through
the BART encoder with BART learned positions (offset 2) over the full
merged sequence and an all-ones attention mask, exactly as FLOWER does
at inference.
"""
from keras import ops
from keras.layers import Embedding, Input, Lambda, LayerNormalization
from keras.models import Model

from paz.models.transformers.embeddings.absolute import embed_position
from paz.models.foundation.florence2.configuration import to_text_args
from paz.models.foundation.florence2.configuration import to_vision_args
from paz.models.foundation.florence2.encoder import build_encoder
from paz.models.foundation.florence2.vision import ImageEncoder

NORM_EPSILON = 1e-5


def build(config):
    vision_args = to_vision_args(config)
    text_args = to_text_args(config)
    size = vision_args.image_size
    static_images = Input((size, size, 3), name="static_images")
    wrist_images = Input((size, size, 3), name="wrist_images")
    token_ids = Input((None,), dtype="int32", name="token_ids")
    image_encoder = ImageEncoder(vision_args)
    static_tokens = image_encoder(static_images)
    wrist_tokens = image_encoder(wrist_images)
    embed = Embedding(text_args.vocabulary_size, text_args.hidden_dim,
                      name="embed_tokens")
    text_tokens = embed(token_ids)
    context = ops.concatenate([static_tokens, wrist_tokens, text_tokens],
                              axis=1)
    x = add_positions(context, text_args)
    x = LayerNormalization(epsilon=NORM_EPSILON,
                           name="layernorm_embedding")(x)
    encoder_args = (text_args.num_layers, text_args.num_heads,
                    text_args.ffn_dim)
    context_tokens = build_encoder(x, None, *encoder_args)
    inputs = [static_images, wrist_images, token_ids]
    return Model(inputs, context_tokens, name="florence2_flower")


def add_positions(x, text_args):
    fn = build_shifted_positions(text_args.position_offset)
    layer = Lambda(fn, output_shape=position_shape,
                   name="context_position_indices")
    positions = layer(x)
    embeddings = embed_position(x, text_args.max_positions, True, positions,
                                "embed_positions")
    return x + embeddings


def build_shifted_positions(offset):
    return lambda x: ops.arange(ops.shape(x)[-2], dtype="int32") + offset


def position_shape(shape):
    return (shape[-2],)
