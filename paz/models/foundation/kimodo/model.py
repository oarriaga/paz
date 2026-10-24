"""Kimodo two-stage denoiser as one functional Keras model.

Defaults are the released ``Kimodo-G1-RP-v1`` architecture. The root stage
predicts the global root block from the masked motion, the body stage
predicts the body block from the local root derived from that prediction.
Both stages are post-norm transformer encoders over the sequence
``[text tokens | time | heading | motion frames]``.
"""
import numpy as np
from keras import Model, ops
from keras.layers import Dense, Input, LayerNormalization, Reshape

from paz.models.foundation.kimodo import features
from paz.models.transformers import attention, feedforward


def build(stats, hidden_dim=1024, ff_dim=2048, num_layers=16, num_heads=8,
          num_text_tokens=50, text_dim=4096, fps=30.0,
          name="kimodo_denoiser"):
    noisy_motion = Input((None, features.FEATURE_DIM), name="noisy_motion")
    motion_mask = Input((None, features.FEATURE_DIM), name="motion_mask")
    observed = Input((None, features.FEATURE_DIM), name="observed_motion")
    text = Input((1, text_dim), name="text")
    timestep = Input((), dtype="int32", name="timestep")
    heading_angle = Input((), name="heading_angle")
    motion = mask_motion(noisy_motion, motion_mask, observed)
    args = (text, timestep, heading_angle, hidden_dim, ff_dim, num_layers,
            num_heads, num_text_tokens)
    root_input = ops.concatenate([motion, motion_mask], axis=-1)
    root = build_stage(root_input, *args, features.GLOBAL_ROOT_DIM, "root")
    local_root = convert_root_to_local(root, stats, fps)
    body_motion = motion[..., features.GLOBAL_ROOT_DIM:]
    body_input = ops.concatenate([local_root, body_motion, motion_mask], -1)
    body = build_stage(body_input, *args, features.BODY_DIM, "body")
    inputs = [noisy_motion, motion_mask, observed, text, timestep,
              heading_angle]
    return Model(inputs, ops.concatenate([root, body], axis=-1), name=name)


def mask_motion(noisy_motion, motion_mask, observed_motion):
    return noisy_motion * (1.0 - motion_mask) + observed_motion * motion_mask


def build_stage(motion, text, timestep, heading_angle, hidden_dim, ff_dim,
                num_layers, num_heads, num_text_tokens, output_dim, name):
    tokens = Dense(hidden_dim, name=f"{name}_input_projection")(motion)
    text_tokens = embed_text(text, num_text_tokens, hidden_dim, name)
    time_token = embed_timestep(timestep, hidden_dim, name)
    heading_token = embed_heading(heading_angle, hidden_dim, name)
    prefix = [text_tokens, time_token, heading_token]
    sequence = ops.concatenate(prefix + [tokens], axis=1)
    sequence = sequence + embed_sequence_positions(sequence)
    for layer in range(num_layers):
        sequence = encoder_block(sequence, ff_dim, num_heads,
                                 f"{name}_layer_{layer}")
    outputs = sequence[:, num_text_tokens + 2:]
    return Dense(output_dim, name=f"{name}_output_projection")(outputs)


def embed_text(text, num_text_tokens, hidden_dim, name):
    # The single pooled vector is zero-padded to the fixed token count
    # before the projection, so padded tokens carry the projection bias.
    padding = [[0, 0], [0, num_text_tokens - 1], [0, 0]]
    padded = ops.pad(text, padding)
    return Dense(hidden_dim, name=f"{name}_text_embedding")(padded)


def embed_timestep(timestep, hidden_dim, name):
    position = ops.cast(timestep, "float32")
    embedding = embed_position(position, hidden_dim, (1, hidden_dim))
    dense_1 = Dense(hidden_dim, activation="silu",
                    name=f"{name}_time_embedding_dense_1")
    dense_2 = Dense(hidden_dim, name=f"{name}_time_embedding_dense_2")
    return dense_2(dense_1(embedding))


def embed_heading(heading_angle, hidden_dim, name):
    direction = ops.stack([ops.cos(heading_angle), ops.sin(heading_angle)], -1)
    embedding = Dense(hidden_dim, name=f"{name}_heading_embedding")(direction)
    return ops.expand_dims(embedding, axis=1)


def embed_sequence_positions(sequence):
    positions = ops.cumsum(ops.ones_like(sequence[..., 0]), axis=1) - 1.0
    dim = sequence.shape[-1]
    return embed_position(positions, dim, (-1, dim))


def embed_position(positions, dim, shape):
    """Interleaved sine/cosine encoding, ``10000 ** (-2 i / dim)``."""
    exponents = -ops.arange(0, dim, 2, dtype="float32") / dim
    angles = ops.expand_dims(positions, -1) * ops.power(10000.0, exponents)
    pairs = ops.stack([ops.sin(angles), ops.cos(angles)], axis=-1)
    return Reshape(shape)(pairs)


def encoder_block(x, ff_dim, num_heads, name):
    context = attend(x, num_heads, f"{name}_self_attention")
    x = add_and_normalize(x, context, f"{name}_norm_1")
    names = (f"{name}_feedforward_intermediate", f"{name}_feedforward_output")
    hidden = feedforward.gelu(x, ff_dim, x.shape[-1], *names)
    return add_and_normalize(x, hidden, f"{name}_norm_2")


def attend(x, num_heads, name):
    hidden_dim = x.shape[-1]
    fused = attention.project_query_key_value(x, hidden_dim, True, name)
    heads = attention.split_query_key_value(fused, num_heads,
                                            hidden_dim // num_heads)
    context = attention.compute_attention(*heads)
    context = attention.merge_attention_heads(context)
    return Dense(hidden_dim, name=f"{name}_output")(context)


def add_and_normalize(x, residual, name):
    # Torch LayerNorm uses 1e-5 where Keras defaults to 1e-3.
    return LayerNormalization(epsilon=1e-5, name=name)(x + residual)


def convert_root_to_local(root, stats, fps):
    """Global root block to the local root block, both normalized."""
    root = unnormalize(root, stats.global_root)
    position = root[..., :3]
    heading = ops.arctan2(root[..., 4], root[..., 3])
    heading_rate = repeat_last(fps * compute_angle_differences(heading))
    velocity = fps * (position[:, 1:] - position[:, :-1])
    planar_velocity = repeat_last(ops.take(velocity, [0, 2], axis=-1))
    local = [ops.expand_dims(heading_rate, -1), planar_velocity,
             position[..., 1:2]]
    return normalize(ops.concatenate(local, axis=-1), stats.local_root)


def compute_angle_differences(angles):
    cosines, sines = ops.cos(angles), ops.sin(angles)
    cosine = cosines[:, 1:] * cosines[:, :-1] + sines[:, 1:] * sines[:, :-1]
    sine = sines[:, 1:] * cosines[:, :-1] - cosines[:, 1:] * sines[:, :-1]
    return ops.arctan2(sine, cosine)


def repeat_last(values):
    return ops.concatenate([values, values[:, -1:]], axis=1)


def normalize(x, normalization):
    mean, stdv = np.asarray(normalization.mean), np.asarray(normalization.stdv)
    return (x - mean) / stdv


def unnormalize(x, normalization):
    return x * np.asarray(normalization.stdv) + np.asarray(normalization.mean)
