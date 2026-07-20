"""FLOWER rectified-flow action DiT (velocity network).

Default parameter values are the ``mbreuss/flower_libero_object``
checkpoint architecture. Inputs are raw Florence-2 encoder hidden states
(the context projection ``cond_norm`` + ``cond_linear`` happens inside),
an all-ones context mask matching the reference, noisy action chunks, and
the flow time. Output is the predicted velocity over the action chunk.
"""
from keras import Model, ops
from keras.layers import Dense, Input

from paz.layers import RMSNormalization
from paz.models.foundation.flower.blocks import flow_block
from paz.models.foundation.flower.embeddings import embed_flow_time
from paz.models.foundation.flower.embeddings import embed_frequency
from paz.models.foundation.flower.embeddings import normalize_features


def build(context_dim=1024, hidden_dim=1024, num_layers=18, num_heads=16,
          head_dim=64, mlp_dim=2816, adaln_dim=256, action_dim=7,
          num_actions=10, rope_max_positions=100, rope_wavelength=32.0,
          control_frequency=3.0, name="flower_dit"):
    context_tokens = Input((None, context_dim), name="context_tokens")
    context_mask = Input((None,), name="context_mask")
    noisy_actions = Input((num_actions, action_dim), name="noisy_actions")
    flow_time = Input((), name="flow_time")
    x = encode_actions(noisy_actions, hidden_dim)
    condition = build_condition(flow_time, hidden_dim, control_frequency)
    context = build_context(context_tokens, hidden_dim)
    shared_signals = compute_shared_signals(condition, hidden_dim)
    for block in range(num_layers):
        args = (x, condition, shared_signals, context, context_mask,
                num_heads, head_dim, mlp_dim, adaln_dim, rope_max_positions,
                rope_wavelength)
        x = flow_block(*args, f"block_{block}")
    velocity = Dense(action_dim, name="action_decoder")(x)
    inputs = [context_tokens, context_mask, noisy_actions, flow_time]
    return Model(inputs, velocity, name=name)


def encode_actions(actions, hidden_dim):
    encode = Dense(hidden_dim, activation="gelu", name="action_encoder_fc1")
    return Dense(hidden_dim, name="action_encoder_fc2")(encode(actions))


def build_condition(flow_time, hidden_dim, control_frequency):
    time_embedding = embed_flow_time(flow_time, hidden_dim)
    frequency = ops.full_like(flow_time, control_frequency)
    frequency_embedding = embed_frequency(frequency, hidden_dim)
    normalized_time = normalize_features(time_embedding)
    return normalized_time + normalize_features(frequency_embedding)


def build_context(context_tokens, hidden_dim):
    hidden = RMSNormalization(name="context_norm")(context_tokens)
    project = Dense(hidden_dim, use_bias=False, name="context_projection")
    return project(hidden)


def compute_shared_signals(condition, hidden_dim):
    hidden = ops.silu(condition)
    project = Dense(6 * hidden_dim, use_bias=False, name="shared_adaln")
    return ops.split(project(hidden), 6, axis=-1)
