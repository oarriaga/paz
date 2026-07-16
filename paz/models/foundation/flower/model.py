"""FLOWER rectified-flow action DiT (velocity network).

Inputs are raw Florence-2 encoder hidden states (the context projection
``cond_norm`` + ``cond_linear`` happens inside), an all-ones context mask
matching the reference, noisy action chunks, and the flow time. Output is
the predicted velocity over the action chunk.
"""
from keras import Model, ops
from keras.layers import Dense, Input

from paz.layers import RMSNormalization
from paz.models.foundation.flower.blocks import flow_block
from paz.models.foundation.flower.embeddings import embed_flow_time
from paz.models.foundation.flower.embeddings import embed_frequency
from paz.models.foundation.flower.embeddings import normalize_features


def build(config, name="flower_dit"):
    context_tokens = Input((None, config.context_dim), name="context_tokens")
    context_mask = Input((None,), name="context_mask")
    action_shape = (config.num_actions, config.action_dim)
    noisy_actions = Input(action_shape, name="noisy_actions")
    flow_time = Input((), name="flow_time")
    x = encode_actions(noisy_actions, config)
    condition = build_condition(flow_time, config)
    context = build_context(context_tokens, config)
    shared_signals = compute_shared_signals(condition, config)
    for block in range(config.num_layers):
        args = (x, condition, shared_signals, context, context_mask, config)
        x = flow_block(*args, f"block_{block}")
    velocity = Dense(config.action_dim, name="action_decoder")(x)
    inputs = [context_tokens, context_mask, noisy_actions, flow_time]
    return Model(inputs, velocity, name=name)


def encode_actions(actions, config):
    hidden = Dense(config.hidden_dim, activation="gelu",
                   name="action_encoder_fc1")(actions)
    return Dense(config.hidden_dim, name="action_encoder_fc2")(hidden)


def build_condition(flow_time, config):
    time_embedding = embed_flow_time(flow_time, config)
    frequency = ops.full_like(flow_time, config.control_frequency)
    frequency_embedding = embed_frequency(frequency, config)
    normalized_time = normalize_features(time_embedding)
    return normalized_time + normalize_features(frequency_embedding)


def build_context(context_tokens, config):
    hidden = RMSNormalization(name="context_norm")(context_tokens)
    return Dense(config.hidden_dim, use_bias=False,
                 name="context_projection")(hidden)


def compute_shared_signals(condition, config):
    hidden = ops.silu(condition)
    signals = Dense(6 * config.hidden_dim, use_bias=False,
                    name="shared_adaln")(hidden)
    return ops.split(signals, 6, axis=-1)
