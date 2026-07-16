"""Keras architecture for the ported SONIC deploy actor."""

import keras
from keras import layers

from paz.models.foundation.sonic.layout import compute_decoder_input_dim
from paz.models.foundation.sonic.layout import compute_encoder_input_dim
from paz.models.foundation.sonic.layout import compute_mode_scalar_index
from paz.models.foundation.sonic.layout import compute_policy_tail_dim


def build_actor(layout, encoder, decoder):
    encoder_dim = compute_encoder_input_dim(layout)
    encoder_obs = keras.Input((encoder_dim,), name="encoder_obs")
    tail_dim = compute_policy_tail_dim(layout)
    policy_obs = keras.Input((tail_dim,), name="policy_obs_tail")
    encoded_tokens = encoder(encoder_obs)
    decoder_input = compute_cat(encoded_tokens, policy_obs)
    action = decoder(decoder_input)
    inputs = {"encoder_obs": encoder_obs, "policy_obs_tail": policy_obs}
    return keras.Model(inputs, action, name="sonic_actor")


def build_encoder(layout):
    encoder_dim = compute_encoder_input_dim(layout)
    encoder_input = keras.Input((encoder_dim,), name="obs_dict")
    mode_index = compute_mode_scalar_index(layout)
    mode_id = compute_mode_id(encoder_input, mode_index)
    branch_tokens = compute_branch_tokens_list(encoder_input, layout)
    args = branch_tokens, mode_id, len(layout.encoder_modes)
    selected_tokens = compute_selected_tokens(*args)
    return keras.Model(encoder_input, selected_tokens, name="sonic_encoder")


def build_decoder(layout):
    decoder_dim = compute_decoder_input_dim(layout)
    decoder_input = keras.Input((decoder_dim,), name="obs_dict")
    tail_dim = compute_policy_tail_dim(layout)
    token_dim = layout.token_dim
    features = compute_decoder_features(decoder_input, token_dim, tail_dim)
    hidden_dims = (2048, 2048, 1024, 1024, 512, 512)
    args = (features, hidden_dims, layout.action_dim, "g1_dyn")
    action = compute_dense_stack(*args)
    return keras.Model(decoder_input, action, name="sonic_decoder")


def compute_mode_id(encoder_input, mode_index):
    mode_id = encoder_input[:, mode_index]
    return keras.ops.cast(mode_id, "int32")


def compute_branch_tokens_list(encoder_input, layout):
    branch_tokens = []
    for mode_layout in layout.encoder_modes:
        args = encoder_input, mode_layout, layout.token_dim
        branch_tokens.append(compute_branch_tokens(*args))
    return branch_tokens


def compute_branch_tokens(encoder_input, mode_layout, token_dim):
    mode_input = compute_mode_input(encoder_input, mode_layout)
    hidden_dims = (2048, 1024, 512, 512)
    args = mode_input, hidden_dims, token_dim, mode_layout.name
    branch_latent = compute_dense_stack(*args)
    fsq_args = branch_latent, 2, 32, 0.032237, 15.515501, 0.5, 16.0
    return compute_release_fsq(*fsq_args)


def compute_selected_tokens(branch_tokens, mode_id, num_modes):
    stacked_tokens = keras.ops.stack(branch_tokens, axis=1)
    mode_weights = keras.ops.one_hot(mode_id, num_modes)
    mode_weights = keras.ops.cast(mode_weights, "float32")
    mode_weights = keras.ops.expand_dims(mode_weights, axis=-1)
    return keras.ops.sum(stacked_tokens * mode_weights, axis=1)


def compute_decoder_features(decoder_input, token_dim, tail_dim):
    token_state = decoder_input[:, :token_dim]
    policy_tail = decoder_input[:, -tail_dim:]
    return compute_cat(token_state, policy_tail)


def compute_dense_stack(inputs, hidden_dims, output_dim, prefix):
    x, layer_index = inputs, 0
    for units in hidden_dims:
        x = layers.Dense(units, name=f"{prefix}_module_{layer_index}")(x)
        x = layers.Activation("swish", name=f"{prefix}_silu_{layer_index}")(x)
        layer_index = layer_index + 2
    return layers.Dense(output_dim, name=f"{prefix}_module_{layer_index}")(x)


def compute_mode_input(encoder_input, mode_layout):
    if not mode_layout.feature_spans:
        raise ValueError(f"Encoder mode {mode_layout.name} has no spans")
    if mode_layout.temporal_frames is None:
        spans = mode_layout.feature_spans
        mode_input = compute_flat_mode_input(encoder_input, spans)
    else:
        mode_input = compute_temporal_mode_input(encoder_input, mode_layout)
    return mode_input


def compute_flat_mode_input(encoder_input, spans):
    parts = compute_parts(encoder_input, spans)
    if len(parts) == 1:
        flat_input = parts[0]
    else:
        flat_input = keras.ops.concatenate(parts, axis=-1)
    return flat_input


def compute_temporal_mode_input(encoder_input, mode_layout):
    temporal_parts = []
    num_frames = mode_layout.temporal_frames
    for group in compute_feature_groups(mode_layout):
        args = encoder_input, group, num_frames
        temporal_parts.append(compute_temporal_part(*args))
    merged = keras.ops.concatenate(temporal_parts, axis=-1)
    total_dim = compute_total_dim(mode_layout.feature_spans)
    return keras.ops.reshape(merged, (-1, total_dim))


def compute_feature_groups(mode_layout):
    if mode_layout.feature_groups:
        feature_groups = mode_layout.feature_groups
    else:
        feature_groups = tuple((span,) for span in mode_layout.feature_spans)
    return feature_groups


def compute_temporal_part(encoder_input, group, num_frames):
    # Concatenates each span in the group before reshaping into frames, so a
    # multi-span group (e.g. position+velocity) does not interleave per
    # timestep. This matches the released model's own training-time layout
    # exactly (verified against the release ONNX in sonic_test.py) even
    # though it looks unintuitive; do not "fix" the ordering here.
    merged_group = compute_flat_mode_input(encoder_input, group)
    group_dim = compute_total_dim(group)
    part_dim = group_dim // num_frames
    return keras.ops.reshape(merged_group, (-1, num_frames, part_dim))


def compute_parts(encoder_input, spans):
    return [encoder_input[:, span.start:span.end] for span in spans]


def compute_total_dim(spans):
    return sum(span.dim for span in spans)


def compute_cat(left, right):
    return keras.ops.concatenate([left, right], axis=-1)


def compute_release_fsq(
        inputs, num_tokens, token_dim, offset, scale, rounding_shift,
        divisor):
    inputs = keras.ops.cast(inputs, "float32")
    token_shape = (-1, num_tokens, token_dim)
    inputs = keras.ops.reshape(inputs, token_shape)
    inputs = keras.ops.tanh(inputs + offset) * scale
    inputs = keras.ops.round(inputs - rounding_shift)
    inputs = inputs / divisor
    flat_shape = (-1, num_tokens * token_dim)
    return keras.ops.reshape(inputs, flat_shape)
