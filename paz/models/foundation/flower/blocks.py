"""FLOWER flow-matching DiT block.

Per block: adaLN-zero modulated causal self-attention with RoPE, ungated
unmasked cross-attention to the projected VLM context, and an adaLN-zero
modulated SwiGLU MLP. The six modulation signals are the sum of the
per-block adaLN projection and the shared per-action-space controller.
"""
import keras
from keras import ops
from keras.layers import Dense, Reshape

from paz.layers import RMSNormalization
from paz.models.transformers import conditioning, feedforward, mask
from paz.models.transformers.attention import apply_attention
from paz.models.transformers.attention import compute_scores
from paz.models.transformers.attention import expand_mask_for_heads
from paz.models.transformers.attention import mask_scores, merge_heads
from paz.models.transformers.attention import transpose_to_heads


def flow_block(x, condition, shared_signals, context, context_mask, config,
               name):
    signals = compute_block_signals(condition, shared_signals, config, name)
    shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = signals
    hidden = RMSNormalization(name=f"{name}_norm_1")(x)
    hidden = conditioning.modulate(hidden, shift_attn, scale_attn)
    hidden = attend_self(hidden, config, f"{name}_self_attention")
    x = x + conditioning.gate(hidden, gate_attn)
    hidden = RMSNormalization(name=f"{name}_norm_2")(x)
    cross_args = (hidden, context, context_mask, config)
    x = x + attend_context(*cross_args, f"{name}_cross_attention")
    hidden = RMSNormalization(name=f"{name}_norm_3")(x)
    hidden = conditioning.modulate(hidden, shift_mlp, scale_mlp)
    names = (f"{name}_mlp_gate", f"{name}_mlp_up", f"{name}_mlp_down")
    hidden = feedforward.swiglu(hidden, config.mlp_dim, config.hidden_dim,
                                *names)
    return x + conditioning.gate(hidden, gate_mlp)


def compute_block_signals(condition, shared_signals, config, name):
    hidden = ops.silu(condition)
    hidden = Dense(config.adaln_dim, name=f"{name}_adaln_dense_1")(hidden)
    up_project = Dense(6 * config.hidden_dim, name=f"{name}_adaln_dense_2")
    signals = ops.split(up_project(hidden), 6, axis=-1)
    return [block + shared for block, shared in zip(signals, shared_signals)]


def attend_self(x, config, name):
    query = project_heads(x, config, f"{name}_query")
    key = project_heads(x, config, f"{name}_key")
    value = project_heads(x, config, f"{name}_value")
    query = RMSNormalization(name=f"{name}_query_norm")(query)
    key = RMSNormalization(name=f"{name}_key_norm")(key)
    rotary_args = (config.rope_max_positions, config.rope_wavelength)
    rotate = RotaryTable(*rotary_args, name=f"{name}_rotary")
    query, key = rotate(query), rotate(key)
    causal_mask = build_causal_mask(x)
    return attend_heads(query, key, value, causal_mask, config, name)


def attend_context(x, context, context_mask, config, name):
    query = project_heads(x, config, f"{name}_query")
    key = project_heads(context, config, f"{name}_key")
    value = project_heads(context, config, f"{name}_value")
    query = RMSNormalization(name=f"{name}_query_norm")(query)
    key = RMSNormalization(name=f"{name}_key_norm")(key)
    context_mask = expand_mask_for_heads(context_mask)
    return attend_heads(query, key, value, context_mask, config, name)


def attend_heads(query, key, value, attention_mask, config, name):
    query = transpose_to_heads(query)
    key = transpose_to_heads(key)
    value = transpose_to_heads(value)
    scores = compute_scores(query, key, config.head_dim)
    scores = mask_scores(scores, attention_mask)
    output = merge_heads(apply_attention(scores, value, 0.0, name))
    flat_shape = (-1, config.num_heads * config.head_dim)
    output = Reshape(flat_shape, name=f"{name}_merge_heads")(output)
    return Dense(config.hidden_dim, use_bias=False,
                 name=f"{name}_output")(output)


def project_heads(x, config, name):
    hidden = Dense(config.hidden_dim, use_bias=False, name=name)(x)
    heads_shape = (-1, config.num_heads, config.head_dim)
    return Reshape(heads_shape, name=f"{name}_split_heads")(hidden)


def build_causal_mask(x):
    positions = ops.arange(x.shape[1])
    causal = mask.causal(positions, positions)
    return expand_mask_for_heads(ops.expand_dims(causal, axis=0))


@keras.saving.register_keras_serializable("flower")
class RotaryTable(keras.layers.Layer):
    """Half-split RoPE with stored cosine/sine tables.

    The FLOWER checkpoint carries the tables as buffers and their values do
    not all match the configured wavelength (blocks inherited from an
    earlier pretraining stage use a different one), so the tables are
    weights loaded from the checkpoint rather than recomputed.
    """

    def __init__(self, max_positions, wavelength, **kwargs):
        super().__init__(**kwargs)
        self.max_positions = max_positions
        self.wavelength = wavelength

    def get_config(self):
        config = super().get_config()
        config["max_positions"] = self.max_positions
        config["wavelength"] = self.wavelength
        return config

    def build(self, input_shape):
        angles = self.build_angles(input_shape[-1])
        shape = (self.max_positions, input_shape[-1] // 2)
        cosine_init = lambda *args, **_: ops.cos(angles)
        sine_init = lambda *args, **_: ops.sin(angles)
        cosine_kwargs = {"shape": shape, "trainable": False,
                         "name": "cosine", "initializer": cosine_init}
        sine_kwargs = {"shape": shape, "trainable": False,
                       "name": "sine", "initializer": sine_init}
        self.cosine = self.add_weight(**cosine_kwargs)
        self.sine = self.add_weight(**sine_kwargs)
        self.built = True

    def build_angles(self, head_dim):
        indices = ops.arange(0, head_dim, 2, dtype="float32")
        frequencies = ops.power(self.wavelength, -indices / head_dim)
        positions = ops.arange(self.max_positions, dtype="float32")
        return ops.expand_dims(positions, axis=1) * frequencies

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        num_positions = x.shape[1]
        table_shape = (1, num_positions, 1, -1)
        cosine = ops.reshape(self.cosine[:num_positions], table_shape)
        sine = ops.reshape(self.sine[:num_positions], table_shape)
        first, second = ops.split(x, 2, axis=-1)
        rotated_first = first * cosine - second * sine
        rotated_second = second * cosine + first * sine
        return ops.concatenate((rotated_first, rotated_second), axis=-1)
