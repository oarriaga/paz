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
from paz.models.transformers.attention import compute_masked_attention
from paz.models.transformers.attention import expand_mask_for_heads
from paz.models.transformers.attention import merge_attention_heads
from paz.models.transformers.attention import project_query_key_value
from paz.models.transformers.attention import rms_normalize_query_key
from paz.models.transformers.attention import split_query_key_value


def flow_block(x, condition, shared_signals, context, context_mask,
               num_heads, head_dim, mlp_dim, adaln_dim, max_positions,
               wavelength, name):
    hidden_dim = num_heads * head_dim
    signal_args = (condition, shared_signals, adaln_dim, hidden_dim, name)
    signals = compute_block_signals(*signal_args)
    shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = signals
    hidden = RMSNormalization(name=f"{name}_norm_1")(x)
    hidden = conditioning.modulate(hidden, shift_attn, scale_attn)
    self_args = (hidden, num_heads, head_dim, max_positions, wavelength)
    hidden = attend_self(*self_args, f"{name}_self_attention")
    x = x + conditioning.gate(hidden, gate_attn)
    hidden = RMSNormalization(name=f"{name}_norm_2")(x)
    cross_args = (hidden, context, context_mask, num_heads, head_dim)
    x = x + attend_context(*cross_args, f"{name}_cross_attention")
    hidden = RMSNormalization(name=f"{name}_norm_3")(x)
    hidden = conditioning.modulate(hidden, shift_mlp, scale_mlp)
    names = (f"{name}_mlp_gate", f"{name}_mlp_up", f"{name}_mlp_down")
    hidden = feedforward.swiglu(hidden, mlp_dim, hidden_dim, *names)
    return x + conditioning.gate(hidden, gate_mlp)


def compute_block_signals(condition, shared_signals, adaln_dim, hidden_dim,
                          name):
    hidden = ops.silu(condition)
    hidden = Dense(adaln_dim, name=f"{name}_adaln_dense_1")(hidden)
    up_project = Dense(6 * hidden_dim, name=f"{name}_adaln_dense_2")
    signals = ops.split(up_project(hidden), 6, axis=-1)
    return [block + shared for block, shared in zip(signals, shared_signals)]


def attend_self(x, num_heads, head_dim, max_positions, wavelength, name):
    hidden_dim = num_heads * head_dim
    fused = project_query_key_value(x, hidden_dim, False, name)
    query, key, value = split_query_key_value(fused, num_heads, head_dim)
    query, key = rms_normalize_query_key(query, key, name)
    rotate = RotaryTable(max_positions, wavelength, name=f"{name}_rotary")
    query, key = rotate(query), rotate(key)
    causal_mask = build_causal_mask(x)
    output = compute_masked_attention(query, key, value, causal_mask)
    output = merge_attention_heads(output)
    return Dense(hidden_dim, use_bias=False, name=f"{name}_output")(output)


def attend_context(x, context, context_mask, num_heads, head_dim, name):
    query = project_heads(x, num_heads, head_dim, f"{name}_query")
    key = project_heads(context, num_heads, head_dim, f"{name}_key")
    value = project_heads(context, num_heads, head_dim, f"{name}_value")
    query, key = rms_normalize_query_key(query, key, name)
    attention_mask = expand_mask_for_heads(context_mask)
    output = compute_masked_attention(query, key, value, attention_mask)
    output = merge_attention_heads(output)
    hidden_dim = num_heads * head_dim
    return Dense(hidden_dim, use_bias=False, name=f"{name}_output")(output)


def project_heads(x, num_heads, head_dim, name):
    hidden = Dense(num_heads * head_dim, use_bias=False, name=name)(x)
    heads_shape = (-1, num_heads, head_dim)
    heads = Reshape(heads_shape, name=f"{name}_split_heads")(hidden)
    return ops.transpose(heads, (0, 2, 1, 3))


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
    weights loaded from the checkpoint rather than recomputed. Inputs are
    head-major: (batch, heads, tokens, head_dim).
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
        num_positions = x.shape[2]
        table_shape = (1, 1, num_positions, -1)
        cosine = ops.reshape(self.cosine[:num_positions], table_shape)
        sine = ops.reshape(self.sine[:num_positions], table_shape)
        first, second = ops.split(x, 2, axis=-1)
        rotated_first = first * cosine - second * sine
        rotated_second = second * cosine + first * sine
        return ops.concatenate((rotated_first, rotated_second), axis=-1)
