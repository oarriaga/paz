import keras
from keras import ops
from keras.layers import EinsumDense, Layer

from paz.models.foundation.gemma4.configuration import (
    build_cache_head_dim, build_feedforward_dim, build_head_dim,
    build_partial_rotary_factor, build_rope_scaling_factor,
    build_rope_wavelength, is_global_attention_layer, use_sliding_window)
from paz.models.transformers.numerics import add_residual, clip_float16
from paz.models.foundation.gemma4.configuration import TextBackboneArgs
from paz.models.foundation.gemma4.layers import attention as attend_ops
from paz.models.foundation.gemma4.layers.core import build_attention_mask
from paz.models.foundation.gemma4.layers.normalization import (
    build_rms_norm, build_scalar_multiply, build_v_norm)


@keras.saving.register_keras_serializable(package="gemma4")
class Gemma4DecoderLayer(Layer):
    """One Gemma4 transformer block owning its sublayers.

    `call` runs the full-sequence forward; `call_with_cache` runs the cached
    single/multi-step forward. Both thread the owned sublayers into the pure
    helpers in `attention.py`; the transformer math is shared, not duplicated.
    """

    def __init__(self, config, layer_index, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layer_index = layer_index
        self.is_global = is_global_attention_layer(config, layer_index)
        self.head_dim = build_head_dim(config, self.is_global)
        self.cache_head_dim = build_cache_head_dim(config)
        self.wavelength = build_rope_wavelength(config, self.is_global)
        self.scaling_factor = build_rope_scaling_factor(config, self.is_global)
        self.partial_rotary = build_partial_rotary_factor(
            config, self.is_global)
        self.window = window_size(config, self.is_global)
        self.feedforward_dim = build_feedforward_dim(config, layer_index)
        self.has_per_layer = bool(config.hidden_size_per_layer_input)
        self.build_attention_layers()
        self.build_feedforward_layers()
        if self.has_per_layer:
            self.build_per_layer_layers()
        self.layer_scalar = build_scalar_multiply("layer_scalar")

    def build_attention_layers(self):
        config, head_dim = self.config, self.head_dim
        epsilon, dtype = config.layer_norm_epsilon, config.dtype
        self.pre_attention_norm = build_rms_norm(
            epsilon, dtype, "pre_attention_norm")
        self.query_proj = head_projection(
            "btd,ndh->btnh", config.num_query_heads, head_dim, dtype,
            "attention_query")
        self.query_norm = build_rms_norm(epsilon, dtype, "attention_query_norm")
        self.key_proj = head_projection(
            "btd,kdh->btkh", config.num_key_value_heads, head_dim, dtype,
            "attention_key")
        self.key_norm = build_rms_norm(epsilon, dtype, "attention_key_norm")
        self.value_proj = head_projection(
            "btd,kdh->btkh", config.num_key_value_heads, head_dim, dtype,
            "attention_value")
        self.value_norm = build_v_norm(epsilon, dtype, "attention_value_norm")
        self.output_proj = dense(
            "btnh,nhd->btd", config.hidden_dim, dtype, "attention_output")
        self.post_attention_norm = build_rms_norm(
            epsilon, dtype, "post_attention_norm")

    def build_feedforward_layers(self):
        config = self.config
        epsilon, dtype = config.layer_norm_epsilon, config.dtype
        self.pre_ffw_norm = build_rms_norm(epsilon, dtype, "pre_ffw_norm")
        self.ffw_gating = dense(
            "btd,df->btf", self.feedforward_dim, dtype, "ffw_gating")
        self.ffw_gating_2 = dense(
            "btd,df->btf", self.feedforward_dim, dtype, "ffw_gating_2")
        self.ffw_linear = dense(
            "btf,fd->btd", config.hidden_dim, dtype, "ffw_linear")
        self.post_ffw_norm = build_rms_norm(epsilon, dtype, "post_ffw_norm")

    def build_per_layer_layers(self):
        config = self.config
        epsilon, dtype = config.layer_norm_epsilon, config.dtype
        per_layer_dim = config.hidden_size_per_layer_input
        self.per_layer_gate = dense(
            "btd,dp->btp", per_layer_dim, dtype, "per_layer_gate")
        self.per_layer_projection = dense(
            "btp,pd->btd", config.hidden_dim, dtype, "per_layer_projection")
        self.post_per_layer_norm = build_rms_norm(
            epsilon, dtype, "post_per_layer_norm")

    def call(self, x, padding_mask=None, per_layer_embedding=None,
             shared_kv=None):
        x = clip_bfloat_guard(x)
        hidden, kv = self.attention(x, padding_mask, shared_kv)
        hidden = self.feedforward(hidden)
        if per_layer_embedding is not None:
            hidden = self.per_layer_input(hidden, per_layer_embedding)
        return self.layer_scalar(hidden), kv

    def call_with_cache(self, x, cache, index, positions=None,
                        per_layer_embedding=None, shared_kv_cache=None):
        x = clip_bfloat_guard(x)
        hidden, cache = self.cached_attention(
            x, cache, index, positions, shared_kv_cache)
        hidden = self.feedforward(hidden)
        if per_layer_embedding is not None:
            hidden = self.per_layer_input(hidden, per_layer_embedding)
        return self.layer_scalar(hidden), cache

    def attention(self, x, padding_mask, shared_kv):
        normed = self.pre_attention_norm(x)
        mask = self.attention_mask(padding_mask)
        hidden, kv = self.attend(normed, mask, shared_kv)
        hidden = self.post_attention_norm(hidden)
        return add_residual(x, hidden), kv

    def attend(self, x, mask, shared_kv):
        query = self.query_with_rope(x)
        if shared_kv is not None:
            key, value = shared_kv[:, 0, ...], shared_kv[:, 1, ...]
        else:
            key = self.key_with_rope(x)
            value = attend_ops.project(x, self.value_proj, self.value_norm)
        kv = ops.stack((key, value), axis=1)
        output = self.mix(query, key, value, mask)
        output = attend_ops.zero_masked_positions(output, mask)
        return self.output_proj(output), kv

    def cached_attention(self, x, cache, index, positions, shared_kv_cache):
        normed = self.pre_attention_norm(x)
        hidden, cache = self.cached_attend(
            normed, cache, index, positions, shared_kv_cache)
        hidden = self.post_attention_norm(hidden)
        return add_residual(x, hidden), cache

    def cached_attend(self, x, cache, index, positions, shared_kv_cache):
        cache_positions = attend_ops.build_cache_positions(index, positions)
        query = self.query_with_rope(x, cache_positions)
        if shared_kv_cache is not None:
            kv_source, updated_cache = shared_kv_cache, cache
        else:
            key = self.key_with_rope(x, cache_positions)
            value = attend_ops.project(x, self.value_proj, self.value_norm)
            updated_cache = attend_ops.update_kv_cache(
                cache, index, key, value, self.head_dim, self.cache_head_dim)
            kv_source = updated_cache
        key, value = attend_ops.read_kv_cache(
            kv_source, self.head_dim, self.cache_head_dim)
        mask = attend_ops.build_cache_mask(key, index, positions, self.window)
        output = self.mix(query, key, value, mask)
        return self.output_proj(output), updated_cache

    def query_with_rope(self, x, positions=None):
        query = attend_ops.project(x, self.query_proj, self.query_norm)
        return self.rope(query, positions)

    def key_with_rope(self, x, positions=None):
        key = attend_ops.project(x, self.key_proj, self.key_norm)
        return self.rope(key, positions)

    def rope(self, x, positions=None):
        return attend_ops.apply_rope(
            x, self.wavelength, self.scaling_factor, self.partial_rotary,
            positions)

    def mix(self, query, key, value, mask):
        args = (query, key, value, mask, self.config.num_query_heads,
                self.config.num_key_value_heads, self.head_dim,
                self.config.attention_logit_soft_cap, self.config.dropout,
                self.config.dtype, self.name + "_attention")
        return attend_ops.compute_attention(*args)

    def attention_mask(self, padding_mask):
        return build_attention_mask(
            padding_mask, self.config.use_bidirectional_attention, self.window)

    def feedforward(self, x):
        hidden = self.pre_ffw_norm(x)
        gate = keras.activations.gelu(self.ffw_gating(hidden), approximate=True)
        hidden = self.ffw_linear(gate * self.ffw_gating_2(hidden))
        hidden = self.post_ffw_norm(hidden)
        return add_residual(x, hidden)

    def per_layer_input(self, x, per_layer_embedding):
        gate = keras.activations.gelu(
            self.per_layer_gate(x), approximate=True)
        hidden = self.per_layer_projection(gate * per_layer_embedding)
        hidden = self.post_per_layer_norm(hidden)
        return add_residual(x, hidden)

    def get_config(self):
        config = super().get_config()
        config["config"] = self.config._asdict()
        config["layer_index"] = self.layer_index
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["config"] = TextBackboneArgs(**config["config"])
        return cls(**config)


def window_size(config, is_global):
    if use_sliding_window(config, is_global):
        return config.sliding_window_size
    return None


def clip_bfloat_guard(x):
    if keras.backend.standardize_dtype(x.dtype) == "float16":
        return clip_float16(x)
    return x


def head_projection(equation, num_heads, head_dim, dtype, name):
    shape = (None, num_heads, head_dim)
    return EinsumDense(equation, shape, dtype=dtype, name=name)


def dense(equation, output_dim, dtype, name):
    return EinsumDense(equation, (None, output_dim), dtype=dtype, name=name)
