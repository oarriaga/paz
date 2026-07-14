import keras
from keras import ops
from keras.initializers import VarianceScaling
from keras.layers import Embedding, EinsumDense

from paz.models.transformers.embeddings.reversible import ReversibleEmbedding
from paz.models.transformers import cache as kv_cache
from paz.models.foundation.gemma4.configuration import TextBackboneArgs
from paz.models.foundation.gemma4.configuration import build_cache_head_dim
from paz.models.foundation.gemma4.configuration import build_kv_source_map
from paz.models.foundation.gemma4.layers.decoder import Gemma4DecoderLayer
from paz.models.foundation.gemma4.layers.normalization import build_rms_norm

BACKBONE_NAME = "gemma4_backbone"


@keras.saving.register_keras_serializable(package="gemma4")
class Gemma4Backbone(keras.Model):
    """Subclassed Gemma4 text backbone owning all transformer weights once.

    `call({token_ids, padding_mask})` is the full-sequence forward.
    `call_with_cache(input_embedding, cache, index, ...)` is the cached
    single/multi-step forward used for generation. Both share the same owned
    decoder layers, token embedding and per-layer embedding table.
    """

    def __init__(self, config, name=BACKBONE_NAME, **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = config
        self.has_per_layer = bool(config.hidden_size_per_layer_input)
        self.kv_source_map = build_kv_source_map(config)
        self.token_embedding = build_token_embedding(
            config.vocabulary_size, config.hidden_dim, config.dtype)
        self.decoder_layers = build_decoder_layers(config)
        self.final_normalization = build_rms_norm(
            config.layer_norm_epsilon, config.dtype, "final_normalization")
        if self.has_per_layer:
            self.build_per_layer_embedding()

    def build_per_layer_embedding(self):
        config = self.config
        dim = config.hidden_size_per_layer_input * config.num_layers
        self.per_layer_embeddings = build_per_layer_embedding(
            config.vocabulary_size, dim, config.dtype)
        self.per_layer_model_projection = EinsumDense(
            "btd,dn->btn", (None, dim), dtype=config.dtype,
            name="per_layer_model_projection")
        self.per_layer_projection_norm = build_rms_norm(
            config.layer_norm_epsilon, config.dtype,
            "per_layer_projection_norm")

    def call(self, inputs):
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]
        embedding = self.token_embedding(token_ids)
        return self.forward_from_embedding(embedding, padding_mask, token_ids)

    def forward_from_embedding(self, embedding, padding_mask, token_ids):
        per_layer = self.per_layer_inputs(
            embedding, self.per_layer_lookup(token_ids))
        hidden = scale_embeddings(embedding, self.config.hidden_dim)
        hidden = self.run_layers(hidden, padding_mask, per_layer)
        return self.final_normalization(hidden)

    def run_layers(self, hidden, padding_mask, per_layer):
        layer_kvs = []
        for index, layer in enumerate(self.decoder_layers):
            shared = self.shared_kv(layer_kvs, index)
            hidden, kv = layer(hidden, padding_mask=padding_mask,
                               per_layer_embedding=per_layer[index],
                               shared_kv=shared)
            layer_kvs.append(kv)
        return hidden

    def shared_kv(self, layer_kvs, index):
        source = self.kv_source_map.get(index)
        return layer_kvs[source] if source is not None else None

    def call_with_cache(self, input_embedding, cache, index, positions=None,
                        per_layer_full=None):
        per_layer = self.per_layer_inputs(input_embedding, per_layer_full)
        hidden = scale_embeddings(input_embedding, self.config.hidden_dim)
        hidden, updated = self.run_cached_layers(
            hidden, cache, index, positions, per_layer)
        updated = ops.cast(updated, self.config.dtype)
        return self.final_normalization(hidden), updated

    def run_cached_layers(self, hidden, cache, index, positions, per_layer):
        updated_caches = []
        for i, layer in enumerate(self.decoder_layers):
            shared = self.shared_kv_cache(updated_caches, i)
            hidden, layer_cache = layer.call_with_cache(
                hidden, cache[:, i, ...], index, positions=positions,
                per_layer_embedding=per_layer[i], shared_kv_cache=shared)
            updated_caches.append(ops.expand_dims(layer_cache, axis=1))
        return hidden, ops.concatenate(updated_caches, axis=1)

    def shared_kv_cache(self, updated_caches, index):
        source = self.kv_source_map.get(index)
        if source is None:
            return None
        return ops.squeeze(updated_caches[source], axis=1)

    def per_layer_lookup(self, token_ids):
        if not self.has_per_layer:
            return None
        embedding = self.per_layer_embeddings(token_ids)
        dim = self.config.hidden_size_per_layer_input
        return scale_per_layer_embedding(embedding, dim, self.config.dtype)

    def per_layer_inputs(self, unscaled_embedding, per_layer_full):
        if not self.has_per_layer or per_layer_full is None:
            return [None] * self.config.num_layers
        projection = self.per_layer_model_projection(unscaled_embedding)
        return self.combine_per_layer(projection, per_layer_full)

    def combine_per_layer(self, projection_full, embedding_full):
        dim = self.config.hidden_size_per_layer_input
        scale = ops.cast(2 ** -0.5, self.config.dtype)
        inputs = []
        for index in range(self.config.num_layers):
            projection = slice_per_layer(projection_full, index, dim)
            embedding = slice_per_layer(embedding_full, index, dim)
            normed = self.per_layer_projection_norm(projection)
            inputs.append((normed + embedding) * scale)
        return inputs

    def build_cache(self, max_length, batch_size=1):
        return build_empty_cache(self.config, max_length, batch_size)

    def get_config(self):
        config = super().get_config()
        config["config"] = self.config._asdict()
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["config"] = TextBackboneArgs(**config["config"])
        return cls(**config)


def build_decoder_layers(config):
    layers = []
    for index in range(config.num_layers):
        name = "decoder_block_{}".format(index)
        layers.append(Gemma4DecoderLayer(config, index, name=name))
    return layers


def build_token_embedding(vocabulary_size, hidden_dim, dtype):
    initializer = VarianceScaling(1.0, "fan_in", "untruncated_normal")
    keys = ("tie_weights", "embeddings_initializer", "dtype", "name")
    values = (True, initializer, dtype, "token_embedding")
    kwargs = dict(zip(keys, values))
    return ReversibleEmbedding(vocabulary_size, hidden_dim, **kwargs)


def build_per_layer_embedding(vocabulary_size, dim, dtype):
    # Zeros initializer avoids a large float32 temporary during construction;
    # pretrained weights are always loaded from file afterwards.
    return Embedding(vocabulary_size, dim, dtype=dtype,
                     embeddings_initializer="zeros",
                     name="per_layer_embeddings")


def scale_embeddings(hidden, hidden_dim):
    return hidden * ops.cast(hidden_dim ** 0.5, hidden.dtype)


def scale_per_layer_embedding(full_embedding, per_layer_dim, dtype):
    scale = ops.cast(float(per_layer_dim) ** 0.5, dtype)
    return ops.cast(full_embedding, dtype) * scale


def slice_per_layer(tensor, layer_index, per_layer_dim):
    start = layer_index * per_layer_dim
    end = (layer_index + 1) * per_layer_dim
    return tensor[..., start:end]


def build_empty_cache(config, max_length, batch_size=1):
    cache_head_dim = build_cache_head_dim(config)
    args = (batch_size, config.num_layers, max_length,
            config.num_key_value_heads, cache_head_dim, config.dtype)
    return kv_cache.build(*args)


def build_text_backbone_args(**overrides):
    values = {
        "vocabulary_size": 256, "image_size": 8, "num_layers": 2,
        "num_query_heads": 2, "num_key_value_heads": 1, "hidden_dim": 8,
        "intermediate_dim": 16, "head_dim": 4,
        "attention_logit_soft_cap": None, "final_logit_soft_cap": None,
        "use_sliding_window_attention": True, "sliding_window_size": 16,
        "sliding_window_pattern": 6, "global_head_dim": None,
        "local_rope_wavelength": 10_000.0,
        "global_rope_wavelength": 1_000_000.0,
        "local_rope_scaling_factor": 1.0, "global_rope_scaling_factor": 1.0,
        "global_rope_partial_rotary_factor": 1.0,
        "use_bidirectional_attention": False, "layer_norm_epsilon": 1e-6,
        "dropout": 0.0, "dtype": "float32", "hidden_size_per_layer_input": None,
        "num_kv_shared_layers": 0, "global_layer_indices": None,
        "use_double_wide_mlp": False,
    }
    values.update(overrides)
    return TextBackboneArgs(**values)
