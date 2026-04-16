from collections import namedtuple
from pathlib import Path

from keras import Model, ops
from keras.initializers import VarianceScaling
from keras.layers import Embedding, EinsumDense, Input, ReversibleEmbedding

from .layers.decoder import decoder_block
from .layers.normalization import build_rms_norm

TEXT_BACKBONE_FIELDS = (
    "vocabulary_size image_size num_layers num_query_heads "
    "num_key_value_heads hidden_dim intermediate_dim head_dim "
    "attention_logit_soft_cap final_logit_soft_cap "
    "use_sliding_window_attention sliding_window_size "
    "sliding_window_pattern global_head_dim "
    "local_rope_scaling_factor global_rope_scaling_factor "
    "global_rope_partial_rotary_factor "
    "use_bidirectional_attention layer_norm_epsilon dropout dtype "
    "hidden_size_per_layer_input num_kv_shared_layers global_layer_indices"
)
TextBackboneArgs = namedtuple("TextBackboneArgs", TEXT_BACKBONE_FIELDS)
TextIntermediates = namedtuple(
    "TextIntermediates",
    "embedding_output block_outputs final_output",
)


def build_text_backbone_args(**overrides):
    values = {
        "vocabulary_size": 256,
        "image_size": 8,
        "num_layers": 2,
        "num_query_heads": 2,
        "num_key_value_heads": 1,
        "hidden_dim": 8,
        "intermediate_dim": 16,
        "head_dim": 4,
        "attention_logit_soft_cap": None,
        "final_logit_soft_cap": None,
        "use_sliding_window_attention": True,
        "sliding_window_size": 16,
        "sliding_window_pattern": 6,
        "global_head_dim": None,
        "local_rope_scaling_factor": 1.0,
        "global_rope_scaling_factor": 1.0,
        "global_rope_partial_rotary_factor": 1.0,
        "use_bidirectional_attention": False,
        "layer_norm_epsilon": 1e-6,
        "dropout": 0.0,
        "dtype": "float32",
        "hidden_size_per_layer_input": None,
        "num_kv_shared_layers": 0,
        "global_layer_indices": None,
    }
    values.update(overrides)
    return TextBackboneArgs(**values)


class Gemma4TextBackbone(Model):
    def __init__(self, config, name="gemma4_text_backbone"):
        self.config = config
        token_embedding = build_token_embedding(
            config.vocabulary_size, config.hidden_dim, config.dtype)
        token_ids = Input((None,), dtype="int32", name="token_ids")
        padding_mask = Input(
            (None,), dtype="int32", name="padding_mask")
        hidden = token_embedding(token_ids)
        hidden = scale_token_embeddings(hidden, config.hidden_dim)
        embedding_output = hidden
        if config.hidden_size_per_layer_input:
            p = config.hidden_size_per_layer_input
            per_layer_embedding = build_per_layer_embedding(
                config.vocabulary_size,
                p * config.num_layers,
                config.dtype)
            full_embedding = per_layer_embedding(token_ids)
            per_layer_embeddings = build_per_layer_slices(
                full_embedding, config.num_layers, p)
        else:
            per_layer_embeddings = [None] * config.num_layers
        block_outputs = []
        for layer_index in range(config.num_layers):
            block_name = "decoder_block_{}".format(layer_index)
            args = (hidden, padding_mask, config, layer_index, block_name)
            kwargs = {"per_layer_embedding": per_layer_embeddings[layer_index]}
            hidden = decoder_block(*args, **kwargs)
            block_outputs.append(hidden)
        norm_args = (config.layer_norm_epsilon, config.dtype,
                     "final_normalization")
        final_output = build_rms_norm(*norm_args)(hidden)
        inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
        super().__init__(inputs, final_output, name=name)
        self._embedding_output = embedding_output
        self._block_outputs = tuple(block_outputs)
        self._final_output = final_output


def build_text_backbone(config, weights_path=None, name="gemma4_text_backbone"):  # fmt: skip
    model = Gemma4TextBackbone(config, name=name)
    if weights_path is not None:
        model.load_weights(str(Path(weights_path)))
    return model


def compute_text_intermediates(model, token_ids, padding_mask):
    outputs = [model._embedding_output]
    outputs.extend(model._block_outputs)
    outputs.append(model._final_output)
    debug = Model(model.input, outputs, name="debug_intermediates")
    inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
    results = debug(inputs)
    embedding_output = results[0]
    block_outputs = tuple(results[1:-1])
    final_output = results[-1]
    return TextIntermediates(embedding_output, block_outputs, final_output)


def build_token_embedding(vocabulary_size, hidden_dim, dtype):
    initializer = VarianceScaling(1.0, "fan_in", "untruncated_normal")
    keys = ("tie_weights", "embeddings_initializer", "dtype", "name")
    values = (True, initializer, dtype, "token_embedding")
    kwargs = dict(zip(keys, values))
    return ReversibleEmbedding(vocabulary_size, hidden_dim, **kwargs)


def scale_token_embeddings(hidden, hidden_dim):
    scale = ops.cast(hidden_dim ** 0.5, hidden.dtype)
    return hidden * scale


def build_per_layer_embedding(vocabulary_size, dim, dtype):
    # Use zeros initializer to avoid a large float32 temporary buffer
    # during construction (relevant when vocab_size is large).
    # Pre-trained weights are always loaded from file, so the initial
    # values do not matter.
    return Embedding(vocabulary_size, dim, dtype=dtype,
                     embeddings_initializer="zeros",
                     name="per_layer_embeddings")


def slice_per_layer(tensor, layer_index, per_layer_dim):
    start = layer_index * per_layer_dim
    end = (layer_index + 1) * per_layer_dim
    return tensor[..., start:end]


def build_per_layer_slices(full_embedding, num_layers, per_layer_dim):
    """Split per-layer embedding into one slice per layer."""
    return [slice_per_layer(full_embedding, i, per_layer_dim)
            for i in range(num_layers)]


def build_per_layer_model_projection(hidden, num_layers, per_layer_dim, dtype):
    """Project initial hidden state to (num_layers * per_layer_dim) dimensions.

    This is the 'model projection' component of the per-layer input — it
    provides a context-dependent conditioning signal derived from the scaled
    token embedding (before any decoder blocks).
    """
    n = num_layers * per_layer_dim
    eq = "btd,dn->btn"
    proj = EinsumDense(eq, (None, n), dtype=dtype, name="per_layer_model_projection")
    return proj(hidden)


def combine_projection_and_embedding(projection, embedding, scale):
    return (projection + embedding) * scale


def build_per_layer_combined_inputs(projection_full, embedding_full,
                                     num_layers, per_layer_dim,
                                     epsilon, dtype):
    """Combine per-layer projection and token embedding.

    For each layer i:
        per_layer_i = (rms_norm(proj_i) + embedding_i) * 2^-0.5

    The projection norm is shared across all layers.
    """
    proj_norm = build_rms_norm(
        epsilon, dtype, "per_layer_projection_norm")
    scale = ops.cast(2 ** -0.5, dtype)
    inputs = []
    for i in range(num_layers):
        proj_i = slice_per_layer(projection_full, i, per_layer_dim)
        embedding_i = slice_per_layer(
            embedding_full, i, per_layer_dim)
        proj_i_normed = proj_norm(proj_i)
        combined = combine_projection_and_embedding(
            proj_i_normed, embedding_i, scale)
        inputs.append(combined)
    return inputs


def build_cache_head_dim(config):
    if config.global_head_dim is not None:
        return config.global_head_dim
    return config.head_dim


def is_global_attention_layer(config, layer_index):
    if config.global_layer_indices is not None:
        return layer_index in config.global_layer_indices
    pattern_index = layer_index % config.sliding_window_pattern
    return pattern_index == config.sliding_window_pattern - 1


def is_kv_shared_layer(config, layer_index):
    if not config.num_kv_shared_layers:
        return False
    return layer_index >= config.num_layers - config.num_kv_shared_layers


def build_kv_source_map(config):
    """Map each kv_shared layer index to its K/V source layer index.

    Returns a dict {layer_index: source_index} for kv_shared layers only.
    Mirrors the keras-hub backbone precomputation of _kv_source.
    """
    num_kv_shared = config.num_kv_shared_layers
    if not num_kv_shared:
        return {}
    first_shared = config.num_layers - num_kv_shared
    non_shared_types = [
        "global" if is_global_attention_layer(config, j) else "local"
        for j in range(first_shared)
    ]
    kv_source = {}
    for j in range(first_shared, config.num_layers):
        layer_type = "global" if is_global_attention_layer(config, j) else "local"
        for k in range(len(non_shared_types) - 1, -1, -1):
            if non_shared_types[k] == layer_type:
                kv_source[j] = k
                break
    return kv_source


def build_feedforward_dim(config, layer_index):
    if is_kv_shared_layer(config, layer_index):
        return config.intermediate_dim * 2
    return config.intermediate_dim


def build_head_dim(config, is_global):
    if is_global and config.global_head_dim is not None:
        return config.global_head_dim
    return config.head_dim


def use_sliding_window(config, is_global):
    return config.use_sliding_window_attention and not is_global


def build_rope_wavelength(is_global):
    if is_global:
        return 1_000_000.0
    return 10_000.0


def build_rope_scaling_factor(config, is_global):
    if is_global:
        return config.global_rope_scaling_factor
    return config.local_rope_scaling_factor


def build_partial_rotary_factor(config, is_global):
    if is_global:
        return config.global_rope_partial_rotary_factor
    return 1.0
