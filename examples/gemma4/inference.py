from keras import Model, ops
from keras.layers import Input

from .layers.core import apply_tanh_soft_cap
from .layers.decoder import cached_decoder_block
from .layers.normalization import build_rms_norm
from .model import (build_cache_head_dim,
                    build_per_layer_combined_inputs,
                    build_per_layer_embedding,
                    build_per_layer_model_projection,
                    build_token_embedding,
                    scale_token_embeddings)


def Gemma4PerLayerEmbeddingStep(
        config, name="gemma4_per_layer_embedding_step"):
    """Per-layer embedding lookup only (4.7 GB for E2B).

    Kept as a separate Keras model so it can be built and loaded
    before Gemma4DecoderStep.  Loading the large embedding table in
    isolation (peak ~9.4 GB) avoids the ~14 GB peak that would occur
    if it were part of the full 9.27 GB decoder step model.

    Output shape: (batch, 1, num_layers * hidden_size_per_layer_input).
    The output is pre-scaled by sqrt(hidden_size_per_layer_input).

    Pass this output directly as the per_layer_full_embedding input
    of Gemma4DecoderStep.
    """
    p = config.hidden_size_per_layer_input
    token_ids = Input((1,), dtype="int32", name="token_ids")
    per_layer_embedding = build_per_layer_embedding(
        config.vocabulary_size, p * config.num_layers, config.dtype)
    full_embedding = per_layer_embedding(token_ids)
    full_embedding = scale_per_layer_embedding(
        full_embedding, p, config.dtype)
    return Model(token_ids, full_embedding, name=name)


def Gemma4DecoderStep(config, name="gemma4_decoder_step"):
    num_kv_heads = config.num_key_value_heads
    cache_head_dim = build_cache_head_dim(config)
    cache_shape = (
        config.num_layers, 2, None, num_kv_heads, cache_head_dim)
    token_ids = Input((1,), dtype="int32", name="token_ids")
    cache = Input(
        cache_shape, dtype=config.dtype,
        name="self_attention_cache")
    cache_index = Input(
        (), dtype="int32", name="cache_update_index")
    index_scalar = extract_cache_index(cache_index)
    embedding = build_token_embedding(
        config.vocabulary_size, config.hidden_dim, config.dtype)
    hidden = embedding(token_ids)
    if config.hidden_size_per_layer_input:
        p = config.hidden_size_per_layer_input
        # Per-layer token embeddings come from
        # Gemma4PerLayerEmbeddingStep, keeping the 4.7 GB table
        # out of this model.
        per_layer_full_embedding = Input(
            (1, p * config.num_layers),
            dtype=config.dtype,
            name="per_layer_full_embedding")
        # Per-layer model projection of the UNSCALED token
        # embedding (27 MB weight). Must be computed before
        # scale_token_embeddings.
        projection_full = build_per_layer_model_projection(
            hidden, config.num_layers, p, config.dtype)
        per_layer_embeddings = build_per_layer_combined_inputs(
            projection_full, per_layer_full_embedding,
            config.num_layers, p,
            config.layer_norm_epsilon, config.dtype)
    else:
        per_layer_full_embedding = None
        per_layer_embeddings = None
    hidden = scale_token_embeddings(hidden, config.hidden_dim)
    hidden, updated_cache = build_cached_decoder_blocks(
        hidden, cache, index_scalar, config,
        per_layer_embeddings=per_layer_embeddings)
    updated_cache = ops.cast(updated_cache, config.dtype)
    norm_args = (config.layer_norm_epsilon, config.dtype,
                 "final_normalization")
    hidden = build_rms_norm(*norm_args)(hidden)
    logits = embedding(hidden, reverse=True)
    logits = apply_tanh_soft_cap(
        logits, config.final_logit_soft_cap)
    inputs = [token_ids, cache, cache_index]
    if per_layer_full_embedding is not None:
        inputs.append(per_layer_full_embedding)
    outputs = [logits, updated_cache]
    return Model(inputs, outputs, name=name)


def scale_per_layer_embedding(full_embedding, per_layer_dim, dtype):
    scale = ops.cast(float(per_layer_dim) ** 0.5, dtype)
    return ops.cast(full_embedding, dtype) * scale


def extract_cache_index(cache_index):
    return ops.cast(cache_index[0], "int32")


def squeeze_shared_cache(cache):
    return ops.squeeze(cache, axis=1)


def slice_layer_cache(cache, layer_index):
    return cache[:, layer_index, ...]


def expand_layer_cache(layer_cache):
    return ops.expand_dims(layer_cache, axis=1)


def concat_layer_caches(caches):
    return ops.concatenate(caches, axis=1)


def build_cached_decoder_blocks(hidden, cache, index, config,
                                 per_layer_embeddings=None):
    from .model import build_kv_source_map
    kv_source_map = build_kv_source_map(config)
    updated_caches = []
    for layer_index in range(config.num_layers):
        block_name = "decoder_block_{}".format(layer_index)
        layer_cache = slice_layer_cache(cache, layer_index)
        per_layer_embedding = None
        if per_layer_embeddings is not None:
            per_layer_embedding = per_layer_embeddings[layer_index]
        shared_kv_cache = None
        source = kv_source_map.get(layer_index)
        if source is not None:
            shared_kv_cache = squeeze_shared_cache(
                updated_caches[source])
        args = (hidden, layer_cache, index,
                config, layer_index, block_name)
        kwargs = {
            "per_layer_embedding": per_layer_embedding,
            "shared_kv_cache": shared_kv_cache,
        }
        hidden, layer_cache = cached_decoder_block(
            *args, **kwargs)
        updated_caches.append(expand_layer_cache(layer_cache))
    updated = concat_layer_caches(updated_caches)
    return hidden, updated


def build_empty_cache(config, max_length, batch_size=1):
    num_kv_heads = config.num_key_value_heads
    cache_head_dim = build_cache_head_dim(config)
    shape = (batch_size, config.num_layers, 2, max_length,
             num_kv_heads, cache_head_dim)
    return ops.zeros(shape, dtype=config.dtype)
