import hashlib
import json
import shutil
from collections import namedtuple
from pathlib import Path

from keras import Model, ops
from keras.initializers import VarianceScaling
from keras.layers import Embedding, EinsumDense, Input
from keras.utils import get_file
from paz.models.transformers.embeddings.reversible import ReversibleEmbedding

from paz.models.transformers import cache as kv_cache
from paz.models.transformers.logits import soft_cap as apply_soft_cap
from paz.models.foundation.gemma4.configuration import TextBackboneArgs
from paz.models.foundation.gemma4.configuration import build_cache_head_dim
from paz.models.foundation.gemma4.configuration import build_kv_source_map
from paz.models.foundation.gemma4.configuration import load_config
from paz.models.foundation.gemma4.layers.decoder import decoder_block
from paz.models.foundation.gemma4.layers.decoder import cached_decoder_block
from paz.models.foundation.gemma4.layers.normalization import build_rms_norm
from paz.models.foundation.gemma4.vision import VisionEncoderArgs
from paz.models.foundation.gemma4.vision import build_vision_encoder

BACKBONE_NAME = "gemma4_text_backbone"
TextIntermediates = namedtuple(
    "TextIntermediates", "embedding_output block_outputs final_output")


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
        "local_rope_wavelength": 10_000.0,
        "global_rope_wavelength": 1_000_000.0,
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
        "use_double_wide_mlp": False,
    }
    values.update(overrides)
    return TextBackboneArgs(**values)


def build_text_backbone(config, weights_path=None, name=BACKBONE_NAME):
    token_embedding = build_token_embedding(
        config.vocabulary_size, config.hidden_dim, config.dtype)
    token_ids = Input((None,), dtype="int32", name="token_ids")
    padding_mask = Input((None,), dtype="int32", name="padding_mask")
    embedding = token_embedding(token_ids)
    inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
    return build_backbone_from_embedding(
        embedding, token_ids, padding_mask, inputs, config, name, weights_path)


def build_backbone_from_embedding(embedding, token_ids, padding_mask, inputs,
                                  config, name, weights_path=None):
    hidden = scale_token_embeddings(embedding, config.hidden_dim)
    embedding_output = hidden
    per_layer = build_backbone_per_layer_embeddings(
        config, token_ids, embedding)
    kv_source = build_kv_source_map(config)
    block_outputs, layer_kvs = [], []
    for layer_index in range(config.num_layers):
        block_name = "decoder_block_{}".format(layer_index)
        source = kv_source.get(layer_index)
        shared_kv = layer_kvs[source] if source is not None else None
        args = (hidden, padding_mask, config, layer_index, block_name)
        kwargs = {"per_layer_embedding": per_layer[layer_index],
                  "shared_kv": shared_kv}
        hidden, kv = decoder_block(*args, **kwargs)
        block_outputs.append(hidden)
        layer_kvs.append(kv)
    norm_args = (config.layer_norm_epsilon, config.dtype,
                 "final_normalization")
    final_output = build_rms_norm(*norm_args)(hidden)
    model = Model(inputs, final_output, name=name)
    attach_intermediates(model, embedding_output, block_outputs, final_output)
    if weights_path is not None:
        model.load_weights(str(Path(weights_path)))
    return model


def build_backbone_per_layer_embeddings(config, token_ids, token_embedding):
    if not config.hidden_size_per_layer_input:
        return [None] * config.num_layers
    p = config.hidden_size_per_layer_input
    embedding = build_per_layer_embedding(
        config.vocabulary_size, p * config.num_layers, config.dtype)
    full_embedding = embedding(token_ids)
    full_embedding = scale_per_layer_embedding(full_embedding, p, config.dtype)
    projection = build_per_layer_model_projection(
        token_embedding, config.num_layers, p, config.dtype)
    args = (projection, full_embedding, config.num_layers, p,
            config.layer_norm_epsilon, config.dtype)
    return build_per_layer_combined_inputs(*args)


def scale_per_layer_embedding(full_embedding, per_layer_dim, dtype):
    scale = ops.cast(float(per_layer_dim) ** 0.5, dtype)
    return ops.cast(full_embedding, dtype) * scale


def attach_intermediates(model, embedding_output, block_outputs, final_output):
    model._embedding_output = embedding_output
    model._block_outputs = tuple(block_outputs)
    model._final_output = final_output


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


def build_per_layer_model_projection(hidden, num_layers, per_layer_dim, dtype):
    """Project initial hidden state to (num_layers * per_layer_dim) dimensions.

    This is the 'model projection' component of the per-layer input — it
    provides a context-dependent conditioning signal derived from the scaled
    token embedding (before any decoder blocks).
    """
    combined_dim = num_layers * per_layer_dim
    equation = "btd,dn->btn"
    name = "per_layer_model_projection"
    projection = EinsumDense(
        equation, (None, combined_dim), dtype=dtype, name=name)
    return projection(hidden)


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
    projection_norm = build_rms_norm(
        epsilon, dtype, "per_layer_projection_norm")
    scale = ops.cast(2 ** -0.5, dtype)
    inputs = []
    for layer_index in range(num_layers):
        projection = slice_per_layer(
            projection_full, layer_index, per_layer_dim)
        embedding = slice_per_layer(embedding_full, layer_index, per_layer_dim)
        normed = projection_norm(projection)
        combined = combine_projection_and_embedding(normed, embedding, scale)
        inputs.append(combined)
    return inputs


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
    cache, cache_index = build_cache_inputs(config)
    token_ids = Input((1,), dtype="int32", name="token_ids")
    embedding = build_token_embedding(
        config.vocabulary_size, config.hidden_dim, config.dtype)
    hidden = embedding(token_ids)
    per_layer = build_per_layer_input(config)
    outputs = build_cached_step(
        hidden, embedding, cache, cache_index, per_layer, config)
    inputs = [token_ids, cache, cache_index]
    if per_layer is not None:
        inputs.append(per_layer)
    return Model(inputs, list(outputs), name=name)


def Gemma4MultimodalDecoderStep(
        config, name="gemma4_multimodal_decoder_step"):
    """Cached decoder step fed a precomputed (unscaled) input embedding.

    Image positions feed the vision embedding (pre-scaled by hidden_dim**-0.5);
    text positions feed the token embedding. Otherwise identical to
    Gemma4DecoderStep.
    """
    cache, cache_index = build_cache_inputs(config)
    input_embedding = Input(
        (None, config.hidden_dim), dtype=config.dtype, name="input_embedding")
    positions = Input((None,), dtype="int32", name="positions")
    embedding = build_token_embedding(
        config.vocabulary_size, config.hidden_dim, config.dtype)
    per_layer = None
    if config.hidden_size_per_layer_input:
        dim = config.hidden_size_per_layer_input * config.num_layers
        per_layer = Input(
            (None, dim), dtype=config.dtype, name="per_layer_full_embedding")
    outputs = build_cached_step(
        input_embedding, embedding, cache, cache_index, per_layer, config,
        positions=positions)
    inputs = [input_embedding, cache, cache_index, positions]
    if per_layer is not None:
        inputs.append(per_layer)
    return Model(inputs, list(outputs), name=name)


def build_cache_inputs(config):
    cache_head_dim = build_cache_head_dim(config)
    cache_shape = (config.num_layers, 2, None,
                   config.num_key_value_heads, cache_head_dim)
    cache = Input(cache_shape, dtype=config.dtype, name="self_attention_cache")
    cache_index = Input((), dtype="int32", name="cache_update_index")
    return cache, cache_index


def build_per_layer_input(config):
    if not config.hidden_size_per_layer_input:
        return None
    dim = config.hidden_size_per_layer_input * config.num_layers
    return Input((1, dim), dtype=config.dtype, name="per_layer_full_embedding")


def build_cached_step(hidden, embedding, cache, cache_index, per_layer, config,
                      positions=None):
    index_scalar = extract_cache_index(cache_index)
    per_layer_embeddings = None
    if per_layer is not None:
        p = config.hidden_size_per_layer_input
        # Per-layer model projection of the UNSCALED embedding (before scaling).
        projection_full = build_per_layer_model_projection(
            hidden, config.num_layers, p, config.dtype)
        per_layer_embeddings = build_per_layer_combined_inputs(
            projection_full, per_layer, config.num_layers, p,
            config.layer_norm_epsilon, config.dtype)
    hidden = scale_token_embeddings(hidden, config.hidden_dim)
    hidden, updated_cache = build_cached_decoder_blocks(
        hidden, cache, index_scalar, config,
        per_layer_embeddings=per_layer_embeddings, positions=positions)
    updated_cache = ops.cast(updated_cache, config.dtype)
    norm_args = (config.layer_norm_epsilon, config.dtype,
                 "final_normalization")
    hidden = build_rms_norm(*norm_args)(hidden)
    logits = embedding(hidden, reverse=True)
    logits = apply_soft_cap(logits, config.final_logit_soft_cap)
    return logits, updated_cache


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
                                 per_layer_embeddings=None, positions=None):
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
            "positions": positions,
        }
        hidden, layer_cache = cached_decoder_block(
            *args, **kwargs)
        updated_caches.append(expand_layer_cache(layer_cache))
    updated = concat_layer_caches(updated_caches)
    return hidden, updated


def build_empty_cache(config, max_length, batch_size=1):
    num_kv_heads = config.num_key_value_heads
    cache_head_dim = build_cache_head_dim(config)
    args = (batch_size, config.num_layers, max_length, num_kv_heads,
            cache_head_dim, config.dtype)
    return kv_cache.build(*args)


GEMMA4_WEIGHTS_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.24/"  # fmt: skip
GEMMA4_CACHE = "paz/models/gemma4"
# GitHub release assets are capped at 2 GB, so larger files are uploaded as
# byte-identical parts and reassembled on download (scripts/shard_gemma4.py).
PART_BYTES = 1_900_000_000
GEMMA4_WEIGHT_FILES = (
    "config.json", "tokenizer.json", "vision_config.json",
    "vision_encoder.weights.h5", "decoder_step.weights.h5",
    "embedding_step.weights.h5",
)
Gemma4Models = namedtuple(
    "Gemma4", "config decoder_step per_layer_step vision_encoder")


def Gemma4(model_name="gemma4_2b", weights="pretrained", models_path=None):
    model_dir = resolve_dir(model_name, models_path)
    config = load_config(model_dir / "config.json")
    decoder_step = Gemma4MultimodalDecoderStep(config)
    per_layer_step = build_per_layer_step(config)
    vision_encoder = build_vision_encoder_from_dir(model_dir)
    if weights is not None:
        decoder_step.load_weights(str(model_dir / "decoder_step.weights.h5"))
        load_optional_weights(model_dir, per_layer_step, vision_encoder)
    return Gemma4Models(config, decoder_step, per_layer_step, vision_encoder)


def resolve_dir(model_name, models_path):
    if models_path is not None:
        return Path(models_path)
    return download_weights(model_name)


def download_weights(model_name):
    subdir = "{}/{}".format(GEMMA4_CACHE, model_name)
    asset = "{}.manifest.json".format(model_name)
    manifest_path = Path(get_file(
        asset, GEMMA4_WEIGHTS_URL + asset, cache_subdir=subdir))
    model_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    for filename, entry in manifest.items():
        assemble_weights_file(model_dir / filename, entry, subdir)
    return model_dir


def assemble_weights_file(path, entry, subdir):
    checksum = entry.get("sha256")
    if is_complete(path, checksum):
        return path
    parts = []
    for asset in entry["parts"]:
        parts.append(get_file(
            asset, GEMMA4_WEIGHTS_URL + asset, cache_subdir=subdir))
    concatenate_parts(parts, path)
    if checksum is not None and compute_sha256(path) != checksum:
        raise ValueError("Checksum mismatch after assembling {}".format(path))
    return path


def is_complete(path, checksum):
    if not path.exists():
        return False
    return checksum is None or compute_sha256(path) == checksum


def concatenate_parts(parts, output):
    with open(str(output), "wb") as merged:
        for part in parts:
            with open(str(part), "rb") as chunk:
                shutil.copyfileobj(chunk, merged)
    return output


def compute_sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shard_weights(source_dir, model_name, output_dir, part_bytes=PART_BYTES):
    """Split a local weights dir into <2 GB parts plus an upload manifest."""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for filename in GEMMA4_WEIGHT_FILES:
        source = source_dir / filename
        if not source.exists():
            continue
        prefix = "{}_{}".format(model_name, filename)
        parts = split_file(source, output_dir, prefix, part_bytes)
        manifest[filename] = {"parts": parts, "sha256": compute_sha256(source)}
    manifest_path = output_dir / "{}.manifest.json".format(model_name)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def split_file(source, output_dir, prefix, part_bytes=PART_BYTES):
    output_dir = Path(output_dir)
    parts, index = [], 0
    with open(str(source), "rb") as file:
        while True:
            block = file.read(part_bytes)
            if not block:
                break
            asset = "{}.part{}".format(prefix, index)
            (output_dir / asset).write_bytes(block)
            parts.append(asset)
            index = index + 1
    return parts


def build_per_layer_step(config):
    if not config.hidden_size_per_layer_input:
        return None
    return Gemma4PerLayerEmbeddingStep(config)


def build_vision_encoder_from_dir(model_dir):
    path = Path(model_dir) / "vision_config.json"
    if not path.exists():
        return None
    with open(str(path), encoding="utf-8") as file:
        config = VisionEncoderArgs(**json.load(file))
    return build_vision_encoder(config)


def load_vision_encoder(model_dir, weights="pretrained"):
    """Build and load just the vision encoder, on the current default device.

    Use inside a `jax.default_device(...)` block to place it where you want,
    then swap it into a bundle: `Gemma4(...)._replace(vision_encoder=vision)`.
    """
    model_dir = Path(model_dir)
    vision_encoder = build_vision_encoder_from_dir(model_dir)
    if weights is not None and vision_encoder is not None:
        vision_encoder.load_weights(
            str(model_dir / "vision_encoder.weights.h5"))
    return vision_encoder


def load_optional_weights(model_dir, per_layer_step, vision_encoder):
    if per_layer_step is not None:
        per_layer_step.load_weights(
            str(model_dir / "embedding_step.weights.h5"))
    if vision_encoder is not None:
        vision_encoder.load_weights(
            str(model_dir / "vision_encoder.weights.h5"))
