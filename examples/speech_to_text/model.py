from pathlib import Path

from keras import ops
from keras import Model
from keras.activations import gelu
from keras.layers import Input, Conv1D, Dropout, Add, Lambda
from keras.layers import LayerNormalization, ReversibleEmbedding

from examples.speech_to_text.layers.frontend import frontend
from examples.speech_to_text.layers.frontend import build_mel_filters
from examples.speech_to_text.layers.embedding import Kernel
from examples.speech_to_text.layers.embedding import embed_position
from examples.speech_to_text.layers.attention import build_cache
from examples.speech_to_text.layers.encoder import encoder_block
from examples.speech_to_text.layers.decoder import decoder_block

WHISPER_MODELS_DIR = Path(__file__).resolve().with_name("whisper_models")
DECODER_LAYER = "transformer_decoder_layer_{}"
NORM_KWARGS = {"axis": -1, "epsilon": 1e-5, "dtype": "float32"}

CONFIGS = {
    "whisper_tiny_en": {
        "vocabulary_size": 51864,
        "num_layers": 4,
        "num_heads": 6,
        "hidden_dim": 384,
        "ffn_dim": 1536,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_base_en": {
        "vocabulary_size": 51864,
        "num_layers": 6,
        "num_heads": 8,
        "hidden_dim": 512,
        "ffn_dim": 2048,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_small_en": {
        "vocabulary_size": 51864,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "ffn_dim": 3072,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_medium_en": {
        "vocabulary_size": 51864,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_tiny_multi": {
        "vocabulary_size": 51865,
        "num_layers": 4,
        "num_heads": 6,
        "hidden_dim": 384,
        "ffn_dim": 1536,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_base_multi": {
        "vocabulary_size": 51865,
        "num_layers": 6,
        "num_heads": 8,
        "hidden_dim": 512,
        "ffn_dim": 2048,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_small_multi": {
        "vocabulary_size": 51865,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "ffn_dim": 3072,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_medium_multi": {
        "vocabulary_size": 51865,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_large_multi": {
        "vocabulary_size": 51865,
        "num_layers": 32,
        "num_heads": 20,
        "hidden_dim": 1280,
        "ffn_dim": 5120,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_large_multi_v2": {
        "vocabulary_size": 51865,
        "num_layers": 32,
        "num_heads": 20,
        "hidden_dim": 1280,
        "ffn_dim": 5120,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
}


def build_whisper_model_dir(variant_name):
    return WHISPER_MODELS_DIR / variant_name


def build_whisper_weights_path(variant_name, model_kind):
    filename = "{}.weights.h5".format(model_kind)
    return build_whisper_model_dir(variant_name) / filename


def load_whisper_weights(model, variant_name, model_kind):
    path = build_whisper_weights_path(variant_name, model_kind)
    if not path.exists():
        template = "No {} weights for {} at {}."
        msg = template.format(model_kind, variant_name, path)
        raise FileNotFoundError(msg)
    model.load_weights(str(path))
    return model


def build_conv1d(filters, kernel, stride, padding, name):
    kwargs = {"dtype": "float32", "name": name}
    return Conv1D(filters, kernel, stride, padding, **kwargs)


def build_gelu(x):
    return gelu(x, approximate=False)


def pad_encoder(t):
    return ops.pad(t, [[0, 0], [1, 1], [0, 0]])


def pad_shape(shape):
    return (shape[0], None, shape[2])


def build_encoder_stem(features, hidden_dim):
    conv_name = "encoder_token_embedding_conv_layer"
    x = build_conv1d(hidden_dim, 3, 1, "same", f"{conv_name}_1")(features)
    x = build_gelu(x)
    pad = Lambda(pad_encoder, output_shape=pad_shape, name="encoder_padder")
    x = pad(x)
    x = build_conv1d(hidden_dim, 3, 2, "valid", f"{conv_name}_2")(x)
    return build_gelu(x)


def build_encoder_embeddings(x, max_seq, dropout):
    half_max = max_seq // 2
    pos_name = "encoder_position_embedding"
    positions = embed_position(x, half_max, False, None, pos_name)
    name = "encoder_embeddings_add"
    x = Add(dtype="float32", name=name)((x, positions))
    name = "encoder_embeddings_dropout"
    return Dropout(dropout, dtype="float32", name=name)(x)


def build_encoder_blocks(x, num_layers, num_heads, ffn_dim, dropout):
    for i in range(num_layers):
        name = "transformer_encoder_layer_{}".format(i)
        x = encoder_block(x, num_heads, ffn_dim, name, dropout, 1e-5)
    return LayerNormalization(**NORM_KWARGS, name="encoder_layer_norm")(x)


def WhisperFrontend(name="whisper_frontend"):
    waveform = Input((None,), dtype="float32", name="waveform")
    mel_filters = build_mel_filters(80, 400, 16000)
    mel_filters = ops.convert_to_tensor(mel_filters, dtype="float32")
    features = frontend(waveform, mel_filters)
    return Model(waveform, features, name=name)


def WhisperEncoder(
    num_mels,
    num_layers,
    num_heads,
    hidden_dim,
    ffn_dim,
    max_seq,
    dropout,
    weights=None,
    name=None,
):
    features = Input((None, num_mels), dtype="float32", name="encoder_features")
    x = build_encoder_stem(features, hidden_dim)
    x = build_encoder_embeddings(x, max_seq, dropout)
    args = (x, num_layers, num_heads, ffn_dim, dropout)
    y = build_encoder_blocks(*args)
    model = Model(features, y, name=name)
    if weights is not None:
        load_whisper_weights(model, weights, "encoder")
    return model


def cast_index_scalar(x):
    return ops.cast(x[0], "int32")


def build_position_from_index(x):
    return ops.expand_dims(ops.cast(x, "int32"), axis=-1)


def build_token_embedding(vocab_size, hidden_dim):
    kwargs = {
        "tie_weights": True,
        "embeddings_initializer": Kernel(),
        "mask_zero": False,
        "name": "decoder_token_embedding",
    }
    return ReversibleEmbedding(vocab_size, hidden_dim, **kwargs)


def build_decoder_embeddings(hidden, positions, max_seq, dropout):
    pos_name = "decoder_position_embedding"
    pos_embed = embed_position(hidden, max_seq, True, positions, pos_name)
    name = "decoder_embeddings_add"
    hidden = Add(dtype="float32", name=name)((hidden, pos_embed))
    name = "decoder_embeddings_dropout"
    return Dropout(dropout, dtype="float32", name=name)(hidden)


def build_cache_slicer(index, shape, name):
    fn = lambda x, i=index: x[:, i, ...]
    return Lambda(fn, output_shape=shape, name=name)


def build_expand_lambda(shape, name):
    fn = lambda x: ops.expand_dims(x, axis=1)
    return Lambda(fn, output_shape=shape, name=name)


def build_concat_lambda(shape, name):
    fn = lambda tensors: ops.concatenate(tensors, axis=1)
    return Lambda(fn, output_shape=shape, name=name)


def slice_self_cache(self_cache, index, num_heads, key_dim):
    name = DECODER_LAYER.format(index) + "_self_attention_cache_slice"
    shape = (2, None, num_heads, key_dim)
    return build_cache_slicer(index, shape, name)(self_cache)


def slice_cross_cache(cross_cache, index, num_heads, key_dim):
    name = DECODER_LAYER.format(index) + "_cross_attention_cache_slice"
    shape = (2, None, num_heads, key_dim)
    return build_cache_slicer(index, shape, name)(cross_cache)


def expand_self_cache(cache, index, num_heads, key_dim):
    name = DECODER_LAYER.format(index) + "_self_attention_cache_expand"
    shape = (1, 2, None, num_heads, key_dim)
    return build_expand_lambda(shape, name)(cache)


def concat_self_caches(caches, num_layers, num_heads, key_dim):
    shape = (num_layers, 2, None, num_heads, key_dim)
    name = "updated_self_attention_cache"
    return build_concat_lambda(shape, name)(caches)


def build_decoder_blocks(
    hidden,
    self_cache,
    cross_cache,
    index,
    num_layers,
    num_heads,
    hidden_dim,
    ffn_dim,
    dropout,
):
    key_dim = int(hidden_dim // num_heads)
    config = {
        "num_heads": num_heads,
        "hidden_dim": hidden_dim,
        "ffn_dim": ffn_dim,
        "dropout": dropout,
    }
    updated_caches = []
    for i in range(num_layers):
        layer_self = slice_self_cache(self_cache, i, num_heads, key_dim)
        layer_cross = slice_cross_cache(cross_cache, i, num_heads, key_dim)
        name = DECODER_LAYER.format(i)
        caches = (layer_self, layer_cross)
        hidden, layer_self = decoder_block(hidden, caches, index, config, name)
        layer_self = expand_self_cache(layer_self, i, num_heads, key_dim)
        updated_caches.append(layer_self)
    updated = concat_self_caches(updated_caches, num_layers, num_heads, key_dim)
    return hidden, updated


def WhisperDecoderStep(
    vocab_size,
    num_layers,
    num_heads,
    hidden_dim,
    ffn_dim,
    max_seq,
    dropout,
    weights=None,
    name=None,
):
    key_dim = int(hidden_dim // num_heads)
    tokens = Input((1,), dtype="int32", name="decoder_token_ids")
    cache_shape = (num_layers, 2, None, num_heads, key_dim)
    cache_name = "self_attention_cache"
    self_cache = Input(cache_shape, dtype="float32", name=cache_name)
    cache_name = "cross_attention_cache"
    cross_cache = Input(cache_shape, dtype="float32", name=cache_name)
    cache_index = Input((), dtype="int32", name="cache_update_index")
    scalar_name = "cache_update_index_scalar"
    scalar = Lambda(cast_index_scalar, output_shape=(), name=scalar_name)
    index_scalar = scalar(cache_index)
    pos_fn = build_position_from_index
    pos_name = "decoder_position_indices"
    pos_layer = Lambda(pos_fn, output_shape=(1,), name=pos_name)
    positions = pos_layer(cache_index)
    embeddings = build_token_embedding(vocab_size, hidden_dim)
    hidden = embeddings(tokens)
    hidden = build_decoder_embeddings(hidden, positions, max_seq, dropout)
    cache_args = (hidden, self_cache, cross_cache, index_scalar)
    block_args = (num_layers, num_heads, hidden_dim, ffn_dim, dropout)
    hidden, updated_cache = build_decoder_blocks(*cache_args, *block_args)
    norm_name = "decoder_layer_norm"
    hidden = LayerNormalization(**NORM_KWARGS, name=norm_name)(hidden)
    logits = embeddings(hidden, reverse=True)
    inputs = [tokens, self_cache, cross_cache, cache_index]
    outputs = [logits, updated_cache]
    model = Model(inputs, outputs, name=name)
    if weights is not None:
        load_whisper_weights(model, weights, "decoder_step")
    return model


def build_cross_caches(output, num_layers, num_heads, hidden_dim):
    key_dim = int(hidden_dim // num_heads)
    caches = []
    for i in range(num_layers):
        layer_name = DECODER_LAYER.format(i) + "_cross_attention"
        cache = build_cache(output, None, num_heads, key_dim, layer_name)
        expand_name = layer_name + "_cache_expand"
        expand_shape = (1, 2, None, num_heads, key_dim)
        cache = build_expand_lambda(expand_shape, expand_name)(cache)
        caches.append(cache)
    concat_shape = (num_layers, 2, None, num_heads, key_dim)
    return build_concat_lambda(concat_shape, "cross_attention_cache")(caches)


def WhisperCrossCache(
    num_layers, num_heads, hidden_dim, weights=None, name=None
):
    input_name = "encoder_output"
    encoder_output = Input((None, hidden_dim), dtype="float32", name=input_name)
    args = (encoder_output, num_layers, num_heads, hidden_dim)
    cross_cache = build_cross_caches(*args)
    model = Model(encoder_output, cross_cache, name=name)
    if weights is not None:
        load_whisper_weights(model, weights, "cross_cache")
    return model
