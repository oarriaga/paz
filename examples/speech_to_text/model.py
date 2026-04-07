from pathlib import Path

from keras import ops
from keras import Model
from keras.activations import gelu
from keras.layers import Input, Conv1D, Dropout, Add, Lambda
from keras.layers import LayerNormalization, ReversibleEmbedding

from examples.speech_to_text.configs import CONFIGS
from examples.speech_to_text.layers.frontend import frontend
from examples.speech_to_text.layers.frontend import build_mel_filters
from examples.speech_to_text.layers.embedding import embed_position
from examples.speech_to_text.layers.utils import Kernel
from examples.speech_to_text.layers.attention import build_cache
from examples.speech_to_text.layers.encoder import encoder_block
from examples.speech_to_text.layers.decoder import decoder_block

WHISPER_MODELS_DIR = Path(__file__).resolve().with_name("whisper_models")
DECODER_LAYER = "transformer_decoder_layer_{}"
NORM_KWARGS = {"axis": -1, "epsilon": 1e-5}


def WhisperFrontend(name="whisper_frontend"):
    waveform = Input((None,), name="waveform")
    num_mels, fft_bins, sample_rate = 80, 400, 16000
    stride, max_mel = 160, 45.245640471924965
    num_samples = sample_rate * 30
    mel_filters = build_mel_filters(num_mels, fft_bins, sample_rate, max_mel)
    mel_filters = ops.convert_to_tensor(mel_filters)
    features = frontend(waveform, mel_filters, num_samples, fft_bins, stride)
    return Model(waveform, features, name=name)


def WhisperEncoder(num_mels, num_layers, num_heads, hidden_dim, ffn_dim, max_seq, dropout, weights=None, models_path=None, name=None):  # fmt: skip
    features = Input((None, num_mels), name="encoder_features")
    x = build_encoder_stem(features, hidden_dim)
    x = build_encoder_embeddings(x, max_seq, dropout)
    args = (x, num_layers, num_heads, ffn_dim, dropout)
    y = build_encoder_blocks(*args)
    model = Model(features, y, name=name)
    if weights is not None:
        load_whisper_weights(model, weights, "encoder", models_path)
    return model


def build_encoder_stem(features, hidden_dim):
    conv_name = "encoder_token_embedding_conv_layer"
    x = build_conv1d(hidden_dim, 3, 1, "same", f"{conv_name}_1")(features)
    x = gelu(x, approximate=False)
    x = Lambda(pad_encoder, output_shape=pad_shape, name="encoder_padder")(x)
    x = build_conv1d(hidden_dim, 3, 2, "valid", f"{conv_name}_2")(x)
    return gelu(x, approximate=False)


def build_conv1d(filters, kernel, stride, padding, name):
    return Conv1D(filters, kernel, stride, padding, name=name)


def pad_encoder(t):
    return ops.pad(t, [[0, 0], [1, 1], [0, 0]])


def pad_shape(shape):
    return (shape[0], None, shape[2])


def build_encoder_embeddings(x, max_seq, dropout):
    half_max = max_seq // 2
    pos_name = "encoder_position_embedding"
    positions = embed_position(x, half_max, False, None, pos_name)
    x = Add(name="encoder_embeddings_add")((x, positions))
    return Dropout(dropout, name="encoder_embeddings_dropout")(x)


def build_encoder_blocks(x, num_layers, num_heads, ffn_dim, dropout):
    for layer_num in range(num_layers):
        name = "transformer_encoder_layer_{}".format(layer_num)
        x = encoder_block(x, num_heads, ffn_dim, name, dropout, 1e-5)
    return LayerNormalization(**NORM_KWARGS, name="encoder_layer_norm")(x)


def WhisperDecoderStep(vocab_size, num_layers, num_heads, hidden_dim, ffn_dim, max_seq, dropout, weights=None, models_path=None, name=None):  # fmt: skip
    key_dim = int(hidden_dim // num_heads)
    tokens = Input((1,), dtype="int32", name="decoder_token_ids")
    cache_shape = (num_layers, 2, None, num_heads, key_dim)
    self_name = "self_attention_cache"
    self_cache = Input(cache_shape, name=self_name)
    cross_name = "cross_attention_cache"
    cross_cache = Input(cache_shape, name=cross_name)
    cache_index = Input((), dtype="int32", name="cache_update_index")
    scalar_name = "cache_update_index_scalar"
    scalar = Lambda(cast_index_scalar, output_shape=(), name=scalar_name)
    index_scalar = scalar(cache_index)
    position_name = "decoder_position_indices"
    fn = build_position_from_index
    position_layer = Lambda(fn, output_shape=(1,), name=position_name)
    positions = position_layer(cache_index)
    embeddings = build_token_embedding(vocab_size, hidden_dim)
    hidden = embeddings(tokens)
    hidden = build_decoder_embeddings(hidden, positions, max_seq, dropout)
    cache_args = (hidden, self_cache, cross_cache, index_scalar)
    block_args = (num_layers, num_heads, hidden_dim, ffn_dim, dropout)
    hidden, updated_cache = build_decoder_blocks(*cache_args, *block_args)
    norm = LayerNormalization(**NORM_KWARGS, name="decoder_layer_norm")
    hidden = norm(hidden)
    logits = embeddings(hidden, reverse=True)
    inputs = [tokens, self_cache, cross_cache, cache_index]
    outputs = [logits, updated_cache]
    model = Model(inputs, outputs, name=name)
    if weights is not None:
        load_whisper_weights(model, weights, "decoder_step", models_path)
    return model


def cast_index_scalar(x):
    return ops.cast(x[0], "int32")


def build_position_from_index(x):
    return ops.expand_dims(ops.cast(x, "int32"), axis=-1)


def build_token_embedding(vocab_size, hidden_dim):
    keys = ("tie_weights", "embeddings_initializer", "mask_zero", "name")
    values = (True, Kernel(), False, "decoder_token_embedding")
    kwargs = dict(zip(keys, values))
    return ReversibleEmbedding(vocab_size, hidden_dim, **kwargs)


def build_decoder_embeddings(hidden, positions, max_seq, dropout):
    pos_name = "decoder_position_embedding"
    pos_embed = embed_position(hidden, max_seq, True, positions, pos_name)
    hidden = Add(name="decoder_embeddings_add")((hidden, pos_embed))
    return Dropout(dropout, name="decoder_embeddings_dropout")(hidden)


def build_decoder_blocks(hidden, self_cache, cross_cache, index, num_layers, num_heads, hidden_dim, ffn_dim, dropout):  # fmt: skip
    key_dim = int(hidden_dim // num_heads)
    keys = ("num_heads", "hidden_dim", "ffn_dim", "dropout")
    values = (num_heads, hidden_dim, ffn_dim, dropout)
    config = dict(zip(keys, values))
    updated_caches = []
    for layer_num in range(num_layers):
        cache_args = (layer_num, num_heads, key_dim)
        layer_self = slice_self_cache(self_cache, *cache_args)
        layer_cross = slice_cross_cache(cross_cache, *cache_args)
        name = DECODER_LAYER.format(layer_num)
        caches = (layer_self, layer_cross)
        hidden, layer_self = decoder_block(hidden, caches, index, config, name)
        layer_self = expand_self_cache(layer_self, *cache_args)
        updated_caches.append(layer_self)
    updated = concat_self_caches(updated_caches, num_layers, num_heads, key_dim)
    return hidden, updated


def slice_self_cache(self_cache, index, num_heads, key_dim):
    name = DECODER_LAYER.format(index) + "_self_attention_cache_slice"
    shape = (2, None, num_heads, key_dim)
    return build_cache_slicer(index, shape, name)(self_cache)


def build_cache_slicer(index, shape, name):
    fn = lambda x, i=index: x[:, i, ...]
    return Lambda(fn, output_shape=shape, name=name)


def slice_cross_cache(cross_cache, index, num_heads, key_dim):
    name = DECODER_LAYER.format(index) + "_cross_attention_cache_slice"
    shape = (2, None, num_heads, key_dim)
    return build_cache_slicer(index, shape, name)(cross_cache)


def expand_self_cache(cache, index, num_heads, key_dim):
    name = DECODER_LAYER.format(index) + "_self_attention_cache_expand"
    shape = (1, 2, None, num_heads, key_dim)
    return build_expand_lambda(shape, name)(cache)


def build_expand_lambda(shape, name):
    fn = lambda x: ops.expand_dims(x, axis=1)
    return Lambda(fn, output_shape=shape, name=name)


def concat_self_caches(caches, num_layers, num_heads, key_dim):
    shape = (num_layers, 2, None, num_heads, key_dim)
    name = "updated_self_attention_cache"
    return build_concatenate_lambda(shape, name)(caches)


def build_concatenate_lambda(shape, name):
    fn = lambda tensors: ops.concatenate(tensors, axis=1)
    return Lambda(fn, output_shape=shape, name=name)


def WhisperCrossCache(num_layers, num_heads, hidden_dim, weights=None, models_path=None, name=None):  # fmt: skip
    input_name = "encoder_output"
    encoder_output = Input((None, hidden_dim), name=input_name)
    args = (encoder_output, num_layers, num_heads, hidden_dim)
    cross_cache = build_cross_caches(*args)
    model = Model(encoder_output, cross_cache, name=name)
    if weights is not None:
        load_whisper_weights(model, weights, "cross_cache", models_path)
    return model


def build_cross_caches(output, num_layers, num_heads, hidden_dim):
    key_dim = int(hidden_dim // num_heads)
    caches = []
    for layer_num in range(num_layers):
        layer_name = DECODER_LAYER.format(layer_num) + "_cross_attention"
        cache = build_cache(output, None, num_heads, key_dim, layer_name)
        expand_name = layer_name + "_cache_expand"
        expand_shape = (1, 2, None, num_heads, key_dim)
        cache = build_expand_lambda(expand_shape, expand_name)(cache)
        caches.append(cache)
    concat_shape = (num_layers, 2, None, num_heads, key_dim)
    name = "cross_attention_cache"
    return build_concatenate_lambda(concat_shape, name)(caches)


def load_whisper_weights(model, variant_name, model_kind, models_path=None):
    models_path = models_path or WHISPER_MODELS_DIR
    path = build_whisper_weights_path(variant_name, model_kind, models_path)
    if not path.exists():
        template = "No {} weights for {} at {}."
        msg = template.format(model_kind, variant_name, path)
        raise FileNotFoundError(msg)
    model.load_weights(str(path))
    return model


def build_whisper_weights_path(variant_name, model_kind, models_path):
    filename = "{}.weights.h5".format(model_kind)
    return build_whisper_model_dir(variant_name, models_path) / filename


def build_whisper_model_dir(variant_name, models_path):
    return Path(models_path) / variant_name
