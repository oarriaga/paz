import jax
import jax.numpy as jp
from keras import ops

import paz
from examples.speech_to_text.tokenizer import find_special_token_id

PROMPT_TOKENS = ["<|startoftranscript|>", "<|transcribe|>", "<|notimestamps|>"]


def kv_decode(key, decoder, cache_shape, cross_model, encoder_output, stop_id):
    max_len = decoder.max_decode_length
    batch = int(encoder_output.shape[0])
    cache = build_self_cache(cache_shape, max_len, batch)
    cross_cache = cross_model(encoder_output)
    cache = jp.asarray(cache)
    cross_cache = jp.asarray(cross_cache)
    stop = jp.array(stop_id, dtype=jp.int32)
    buffer, length = decoder(key, cache, cross_cache, stop)
    return ops.convert_to_numpy(buffer[0, :length]).tolist()


def build_self_cache(cache_shape, max_len, batch_size=1):
    num_layers = int(cache_shape[1])
    num_heads = int(cache_shape[4])
    key_dim = int(cache_shape[5])
    return paz.transformers.cache.build(
        batch_size, num_layers, max_len, num_heads, key_dim)


def KVDecoder(decoder_step, prompt_ids, max_tokens, max_seq=448, select=None):
    if select is None:
        select = paz.transformers.search.greedy
    max_len = min(max_seq, len(prompt_ids) + max_tokens)
    prompt = jp.array(prompt_ids, dtype=jp.int32)
    prompt_len = len(prompt_ids)

    @jax.jit
    def decode(key, self_cache, cross_cache, stop_id):
        buffer = jp.zeros((1, max_len), dtype=jp.int32)
        buffer = buffer.at[0, :prompt_len].set(prompt)
        cache = self_cache
        if prompt_len > 1:
            warmup = warmup_step(prompt, decoder_step, cross_cache)
            cache = jax.lax.fori_loop(0, prompt_len - 1, warmup, cache)
        step = build_step(decoder_step, cross_cache)
        run = paz.transformers.search.build(step, select, max_tokens, max_len)
        token = jp.reshape(prompt[prompt_len - 1], (1, 1))
        index = jp.array(prompt_len - 1, dtype=jp.int32)
        return run(key, buffer, token, index, cache, stop_id)

    # Python functions are objects; attributes can be set on them freely.
    decode.max_decode_length = max_len
    return decode


def build_step(decoder, cross_cache):
    def step(cache, token, index, key):
        inputs = build_decoder_inputs(token, cache, cross_cache, index)
        return decoder(inputs)

    return step


def warmup_step(prompt, decoder, cross_cache):
    def step(index, cache):
        token = jp.reshape(prompt[index], (1, 1))
        inputs = build_decoder_inputs(token, cache, cross_cache, index)
        _, cache = decoder(inputs)
        return cache

    return step


def build_decoder_inputs(token, cache, cross_cache, index):
    index_array = jp.array([index], dtype=jp.int32)
    return [token, cache, cross_cache, index_array]


def extract_text_token_ids(ids, prompt_length, stop_id):
    text_ids = ids[prompt_length:]
    if stop_id in text_ids:
        text_ids = text_ids[: text_ids.index(stop_id)]
    return text_ids


def build_whisper_prompt_token_ids(config_path=None):
    return [find_special_token_id(t, config_path) for t in PROMPT_TOKENS]
