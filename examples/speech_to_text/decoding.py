import jax
import jax.numpy as jp
from keras import ops

from examples.speech_to_text.tokenizer import find_special_token_id


PROMPT_TOKENS = ["<|startoftranscript|>", "<|transcribe|>", "<|notimestamps|>"]


def build_whisper_prompt_token_ids(config_path=None):
    return [find_special_token_id(t, config_path) for t in PROMPT_TOKENS]


def build_decoder_inputs(token, cache, cross_cache, index):
    index_array = jp.array([index], dtype=jp.int32)
    return [token, cache, cross_cache, index_array]


def warmup_step(prompt, decoder, cross_cache):
    def step(index, cache):
        token = jp.reshape(prompt[index], (1, 1))
        inputs = build_decoder_inputs(token, cache, cross_cache, index)
        _, cache = decoder(inputs)
        return cache
    return step


def build_initial_state(buf, prompt, prompt_len, cache, stop_id):
    last_token = jp.reshape(prompt[prompt_len - 1], (1, 1))
    index = jp.array(prompt_len - 1, dtype=jp.int32)
    num_generated = jp.array(0, dtype=jp.int32)
    finished = jp.array(False)
    return (buf, last_token, index, cache, num_generated, finished)


def should_continue(max_gen, max_len):
    def check(state):
        _, _, index, _, num_gen, finished = state
        not_done = ~finished
        under_gen = num_gen < max_gen
        under_len = index + 1 < max_len
        return not_done & under_gen & under_len
    return check


def build_next_state(decoder, cross_cache, stop_id):
    def step(state):
        buf, token, index, cache, num_gen, _ = state
        inputs = build_decoder_inputs(token, cache, cross_cache, index)
        logits, cache = decoder(inputs)
        next_id = jp.argmax(logits[:, 0, :], axis=-1).astype(jp.int32)
        next_index = index + 1
        buf = buf.at[0, next_index].set(next_id[0])
        token = jp.expand_dims(next_id, axis=-1)
        finished = next_id[0] == stop_id
        return (buf, token, next_index, cache, num_gen + 1, finished)
    return step


def KVDecoder(decoder_step, prompt_ids, max_tokens, max_seq=448):
    max_len = min(max_seq, len(prompt_ids) + max_tokens)
    prompt = jp.array(prompt_ids, dtype=jp.int32)
    prompt_len = len(prompt_ids)

    @jax.jit
    def decode(self_cache, cross_cache, stop_id):
        buf = jp.zeros((1, max_len), dtype=jp.int32)
        buf = buf.at[0, :prompt_len].set(prompt)
        step = warmup_step(prompt, decoder_step, cross_cache)
        cache = self_cache
        if prompt_len > 1:
            cache = jax.lax.fori_loop(0, prompt_len - 1, step, cache)
        init = build_initial_state(buf, prompt, prompt_len, cache, stop_id)
        cont = should_continue(max_tokens, max_len)
        advance = build_next_state(decoder_step, cross_cache, stop_id)
        result = jax.lax.while_loop(cont, advance, init)
        buf, _, index, _, _, _ = result
        return buf, index + 1

    decode.max_decode_length = max_len
    return decode


def build_self_cache(cache_shape, max_len, batch_size=1):
    num_layers = int(cache_shape[1])
    num_heads = int(cache_shape[4])
    key_dim = int(cache_shape[5])
    shape = (batch_size, num_layers, 2, max_len, num_heads, key_dim)
    return ops.zeros(shape, dtype="float32")


def kv_decode(decoder, cache_shape, cross_model, encoder_output, stop_id):
    max_len = decoder.max_decode_length
    batch = int(encoder_output.shape[0])
    cache = build_self_cache(cache_shape, max_len, batch)
    cross_cache = cross_model(encoder_output)
    cache = jp.asarray(cache)
    cross_cache = jp.asarray(cross_cache)
    stop = jp.array(stop_id, dtype=jp.int32)
    buf, length = decoder(cache, cross_cache, stop)
    return ops.convert_to_numpy(buf[0, :length]).tolist()


def extract_text_token_ids(ids, prompt_length, stop_id):
    text_ids = ids[prompt_length:]
    if stop_id in text_ids:
        text_ids = text_ids[:text_ids.index(stop_id)]
    return text_ids
