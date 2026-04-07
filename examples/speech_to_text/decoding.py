import jax
import jax.numpy as jp
from keras import ops

from examples.speech_to_text.tokenizer import find_special_token_id

PROMPT_TOKENS = ["<|startoftranscript|>", "<|transcribe|>", "<|notimestamps|>"]


def kv_decode(decoder, cache_shape, cross_model, encoder_output, stop_id):
    max_len = decoder.max_decode_length
    batch = int(encoder_output.shape[0])
    cache = build_self_cache(cache_shape, max_len, batch)
    cross_cache = cross_model(encoder_output)
    cache = jp.asarray(cache)
    cross_cache = jp.asarray(cross_cache)
    stop = jp.array(stop_id, dtype=jp.int32)
    buffer, length = decoder(cache, cross_cache, stop)
    return ops.convert_to_numpy(buffer[0, :length]).tolist()


def build_self_cache(cache_shape, max_len, batch_size=1):
    num_layers = int(cache_shape[1])
    num_heads = int(cache_shape[4])
    key_dim = int(cache_shape[5])
    shape = (batch_size, num_layers, 2, max_len, num_heads, key_dim)
    return ops.zeros(shape, dtype="float32")


def KVDecoder(decoder_step, prompt_ids, max_tokens, max_seq=448):
    max_len = min(max_seq, len(prompt_ids) + max_tokens)
    prompt = jp.array(prompt_ids, dtype=jp.int32)
    prompt_len = len(prompt_ids)

    @jax.jit
    def decode(self_cache, cross_cache, stop_id):
        buffer = jp.zeros((1, max_len), dtype=jp.int32)
        buffer = buffer.at[0, :prompt_len].set(prompt)
        step = warmup_step(prompt, decoder_step, cross_cache)
        cache = self_cache
        if prompt_len > 1:
            cache = jax.lax.fori_loop(0, prompt_len - 1, step, cache)
        args = (buffer, prompt, prompt_len, cache, stop_id)
        initial_state = build_initial_state(*args)
        cont = should_continue(max_tokens, max_len)
        advance = build_next_state(decoder_step, cross_cache, stop_id)
        result = jax.lax.while_loop(cont, advance, initial_state)
        buffer, _, index, _, _, _ = result
        return buffer, index + 1

    # Python functions are objects; attributes can be set on them freely.
    decode.max_decode_length = max_len
    return decode


def warmup_step(prompt, decoder, cross_cache):
    def step(index, cache):
        token = jp.reshape(prompt[index], (1, 1))
        inputs = build_decoder_inputs(token, cache, cross_cache, index)
        _, cache = decoder(inputs)
        return cache

    return step


def build_initial_state(buffer, prompt, prompt_len, cache, stop_id):
    last_token = jp.reshape(prompt[prompt_len - 1], (1, 1))
    index = jp.array(prompt_len - 1, dtype=jp.int32)
    num_generated = jp.array(0, dtype=jp.int32)
    finished = jp.array(False)
    return (buffer, last_token, index, cache, num_generated, finished)


def should_continue(max_gen, max_len):
    def check(state):
        _, _, index, _, num_generated, finished = state
        not_done = ~finished
        under_gen = num_generated < max_gen
        under_len = index + 1 < max_len
        return not_done & under_gen & under_len

    return check


def build_next_state(decoder, cross_cache, stop_id):
    def step(state):
        buffer, token, index, cache, num_generated, _ = state
        inputs = build_decoder_inputs(token, cache, cross_cache, index)
        logits, cache = decoder(inputs)
        next_id = jp.argmax(logits[:, 0, :], axis=-1).astype(jp.int32)
        next_index = index + 1
        buffer = buffer.at[0, next_index].set(next_id[0])
        token = jp.expand_dims(next_id, axis=-1)
        finished = next_id[0] == stop_id
        return (buffer, token, next_index, cache, num_generated + 1, finished)

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
