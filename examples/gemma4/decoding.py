import jax
import jax.numpy as jp
from keras import ops

from .inference import build_empty_cache


def kv_decode(step_model, config, prompt_ids, stop_id, max_tokens, max_seq=4096):  # fmt: skip
    decoder = KVDecoder(step_model, prompt_ids, max_tokens, max_seq)
    cache = build_empty_cache(config, decoder.max_decode_length)
    cache = jp.asarray(cache)
    stop = jp.array(stop_id, dtype=jp.int32)
    buffer, length = decoder(cache, stop)
    return ops.convert_to_numpy(buffer[0, :length]).tolist()


def KVDecoder(step_model, prompt_ids, max_tokens, max_seq=4096):
    max_len = min(max_seq, len(prompt_ids) + max_tokens)
    prompt = jp.array(prompt_ids, dtype=jp.int32)
    prompt_len = len(prompt_ids)

    @jax.jit
    def decode(self_cache, stop_id):
        buffer = jp.zeros((1, max_len), dtype=jp.int32)
        buffer = buffer.at[0, :prompt_len].set(prompt)
        step = warmup_step(prompt, step_model)
        cache = self_cache
        if prompt_len > 1:
            cache = jax.lax.fori_loop(
                0, prompt_len - 1, step, cache)
        args = (buffer, prompt, prompt_len, cache, stop_id)
        initial = build_initial_state(*args)
        cont = should_continue(max_tokens, max_len)
        advance = build_next_state(step_model, stop_id)
        result = jax.lax.while_loop(cont, advance, initial)
        buffer, _, index, _, _, _ = result
        return buffer, index + 1

    decode.max_decode_length = max_len
    return decode


def warmup_step(prompt, step_model):
    def step(index, cache):
        token = jp.reshape(prompt[index], (1, 1))
        inputs = build_step_inputs(token, cache, index)
        _, cache = step_model(inputs)
        return cache
    return step


def build_initial_state(buffer, prompt, prompt_len, cache, stop_id):
    last_token = jp.reshape(prompt[prompt_len - 1], (1, 1))
    index = jp.array(prompt_len - 1, dtype=jp.int32)
    num_generated = jp.array(0, dtype=jp.int32)
    finished = jp.array(False)
    return (buffer, last_token, index, cache,
            num_generated, finished)


def should_continue(max_gen, max_len):
    def check(state):
        _, _, index, _, num_generated, finished = state
        not_done = ~finished
        under_gen = num_generated < max_gen
        under_len = index + 1 < max_len
        return not_done & under_gen & under_len
    return check


def build_next_state(step_model, stop_id):
    def step(state):
        buffer, token, index, cache, num_generated, _ = state
        inputs = build_step_inputs(token, cache, index)
        logits, cache = step_model(inputs)
        next_id = jp.argmax(logits[:, 0, :], axis=-1)
        next_id = next_id.astype(jp.int32)
        next_index = index + 1
        buffer = buffer.at[0, next_index].set(next_id[0])
        token = jp.expand_dims(next_id, axis=-1)
        finished = next_id[0] == stop_id
        num_generated = num_generated + 1
        args = (buffer, token, next_index, cache,
                num_generated, finished)
        return args
    return step


def build_step_inputs(token, cache, index):
    index_array = jp.array([index], dtype=jp.int32)
    return [token, cache, index_array]


def extract_generated_ids(ids, prompt_length, stop_id):
    generated = ids[prompt_length:]
    if stop_id in generated:
        generated = generated[:generated.index(stop_id)]
    return generated
