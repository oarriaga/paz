import jax
import jax.numpy as jp
from keras import ops

from .inference import build_empty_cache
from .sampling import sample_logits


def kv_decode(
    step_model, config, prompt_ids, stop_id, max_tokens, max_seq=4096
):
    """Single-sequence jitted greedy decoding (text, no per-layer input)."""
    return run_kv_decode(step_model, config, prompt_ids, stop_id, max_tokens,
                         jax.random.PRNGKey(0), select_greedy, max_seq)


def kv_sample(
    step_model, config, prompt_ids, stop_id, max_tokens, key, sampling,
    max_seq=4096
):
    """Single-sequence jitted sampling decoding seeded by `key`."""
    return run_kv_decode(step_model, config, prompt_ids, stop_id, max_tokens,
                         key, build_sampler(sampling), max_seq)


def run_kv_decode(step_model, config, prompt_ids, stop_id, max_tokens, key,
                  select, max_seq):
    decoder = KVDecoder(step_model, prompt_ids, max_tokens, select, max_seq)
    cache = jp.asarray(build_empty_cache(config, decoder.max_decode_length))
    stop = jp.array(stop_id, dtype=jp.int32)
    buffer, length = decoder(cache, stop, key)
    return ops.convert_to_numpy(buffer[0, :length]).tolist()


def select_greedy(logits, key):
    return jp.argmax(logits, axis=-1).astype(jp.int32)


def build_sampler(sampling):
    return lambda logits, key: sample_logits(logits, key, sampling)


def KVDecoder(step_model, prompt_ids, max_tokens, select, max_seq=4096):
    max_len = min(max_seq, len(prompt_ids) + max_tokens)
    prompt = jp.array(prompt_ids, dtype=jp.int32)
    prompt_len = len(prompt_ids)

    @jax.jit
    def decode(self_cache, stop_id, key):
        buffer = jp.zeros((1, max_len), dtype=jp.int32)
        buffer = buffer.at[0, :prompt_len].set(prompt)
        step = warmup_step(prompt, step_model)
        cache = self_cache
        if prompt_len > 1:
            cache = jax.lax.fori_loop(0, prompt_len - 1, step, cache)
        args = (buffer, prompt, prompt_len, cache, stop_id, key)
        state = build_initial_state(*args)
        cont = should_continue(max_tokens, max_len)
        step_fn = build_next_state(step_model, stop_id, select)
        buffer, _, index, _, _, _, _ = jax.lax.while_loop(cont, step_fn, state)
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


def build_initial_state(buffer, prompt, prompt_len, cache, stop_id, key):
    last_token = jp.reshape(prompt[prompt_len - 1], (1, 1))
    index = jp.array(prompt_len - 1, dtype=jp.int32)
    num_generated = jp.array(0, dtype=jp.int32)
    finished = jp.array(False)
    return (buffer, last_token, index, cache, num_generated, finished, key)


def should_continue(max_gen, max_len):
    def check(state):
        _, _, index, _, num_generated, finished, _ = state
        not_done = ~finished
        under_gen = num_generated < max_gen
        under_len = index + 1 < max_len
        return not_done & under_gen & under_len
    return check


def build_next_state(step_model, stop_id, select):
    def step(state):
        buffer, token, index, cache, num_generated, _, key = state
        inputs = build_step_inputs(token, cache, index)
        logits, cache = step_model(inputs)
        key, step_key = jax.random.split(key)
        next_id = select(logits[:, 0, :], step_key)
        next_index = index + 1
        buffer = buffer.at[0, next_index].set(next_id[0])
        token = jp.expand_dims(next_id, axis=-1)
        finished = next_id[0] == stop_id
        num_generated = num_generated + 1
        return (buffer, token, next_index, cache, num_generated, finished, key)
    return step


def build_step_inputs(token, cache, index):
    index_array = jp.array([index], dtype=jp.int32)
    return [token, cache, index_array]


def extract_generated_ids(ids, prompt_length, stop_id):
    generated = ids[prompt_length:]
    if stop_id in generated:
        generated = generated[:generated.index(stop_id)]
    return generated
