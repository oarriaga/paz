import jax
import jax.numpy as jp
from keras import ops

from paz.models.transformers import search

from paz.models.foundation.gemma4.inference import build_empty_cache


def kv_decode(step_model, config, prompt_ids, stop_id, max_tokens,
              max_seq=4096):
    """Single-sequence jitted greedy decoding (text, no per-layer input)."""
    args = (step_model, config, prompt_ids, stop_id, max_tokens,
            jax.random.PRNGKey(0), search.greedy, max_seq)
    return run_kv_decode(*args)


def kv_sample(step_model, config, prompt_ids, stop_id, max_tokens, key,
              sampling, max_seq=4096):
    """Single-sequence jitted sampling decoding seeded by `key`."""
    select = search.build_sampler(
        sampling.temperature, sampling.top_k, sampling.top_p)
    args = (step_model, config, prompt_ids, stop_id, max_tokens, key, select,
            max_seq)
    return run_kv_decode(*args)


def run_kv_decode(step_model, config, prompt_ids, stop_id, max_tokens, key,
                  select, max_seq):
    decoder = KVDecoder(step_model, prompt_ids, max_tokens, select, max_seq)
    cache = jp.asarray(build_empty_cache(config, decoder.max_decode_length))
    stop = jp.array(stop_id, dtype=jp.int32)
    buffer, length = decoder(cache, stop, key)
    return ops.convert_to_numpy(buffer[0, :length]).tolist()


def KVDecoder(step_model, prompt_ids, max_tokens, select, max_seq=4096,
              emit=search.discard):
    max_len = min(max_seq, len(prompt_ids) + max_tokens)
    prompt = jp.array(prompt_ids, dtype=jp.int32)
    prompt_len = len(prompt_ids)

    @jax.jit
    def decode(self_cache, stop_id, key):
        buffer = jp.zeros((1, max_len), dtype=jp.int32)
        buffer = buffer.at[0, :prompt_len].set(prompt)
        cache = self_cache
        if prompt_len > 1:
            warmup = warmup_step(prompt, step_model)
            cache = jax.lax.fori_loop(0, prompt_len - 1, warmup, cache)
        step = build_step(step_model)
        run = search.build_streaming(step, select, max_tokens, max_len, emit)
        token = jp.reshape(prompt[prompt_len - 1], (1, 1))
        index = jp.array(prompt_len - 1, dtype=jp.int32)
        return run(key, buffer, token, index, cache, stop_id)

    decode.max_decode_length = max_len
    return decode


def build_step(step_model):
    def step(cache, token, index, key):
        return step_model(build_step_inputs(token, cache, index))

    return step


def warmup_step(prompt, step_model):
    def step(index, cache):
        token = jp.reshape(prompt[index], (1, 1))
        inputs = build_step_inputs(token, cache, index)
        _, cache = step_model(inputs)
        return cache

    return step


def build_step_inputs(token, cache, index):
    index_array = jp.array([index], dtype=jp.int32)
    return [token, cache, index_array]


def extract_generated_ids(ids, prompt_length, stop_id):
    generated = ids[prompt_length:]
    if stop_id in generated:
        generated = generated[:generated.index(stop_id)]
    return generated
