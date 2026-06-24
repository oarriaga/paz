"""Autoregressive token search over a cached decoder step.

``build`` returns a ``run`` that drives a ``jax.lax.while_loop`` writing
generated tokens into ``buffer`` starting after ``index``. The model supplies
``step(cache, token, index, key) -> (logits, cache)`` (it owns the forward and
the constant cross-cache); ``select(logits, key) -> token_id`` is ``greedy`` or
a ``build_sampler`` closure. Prefill/warmup stays model-side and seeds the call.
"""
import jax
import jax.numpy as jp

from paz.models.transformers.logits import apply_temperature
from paz.models.transformers.logits import apply_top_k
from paz.models.transformers.logits import apply_top_p


def build(step, select, max_tokens, max_length):
    def run(key, buffer, token, index, cache, stop_id):
        count = jp.array(0, dtype=jp.int32)
        finished = jp.array(False)
        state = (buffer, token, index, cache, count, finished, key)
        cont = build_should_continue(max_tokens, max_length)
        advance = build_advance(step, select, stop_id)
        state = jax.lax.while_loop(cont, advance, state)
        return state[0], state[2] + 1

    return run


def build_should_continue(max_tokens, max_length):
    def check(state):
        _, _, index, _, count, finished, _ = state
        return (~finished) & (count < max_tokens) & (index + 1 < max_length)

    return check


def build_advance(step, select, stop_id):
    def advance(state):
        buffer, token, index, cache, count, _, key = state
        logits, cache = step(cache, token, index, key)
        key, step_key = jax.random.split(key)
        next_id = select(logits, step_key)
        next_index = index + 1
        buffer = buffer.at[0, next_index].set(next_id[0])
        token = jp.expand_dims(next_id, axis=-1)
        finished = next_id[0] == stop_id
        return (buffer, token, next_index, cache, count + 1, finished, key)

    return advance


def greedy(logits, key):
    return jp.argmax(logits[:, 0, :], axis=-1).astype(jp.int32)


def build_sampler(temperature, top_k, top_p):
    def select(logits, key):
        row = logits[:, 0, :]
        row = apply_temperature(row, temperature)
        row = apply_top_k(row, top_k)
        row = apply_top_p(row, top_p)
        return jax.random.categorical(key, row, axis=-1).astype(jp.int32)

    return select
