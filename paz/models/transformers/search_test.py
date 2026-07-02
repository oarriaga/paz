import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
import jax.numpy as jp

from paz.models.transformers import search

VOCAB = 10


def build_counting_step():
    # Deterministic toy decoder: next id = (current token + 1) % VOCAB.
    def step(cache, token, index, key):
        nxt = (token[0, 0] + 1) % VOCAB
        return jax.nn.one_hot(nxt, VOCAB)[None, None, :], cache

    return step


def run_counting(stop_id, max_tokens=4, max_length=8, seed=0):
    run = search.build_streaming(build_counting_step(), search.greedy,
                                 max_tokens, max_length, search.discard)
    buffer = jp.zeros((1, max_length), dtype=jp.int32).at[0, 0].set(2)
    token = jp.reshape(jp.array(2, dtype=jp.int32), (1, 1))
    index = jp.array(0, dtype=jp.int32)
    cache = jp.zeros((1,))
    return run(jax.random.PRNGKey(seed), buffer, token, index, cache,
               jp.array(stop_id, dtype=jp.int32))


def test_greedy_loop_counts_up_to_max_tokens():
    buffer, length = run_counting(stop_id=999)
    assert int(length) == 5
    assert buffer[0, :5].tolist() == [2, 3, 4, 5, 6]


def test_loop_stops_at_stop_id():
    buffer, length = run_counting(stop_id=4)
    # seed 2 -> 3 -> 4(stop); generated up to and including the stop token.
    assert buffer[0, :int(length)].tolist() == [2, 3, 4]


def test_streaming_emits_generated_tokens_in_order():
    seen = []
    run = search.build_streaming(build_counting_step(), search.greedy, 4, 8,
                                 lambda token_id: seen.append(int(token_id)))
    buffer = jp.zeros((1, 8), dtype=jp.int32).at[0, 0].set(2)
    token = jp.reshape(jp.array(2, dtype=jp.int32), (1, 1))
    index = jp.array(0, dtype=jp.int32)
    buffer, length = run(jax.random.PRNGKey(0), buffer, token, index,
                         jp.zeros((1,)), jp.array(999, dtype=jp.int32))
    assert seen == buffer[0, 1:int(length)].tolist()
    assert seen == [3, 4, 5, 6]


def test_sampler_top_k_one_matches_greedy_and_is_seeded():
    row = jp.array([[1.0, 5.0, 2.0, 0.0]])[:, None, :]
    sampler = search.build_sampler(1.0, 1, 1.0)
    key = jax.random.PRNGKey(3)
    assert int(sampler(row, key)[0]) == int(search.greedy(row, key)[0])
    assert int(sampler(row, key)[0]) == int(sampler(row, key)[0])
