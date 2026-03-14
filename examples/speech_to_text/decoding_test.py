import numpy as np
from keras import ops

from examples.speech_to_text.decoding import (
    build_whisper_base_en_prompt_token_ids,
)
from examples.speech_to_text.decoding import (
    compute_argmax_token_id,
)
from examples.speech_to_text.decoding import decode_token_ids
from examples.speech_to_text.decoding import extract_text_token_ids


def test_base_en_decode_prompt_is_explicit():
    assert build_whisper_base_en_prompt_token_ids() == [50257, 50357, 50362]


def test_next_token_id_uses_last_active_position():
    logits = np.zeros((1, 3, 5), dtype="float32")
    logits[0, 1, 4] = 10.0
    logits[0, 2, 1] = 20.0
    next_token_id = compute_argmax_token_id(ops.convert_to_tensor(logits), 2)
    assert next_token_id == 4


def test_greedy_decode_stops_on_stop_token():
    initial_token_ids = [1, 2]
    next_token_ids = [9, 8, 7]

    def next_token_function(token_ids):
        return next_token_ids[len(token_ids) - len(initial_token_ids)]

    token_ids = decode_token_ids(next_token_function, initial_token_ids, 8)
    assert token_ids == [1, 2, 9, 8]


def test_greedy_decode_stops_on_max_generated_tokens():
    initial_token_ids = [1, 2]

    def next_token_function(token_ids):
        return 7

    token_ids = decode_token_ids(
        next_token_function,
        initial_token_ids,
        8,
        max_generated_tokens=3,
    )
    assert token_ids == [1, 2, 7, 7, 7]


def test_build_text_token_ids_strips_prompt_and_stop():
    token_ids = [50257, 50357, 50362, 31373, 13, 50256, 11]
    text_token_ids = extract_text_token_ids(token_ids, 3, 50256)
    assert text_token_ids == [31373, 13]
