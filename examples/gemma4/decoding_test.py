import jax.numpy as jp

from .decoding import KVDecoder, build_step_inputs, extract_generated_ids
from .inference import Gemma4DecoderStep, build_empty_cache
from .model import build_text_backbone_args


def build_test_config():
    return build_text_backbone_args(
        use_sliding_window_attention=False)


def test_kv_decoder_generates_tokens():
    config = build_test_config()
    step_model = Gemma4DecoderStep(config)
    prompt = [1, 2, 3]
    max_tokens = 5
    max_seq = 16
    decoder = KVDecoder(step_model, prompt, max_tokens, max_seq)
    cache = build_empty_cache(config, decoder.max_decode_length)
    cache = jp.asarray(cache)
    stop_id = jp.array(config.vocabulary_size - 1, dtype=jp.int32)
    buffer, length = decoder(cache, stop_id)
    ids = buffer[0, :length].tolist()
    assert len(ids) >= len(prompt)
    assert ids[:len(prompt)] == prompt


def test_extract_generated_ids():
    ids = [1, 2, 3, 50, 60, 255, 70]
    result = extract_generated_ids(ids, 3, 255)
    assert result == [50, 60]


def test_extract_generated_ids_no_stop():
    ids = [1, 2, 3, 50, 60, 70]
    result = extract_generated_ids(ids, 3, 255)
    assert result == [50, 60, 70]
