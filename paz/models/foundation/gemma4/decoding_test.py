import jax
import jax.numpy as jp

from paz.models.transformers import search

from paz.models.foundation.gemma4.decoding import (
    KVDecoder, extract_generated_ids, kv_decode, kv_sample)
from paz.models.foundation.gemma4.model import (
    Gemma4DecoderStep, build_empty_cache)
from paz.models.foundation.gemma4.model import build_text_backbone_args
from paz.models.foundation.gemma4.sampling import SamplingArgs


def build_test_config():
    return build_text_backbone_args(use_sliding_window_attention=False)


def test_kv_decoder_generates_tokens():
    config = build_test_config()
    step_model = Gemma4DecoderStep(config)
    prompt = [1, 2, 3]
    decoder = KVDecoder(step_model, prompt, 5, search.greedy, 16)
    cache = jp.asarray(build_empty_cache(config, decoder.max_decode_length))
    stop_id = jp.array(config.vocabulary_size - 1, dtype=jp.int32)
    buffer, length = decoder(cache, stop_id, jax.random.PRNGKey(0))
    ids = buffer[0, :length].tolist()
    assert len(ids) >= len(prompt)
    assert ids[:len(prompt)] == prompt


def test_kv_decoder_streams_generated_tokens_in_order():
    config = build_test_config()
    step_model = Gemma4DecoderStep(config)
    prompt = [1, 2, 3]
    seen = []
    decoder = KVDecoder(step_model, prompt, 5, search.greedy, 16,
                        emit=lambda token_id: seen.append(int(token_id)))
    cache = jp.asarray(build_empty_cache(config, decoder.max_decode_length))
    stop_id = jp.array(config.vocabulary_size - 1, dtype=jp.int32)
    buffer, length = decoder(cache, stop_id, jax.random.PRNGKey(0))
    assert seen == buffer[0, len(prompt):int(length)].tolist()


def test_kv_sample_seeded_and_greedy_matches_kv_decode():
    config = build_test_config()
    step_model = Gemma4DecoderStep(config)
    prompt = [1, 2, 3]
    stop = config.vocabulary_size - 1
    greedy = kv_decode(step_model, config, prompt, stop, 6)
    # top_k=1 sampling reduces to argmax here (no exact ties in float32 logits).
    peaked = kv_sample(step_model, config, prompt, stop, 6,
                       jax.random.PRNGKey(0), SamplingArgs(1.0, 1, 1.0))
    assert peaked == greedy
    args = SamplingArgs(temperature=1.0, top_k=0, top_p=1.0)
    key = jax.random.PRNGKey(2)
    a = kv_sample(step_model, config, prompt, stop, 6, key, args)
    b = kv_sample(step_model, config, prompt, stop, 6, key, args)
    assert a == b and a[:len(prompt)] == prompt


def test_extract_generated_ids():
    ids = [1, 2, 3, 50, 60, 255, 70]
    result = extract_generated_ids(ids, 3, 255)
    assert result == [50, 60]


def test_extract_generated_ids_no_stop():
    ids = [1, 2, 3, 50, 60, 70]
    result = extract_generated_ids(ids, 3, 255)
    assert result == [50, 60, 70]
