import jax.numpy as jp

from .causal_lm import Gemma4CausalLM
from .inference import Gemma4DecoderStep, build_empty_cache
from .model import build_text_backbone_args


def build_test_config():
    return build_text_backbone_args(use_sliding_window_attention=False)


def assert_close(left, right, tol=1e-3):
    diff = jp.max(jp.abs(left - right))
    assert float(diff) <= tol


def test_decoder_step_output_shape():
    config = build_test_config()
    step_model = Gemma4DecoderStep(config)
    cache = build_empty_cache(config, 8)
    token = jp.array([[1]], dtype=jp.int32)
    index = jp.array([0], dtype=jp.int32)
    logits, new_cache = step_model([token, cache, index])
    assert logits.shape == (1, 1, config.vocabulary_size)
    assert new_cache.shape == cache.shape


def test_cached_step_matches_full_sequence():
    config = build_test_config()
    full_model = Gemma4CausalLM(config)
    step_model = Gemma4DecoderStep(config)
    copy_decoder_step_weights(step_model, full_model)
    token_ids = jp.array([[5, 10, 15]], dtype=jp.int32)
    padding_mask = jp.ones_like(token_ids)
    inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
    full_logits = full_model(inputs)
    cache = build_empty_cache(config, 8)
    for position in range(3):
        token = token_ids[:, position:position + 1]
        index = jp.array([position], dtype=jp.int32)
        step_logits, cache = step_model([token, cache, index])
    step_last = step_logits[0, 0, :]
    full_last = full_logits[0, 2, :]
    assert_close(step_last, full_last)


def copy_decoder_step_weights(step_model, full_model):
    full_by_path = {weight.path: weight for weight in full_model.weights}
    for step_weight in step_model.weights:
        step_weight.assign(full_by_path[step_weight.path])
