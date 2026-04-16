import numpy as np

from .inference import Gemma4DecoderStep, build_empty_cache
from .causal_lm import Gemma4CausalLM
from .model import build_text_backbone_args
from .reference import copy_text_backbone_weights


def build_test_config():
    return build_text_backbone_args(
        use_sliding_window_attention=False)


def test_decoder_step_output_shape():
    config = build_test_config()
    step = Gemma4DecoderStep(config)
    cache = np.zeros(
        (1, config.num_layers, 2, 8,
         config.num_key_value_heads, config.head_dim),
        dtype="float32")
    token = np.array([[1]], dtype="int32")
    index = np.array([0], dtype="int32")
    logits, new_cache = step([token, cache, index])
    assert logits.shape == (1, 1, config.vocabulary_size)
    assert new_cache.shape == cache.shape


def test_cached_step_matches_full_sequence():
    config = build_test_config()
    full_model = Gemma4CausalLM(config)
    step_model = Gemma4DecoderStep(config)
    copy_decoder_step_weights(step_model, full_model)
    token_ids = np.array([[5, 10, 15]], dtype="int32")
    padding_mask = np.ones_like(token_ids)
    full_inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
    full_logits = np.array(full_model(full_inputs))
    max_len = 8
    cache = np.zeros(
        (1, config.num_layers, 2, max_len,
         config.num_key_value_heads, config.head_dim),
        dtype="float32")
    for position in range(3):
        token = token_ids[:, position:position + 1]
        index = np.array([position], dtype="int32")
        step_logits, cache = step_model([token, cache, index])
        cache = np.array(cache)
    step_last = np.array(step_logits[0, 0, :])
    full_last = full_logits[0, 2, :]
    np.testing.assert_allclose(step_last, full_last, 1e-3, 1e-3)


def copy_decoder_step_weights(step_model, full_model):
    full_by_path = {w.path: w for w in full_model.weights}
    for step_w in step_model.weights:
        full_w = full_by_path[step_w.path]
        step_w.assign(full_w)
