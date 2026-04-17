import jax.numpy as jp

from .causal_lm import Gemma4CausalLM
from .model import build_text_backbone_args


def build_test_inputs():
    token_ids = jp.array([[1, 2, 3, 4, 0]], dtype=jp.int32)
    padding_mask = jp.array([[1, 1, 1, 1, 0]], dtype=jp.int32)
    return {"token_ids": token_ids, "padding_mask": padding_mask}


def assert_close(left, right, tol=1e-6):
    diff = jp.max(jp.abs(left - right))
    assert float(diff) <= tol


def test_causal_lm_output_shape():
    config = build_text_backbone_args()
    model = Gemma4CausalLM(config)
    logits = model(build_test_inputs())
    assert logits.shape == (1, 5, config.vocabulary_size)


def test_causal_lm_logits_are_raw_scores():
    config = build_text_backbone_args()
    model = Gemma4CausalLM(config)
    logits = model(build_test_inputs())
    total = float(jp.sum(jp.exp(logits[0, 0, :])))
    assert abs(total - 1.0) > 0.1


def test_causal_lm_soft_cap():
    config = build_text_backbone_args(final_logit_soft_cap=30.0)
    model = Gemma4CausalLM(config)
    logits = model(build_test_inputs())
    assert bool(jp.all(jp.abs(logits) < 30.0))


def test_causal_lm_save_and_load(tmp_path):
    config = build_text_backbone_args()
    model = Gemma4CausalLM(config)
    inputs = build_test_inputs()
    output = model(inputs)
    path = tmp_path / "causal_lm.weights.h5"
    model.save_weights(str(path))
    loaded = Gemma4CausalLM(config, weights_path=path)
    assert_close(output, loaded(inputs))
