import jax.numpy as jp

from paz.models.foundation.gemma4.causal_lm import Gemma4CausalLM
from paz.models.foundation.gemma4.model import build_text_backbone_args


def build_test_inputs():
    token_ids = jp.array([[1, 2, 3, 4, 0]], dtype=jp.int32)
    padding_mask = jp.array([[1, 1, 1, 1, 0]], dtype=jp.int32)
    return {"token_ids": token_ids, "padding_mask": padding_mask}


def assert_close(left, right, tol=1e-6):
    diff = jp.max(jp.abs(left - right))
    assert float(diff) <= tol


def test_causal_lm_logits_are_raw_scores():
    config = build_text_backbone_args()
    model = Gemma4CausalLM(config)
    logits = model(build_test_inputs())
    total = float(jp.sum(jp.exp(logits[0, 0, :])))
    assert abs(total - 1.0) > 0.1


def test_causal_lm_soft_cap_squashes_raw_logits_with_tanh():
    config = build_text_backbone_args()
    model = Gemma4CausalLM(config)
    inputs = build_test_inputs()
    raw = model(inputs)
    cap = float(jp.max(jp.abs(raw))) / 2.0
    capped_model = Gemma4CausalLM(config._replace(final_logit_soft_cap=cap))
    capped_model(inputs)
    capped_model.set_weights(model.get_weights())
    capped = capped_model(inputs)
    assert float(jp.max(jp.abs(capped))) < cap
    assert_close(capped, cap * jp.tanh(raw / cap), tol=1e-5)


def test_causal_lm_save_and_load(tmp_path):
    config = build_text_backbone_args()
    model = Gemma4CausalLM(config)
    inputs = build_test_inputs()
    output = model(inputs)
    path = tmp_path / "causal_lm.weights.h5"
    model.save_weights(str(path))
    loaded = Gemma4CausalLM(config)
    loaded(inputs)
    loaded.load_weights(str(path))
    assert_close(output, loaded(inputs))
