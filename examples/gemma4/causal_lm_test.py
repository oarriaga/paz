import numpy as np

from .causal_lm import Gemma4CausalLM
from .model import build_text_backbone_args


def build_test_inputs():
    token_ids = np.array([[1, 2, 3, 4, 0]], dtype="int32")
    padding_mask = np.array([[1, 1, 1, 1, 0]], dtype="int32")
    return {"token_ids": token_ids, "padding_mask": padding_mask}


def test_causal_lm_output_shape():
    config = build_text_backbone_args()
    model = Gemma4CausalLM(config)
    inputs = build_test_inputs()
    logits = model(inputs)
    assert logits.shape == (1, 5, config.vocabulary_size)


def test_causal_lm_logits_are_raw_scores():
    config = build_text_backbone_args()
    model = Gemma4CausalLM(config)
    inputs = build_test_inputs()
    logits = model(inputs)
    row_sums = np.sum(np.exp(np.array(logits[0, 0, :])))
    assert not np.isclose(row_sums, 1.0, atol=0.1)


def test_causal_lm_soft_cap():
    config = build_text_backbone_args(final_logit_soft_cap=30.0)
    model = Gemma4CausalLM(config)
    inputs = build_test_inputs()
    logits = np.array(model(inputs))
    assert np.all(np.abs(logits) < 30.0)


def test_causal_lm_save_and_load(tmp_path):
    config = build_text_backbone_args()
    model = Gemma4CausalLM(config)
    inputs = build_test_inputs()
    original = np.array(model(inputs))
    path = tmp_path / "causal_lm.weights.h5"
    model.save_weights(str(path))
    loaded = Gemma4CausalLM(config, weights_path=path)
    reloaded = np.array(loaded(inputs))
    np.testing.assert_allclose(original, reloaded, 1e-6, 1e-6)
