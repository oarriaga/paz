import importlib
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax.numpy as jp
import numpy as np
import pytest

from paz.models.foundation.gemma4.causal_lm import Gemma4CausalLM
from paz.models.foundation.gemma4.conversion import build_target_backbone
from paz.models.foundation.gemma4.huggingface import save_backbone
from paz.models.foundation.gemma4.model import build_text_backbone_args

pytest.importorskip("transformers")
# transformers exposes models lazily; import the submodule directly so
# pytest's import hook cannot shadow the attribute lookup.
unified = importlib.import_module("transformers.models.gemma4_unified")
torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

LAYER_TYPES = ["sliding_attention"] * 2 + ["full_attention"]
TOKENS = np.array([[5, 10, 15, 20, 25, 30, 2, 40]], dtype="int32")


def build_reference(path):
    # Two local layers then one global, twice: the 12B shape in miniature,
    # with one global KV head at a wider head dim and a shared K/V projection.
    text = unified.Gemma4UnifiedTextConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, global_head_dim=16, num_global_key_value_heads=1,
        attention_k_eq_v=True, sliding_window=4, layer_types=LAYER_TYPES * 2,
        hidden_size_per_layer_input=0, num_kv_shared_layers=0,
        final_logit_softcapping=30.0, max_position_embeddings=64)
    config = unified.Gemma4UnifiedConfig(text_config=text)
    torch.manual_seed(0)
    model = unified.Gemma4UnifiedForConditionalGeneration(config)
    model = model.eval().to(torch.float32)
    model.save_pretrained(path)
    return model


def build_config():
    return build_text_backbone_args(
        num_layers=6, sliding_window_pattern=3, head_dim=8, global_head_dim=16,
        num_global_key_value_heads=1,
        global_rope_partial_rotary_factor=0.25, sliding_window_size=4,
        vocabulary_size=64, hidden_dim=16, intermediate_dim=32,
        num_query_heads=4, num_key_value_heads=2, final_logit_soft_cap=30.0)


def compute_reference_logits(reference):
    with torch.no_grad():
        outputs = reference(input_ids=torch.as_tensor(TOKENS, dtype=torch.long))
    return np.asarray(outputs.logits.float().numpy())


def compute_paz_logits(config, path):
    model = Gemma4CausalLM(config)
    tokens = jp.asarray(TOKENS)
    inputs = {"token_ids": tokens, "padding_mask": jp.ones_like(tokens)}
    model(inputs)
    model.backbone.load_weights(str(path / "backbone.weights.h5"))
    return np.asarray(model(inputs))


def test_conversion_matches_hugging_face(tmp_path):
    reference = build_reference(tmp_path)
    config = build_config()
    save_backbone(tmp_path, build_target_backbone(config), tmp_path)
    expected = compute_reference_logits(reference)
    logits = compute_paz_logits(config, tmp_path)
    assert np.array_equal(expected.argmax(-1), logits.argmax(-1))
    assert float(np.max(np.abs(expected - logits))) < 1e-4


def test_transfer_rejects_a_checkpoint_with_unexpected_text_weights(tmp_path):
    build_reference(tmp_path)
    config = build_config()._replace(num_layers=5)
    with pytest.raises(ValueError, match="text weights"):
        save_backbone(tmp_path, build_target_backbone(config), tmp_path)
