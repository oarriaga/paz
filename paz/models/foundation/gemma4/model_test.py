import jax.numpy as jp

from paz.models.foundation.gemma4.model import (
    Gemma4Backbone, build_text_backbone_args)
from paz.models.foundation.gemma4.causal_lm import Gemma4CausalLM
from paz.models.foundation.gemma4.configuration import build_kv_source_map


def build_test_inputs():
    token_ids = jp.array([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]], dtype=jp.int32)
    padding_mask = jp.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=jp.int32)
    return {"token_ids": token_ids, "padding_mask": padding_mask}


def assert_close(left, right, tol=1e-6):
    assert float(jp.max(jp.abs(left - right))) <= tol


def test_backbone_builds_and_shapes():
    config = build_text_backbone_args()
    model = Gemma4Backbone(config)
    output = model(build_test_inputs())
    assert output.shape == (2, 5, config.hidden_dim)


def test_backbone_save_and_load(tmp_path):
    config = build_text_backbone_args()
    model = Gemma4Backbone(config)
    inputs = build_test_inputs()
    output = model(inputs)
    path = tmp_path / "gemma4_backbone.weights.h5"
    model.save_weights(str(path))
    loaded = Gemma4Backbone(config)
    loaded(inputs)
    loaded.load_weights(str(path))
    assert_close(output, loaded(inputs))


def test_backbone_supports_per_layer_inputs():
    config = build_text_backbone_args(hidden_size_per_layer_input=2)
    model = Gemma4Backbone(config)
    output = model(build_test_inputs())
    assert output.shape == (2, 5, config.hidden_dim)


def test_backbone_runs_global_partial_rope():
    config = build_text_backbone_args(
        num_layers=6, sliding_window_pattern=3, head_dim=16, global_head_dim=16,
        global_rope_partial_rotary_factor=0.25)
    model = Gemma4Backbone(config)
    output = model(build_test_inputs())
    assert output.shape == (2, 5, config.hidden_dim)


def test_kv_source_map_shares_tail_layers():
    config = build_text_backbone_args(
        num_layers=6, num_kv_shared_layers=2, sliding_window_pattern=3)
    source_map = build_kv_source_map(config)
    # local layer 4 shares from local layer 3; global layer 5 from global 2.
    assert source_map == {4: 3, 5: 2}
    model = Gemma4Backbone(config)
    assert model(build_test_inputs()).shape == (2, 5, config.hidden_dim)


def full_feature_config():
    return build_text_backbone_args(
        num_layers=6, sliding_window_pattern=3, head_dim=8, global_head_dim=16,
        hidden_size_per_layer_input=4, num_kv_shared_layers=2,
        use_double_wide_mlp=True, global_rope_partial_rotary_factor=0.5,
        use_sliding_window_attention=True, sliding_window_size=4,
        final_logit_soft_cap=30.0, vocabulary_size=64, hidden_dim=16,
        intermediate_dim=32, num_query_heads=4, num_key_value_heads=2)


def test_prefill_parity_call_matches_call_with_cache():
    config = full_feature_config()
    model = Gemma4CausalLM(config)
    token_ids = jp.array([[5, 10, 15, 20, 25, 30, 2, 40]], dtype=jp.int32)
    length = token_ids.shape[1]
    inputs = {"token_ids": token_ids, "padding_mask": jp.ones_like(token_ids)}
    full = model(inputs)
    cache = jp.asarray(model.build_cache(length))
    for position in range(length):
        token = token_ids[:, position:position + 1]
        embedding = model.backbone.token_embedding(token)
        per_layer = model.backbone.per_layer_lookup(token)
        index = jp.array(position, dtype=jp.int32)
        step_logits, cache = model.call_with_cache(
            embedding, cache, index, None, per_layer)
        assert_close(step_logits[0, 0], full[0, position], tol=1e-3)


def test_bfloat16_prefill_parity_beyond_sliding_window():
    # Regression: the cached path once lost precision against the full
    # forward in bfloat16 (float32 RoPE keys truncated by the bfloat16
    # cache, missing __call__ autocast, bfloat16 RoPE positions past 256).
    config = full_feature_config()._replace(dtype="bfloat16")
    model = Gemma4Backbone(config)
    length = 300
    token_ids = jp.arange(length, dtype=jp.int32)[None] % 60 + 2
    embedding = model.token_embedding(token_ids)
    padding_mask = jp.ones_like(token_ids)
    full = model.forward_from_embedding(embedding, padding_mask, token_ids)
    per_layer = model.per_layer_lookup(token_ids)
    cache = jp.asarray(model.build_cache(length))
    positions = jp.arange(length, dtype=jp.int32)[None]
    index = jp.array(0, dtype=jp.int32)
    cached, _ = model.call_with_cache(
        embedding, cache, index, positions, per_layer)
    full = jp.asarray(full, jp.float32)
    cached = jp.asarray(cached, jp.float32)
    assert_close(full, cached, tol=1e-3)


def test_causal_lm_cached_step_shapes():
    config = build_text_backbone_args(hidden_size_per_layer_input=2)
    model = Gemma4CausalLM(config)
    token = jp.array([[1]], dtype=jp.int32)
    cache = jp.asarray(model.build_cache(8))
    embedding = model.backbone.token_embedding(token)
    per_layer = model.backbone.per_layer_lookup(token)
    logits, new_cache = model.call_with_cache(
        embedding, cache, jp.array(0, jp.int32), None, per_layer)
    assert logits.shape == (1, 1, config.vocabulary_size)
    assert new_cache.shape == cache.shape
