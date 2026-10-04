import json

import jax.numpy as jp

from paz.models.foundation.gemma4.model import (
    Gemma4Backbone, build_text_backbone_args)
from paz.models.foundation.gemma4.causal_lm import Gemma4CausalLM
from paz.models.foundation.gemma4.configuration import (
    build_cache_head_dim, build_cache_num_kv_heads, build_head_dim,
    build_kv_source_map, build_num_kv_heads, is_global_attention_layer,
    load_config, save_config, shares_key_and_value, to_backbone_args)


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


def mixed_kv_config():
    # Local layers keep 2 KV heads at head_dim 8; global layers drop to a
    # single KV head at head_dim 16 and reuse the key projection as the value.
    return build_text_backbone_args(
        num_layers=6, sliding_window_pattern=3, head_dim=8, global_head_dim=16,
        num_global_key_value_heads=1,
        global_rope_partial_rotary_factor=0.25, sliding_window_size=4,
        vocabulary_size=64, hidden_dim=16, intermediate_dim=32,
        num_query_heads=4, num_key_value_heads=2)


def test_mixed_kv_heads_resolve_per_layer():
    config = mixed_kv_config()
    model = Gemma4Backbone(config)
    local, global_ = model.decoder_layers[0], model.decoder_layers[2]
    assert (local.num_kv_heads, local.head_dim) == (2, 8)
    assert (global_.num_kv_heads, global_.head_dim) == (1, 16)
    assert model(build_test_inputs()).shape == (2, 5, config.hidden_dim)


def test_shared_key_value_allocates_no_value_weights():
    model = Gemma4Backbone(mixed_kv_config())
    model(build_test_inputs())
    local, global_ = model.decoder_layers[0], model.decoder_layers[2]
    assert global_.value_proj is None
    assert local.value_proj is not None
    paths = [weight.path for weight in global_.weights]
    assert not [path for path in paths if "value_proj" in path]
    assert len(local.weights) == len(global_.weights) + 1


def test_shared_key_value_reuses_the_key_projection():
    model = Gemma4Backbone(mixed_kv_config())
    model(build_test_inputs())
    global_ = model.decoder_layers[2]
    x = jp.asarray(jp.arange(32, dtype=jp.float32).reshape(1, 2, 16) / 32.0)
    key, value = global_.key_and_value(x)
    expected = global_.value_norm(global_.key_proj(x))
    assert key.shape == value.shape == (1, 2, 1, 16)
    assert_close(value, expected)


def test_mixed_kv_cache_pads_the_head_axis():
    config = mixed_kv_config()
    model = Gemma4CausalLM(config)
    cache = jp.asarray(model.build_cache(8))
    # The padded cache carries the widest head count and head dimension.
    assert cache.shape == (1, config.num_layers, 2, 8, 2, 16)


def test_mixed_kv_prefill_parity_call_matches_call_with_cache():
    config = mixed_kv_config()
    model = Gemma4CausalLM(config)
    token_ids = jp.array([[5, 10, 15, 20, 25, 30, 2, 40]], dtype=jp.int32)
    length = token_ids.shape[1]
    inputs = {"token_ids": token_ids, "padding_mask": jp.ones_like(token_ids)}
    full = model(inputs)
    cache = jp.asarray(model.build_cache(length))
    for position in range(length):
        token = token_ids[:, position:position + 1]
        embedding = model.backbone.token_embedding(token)
        index = jp.array(position, dtype=jp.int32)
        step_logits, cache = model.call_with_cache(embedding, cache, index)
        assert_close(step_logits[0, 0], full[0, position], tol=1e-3)


def test_gemma4_12b_matches_the_published_text_configuration():
    config = to_backbone_args("gemma4_12b")
    assert config.vocabulary_size == 262_144
    assert config.num_layers == 48
    assert config.hidden_dim == 3840
    assert config.intermediate_dim == 15360
    assert config.num_query_heads == 16
    assert config.sliding_window_size == 1024
    assert config.local_rope_wavelength == 10_000.0
    assert config.global_rope_wavelength == 1_000_000.0
    assert config.global_rope_partial_rotary_factor == 0.25
    assert config.final_logit_soft_cap == 30.0
    assert config.dtype == "bfloat16"
    assert not config.hidden_size_per_layer_input
    assert config.num_kv_shared_layers == 0
    globals_ = [i for i in range(48) if is_global_attention_layer(config, i)]
    assert globals_ == [5, 11, 17, 23, 29, 35, 41, 47]
    assert (build_num_kv_heads(config, False), build_head_dim(config, False)) \
        == (8, 256)
    assert (build_num_kv_heads(config, True), build_head_dim(config, True)) \
        == (1, 512)
    assert shares_key_and_value(config, True)
    assert not shares_key_and_value(config, False)
    assert build_cache_num_kv_heads(config) == 8
    assert build_cache_head_dim(config) == 512


def test_config_without_the_new_fields_still_loads(tmp_path):
    # Published E2B artifacts predate the 12B fields; they must keep loading.
    path = tmp_path / "config.json"
    save_config(to_backbone_args("gemma4_2b"), path)
    values = json.loads(path.read_text())
    values.pop("num_global_key_value_heads")
    path.write_text(json.dumps(values))
    config = load_config(path)
    assert config.num_global_key_value_heads is None
    assert not shares_key_and_value(config, True)
