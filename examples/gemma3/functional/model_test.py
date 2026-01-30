import numpy as np
from keras import ops

from examples.gemma3.functional.gemma3 import apply_reversible_projection
from examples.gemma3.functional.gemma3 import build_gemma3_backbone
from examples.gemma3.functional.gemma3 import build_vision_encoder
from examples.gemma3.functional import gemma3 as g3
from examples.gemma3.functional.test_utils import collect_backbone_weights
from examples.gemma3.functional.test_utils import copy_backbone_weights
from examples.gemma3.functional.test_utils import count_params

from keras_hub.src.models.gemma3 import gemma3_backbone
from keras_hub.src.models.gemma3 import gemma3_causal_lm
from keras_hub.src.models.gemma3 import gemma3_vision_encoder


def _build_text_only_setup():
    vocabulary_size = 32
    image_size = 16
    num_layers = 2
    num_query_heads = 2
    num_key_value_heads = 1
    hidden_dim = 8
    intermediate_dim = 16
    head_dim = 4
    query_head_dim_normalize = True
    use_query_key_norm = True
    use_post_ffw_norm = False
    use_post_attention_norm = False
    attention_logit_soft_cap = None
    use_sliding_window_attention = False
    sliding_window_size = 8
    local_rope_scaling_factor = 1.0
    global_rope_scaling_factor = 1.0
    use_bidirectional_attention = False
    layer_norm_epsilon = 1e-6
    dropout = 0.0
    dtype = "float32"
    name_prefix = "clean_text"

    hub_model = gemma3_backbone.Gemma3Backbone(
        vocabulary_size=vocabulary_size,
        image_size=image_size,
        num_layers=num_layers,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        head_dim=head_dim,
        query_head_dim_normalize=query_head_dim_normalize,
        use_query_key_norm=use_query_key_norm,
        use_post_ffw_norm=use_post_ffw_norm,
        use_post_attention_norm=use_post_attention_norm,
        attention_logit_soft_cap=attention_logit_soft_cap,
        use_sliding_window_attention=use_sliding_window_attention,
        sliding_window_size=sliding_window_size,
        vision_encoder=None,
        layer_norm_epsilon=layer_norm_epsilon,
        dropout=dropout,
        dtype=dtype,
        name="hub_text",
    )

    apply_backbone, layers = build_gemma3_backbone(
        vocabulary_size=vocabulary_size,
        image_size=image_size,
        num_layers=num_layers,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        head_dim=head_dim,
        query_head_dim_normalize=query_head_dim_normalize,
        use_query_key_norm=use_query_key_norm,
        use_post_ffw_norm=use_post_ffw_norm,
        use_post_attention_norm=use_post_attention_norm,
        attention_logit_soft_cap=attention_logit_soft_cap,
        use_sliding_window_attention=use_sliding_window_attention,
        sliding_window_size=sliding_window_size,
        local_rope_scaling_factor=local_rope_scaling_factor,
        global_rope_scaling_factor=global_rope_scaling_factor,
        use_bidirectional_attention=use_bidirectional_attention,
        layer_norm_epsilon=layer_norm_epsilon,
        dropout=dropout,
        dtype=dtype,
        name_prefix=name_prefix,
        vision_apply=None,
        vision_layers=None,
        vision_num_tokens=None,
    )

    token_ids = ops.convert_to_tensor([[1, 2, 3], [4, 5, 0]], dtype="int32")
    padding_mask = ops.convert_to_tensor([[1, 1, 1], [1, 1, 0]], dtype="int32")
    hub_inputs = {"token_ids": token_ids, "padding_mask": padding_mask}

    hub_model(hub_inputs)
    apply_backbone(token_ids, padding_mask, None, None, None, False)
    return hub_model, apply_backbone, layers, token_ids, padding_mask


def _build_vision_setup():
    vision_image_size = 16
    vision_patch_size = 4
    vision_num_heads = 2
    vision_hidden_dim = 8
    vision_num_layers = 2
    vision_intermediate_dim = 16
    vision_output_dim = 8
    vision_pool_size = 2
    vision_layer_norm_epsilon = 1e-6
    vision_dropout = 0.0
    vision_dtype = "float32"
    vision_name_prefix = "clean_vision"

    vision_apply, vision_layers = build_vision_encoder(
        image_size=vision_image_size,
        patch_size=vision_patch_size,
        num_heads=vision_num_heads,
        hidden_dim=vision_hidden_dim,
        num_layers=vision_num_layers,
        intermediate_dim=vision_intermediate_dim,
        output_dim=vision_output_dim,
        pool_size=vision_pool_size,
        layer_norm_epsilon=vision_layer_norm_epsilon,
        dropout=vision_dropout,
        dtype=vision_dtype,
        name_prefix=vision_name_prefix,
    )
    num_tokens = g3.compute_num_vision_tokens_per_image(16, 4, 2)

    vocabulary_size = 32
    image_size = 16
    num_layers = 2
    num_query_heads = 2
    num_key_value_heads = 1
    hidden_dim = 8
    intermediate_dim = 16
    head_dim = 4
    query_head_dim_normalize = True
    use_query_key_norm = True
    use_post_ffw_norm = False
    use_post_attention_norm = False
    attention_logit_soft_cap = None
    use_sliding_window_attention = False
    sliding_window_size = 8
    local_rope_scaling_factor = 1.0
    global_rope_scaling_factor = 1.0
    use_bidirectional_attention = False
    layer_norm_epsilon = 1e-6
    dropout = 0.0
    dtype = "float32"
    name_prefix = "clean_vision"

    hub_vision = gemma3_vision_encoder.Gemma3VisionEncoder(
        image_size=vision_image_size,
        patch_size=vision_patch_size,
        num_heads=vision_num_heads,
        hidden_dim=vision_hidden_dim,
        num_layers=vision_num_layers,
        intermediate_dim=vision_intermediate_dim,
        output_dim=vision_output_dim,
        pool_size=vision_pool_size,
        layer_norm_epsilon=vision_layer_norm_epsilon,
        dtype=vision_dtype,
        name="hub_vision",
    )

    hub_model = gemma3_backbone.Gemma3Backbone(
        vocabulary_size=vocabulary_size,
        image_size=image_size,
        num_layers=num_layers,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        head_dim=head_dim,
        query_head_dim_normalize=query_head_dim_normalize,
        use_query_key_norm=use_query_key_norm,
        use_post_ffw_norm=use_post_ffw_norm,
        use_post_attention_norm=use_post_attention_norm,
        attention_logit_soft_cap=attention_logit_soft_cap,
        use_sliding_window_attention=use_sliding_window_attention,
        sliding_window_size=sliding_window_size,
        vision_encoder=hub_vision,
        layer_norm_epsilon=layer_norm_epsilon,
        dropout=dropout,
        dtype=dtype,
        name="hub_vision_backbone",
    )

    apply_backbone, layers = build_gemma3_backbone(
        vocabulary_size=vocabulary_size,
        image_size=image_size,
        num_layers=num_layers,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        head_dim=head_dim,
        query_head_dim_normalize=query_head_dim_normalize,
        use_query_key_norm=use_query_key_norm,
        use_post_ffw_norm=use_post_ffw_norm,
        use_post_attention_norm=use_post_attention_norm,
        attention_logit_soft_cap=attention_logit_soft_cap,
        use_sliding_window_attention=use_sliding_window_attention,
        sliding_window_size=sliding_window_size,
        local_rope_scaling_factor=local_rope_scaling_factor,
        global_rope_scaling_factor=global_rope_scaling_factor,
        use_bidirectional_attention=use_bidirectional_attention,
        layer_norm_epsilon=layer_norm_epsilon,
        dropout=dropout,
        dtype=dtype,
        name_prefix=name_prefix,
        vision_apply=vision_apply,
        vision_layers=vision_layers,
        vision_num_tokens=num_tokens,
    )

    token_ids = ops.convert_to_tensor([[1, 2, 3, 4, 5]], dtype="int32")
    padding_mask = ops.convert_to_tensor([[1, 1, 1, 1, 1]], dtype="int32")
    rng = np.random.default_rng(0)
    image_shape = (1, 1, 16, 16, 3)
    images = rng.standard_normal(image_shape).astype("float32")
    images = ops.convert_to_tensor(images)
    vision_indices = ops.convert_to_tensor([[1, 2, 3, 4]], dtype="int32")
    vision_mask = ops.convert_to_tensor([[0, 1, 1, 1, 1]], dtype="int32")
    hub_inputs = {}
    hub_inputs["token_ids"] = token_ids
    hub_inputs["padding_mask"] = padding_mask
    hub_inputs["images"] = images
    hub_inputs["vision_indices"] = vision_indices
    hub_inputs["vision_mask"] = vision_mask

    hub_model(hub_inputs)
    apply_backbone(
        token_ids, padding_mask, images, vision_indices, vision_mask, False
    )
    return hub_model, apply_backbone, layers, hub_inputs


def test_backbone_text_only_param_count_matches_hub():
    hub_model, _, layers, _, _ = _build_text_only_setup()
    hub_count = hub_model.count_params()
    clean_count = count_params(collect_backbone_weights(layers))
    assert hub_count == clean_count


def test_backbone_text_only_output_matches_hub():
    setup = _build_text_only_setup()
    hub_model, apply_backbone, layers, token_ids, padding_mask = setup
    copy_backbone_weights(layers, hub_model)
    hub_inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
    hub_output = hub_model(hub_inputs)
    apply = apply_backbone
    clean_output = apply(token_ids, padding_mask, None, None, None, False)
    clean_np = ops.convert_to_numpy(clean_output)
    hub_np = ops.convert_to_numpy(hub_output)
    np.testing.assert_allclose(clean_np, hub_np, rtol=1e-5, atol=1e-5)


def test_backbone_vision_param_count_matches_hub():
    hub_model, _, layers, _ = _build_vision_setup()
    hub_count = hub_model.count_params()
    clean_count = count_params(collect_backbone_weights(layers))
    assert hub_count == clean_count


def test_backbone_vision_output_matches_hub():
    hub_model, apply_backbone, layers, hub_inputs = _build_vision_setup()
    copy_backbone_weights(layers, hub_model)
    hub_output = hub_model(hub_inputs)
    token_ids = hub_inputs["token_ids"]
    padding_mask = hub_inputs["padding_mask"]
    images = hub_inputs["images"]
    vision_ids = hub_inputs["vision_indices"]
    vision_mask = hub_inputs["vision_mask"]
    tokens = token_ids
    padding = padding_mask
    images_data = images
    ids = vision_ids
    mask = vision_mask
    args = tokens, padding, images_data, ids, mask, False
    clean_output = apply_backbone(*args)
    clean_np = ops.convert_to_numpy(clean_output)
    hub_np = ops.convert_to_numpy(hub_output)
    np.testing.assert_allclose(clean_np, hub_np, rtol=1e-5, atol=1e-5)


def test_causal_lm_logits_match_hub():
    setup = _build_text_only_setup()
    hub_model, apply_backbone, layers, token_ids, padding_mask = setup
    copy_backbone_weights(layers, hub_model)
    lm_kwargs = {}
    lm_kwargs["preprocessor"] = None
    lm_kwargs["backbone"] = hub_model
    hub_lm = gemma3_causal_lm.Gemma3CausalLM(**lm_kwargs)
    hub_inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
    hub_lm(hub_inputs)

    apply = apply_backbone
    clean_hidden = apply(token_ids, padding_mask, None, None, None, False)
    token_embedding = layers[0]
    clean_logits = apply_reversible_projection(token_embedding, clean_hidden)
    hub_output = hub_lm(hub_inputs)
    clean_np = ops.convert_to_numpy(clean_logits)
    hub_np = ops.convert_to_numpy(hub_output)
    np.testing.assert_allclose(clean_np, hub_np, rtol=1e-5, atol=1e-5)
