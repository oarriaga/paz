import numpy as np
from keras import ops

from examples.gemma3.functional.gemma3 import build_decoder_block
from examples.gemma3.functional.test_utils import copy_decoder_block_weights

from keras_hub.src.models.gemma3.gemma3_decoder_block import Gemma3DecoderBlock


def _build_block():
    return build_decoder_block(
        hidden_dim=8,
        intermediate_dim=16,
        head_dim=2,
        num_query_heads=4,
        num_key_value_heads=2,
        query_head_dim_normalize=True,
        use_query_key_norm=True,
        use_post_ffw_norm=True,
        use_post_attention_norm=True,
        gate_dim_reduction=1,
        logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=8,
        layer_norm_epsilon=1e-6,
        rope_wavelength=10000.0,
        rope_scaling_factor=1.0,
        use_bidirectional_attention=False,
        dropout=0.0,
        dtype="float32",
        name_prefix="clean",
    )


def _build_hub_block(name):
    return Gemma3DecoderBlock(
        hidden_dim=8,
        intermediate_dim=16,
        head_dim=2,
        num_query_heads=4,
        num_key_value_heads=2,
        query_head_dim_normalize=True,
        use_query_key_norm=True,
        use_post_ffw_norm=True,
        use_post_attention_norm=True,
        gate_dim_reduction=1,
        logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=8,
        layer_norm_epsilon=1e-6,
        rope_wavelength=10000.0,
        rope_scaling_factor=1.0,
        use_bidirectional_attention=False,
        dropout=0.0,
        name=name,
    )


def test_decoder_block_output_matches_hub():
    rng = np.random.default_rng(9)
    shape = (2, 3, 8)
    values = rng.standard_normal(shape).astype("float32")
    inputs = ops.convert_to_tensor(values)
    padding_mask = ops.ones((2, 3), dtype="int32")

    hub_block = _build_hub_block("hub_block")
    hub_block(inputs, padding_mask=padding_mask)

    apply_block, block_layers = _build_block()
    apply_block(inputs, padding_mask, None, None, False)

    copy_decoder_block_weights(block_layers, hub_block)

    hub_output = hub_block(inputs, padding_mask=padding_mask)
    clean_output = apply_block(inputs, padding_mask, None, None, False)
    clean_np = ops.convert_to_numpy(clean_output)
    hub_np = ops.convert_to_numpy(hub_output)
    np.testing.assert_allclose(clean_np, hub_np, rtol=1e-5, atol=1e-5)
