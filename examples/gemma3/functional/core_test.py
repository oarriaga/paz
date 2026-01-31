import numpy as np
from keras import ops
from keras.layers import RMSNormalization

from examples.gemma3.functional.gemma3 import apply_reversible_projection
from examples.gemma3.functional.gemma3 import apply_rotary_embedding
from examples.gemma3.functional.gemma3 import apply_tanh_soft_cap
from examples.gemma3.functional.gemma3 import build_reversible_embedding
from examples.gemma3.functional.gemma3 import compute_causal_mask
from examples.gemma3.functional.gemma3 import merge_padding_and_attention_mask
from examples.gemma3.functional.keras_hub_utils import ensure_keras_hub

ensure_keras_hub()

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.gemma3.rms_normalization import RMSNormalization as HubRMSNormalization

def test_apply_tanh_soft_cap_matches_formula():
    values = ops.convert_to_tensor([[-2.0, 0.5, 3.0]], dtype="float32")
    soft_cap = 2.0
    output = apply_tanh_soft_cap(values, soft_cap)
    expected = np.tanh(np.array([[-2.0, 0.5, 3.0]]) / soft_cap) * soft_cap
    output_np = ops.convert_to_numpy(output)
    np.testing.assert_allclose(output_np, expected, rtol=1e-6, atol=1e-6)


def test_compute_causal_mask_small():
    batch_size = 1
    input_length = 4
    output_length = 2
    cache_index = 1
    compute_mask = compute_causal_mask
    mask = compute_mask(batch_size, input_length, output_length, cache_index)
    row_1 = [True, True, False, False]
    row_2 = [True, True, True, False]
    expected = np.array([[row_1, row_2]])
    mask_np = ops.convert_to_numpy(mask)
    np.testing.assert_array_equal(mask_np, expected)


def test_merge_padding_and_attention_mask_combines():
    padding_values = [[1, 1, 0]]
    padding_mask = ops.convert_to_tensor(padding_values, dtype="int32")
    attention_values = [[[1, 1, 1], [1, 1, 0], [1, 0, 0]]]
    attention_mask = ops.convert_to_tensor(attention_values, dtype="int32")
    combined = merge_padding_and_attention_mask(padding_mask, attention_mask)
    expected = np.array([[[1, 1, 0], [1, 1, 0], [1, 0, 0]]])
    combined_np = ops.convert_to_numpy(combined)
    np.testing.assert_array_equal(combined_np, expected)


def test_apply_rotary_embedding_matches_hub():
    rng = np.random.default_rng(123)
    shape = (2, 3, 4, 8)
    values = rng.standard_normal(shape).astype("float32")
    inputs = ops.convert_to_tensor(values)
    max_wavelength = 10000.0
    scaling_factor = 2.0
    start_index = 1

    hub_layer = RotaryEmbedding(
        max_wavelength=max_wavelength, scaling_factor=scaling_factor
    )
    hub_output = hub_layer(inputs, start_index=start_index)

    apply_rotary = apply_rotary_embedding
    start = start_index
    wavelength = max_wavelength
    scale = scaling_factor
    clean_output = apply_rotary(inputs, start, wavelength, scale)
    clean_np = ops.convert_to_numpy(clean_output)
    hub_np = ops.convert_to_numpy(hub_output)
    np.testing.assert_allclose(clean_np, hub_np, rtol=1e-5, atol=1e-5)


def test_rms_norm_matches_hub():
    rng = np.random.default_rng(11)
    shape = (2, 3, 5)
    values = rng.standard_normal(shape).astype("float32")
    inputs = ops.convert_to_tensor(values)
    epsilon = 1e-6

    hub_layer = HubRMSNormalization(epsilon=epsilon)
    clean_layer = RMSNormalization(epsilon=epsilon, name="rms_norm")
    hub_layer(inputs)
    clean_layer(inputs)

    scale_values = rng.standard_normal((5,)).astype("float32")
    hub_layer.set_weights([scale_values])
    clean_layer.set_weights([scale_values + 1.0])

    hub_output = hub_layer(inputs)
    clean_output = clean_layer(inputs)
    clean_np = ops.convert_to_numpy(clean_output)
    hub_np = ops.convert_to_numpy(hub_output)
    np.testing.assert_allclose(clean_np, hub_np, rtol=1e-5, atol=1e-5)


def test_reversible_projection_uses_embedding_weights():
    embedding_layer = build_reversible_embedding(7, 4)
    token_ids = ops.convert_to_tensor([[1, 2, 3]], dtype="int32")
    embeddings = embedding_layer(token_ids)
    logits = apply_reversible_projection(embedding_layer, embeddings)
    kernel = embedding_layer.embeddings
    expected = ops.matmul(embeddings, ops.transpose(kernel))
    logits_np = ops.convert_to_numpy(logits)
    expected_np = ops.convert_to_numpy(expected)
    np.testing.assert_allclose(logits_np, expected_np, rtol=1e-6, atol=1e-6)
