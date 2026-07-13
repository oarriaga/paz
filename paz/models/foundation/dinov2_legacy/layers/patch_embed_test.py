import numpy as np
import keras
from keras import Input, Model

from paz.models.foundation.dinov2_legacy.layers.patch_embed import (
    build_patch_embedding,
    to_pair,
)


def make_model(patch_size, input_channels, embedding_dimension):
    inputs = Input(shape=(None, None, input_channels))
    outputs = build_patch_embedding(
        inputs, patch_size, embedding_dimension, "patch_embed"
    )
    return Model(inputs, outputs, name="patch_embed")


def test_to_pair_with_int_returns_square_tuple():
    assert to_pair(14) == (14, 14)


def test_to_pair_with_tuple_returns_same_tuple():
    assert to_pair((7, 9)) == (7, 9)


def test_to_pair_with_list_returns_tuple():
    assert to_pair([3, 5]) == (3, 5)


def test_build_patch_embedding_returns_tensor():
    inputs = Input(shape=(None, None, 3))
    outputs = build_patch_embedding(inputs, 14, 384, "patch_embed")
    assert outputs.shape[-1] == 384


def test_wrapped_model_has_expected_name():
    model = make_model(14, 3, 384)
    assert model.name == "patch_embed"


def test_wrapped_model_has_proj_sublayer():
    model = make_model(14, 3, 384)
    proj = model.get_layer("patch_embed_proj")
    assert isinstance(proj, keras.layers.Conv2D)


def test_proj_kernel_shape_matches_patch_size_and_channels():
    model = make_model(14, 3, 384)
    proj = model.get_layer("patch_embed_proj")
    kernel = proj.kernel
    assert tuple(kernel.shape) == (14, 14, 3, 384)


def test_output_shape_for_square_image():
    model = make_model(14, 3, 384)
    image = np.zeros((1, 224, 224, 3), dtype="float32")
    output = model(image)
    expected_patches = (224 // 14) ** 2
    assert tuple(output.shape) == (1, expected_patches, 384)


def test_output_shape_supports_larger_image():
    model = make_model(14, 3, 384)
    image = np.zeros((2, 518, 518, 3), dtype="float32")
    output = model(image)
    expected_patches = (518 // 14) ** 2
    assert tuple(output.shape) == (2, expected_patches, 384)


def test_output_matches_manual_conv_then_flatten():
    keras.utils.set_random_seed(0)
    model = make_model(14, 3, 64)
    image = np.random.randn(1, 56, 56, 3).astype("float32")
    output = np.array(model(image))
    proj = model.get_layer("patch_embed_proj")
    conv_only = np.array(proj(image))
    flat = conv_only.reshape(1, -1, 64)
    np.testing.assert_allclose(output, flat, atol=1e-6)
