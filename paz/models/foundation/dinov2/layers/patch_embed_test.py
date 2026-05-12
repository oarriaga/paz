import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest
import keras
from keras import ops

from paz.models.foundation.dinov2.layers.patch_embed import (
    PatchEmbed,
    make_2tuple,
    build_projection_layer,
    build_normalize_layer,
    build_patch_embed_layers,
    validate_patch_dims,
    flatten_projected,
    set_all_attributes,
)

# ─── Helpers ────────────────────────────────────────────────────────────────


def make_image(B=1, H=224, W=224, C=3):
    return ops.convert_to_tensor(np.random.randn(B, H, W, C).astype("float32"))


def make_patch_embed(img_size=224, patch_size=16, dim=768, flat=True):
    return PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        embedding_dimension=dim,
        flatten_embedding=flat,
    )


# ─── make_2tuple ────────────────────────────────────────────────────────────


class TestMake2Tuple:
    def test_int_becomes_tuple(self):
        assert make_2tuple(7) == (7, 7)

    def test_list_passes_through(self):
        assert make_2tuple([3, 5]) == (3, 5)

    def test_tuple_passes_through(self):
        assert make_2tuple((4, 8)) == (4, 8)

    def test_wrong_length_list_raises(self):
        with pytest.raises(AssertionError):
            make_2tuple([1, 2, 3])

    def test_non_int_raises(self):
        with pytest.raises(AssertionError):
            make_2tuple(3.5)


# ─── build_projection_layer ─────────────────────────────────────────────────


class TestBuildProjectionLayer:
    def test_returns_conv2d(self):
        layer = build_projection_layer((16, 16), 768)
        assert isinstance(layer, keras.layers.Conv2D)

    def test_correct_filters(self):
        layer = build_projection_layer((16, 16), 384)
        assert layer.filters == 384

    def test_correct_kernel_size(self):
        layer = build_projection_layer((14, 14), 512)
        assert layer.kernel_size == (14, 14)

    def test_correct_strides(self):
        layer = build_projection_layer((16, 16), 768)
        assert layer.strides == (16, 16)

    def test_valid_padding(self):
        layer = build_projection_layer((16, 16), 768)
        assert layer.padding == "valid"

    def test_name_is_proj(self):
        layer = build_projection_layer((16, 16), 768)
        assert layer.name == "proj"


# ─── build_normalize_layer ───────────────────────────────────────────────────


class TestBuildNormalizeLayer:
    def test_none_returns_identity(self):
        layer = build_normalize_layer(None, 768)
        assert isinstance(layer, keras.layers.Identity)

    def test_layer_norm_factory(self):
        norm_cls = lambda _: keras.layers.LayerNormalization(axis=-1)
        layer = build_normalize_layer(norm_cls, 768)
        assert isinstance(layer, keras.layers.LayerNormalization)


# ─── build_patch_embed_layers ────────────────────────────────────────────────


class TestBuildPatchEmbedLayers:
    def test_returns_two_layers(self):
        proj, norm = build_patch_embed_layers((16, 16), 768, None)
        assert isinstance(proj, keras.layers.Conv2D)
        assert isinstance(norm, keras.layers.Identity)

    def test_proj_has_right_dim(self):
        proj, _ = build_patch_embed_layers((14, 14), 384, None)
        assert proj.filters == 384


# ─── validate_patch_dims ────────────────────────────────────────────────────


class TestValidatePatchDims:
    def test_valid_dims_pass(self):
        validate_patch_dims(224, 224, (16, 16))

    def test_invalid_height_raises(self):
        with pytest.raises(AssertionError):
            validate_patch_dims(225, 224, (16, 16))

    def test_invalid_width_raises(self):
        with pytest.raises(AssertionError):
            validate_patch_dims(224, 226, (16, 16))

    def test_non_square_patch(self):
        validate_patch_dims(224, 336, (16, 14))

    def test_different_valid_sizes(self):
        validate_patch_dims(518, 518, (14, 14))


# ─── flatten_projected ───────────────────────────────────────────────────────


class TestFlattenProjected:
    def test_output_shape(self):
        x = ops.ones((2, 14, 14, 768))
        H, W, out = flatten_projected(x, 2, 768)
        assert out.shape == (2, 196, 768)

    def test_H_W_values(self):
        x = ops.ones((1, 10, 8, 64))
        H, W, out = flatten_projected(x, 1, 64)
        assert np.array(H) == 10
        assert np.array(W) == 8

    def test_values_preserved(self):
        arr = np.arange(4 * 2 * 2 * 1, dtype="float32").reshape(4, 2, 2, 1)
        x = ops.convert_to_tensor(arr)
        _, _, out = flatten_projected(x, 4, 1)
        np.testing.assert_array_equal(np.array(out).reshape(-1), arr.reshape(-1))


# ─── PatchEmbed factory ──────────────────────────────────────────────────────


class TestPatchEmbedFunction:
    def test_returns_keras_model(self):
        model = make_patch_embed()
        assert isinstance(model, keras.Model)

    def test_number_of_patches(self):
        model = make_patch_embed(img_size=224, patch_size=16)
        assert model.number_of_patches == 196

    def test_patches_resolution(self):
        model = make_patch_embed(img_size=224, patch_size=16)
        assert model.patches_resolution == (14, 14)

    def test_img_size_attribute(self):
        model = make_patch_embed(img_size=224)
        assert model.img_size == (224, 224)

    def test_patch_size_attribute(self):
        model = make_patch_embed(patch_size=16)
        assert model.patch_size == (16, 16)

    def test_input_channels_attribute(self):
        model = PatchEmbed(input_channels=3)
        assert model.input_channels == 3

    def test_embedding_dimension_attribute(self):
        model = make_patch_embed(dim=384)
        assert model.embedding_dimension == 384

    def test_flatten_embedding_default_true(self):
        model = make_patch_embed()
        assert model.flatten_embedding is True

    def test_flatten_embedding_false(self):
        model = make_patch_embed(flat=False)
        assert model.flatten_embedding is False

    def test_projection_layer_is_conv2d(self):
        model = make_patch_embed()
        assert isinstance(model.projection_layer, keras.layers.Conv2D)

    def test_normalize_is_identity_by_default(self):
        model = make_patch_embed()
        assert isinstance(model.normalize, keras.layers.Identity)

    def test_custom_name(self):
        model = PatchEmbed(name="pe_test")
        assert model.name == "pe_test"

    def test_forward_output_shape_flattened(self):
        model = make_patch_embed(img_size=224, patch_size=16, dim=768)
        x = make_image(B=2, H=224, W=224)
        out = model(x)
        assert out.shape == (2, 196, 768)

    def test_forward_output_shape_not_flattened(self):
        model = make_patch_embed(img_size=224, patch_size=16, dim=768, flat=False)
        x = make_image(B=2, H=224, W=224)
        out = model(x)
        assert out.shape == (2, 14, 14, 768)

    def test_different_patch_sizes(self):
        model = PatchEmbed(img_size=518, patch_size=14, embedding_dimension=1536)
        x = make_image(B=1, H=518, W=518)
        out = model(x)
        n = (518 // 14) ** 2
        assert out.shape == (1, n, 1536)

    def test_number_of_patches_matches_output_tokens(self):
        model = make_patch_embed(img_size=224, patch_size=16)
        x = make_image(H=224, W=224)
        out = model(x)
        assert out.shape[1] == model.number_of_patches

    def test_non_square_image(self):
        model = PatchEmbed(img_size=(224, 336), patch_size=16, embedding_dimension=768)
        x = make_image(B=1, H=224, W=336)
        out = model(x)
        assert out.shape == (1, 14 * 21, 768)

    def test_with_normalization_layer(self):
        norm = lambda _: keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)
        model = PatchEmbed(embedding_dimension=256, normalization_layer=norm)
        x = make_image(H=224, W=224)
        out = model(x)
        assert out.shape[-1] == 256

    def test_projection_layer_is_built(self):
        model = make_patch_embed()
        assert model.projection_layer.built

    def test_no_trainable_variables_except_proj(self):
        model = make_patch_embed()
        assert len(model.trainable_variables) > 0
        proj_vars = model.projection_layer.trainable_variables
        assert len(proj_vars) == 2

    def test_projection_layer_set_weights(self):
        model = make_patch_embed(dim=768)
        proj = model.projection_layer
        w = np.random.randn(16, 16, 3, 768).astype("float32")
        b = np.zeros(768, dtype="float32")
        proj.set_weights([w, b])
        np.testing.assert_array_equal(proj.get_weights()[1], b)

    def test_deterministic_inference(self):
        model = make_patch_embed()
        x = make_image()
        out1 = np.array(model(x, training=False))
        out2 = np.array(model(x, training=False))
        np.testing.assert_allclose(out1, out2, atol=1e-6)

    def test_training_kwarg_accepted(self):
        model = make_patch_embed()
        x = make_image()
        out = model(x, training=True)
        assert out.shape == (1, 196, 768)

    def test_img_size_tuple(self):
        model = PatchEmbed(img_size=(224, 224), patch_size=16)
        assert model.number_of_patches == 196


# ─── hasattr-based layer detection ──────────────────────────────────────────


class TestHasattrPatchEmbedDetection:
    def test_hasattr_number_of_patches_is_true(self):
        model = make_patch_embed()
        assert hasattr(model, "number_of_patches")

    def test_other_keras_model_lacks_number_of_patches(self):
        other = keras.Sequential([keras.layers.Dense(8)])
        assert not hasattr(other, "number_of_patches")

    def test_find_patch_embed_in_sublayers(self):
        model = make_patch_embed()
        found = any(hasattr(layer, "number_of_patches") for layer in [model])
        assert found


# ─── Integration with DinoVisionTransformer ──────────────────────────────────


class TestPatchEmbedInVisionTransformer:
    def test_patch_embedding_number_of_patches_accessible(self):
        from paz.models.foundation.dinov2.models.vision_transformer import (
            DinoVisionTransformer,
        )

        model = DinoVisionTransformer(
            img_size=224,
            patch_size=16,
            embedding_dimension=64,
            depth=1,
            number_of_heads=4,
        )
        assert model.patch_embedding.number_of_patches == 196

    def test_patch_embedding_is_keras_model(self):
        from paz.models.foundation.dinov2.models.vision_transformer import (
            DinoVisionTransformer,
        )

        model = DinoVisionTransformer(
            img_size=224,
            patch_size=16,
            embedding_dimension=64,
            depth=1,
            number_of_heads=4,
        )
        assert isinstance(model.patch_embedding, keras.Model)

    def test_patch_embedding_has_projection_layer(self):
        from paz.models.foundation.dinov2.models.vision_transformer import (
            DinoVisionTransformer,
        )

        model = DinoVisionTransformer(
            img_size=224,
            patch_size=16,
            embedding_dimension=64,
            depth=1,
            number_of_heads=4,
        )
        assert isinstance(model.patch_embedding.projection_layer, keras.layers.Conv2D)

    def test_vision_transformer_forward_pass(self):
        from paz.models.foundation.dinov2.models.vision_transformer import (
            DinoVisionTransformer,
        )

        model = DinoVisionTransformer(
            img_size=224,
            patch_size=16,
            embedding_dimension=64,
            depth=1,
            number_of_heads=4,
        )
        x = make_image(B=1, H=224, W=224)
        out = model(x, training=False)
        assert out.shape[-1] == 64

    def test_patch_embed_found_via_hasattr(self):
        from paz.models.foundation.dinov2.models.vision_transformer import (
            DinoVisionTransformer,
        )

        model = DinoVisionTransformer(
            img_size=224,
            patch_size=16,
            embedding_dimension=64,
            depth=1,
            number_of_heads=4,
        )
        found = None
        for layer in model.layers:
            if hasattr(layer, "number_of_patches"):
                found = layer
                break
        assert found is not None
        assert isinstance(found.projection_layer, keras.layers.Conv2D)

    def test_weight_porting_pattern(self):
        from paz.models.foundation.dinov2.models.vision_transformer import (
            DinoVisionTransformer,
        )

        dim = 64
        model = DinoVisionTransformer(
            img_size=224,
            patch_size=16,
            embedding_dimension=dim,
            depth=1,
            number_of_heads=4,
        )
        found = None
        for layer in model.layers:
            if hasattr(layer, "number_of_patches"):
                found = layer
                break
        assert found is not None
        w = np.ones((16, 16, 3, dim), dtype="float32") * 0.01
        b = np.zeros(dim, dtype="float32")
        found.projection_layer.set_weights([w, b])
        np.testing.assert_array_equal(found.projection_layer.get_weights()[1], b)
