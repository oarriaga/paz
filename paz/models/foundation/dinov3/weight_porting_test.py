import os
import pytest
import torch
import numpy as np
import keras
from paz.models.foundation.dinov3.models.vision_transformer import vit_small
from paz.models.foundation.dinov3.models.torch_vision_transformer_for_testing import (
    PT_vit_small,
)
from paz.models.foundation.dinov3.port_dino_weights_from_torch_to_keras import (
    transfer_weights_from_pt_to_keras,
)


PT_WEIGHT_PATH = "/home/octavio/Storage/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
weights_exist = os.path.isfile(PT_WEIGHT_PATH)

MODEL_KWARGS = {
    "img_size": 224,
    "patch_size": 16,
    "ffn_layer": "mlp",
    "untie_cls_and_patch_norms": False,
    "norm_layer": "layernorm",
    "layerscale_init": 1e-6,
    "n_storage_tokens": 4,
    "pos_embed_rope_dtype": "float32",
}


@pytest.fixture(scope="module")
def pytorch_model_and_weights():
    if not weights_exist:
        pytest.skip(f"PyTorch weight file not found: {PT_WEIGHT_PATH}")

    pt_model = PT_vit_small(**MODEL_KWARGS)
    state_dict = torch.load(PT_WEIGHT_PATH, map_location=torch.device("cpu"))
    pt_model.load_state_dict(state_dict, strict=False)
    pt_model.eval()

    return pt_model, state_dict


@pytest.fixture(scope="module")
def keras_model():
    model = vit_small(**MODEL_KWARGS)
    dummy_input = np.zeros((1, 224, 224, 3), dtype="float32")
    model(dummy_input, training=False)
    return model


def test_cls_token_porting(pytorch_model_and_weights, keras_model):
    """Tests that cls_token is correctly ported from PyTorch to Keras."""
    _, state_dict = pytorch_model_and_weights

    original_cls_token = keras_model.cls_token.numpy().copy()
    keras_model.cls_token.assign(state_dict["cls_token"].numpy())

    np.testing.assert_array_equal(
        keras_model.cls_token.numpy(),
        state_dict["cls_token"].numpy(),
        err_msg="cls_token not correctly assigned"
    )
    assert not np.array_equal(original_cls_token, keras_model.cls_token.numpy()), \
        "cls_token was not modified"


def test_mask_token_porting(pytorch_model_and_weights, keras_model):
    """Tests that mask_token is correctly ported from PyTorch to Keras."""
    _, state_dict = pytorch_model_and_weights

    original_mask_token = keras_model.mask_token.numpy().copy()
    keras_model.mask_token.assign(state_dict["mask_token"].numpy())

    np.testing.assert_array_equal(
        keras_model.mask_token.numpy(),
        state_dict["mask_token"].numpy(),
        err_msg="mask_token not correctly assigned"
    )
    assert not np.array_equal(original_mask_token, keras_model.mask_token.numpy()), \
        "mask_token was not modified"


def test_storage_tokens_porting(pytorch_model_and_weights, keras_model):
    """Tests that storage_tokens is correctly ported from PyTorch to Keras."""
    _, state_dict = pytorch_model_and_weights

    if "storage_tokens" not in state_dict or keras_model.storage_tokens is None:
        pytest.skip("storage_tokens not present in model")

    original_storage = keras_model.storage_tokens.numpy().copy()
    keras_model.storage_tokens.assign(state_dict["storage_tokens"].numpy())

    np.testing.assert_array_equal(
        keras_model.storage_tokens.numpy(),
        state_dict["storage_tokens"].numpy(),
        err_msg="storage_tokens not correctly assigned"
    )
    assert not np.array_equal(original_storage, keras_model.storage_tokens.numpy()), \
        "storage_tokens was not modified"


def test_patch_embed_weight_shapes(pytorch_model_and_weights, keras_model):
    """Tests that patch embedding weights have compatible shapes after transposition."""
    _, state_dict = pytorch_model_and_weights

    pt_weight = state_dict["patch_embed.proj.weight"]
    pt_weight_transposed = pt_weight.permute(2, 3, 1, 0).numpy()

    keras_weight_shape = keras_model.patch_embed.projection.kernel.shape

    assert pt_weight_transposed.shape == tuple(keras_weight_shape), \
        f"Patch embed weight shapes incompatible: PT {pt_weight_transposed.shape} vs Keras {keras_weight_shape}"


def test_patch_embed_porting(pytorch_model_and_weights, keras_model):
    """Tests that patch embedding weights are correctly ported."""
    _, state_dict = pytorch_model_and_weights

    pt_weight = state_dict["patch_embed.proj.weight"].permute(2, 3, 1, 0).numpy()
    pt_bias = state_dict["patch_embed.proj.bias"].numpy() if "patch_embed.proj.bias" in state_dict else None

    keras_model.patch_embed.projection.kernel.assign(pt_weight)
    if pt_bias is not None:
        keras_model.patch_embed.projection.bias.assign(pt_bias)

    np.testing.assert_allclose(
        keras_model.patch_embed.projection.kernel.numpy(),
        pt_weight,
        rtol=1e-6,
        err_msg="Patch embed weights not correctly ported"
    )


def test_rope_embed_porting(pytorch_model_and_weights, keras_model):
    """Tests that RoPE embedding periods are correctly ported."""
    _, state_dict = pytorch_model_and_weights

    if "rope_embed.periods" not in state_dict:
        pytest.skip("rope_embed.periods not in state_dict")

    rope_periods = state_dict["rope_embed.periods"].float().numpy()
    keras_model.rope_embed.set_weights([rope_periods])

    np.testing.assert_allclose(
        keras_model.rope_embed.get_weights()[0],
        rope_periods,
        rtol=1e-6,
        err_msg="RoPE periods not correctly ported"
    )


def test_block_attention_weight_shapes(pytorch_model_and_weights, keras_model):
    """Tests that attention weights in blocks have compatible shapes."""
    _, state_dict = pytorch_model_and_weights

    for i in range(len(keras_model.blocks)):
        pt_qkv_weight = state_dict[f"blocks.{i}.attn.qkv.weight"]
        pt_proj_weight = state_dict[f"blocks.{i}.attn.proj.weight"]

        keras_qkv_shape = keras_model.blocks[i].attn.qkv.kernel.shape
        keras_proj_shape = keras_model.blocks[i].attn.proj.kernel.shape

        assert pt_qkv_weight.T.shape == tuple(keras_qkv_shape), \
            f"Block {i} QKV weight shapes incompatible"
        assert pt_proj_weight.T.shape == tuple(keras_proj_shape), \
            f"Block {i} projection weight shapes incompatible"


def test_block_norm_weight_shapes(pytorch_model_and_weights, keras_model):
    """Tests that normalization layers in blocks have compatible shapes."""
    _, state_dict = pytorch_model_and_weights

    for i in range(len(keras_model.blocks)):
        pt_norm1_weight = state_dict[f"blocks.{i}.norm1.weight"]
        pt_norm2_weight = state_dict[f"blocks.{i}.norm2.weight"]

        keras_norm1_gamma_shape = keras_model.blocks[i].norm1.gamma.shape
        keras_norm2_gamma_shape = keras_model.blocks[i].norm2.gamma.shape

        assert pt_norm1_weight.shape == tuple(keras_norm1_gamma_shape), \
            f"Block {i} norm1 weight shapes incompatible"
        assert pt_norm2_weight.shape == tuple(keras_norm2_gamma_shape), \
            f"Block {i} norm2 weight shapes incompatible"


def test_block_mlp_weight_shapes(pytorch_model_and_weights, keras_model):
    """Tests that MLP weights in blocks have compatible shapes."""
    _, state_dict = pytorch_model_and_weights

    for i in range(len(keras_model.blocks)):
        if f"blocks.{i}.mlp.fc1.weight" in state_dict:
            pt_fc1_weight = state_dict[f"blocks.{i}.mlp.fc1.weight"]
            pt_fc2_weight = state_dict[f"blocks.{i}.mlp.fc2.weight"]

            keras_fc1_shape = keras_model.blocks[i].mlp.fc1.kernel.shape
            keras_fc2_shape = keras_model.blocks[i].mlp.fc2.kernel.shape

            assert pt_fc1_weight.T.shape == tuple(keras_fc1_shape), \
                f"Block {i} MLP fc1 weight shapes incompatible"
            assert pt_fc2_weight.T.shape == tuple(keras_fc2_shape), \
                f"Block {i} MLP fc2 weight shapes incompatible"


def test_block_layerscale_presence(pytorch_model_and_weights, keras_model):
    """Tests that LayerScale is present when init_values is set."""
    for i in range(len(keras_model.blocks)):
        block = keras_model.blocks[i]

        if keras_model.blocks[i].init_values is not None:
            assert hasattr(block.ls1, 'gamma'), \
                f"Block {i} ls1 should have gamma when init_values is set"
            assert hasattr(block.ls2, 'gamma'), \
                f"Block {i} ls2 should have gamma when init_values is set"


def test_final_norm_weight_shapes(pytorch_model_and_weights, keras_model):
    """Tests that final normalization layers have compatible shapes."""
    _, state_dict = pytorch_model_and_weights

    pt_norm_weight = state_dict["norm.weight"]
    keras_norm_gamma_shape = keras_model.norm.gamma.shape

    assert pt_norm_weight.shape == tuple(keras_norm_gamma_shape), \
        f"Final norm weight shapes incompatible"

    if keras_model.cls_norm:
        pt_cls_norm_weight = state_dict["cls_norm.weight"]
        keras_cls_norm_gamma_shape = keras_model.cls_norm.gamma.shape
        assert pt_cls_norm_weight.shape == tuple(keras_cls_norm_gamma_shape), \
            "cls_norm weight shapes incompatible"


def test_full_weight_porting_runs_without_error(pytorch_model_and_weights, keras_model):
    """Tests that the full weight porting function runs without errors."""
    _, state_dict = pytorch_model_and_weights

    try:
        transfer_weights_from_pt_to_keras(state_dict, keras_model)
    except Exception as e:
        pytest.fail(f"Weight porting failed with error: {e}")


def test_ported_model_output_matches_pytorch(pytorch_model_and_weights, keras_model):
    """Tests that after porting weights, model outputs match PyTorch."""
    pt_model, state_dict = pytorch_model_and_weights

    transfer_weights_from_pt_to_keras(state_dict, keras_model)

    np.random.seed(42)
    dummy_input_np = np.random.randn(1, 224, 224, 3).astype("float32")
    keras_input = keras.ops.convert_to_tensor(dummy_input_np)
    pt_input = torch.from_numpy(dummy_input_np.transpose(0, 3, 1, 2))

    keras_output = keras_model(keras_input, training=False)
    with torch.no_grad():
        pt_output = pt_model(pt_input, is_training=False)

    keras_output_np = keras.ops.convert_to_numpy(keras_output)
    pt_output_np = pt_output.detach().cpu().numpy()

    np.testing.assert_allclose(
        keras_output_np,
        pt_output_np,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Ported model outputs do not match PyTorch outputs"
    )


def test_weight_porting_is_deterministic(pytorch_model_and_weights):
    """Tests that weight porting produces the same results when run multiple times."""
    _, state_dict = pytorch_model_and_weights

    model1 = vit_small(**MODEL_KWARGS)
    dummy_input = np.zeros((1, 224, 224, 3), dtype="float32")
    model1(dummy_input, training=False)
    transfer_weights_from_pt_to_keras(state_dict, model1)

    model2 = vit_small(**MODEL_KWARGS)
    model2(dummy_input, training=False)
    transfer_weights_from_pt_to_keras(state_dict, model2)

    for w1, w2 in zip(model1.weights, model2.weights):
        np.testing.assert_array_equal(
            w1.numpy(),
            w2.numpy(),
            err_msg=f"Weight porting is not deterministic for weight {w1.name}"
        )


def test_weight_count_matches(pytorch_model_and_weights, keras_model):
    """Tests that the number of trainable parameters matches between PyTorch and Keras."""
    pt_model, _ = pytorch_model_and_weights

    pt_params = sum(p.numel() for p in pt_model.parameters() if p.requires_grad)
    keras_params = sum(np.prod(w.shape) for w in keras_model.trainable_weights)

    tolerance = 0.01
    assert abs(pt_params - keras_params) / pt_params < tolerance, \
        f"Parameter count mismatch: PyTorch={pt_params}, Keras={keras_params}"
