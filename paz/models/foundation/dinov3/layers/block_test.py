import os
import torch
import numpy as np
from torch import nn
from functools import partial
import keras
import jax
import jax.numpy as jnp
from keras import layers
import pytest

from paz.models.foundation.dinov3.layers.block import (
    SelfAttentionBlock,
    CausalSelfAttentionBlock,
)

from paz.models.foundation.dinov3.layers.torch_layers_for_testing import (
    PT_SelfAttention,
    PT_CausalSelfAttention,
    PT_Mlp,
    PT_SelfAttentionBlock,
    PT_CausalSelfAttentionBlock,
)


def transfer_weights_pt_to_pt_causal(pt_ref_block, pt_causal_block):
    """
    Helper to transfer weights from a reference PT SelfAttentionBlock to a
    PT CausalSelfAttentionBlock. This function is specific and not redundant.
    """
    with torch.no_grad():
        pt_causal_block.attention_norm.weight.data = (
            pt_ref_block.norm1.weight.detach().clone()
        )
        pt_causal_block.attention_norm.bias.data = (
            pt_ref_block.norm1.bias.detach().clone()
        )
        pt_causal_block.ffn_norm.weight.data = (
            pt_ref_block.norm2.weight.detach().clone()
        )
        pt_causal_block.ffn_norm.bias.data = pt_ref_block.norm2.bias.detach().clone()
        pt_causal_block.attention.qkv.weight.data = (
            pt_ref_block.attn.qkv.weight.detach().clone()
        )
        if (
            pt_ref_block.attn.qkv.bias is not None
            and pt_causal_block.attention.qkv.bias is not None
        ):
            pt_causal_block.attention.qkv.bias.data = (
                pt_ref_block.attn.qkv.bias.detach().clone()
            )
        pt_causal_block.attention.proj.weight.data = (
            pt_ref_block.attn.proj.weight.detach().clone()
        )
        if (
            pt_ref_block.attn.proj.bias is not None
            and pt_causal_block.attention.proj.bias is not None
        ):
            pt_causal_block.attention.proj.bias.data = (
                pt_ref_block.attn.proj.bias.detach().clone()
            )
        pt_causal_block.feed_forward.fc1.weight.data = (
            pt_ref_block.mlp.fc1.weight.detach().clone()
        )
        if (
            pt_ref_block.mlp.fc1.bias is not None
            and pt_causal_block.feed_forward.fc1.bias is not None
        ):
            pt_causal_block.feed_forward.fc1.bias.data = (
                pt_ref_block.mlp.fc1.bias.detach().clone()
            )
        pt_causal_block.feed_forward.fc2.weight.data = (
            pt_ref_block.mlp.fc2.weight.detach().clone()
        )
        if (
            pt_ref_block.mlp.fc2.bias is not None
            and pt_causal_block.feed_forward.fc2.bias is not None
        ):
            pt_causal_block.feed_forward.fc2.bias.data = (
                pt_ref_block.mlp.fc2.bias.detach().clone()
            )
        if not isinstance(pt_ref_block.ls1, nn.Identity):
            pt_causal_block.ls1.gamma.data = pt_ref_block.ls1.gamma.detach().clone()
        if not isinstance(pt_ref_block.ls2, nn.Identity):
            pt_causal_block.ls2.gamma.data = pt_ref_block.ls2.gamma.detach().clone()


def transfer_weights_generic(keras_block, pt_block, layer_name_map):
    """
    A generic function to transfer weights from a PyTorch block to a Keras block,
    replacing the two previous redundant functions.
    """
    for keras_attr_str, pt_attr_str in layer_name_map.items():
        pt_layer = pt_block
        for part in pt_attr_str.split("."):
            pt_layer = getattr(pt_layer, part)

        keras_layer = keras_block
        for part in keras_attr_str.split("."):
            keras_layer = getattr(keras_layer, part)

        weights_to_set = []
        if hasattr(pt_layer, "weight"):
            weight_data = pt_layer.weight.detach().numpy()
            if isinstance(keras_layer, layers.Dense):
                weights_to_set.append(weight_data.T)
            else:
                weights_to_set.append(weight_data)

        if hasattr(pt_layer, "bias") and pt_layer.bias is not None:
            weights_to_set.append(pt_layer.bias.detach().numpy())

        if "ls" in keras_attr_str and hasattr(pt_layer, "gamma"):
            weights_to_set.append(pt_layer.gamma.detach().numpy())

        if weights_to_set:
            keras_layer.set_weights(weights_to_set)


DINO_WEIGHT_PATH = "/path/that/does/not/exist/dinov3_vits16_pretrain.pth"
dinov3_files_exist = os.path.isfile(DINO_WEIGHT_PATH)


@pytest.mark.parametrize(
    "dim, num_heads, mlp_ratio, qkv_bias, init_values",
    [
        (64, 8, 4.0, True, None),
        (128, 4, 2.0, False, None),
        (96, 6, 4.0, True, 1e-4),
        (32, 4, 3.0, False, 1e-4),
    ],
)
def test_self_attention_block_consistency(
    dim, num_heads, mlp_ratio, qkv_bias, init_values
):
    np.random.seed(42)
    torch.manual_seed(42)
    pt_model = PT_SelfAttentionBlock(
        dim=dim,
        num_heads=num_heads,
        ffn_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        proj_bias=True,
        ffn_bias=True,
        init_values=init_values,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_class=PT_SelfAttention,
        ffn_layer=PT_Mlp,
        act_layer=nn.GELU,
    )
    if init_values:
        pt_model.ls1.reset_parameters()
        pt_model.ls2.reset_parameters()
    pt_model.eval()

    keras_model = SelfAttentionBlock(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        init_values=init_values,
    )

    np_input = np.random.rand(4, 16, dim).astype("float32")
    _ = keras_model(np_input)

    layer_map = {
        "norm1": "norm1",
        "norm2": "norm2",
        "attn.qkv": "attn.qkv",
        "attn.proj": "attn.proj",
        "mlp.fc1": "mlp.fc1",
        "mlp.fc2": "mlp.fc2",
    }
    if init_values:
        layer_map.update({"ls1": "ls1", "ls2": "ls2"})
    transfer_weights_generic(keras_model, pt_model, layer_map)

    pt_output = pt_model(torch.from_numpy(np_input)).detach().numpy()
    keras_output = keras_model(np_input, training=False)
    if hasattr(keras_output, "to_numpy"):
        keras_output = keras_output.to_numpy()

    np.testing.assert_allclose(pt_output, keras_output, atol=1e-5, rtol=1e-5)
    print(f"Consistency test passed for SelfAttentionBlock with dim={dim}")


@pytest.mark.parametrize(
    "dim, num_heads, mlp_ratio, init_values, qkv_bias",
    [(64, 8, 4.0, None, True), (128, 4, 2.0, 1e-5, False)],
)
def test_causal_self_attention_block_consistency(
    dim, num_heads, mlp_ratio, init_values, qkv_bias
):
    np.random.seed(42)
    torch.manual_seed(42)
    pt_model = PT_CausalSelfAttentionBlock(
        dim=dim,
        num_heads=num_heads,
        ffn_ratio=mlp_ratio,
        ls_init_value=init_values,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    pt_model.attention = PT_CausalSelfAttention(dim, num_heads, qkv_bias=qkv_bias)
    if init_values:
        pt_model.ls1.reset_parameters()
        pt_model.ls2.reset_parameters()
    pt_model.eval()

    keras_model = CausalSelfAttentionBlock(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        init_values=init_values,
        qkv_bias=qkv_bias,
    )

    np_input = np.random.rand(4, 16, dim).astype("float32")
    _ = keras_model(np_input)

    layer_map = {
        "attention_norm": "attention_norm",
        "ffn_norm": "ffn_norm",
        "attention.qkv": "attention.qkv",
        "attention.proj": "attention.proj",
        "feed_forward.fc1": "feed_forward.fc1",
        "feed_forward.fc2": "feed_forward.fc2",
    }
    if init_values:
        layer_map.update({"ls1": "ls1", "ls2": "ls2"})
    transfer_weights_generic(keras_model, pt_model, layer_map)

    pt_output = pt_model(torch.from_numpy(np_input)).detach().numpy()
    keras_output = np.asarray(keras_model(np_input, training=False))

    np.testing.assert_allclose(pt_output, keras_output, atol=1e-5, rtol=1e-5)
    print(f"Consistency test passed for CausalSelfAttentionBlock with dim={dim}")


@pytest.mark.parametrize(
    "block_class, dim, num_heads, drop_rate",
    [
        (SelfAttentionBlock, 64, 8, 0.1),
        (SelfAttentionBlock, 96, 6, 0.5),
        (SelfAttentionBlock, 128, 4, 0.5),
        (CausalSelfAttentionBlock, 64, 8, 0.1),
        (CausalSelfAttentionBlock, 96, 6, 0.5),
        (CausalSelfAttentionBlock, 128, 4, 0.5),
    ],
)
def test_block_training_mode(block_class, dim, num_heads, drop_rate):
    """A single, parametrized test for training mode behavior."""
    np.random.seed(123)
    init_kwargs = {"dim": dim, "num_heads": num_heads, "drop": drop_rate}
    if block_class == SelfAttentionBlock:
        init_kwargs.update({"attn_drop": drop_rate, "drop_path": drop_rate})

    keras_model = block_class(**init_kwargs)
    np_input = np.random.rand(4, 16, dim).astype("float32")

    train_raw = keras_model(np_input, training=True)
    train_output = train_raw[0] if isinstance(train_raw, tuple) else train_raw

    eval_raw = keras_model(np_input, training=False)
    eval_output = eval_raw[0] if isinstance(eval_raw, tuple) else eval_raw

    if hasattr(train_output, "to_numpy"):
        train_output = train_output.to_numpy()
    if hasattr(eval_output, "to_numpy"):
        eval_output = eval_output.to_numpy()

    assert train_output.shape == np_input.shape
    assert not np.allclose(train_output, eval_output)
    print(f"Training mode test passed for {block_class.__name__}")


@pytest.mark.skipif(not dinov3_files_exist, reason="DINOv3 model/weights not found.")
def test_dinov3_block_pretrained_weights_match():
    """Validates the Keras SelfAttentionBlock against pre-trained DINOv3 weights."""
    from paz.models.foundation.dinov3.models.torch_vision_transformer_for_testing import (
        PT_vit_small,
    )

    model_kwargs = {
        "img_size": 224,
        "patch_size": 16,
        "ffn_layer": "mlp",
        "untie_cls_and_patch_norms": False,
        "norm_layer": "layernorm",
        "layerscale_init": 1e-6,
        "n_storage_tokens": 4,
        "pos_embed_rope_dtype": "float32",
    }
    dinov3_model = PT_vit_small(**model_kwargs)
    state_dict = torch.load(DINO_WEIGHT_PATH, map_location=torch.device("cpu"))
    dinov3_model.load_state_dict(state_dict, strict=False)
    dinov3_model.eval()
    torch_block = dinov3_model.blocks[0]

    PRETRAINED_DIM = dinov3_model.embed_dim
    PRETRAINED_NUM_HEADS = torch_block.attn.num_heads
    QKV_BIAS = torch_block.attn.qkv.bias is not None
    MLP_RATIO = torch_block.mlp.fc1.out_features / PRETRAINED_DIM
    INIT_VALUES = (
        torch_block.ls1.init_values
        if not isinstance(torch_block.ls1, nn.Identity)
        else None
    )

    keras_block = SelfAttentionBlock(
        dim=PRETRAINED_DIM,
        num_heads=PRETRAINED_NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        qkv_bias=QKV_BIAS,
        init_values=INIT_VALUES,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    )

    x_torch = torch.rand(2, 201, PRETRAINED_DIM)
    x_keras = x_torch.numpy()
    _ = keras_block(np.zeros_like(x_keras, dtype="float32"))

    layer_map = {
        "norm1": "norm1",
        "norm2": "norm2",
        "attn.qkv": "attn.qkv",
        "attn.proj": "attn.proj",
        "mlp.fc1": "mlp.fc1",
        "mlp.fc2": "mlp.fc2",
    }
    if INIT_VALUES:
        layer_map.update({"ls1": "ls1", "ls2": "ls2"})
    transfer_weights_generic(keras_block, torch_block, layer_map)

    torch_out = torch_block(x_torch).detach().numpy()
    keras_out = keras_block(x_keras, training=False)
    if hasattr(keras_out, "to_numpy"):
        keras_out = keras_out.to_numpy()

    mean_abs_diff = np.mean(np.abs(torch_out - keras_out))
    print(f"\nPre-trained SelfAttentionBlock Mean Absolute Difference: {mean_abs_diff}")
    np.testing.assert_allclose(torch_out, keras_out, atol=1e-3)
    assert (
        mean_abs_diff < 1e-5
    ), f"Mean absolute difference {mean_abs_diff} exceeds tolerance."


@pytest.mark.skipif(not dinov3_files_exist, reason="DINOv3 model/weights not found.")
def test_dinov3_causal_block_pretrained_weights_match():
    """Validates the Keras CausalSelfAttentionBlock against pre-trained DINOv3 weights."""
    from paz.models.foundation.dinov3.models.torch_vision_transformer_for_testing import (
        PT_vit_small,
    )

    model_kwargs = {
        "img_size": 224,
        "patch_size": 16,
        "ffn_layer": "mlp",
        "untie_cls_and_patch_norms": False,
        "norm_layer": "layernorm",
        "layerscale_init": 1e-6,
        "n_storage_tokens": 4,
        "pos_embed_rope_dtype": "float32",
    }
    dinov3_model = PT_vit_small(**model_kwargs)
    state_dict = torch.load(DINO_WEIGHT_PATH, map_location=torch.device("cpu"))
    dinov3_model.load_state_dict(state_dict, strict=False)
    dinov3_model.eval()
    ref_block = dinov3_model.blocks[0]

    dim = dinov3_model.embed_dim
    heads = ref_block.attn.num_heads
    qkv_bias = ref_block.attn.qkv.bias is not None
    mlp_ratio = ref_block.mlp.fc1.out_features / dim
    init_values = (
        ref_block.ls1.init_values if hasattr(ref_block.ls1, "init_values") else None
    )

    pt_causal_block = PT_CausalSelfAttentionBlock(
        dim, heads, mlp_ratio, init_values, norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    pt_causal_block.attention = PT_CausalSelfAttention(dim, heads, qkv_bias=qkv_bias)
    transfer_weights_pt_to_pt_causal(ref_block, pt_causal_block)
    pt_causal_block.eval()

    keras_block = CausalSelfAttentionBlock(
        dim, heads, mlp_ratio, init_values, qkv_bias=qkv_bias
    )
    np_input = np.random.rand(2, 201, dim).astype("float32")
    _ = keras_block(np_input)

    layer_map = {
        "attention_norm": "attention_norm",
        "ffn_norm": "ffn_norm",
        "attention.qkv": "attention.qkv",
        "attention.proj": "attention.proj",
        "feed_forward.fc1": "feed_forward.fc1",
        "feed_forward.fc2": "feed_forward.fc2",
    }
    if init_values:
        layer_map.update({"ls1": "ls1", "ls2": "ls2"})
    transfer_weights_generic(keras_block, pt_causal_block, layer_map)

    pt_output = pt_causal_block(torch.from_numpy(np_input)).detach().numpy()
    keras_output = np.asarray(keras_block(np_input, training=False))
    mean_abs_diff = np.mean(np.abs(pt_output - keras_output))

    print(
        f"\nPre-trained CausalSelfAttentionBlock Mean Absolute Difference: {mean_abs_diff}"
    )
    np.testing.assert_allclose(pt_output, keras_output, atol=1e-4)
    assert mean_abs_diff < 1e-5


@pytest.mark.parametrize(
    "dim, num_heads, sequence_length",
    [
        (64, 8, 32),
        (128, 4, 16),
    ],
)
def test_causality_functional(dim, num_heads, sequence_length):
    """
    Tests the CausalSelfAttentionBlock for causal masking.
    An output at timestep `t` should not depend on inputs from `t+1`.
    """
    np.random.seed(42)
    # 1. Initialize the model
    model = CausalSelfAttentionBlock(dim=dim, num_heads=num_heads)

    # 2. Create an initial input and get the output
    original_input = np.random.rand(1, sequence_length, dim).astype("float32")
    original_output = model(original_input, training=False)
    if hasattr(original_output, "to_numpy"):
        original_output = original_output.to_numpy()

    # 3. Create a modified input where only the LAST token is changed
    modified_input = np.copy(original_input)
    modified_input[:, -1, :] = np.random.rand(1, dim)

    # 4. Get the output from the modified input
    modified_output = model(modified_input, training=False)
    if hasattr(modified_output, "to_numpy"):
        modified_output = modified_output.to_numpy()

    # 5. Assert that all outputs EXCEPT the last one are identical
    np.testing.assert_allclose(
        original_output[:, :-1, :],
        modified_output[:, :-1, :],
        atol=1e-6,
        err_msg="Outputs before the last token should be identical, causality is broken.",
    )

    # 6. Assert that the last output token IS different
    assert not np.allclose(
        original_output[:, -1, :],
        modified_output[:, -1, :],
    ), "The last output token should have changed."

    print(f"\nCausality test passed for dim={dim}, heads={num_heads}")


@pytest.mark.parametrize(
    "block_class, dim, num_heads",
    [
        (SelfAttentionBlock, 64, 8),
        (CausalSelfAttentionBlock, 32, 4),
    ],
)
def test_serialization(block_class, dim, num_heads, tmp_path):
    """
    Tests if the custom blocks can be saved and loaded correctly.
    """
    # 1. Create a model with the custom layer
    np_input = np.random.rand(2, 16, dim).astype("float32")
    layer = block_class(dim=dim, num_heads=num_heads, name="custom_block")
    model = keras.Sequential([keras.layers.Input(shape=(16, dim)), layer])

    # 2. Get the output of the original model
    original_output = model(np_input)

    # 3. Save and load the model
    model_path = tmp_path / "model.keras"
    model.save(model_path)
    loaded_model = keras.models.load_model(
        model_path, custom_objects={block_class.__name__: block_class}
    )

    # 4. Get the output of the loaded model
    loaded_output = loaded_model(np_input)

    # 5. Assert that outputs are identical
    np.testing.assert_allclose(
        original_output,
        loaded_output,
        atol=1e-6,
        err_msg="Output of loaded model does not match original.",
    )

    # 6. Assert that weights are identical
    for w1, w2 in zip(model.weights, loaded_model.weights):
        np.testing.assert_allclose(
            w1.numpy(),
            w2.numpy(),
            err_msg="Weights of loaded model do not match original.",
        )

    print(f"\nSerialization test passed for {block_class.__name__}")


@pytest.mark.parametrize(
    "dim, num_heads, qkv_bias",
    [(32, 4, True), (32, 4, False)],
)
def test_gradients_consistency(dim, num_heads, qkv_bias):
    """
    Verifies that the backward pass (gradients) is consistent
    between the Keras/JAX and PyTorch implementations.
    """
    np.random.seed(123)
    torch.manual_seed(123)

    # --- PyTorch Setup ---
    pt_model = PT_CausalSelfAttentionBlock(
        dim=dim,
        num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    pt_model.attention = PT_CausalSelfAttention(dim, num_heads, qkv_bias=qkv_bias)
    pt_model.eval()

    # --- Keras Setup ---
    keras_model = CausalSelfAttentionBlock(
        dim=dim, num_heads=num_heads, qkv_bias=qkv_bias
    )
    np_input = np.random.rand(2, 16, dim).astype("float32")
    _ = keras_model(np_input)  # Build layer

    # --- COMPLETE Weight Transfer ---
    layer_map = {
        "attention_norm": "attention_norm",
        "ffn_norm": "ffn_norm",
        "attention.qkv": "attention.qkv",
        "attention.proj": "attention.proj",
        "feed_forward.fc1": "feed_forward.fc1",
        "feed_forward.fc2": "feed_forward.fc2",
    }
    transfer_weights_generic(keras_model, pt_model, layer_map)

    # --- PyTorch Gradient Calculation ---
    pt_input = torch.from_numpy(np_input).requires_grad_(True)
    pt_output = pt_model(pt_input)
    pt_loss = pt_output.sum()
    pt_loss.backward()

    pt_grad_input = pt_input.grad.numpy()
    pt_grad_qkv_w = pt_model.attention.qkv.weight.grad.numpy()

    # --- Keras/JAX Gradient Calculation ---
    jax_input = jnp.array(np_input)

    def get_keras_loss(weights, inp):
        keras_model.set_weights(weights)
        output = keras_model(inp, training=False)
        return jnp.sum(output)

    grad_fn = jax.grad(get_keras_loss, argnums=[0, 1])

    initial_weights_vars = keras_model.weights
    initial_weights_np = keras_model.get_weights()  # This gives NumPy arrays

    keras_grads_w, keras_grad_input = grad_fn(initial_weights_np, jax_input)

    qkv_kernel_path = keras_model.attention.qkv.kernel.path
    qkv_weight_index = -1
    for i, weight_var in enumerate(initial_weights_vars):
        if weight_var.path == qkv_kernel_path:
            qkv_weight_index = i
            break

    if qkv_weight_index == -1:
        raise ValueError("Could not find QKV kernel in the model's weights.")

    keras_grad_qkv_w = keras_grads_w[qkv_weight_index]

    # --- Comparison ---
    print("\nComparing gradients...")
    np.testing.assert_allclose(
        pt_grad_input,
        np.array(keras_grad_input),
        atol=1e-5,
        rtol=1e-5,
        err_msg="Input gradients do not match.",
    )
    np.testing.assert_allclose(
        pt_grad_qkv_w.T,
        np.array(keras_grad_qkv_w),
        atol=1e-5,
        rtol=1e-5,
        err_msg="QKV weight gradients do not match.",
    )
    print(
        f"Gradient consistency test passed for dim={dim}, heads={num_heads}, bias={qkv_bias}"
    )
