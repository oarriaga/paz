import os
import torch
import pytest
import numpy as np

from paz.models.foundation.dinov3.layers.rope_position_encoding import (
    RopePositionEmbedding,
)

from paz.models.foundation.dinov3.layers.torch_layers_for_testing import (
    PT_RopePositionEmbedding,
)


@pytest.fixture
def params():
    """Provides common parameters for tests."""
    return {
        "embed_dim": 256,
        "num_heads": 8,
        "H": 32,
        "W": 24,
    }


def run_and_compare(keras_model, torch_model, H, W, training, atol=1e-5):
    """Helper function to execute models and assert closeness."""
    # Set training mode for PyTorch model
    torch_model.train(training)

    # PyTorch execution
    sin_torch, cos_torch = torch_model(H=H, W=W)
    sin_torch_np = sin_torch.detach().cpu().numpy()
    cos_torch_np = cos_torch.detach().cpu().numpy()

    # Keras execution - Pass the training flag here
    sin_keras, cos_keras = keras_model(H=H, W=W, training=training)
    sin_keras_np = np.array(sin_keras)
    cos_keras_np = np.array(cos_keras)

    # Compare shapes and values
    assert sin_keras_np.shape == sin_torch_np.shape
    assert cos_keras_np.shape == cos_torch_np.shape
    np.testing.assert_allclose(sin_keras_np, sin_torch_np, atol=atol)
    np.testing.assert_allclose(cos_keras_np, cos_torch_np, atol=atol)


def test_base_parametrization(params):
    """Tests equivalence when using the `base` parameter."""
    torch.manual_seed(0)
    torch_model = PT_RopePositionEmbedding(
        embed_dim=params["embed_dim"],
        num_heads=params["num_heads"],
        base=100.0,
        dtype=torch.float32,
    )
    keras_model = RopePositionEmbedding(
        embed_dim=params["embed_dim"],
        num_heads=params["num_heads"],
        base=100.0,
        dtype="float32",
    )
    run_and_compare(keras_model, torch_model, params["H"], params["W"], training=False)


def test_period_parametrization(params):
    """Tests equivalence when using `min_period` and `max_period`."""
    torch.manual_seed(0)
    torch_model = PT_RopePositionEmbedding(
        embed_dim=params["embed_dim"],
        num_heads=params["num_heads"],
        base=None,
        min_period=1.0,
        max_period=100.0,
        dtype=torch.float32,
    )
    keras_model = RopePositionEmbedding(
        embed_dim=params["embed_dim"],
        num_heads=params["num_heads"],
        base=None,
        min_period=1.0,
        max_period=100.0,
        dtype="float32",
    )
    run_and_compare(keras_model, torch_model, params["H"], params["W"], training=False)


@pytest.mark.parametrize("normalize", ["max", "min", "separate"])
def test_normalize_coords(params, normalize):
    """Tests all `normalize_coords` options."""
    torch.manual_seed(0)
    torch_model = PT_RopePositionEmbedding(
        embed_dim=params["embed_dim"],
        num_heads=params["num_heads"],
        normalize_coords=normalize,
        dtype=torch.float32,
    )
    keras_model = RopePositionEmbedding(
        embed_dim=params["embed_dim"],
        num_heads=params["num_heads"],
        normalize_coords=normalize,
        dtype="float32",
    )
    run_and_compare(keras_model, torch_model, params["H"], params["W"], training=False)


def test_training_augmentations_are_different(params):
    """
    Tests that training mode produces different (randomized) outputs
    compared to eval mode.
    """
    torch.manual_seed(42)
    init_args = {
        "embed_dim": params["embed_dim"],
        "num_heads": params["num_heads"],
        "shift_coords": 0.1,
        "jitter_coords": 2.0,
        "rescale_coords": 2.0,
        "dtype": "float32",
    }
    torch_init_args = init_args.copy()
    torch_init_args["dtype"] = torch.float32
    keras_model = RopePositionEmbedding(**init_args)
    torch_model = PT_RopePositionEmbedding(**torch_init_args)
    sin_eval_keras, cos_eval_keras = keras_model(
        H=params["H"], W=params["W"], training=False
    )
    sin_train_keras, cos_train_keras = keras_model(
        H=params["H"], W=params["W"], training=True
    )
    assert not np.allclose(np.array(sin_eval_keras), np.array(sin_train_keras))
    assert not np.allclose(np.array(cos_eval_keras), np.array(cos_train_keras))
    torch_model.eval()
    sin_eval_torch, cos_eval_torch = torch_model(H=params["H"], W=params["W"])
    torch_model.train()
    sin_train_torch, cos_train_torch = torch_model(H=params["H"], W=params["W"])
    assert not np.allclose(sin_eval_torch.numpy(), sin_train_torch.numpy())
    assert not np.allclose(cos_eval_torch.numpy(), cos_train_torch.numpy())
    assert sin_train_keras.shape == sin_eval_keras.shape
    assert cos_train_keras.shape == cos_eval_keras.shape


DINO_REPO_PATH = ""  # Not needed
DINO_WEIGHT_PATH = "/path/that/does/not/exist/dinov3_vits16_pretrain.pth"
dinov3_files_exist = os.path.isfile(DINO_WEIGHT_PATH)


@pytest.mark.skipif(
    not dinov3_files_exist, reason="DINOv3 local model/weight files not found."
)
def test_dinov3_real_pretrained_weights(params):
    dinov3_model = torch.hub.load(
        DINO_REPO_PATH, "dinov3_vits16", source="local", weights=DINO_WEIGHT_PATH
    )
    dinov3_model.eval()
    torch_pretrained_rope = dinov3_model.rope_embed
    dino_embed_dim = dinov3_model.embed_dim
    dino_num_heads = dinov3_model.blocks[0].attn.num_heads
    keras_model = RopePositionEmbedding(
        embed_dim=dino_embed_dim,
        num_heads=dino_num_heads,
        base=torch_pretrained_rope.base,
        min_period=torch_pretrained_rope.min_period,
        max_period=torch_pretrained_rope.max_period,
        normalize_coords=torch_pretrained_rope.normalize_coords,
        shift_coords=torch_pretrained_rope.shift_coords,
        jitter_coords=torch_pretrained_rope.jitter_coords,
        rescale_coords=torch_pretrained_rope.rescale_coords,
        dtype="float32",
    )
    keras_model.build(input_shape=None)
    torch_periods_np = torch_pretrained_rope.periods.cpu().numpy()
    keras_model.set_weights([torch_periods_np])
    run_and_compare(
        keras_model,
        torch_pretrained_rope,
        H=params["H"],
        W=params["W"],
        training=False,
        atol=1e-5,
    )
