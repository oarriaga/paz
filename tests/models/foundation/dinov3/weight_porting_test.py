import os
import pytest

torch = pytest.importorskip("torch")

import numpy as np
import keras
from paz.models.foundation.dinov3.models.vision_transformer import vit_small
from torch_vision_transformer_for_testing import (
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


def test_weight_count_matches(pytorch_model_and_weights, keras_model):
    """Tests that the number of trainable parameters matches between PyTorch and Keras."""
    pt_model, _ = pytorch_model_and_weights

    pt_params = sum(p.numel() for p in pt_model.parameters() if p.requires_grad)
    keras_params = sum(np.prod(w.shape) for w in keras_model.trainable_weights)

    tolerance = 0.01
    assert abs(pt_params - keras_params) / pt_params < tolerance, \
        f"Parameter count mismatch: PyTorch={pt_params}, Keras={keras_params}"
