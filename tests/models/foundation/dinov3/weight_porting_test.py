import numpy as np
import pytest

torch = pytest.importorskip("torch")

from paz.models.foundation.dinov3 import port_dino_weights_from_torch_to_keras
from paz.models.foundation.dinov3.models.vision_transformer import vit_small
from torch_vision_transformer_for_testing import PT_vit_small

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


def build_random_reference():
    torch.manual_seed(0)
    reference = PT_vit_small(**MODEL_KWARGS)
    reference.init_weights()
    # The reference leaves the layer scales uninitialized; values near one
    # make every block contribute to the compared output.
    for name, parameter in reference.named_parameters():
        if name.endswith("gamma"):
            torch.nn.init.uniform_(parameter, 0.5, 1.0)
    return reference.eval()


def build_model():
    model = vit_small(**MODEL_KWARGS)
    model(np.zeros((1, 224, 224, 3), "float32"), training=False)
    return model


def port_weights(state_dict, model):
    porting = port_dino_weights_from_torch_to_keras
    return porting.transfer_weights_from_pt_to_keras(state_dict, model)


def assert_outputs_match(reference, model):
    shape = (1, 224, 224, 3)
    image = np.random.RandomState(42).randn(*shape).astype("float32")
    with torch.no_grad():
        expected = reference(torch.from_numpy(image.transpose(0, 3, 1, 2)))
    output = np.array(model(image, training=False))
    np.testing.assert_allclose(output, expected.numpy(), rtol=1e-5, atol=1e-5)


def test_ported_random_weights_match_reference():
    reference = build_random_reference()
    model = port_weights(reference.state_dict(), build_model())
    assert_outputs_match(reference, model)


def test_ported_pretrained_weights_match_reference(torch_weights_path):
    filename = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    path = torch_weights_path(filename)
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    reference = PT_vit_small(**MODEL_KWARGS)
    reference.load_state_dict(state_dict, strict=False)
    model = port_weights(state_dict, build_model())
    assert_outputs_match(reference.eval(), model)


def test_trainable_parameter_count_matches_reference():
    reference = PT_vit_small(**MODEL_KWARGS)
    parameters = [p for p in reference.parameters() if p.requires_grad]
    expected = sum(parameter.numel() for parameter in parameters)
    weights = build_model().trainable_weights
    assert sum(int(np.prod(weight.shape)) for weight in weights) == expected
