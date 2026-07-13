import os

import numpy as np
import pytest
from keras import Model

from paz.models import DINOv2Small
from paz.models.foundation.dinov2.models import count_positions
from paz.models.foundation.dinov2.port_weights import port_weights
from paz.models.foundation.dinov2_legacy.models.vision_transformer import (
    DINOV2Small as LegacyDINOv2Small,
)

IMAGE_SHAPE = (98, 98, 3)
HIDDEN_SIZE = 384
DEPTH = 12


def copy_weights_by_name(source, target):
    for layer in target.layers:
        weights = layer.get_weights()
        if weights:
            layer.set_weights(source.get_layer(layer.name).get_weights())


def test_matches_legacy_dinov2():
    canonical = DINOv2Small(image_shape=IMAGE_SHAPE)
    legacy = LegacyDINOv2Small(img_size=IMAGE_SHAPE[0])
    copy_weights_by_name(legacy, canonical)
    legacy_norm = Model(legacy.input, legacy.get_layer("norm").output)
    data = np.random.RandomState(0).randn(2, *IMAGE_SHAPE).astype("float32")
    class_token, patch_tokens = canonical(data)
    normalized = np.array(legacy_norm(data))
    assert np.allclose(np.array(class_token), normalized[:, 0], atol=1e-5)
    assert np.allclose(np.array(patch_tokens), normalized[:, 1:], atol=1e-5)


def build_torch_state_dict(model):
    state_dict = {}
    for layer in model.layers:
        add_torch_parameters(state_dict, layer)
    return state_dict


def add_torch_parameters(state_dict, layer):
    name = layer.name
    weights = layer.get_weights()
    if name == "patch_embed_proj":
        state_dict["patch_embed.proj.weight"] = np.transpose(weights[0], (3, 2, 0, 1))  # noqa: E501
        state_dict["patch_embed.proj.bias"] = weights[1]
    elif name == "cls_token":
        state_dict["cls_token"] = weights[0].reshape(1, 1, -1)
    elif name == "pos_embed":
        state_dict["pos_embed"] = weights[0].reshape(1, -1, HIDDEN_SIZE)
    elif name == "norm":
        state_dict["norm.weight"], state_dict["norm.bias"] = weights
    elif name.startswith("block_"):
        add_block_parameters(state_dict, name, weights)


def add_block_parameters(state_dict, name, weights):
    index = name.split("_")[1]
    kinds = {"norm1": "norm1", "norm2": "norm2", "qkv": "attn.qkv",
             "proj": "attn.proj", "mlp_fc1": "mlp.fc1", "mlp_fc2": "mlp.fc2"}
    suffix = name[len(f"block_{index}_"):]
    if suffix in ("norm1", "norm2"):
        gamma, beta = weights
        state_dict[f"blocks.{index}.{kinds[suffix]}.weight"] = gamma
        state_dict[f"blocks.{index}.{kinds[suffix]}.bias"] = beta
    elif suffix in ("qkv", "proj", "mlp_fc1", "mlp_fc2"):
        kernel, bias = weights
        state_dict[f"blocks.{index}.{kinds[suffix]}.weight"] = kernel.T
        state_dict[f"blocks.{index}.{kinds[suffix]}.bias"] = bias
    elif suffix in ("ls1", "ls2"):
        state_dict[f"blocks.{index}.{suffix}.gamma"] = weights[0]


def test_converter_reproduces_source_model(tmp_path):
    source = DINOv2Small(image_shape=IMAGE_SHAPE)
    state_dict = build_torch_state_dict(source)
    positions = count_positions(IMAGE_SHAPE, 14)
    target = DINOv2Small(image_shape=IMAGE_SHAPE)
    port_weights(target, state_dict, positions, HIDDEN_SIZE, DEPTH)
    path = os.path.join(tmp_path, "ported.weights.h5")
    target.save_weights(path)
    reloaded = DINOv2Small(image_shape=IMAGE_SHAPE)
    reloaded.load_weights(path)
    data = np.random.RandomState(1).randn(1, *IMAGE_SHAPE).astype("float32")
    expected = np.array(source(data)[0])
    assert np.allclose(np.array(reloaded(data)[0]), expected, atol=1e-5)


def test_converter_rejects_missing_key():
    source = DINOv2Small(image_shape=IMAGE_SHAPE)
    state_dict = build_torch_state_dict(source)
    del state_dict["norm.weight"]
    positions = count_positions(IMAGE_SHAPE, 14)
    target = DINOv2Small(image_shape=IMAGE_SHAPE)
    with pytest.raises(KeyError):
        port_weights(target, state_dict, positions, HIDDEN_SIZE, DEPTH)


def test_converter_rejects_unexpected_key():
    source = DINOv2Small(image_shape=IMAGE_SHAPE)
    state_dict = build_torch_state_dict(source)
    state_dict["blocks.0.attn.extra"] = np.zeros((3,), "float32")
    positions = count_positions(IMAGE_SHAPE, 14)
    target = DINOv2Small(image_shape=IMAGE_SHAPE)
    with pytest.raises(KeyError):
        port_weights(target, state_dict, positions, HIDDEN_SIZE, DEPTH)


@pytest.mark.skipif(not os.environ.get("DINOV2_REFERENCE"),
                    reason="set DINOV2_REFERENCE to run torch-hub parity")
def test_reference_torch_parity():
    import torch
    reference = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                               verbose=False).eval()
    state_dict = {key: value.detach().cpu().numpy()
                  for key, value in reference.state_dict().items()}
    model = DINOv2Small(image_shape=(518, 518, 3))
    positions = count_positions((518, 518, 3), 14)
    port_weights(model, state_dict, positions, HIDDEN_SIZE, DEPTH)
    data = np.random.RandomState(0).randn(1, 518, 518, 3).astype("float32")
    tensor = torch.from_numpy(data).permute(0, 3, 1, 2)
    with torch.no_grad():
        expected = reference.forward_features(tensor)["x_norm_clstoken"].numpy()
    class_token = np.array(model(data)[0])
    assert np.allclose(class_token, expected, atol=1e-3)
