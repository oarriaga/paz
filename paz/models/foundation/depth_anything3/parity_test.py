"""Optional bit-parity check against the official DA3 reference.

Runs only when DA3_REFERENCE is set and the reference package and Apache-2.0
checkpoints are importable. It ports the real checkpoints into the Keras
models and compares depth, cameras, and rays.
"""
import os
import glob
import json

import numpy as np
import pytest

REFERENCE = os.environ.get("DA3_REFERENCE")


def load_reference_backbone(model_name="DA3-SMALL"):
    from depth_anything_3.cfg import import_item
    from safetensors.torch import load_file
    base = os.path.expanduser("~/.cache/huggingface/hub")
    pattern = f"{base}/models--depth-anything--{model_name}/snapshots/*"
    snapshots = glob.glob(pattern)
    if not snapshots:
        pytest.skip(f"{model_name} checkpoint not cached")
    config_path = os.path.join(snapshots[0], "config.json")
    config = json.load(open(config_path))["config"]
    net = build_object(config, import_item).eval()
    weights = load_file(os.path.join(snapshots[0], "model.safetensors"))
    net.load_state_dict({k[6:]: v for k, v in weights.items()}, strict=False)
    return net, {k: v.numpy() for k, v in weights.items()}


def build_object(node, import_item):
    if isinstance(node, dict) and "__object__" in node:
        specification = node["__object__"]
        item = import_item(specification["path"], specification["name"])
        arguments = {key: build_object(value, import_item)
                     for key, value in node.items() if key != "__object__"}
        return item(**arguments)
    return node


@pytest.mark.skipif(not REFERENCE, reason="set DA3_REFERENCE to run")
def test_backbone_matches_reference():
    import torch
    from paz.models.foundation.depth_anything3 import build_da3_small_backbone
    from paz.models.foundation.depth_anything3 import port_weights
    net, state = load_reference_backbone()
    views, size = 2, 518
    shape = 1, views, 3, size, size
    image = np.random.RandomState(0).randn(*shape).astype("float32")
    with torch.no_grad():
        features = net.backbone(torch.from_numpy(image))[0]
    reference_features = [tensor[0].numpy() for tensor in features]
    reference_cameras = [tensor[1].numpy() for tensor in features]

    model = build_da3_small_backbone(views, (size, size, 3))
    port_weights.port_backbone_weights(model, state, 12, 1370, 384)
    outputs = model(np.transpose(image, (0, 1, 3, 4, 2)))
    for index in range(4):
        assert np.allclose(np.array(outputs[index]),
                           reference_features[index], atol=2e-3)
        assert np.allclose(np.array(outputs[4 + index]),
                           reference_cameras[index], atol=1e-3)


@pytest.mark.skipif(not REFERENCE, reason="set DA3_REFERENCE to run")
def test_da3_small_matches_reference():
    from paz.models import DepthAnything3Small
    check_da3_reference("DA3-SMALL", DepthAnything3Small, 384)


@pytest.mark.skipif(not REFERENCE, reason="set DA3_REFERENCE to run")
def test_da3_base_matches_reference():
    from paz.models import DepthAnything3Base
    check_da3_reference("DA3-BASE", DepthAnything3Base, 768)


@pytest.mark.skipif(not REFERENCE, reason="set DA3_REFERENCE to run")
def test_da3_mono_large_matches_reference():
    import torch
    from paz.models import DepthAnything3MonoLarge
    from paz.models.foundation.depth_anything3 import port_weights
    net, state = load_reference_backbone("DA3MONO-LARGE")
    height, width = 154, 154
    shape = 1, 1, 3, height, width
    image = np.random.RandomState(8).randn(*shape).astype("float32")
    with torch.no_grad():
        feats, _ = net.backbone(torch.from_numpy(image))
        head = net.head(feats, height, width, patch_start_idx=0)
    model = DepthAnything3MonoLarge((height, width, 3))
    positions = (height // 14) ** 2 + 1
    args = model, state, 24, positions, 1024
    port_weights.port_backbone_weights(*args, use_camera=False,
                                       use_qk_norm=False)
    port_weights.port_dpt_head_weights(model, state)
    depth, sky = model(np.transpose(image[:, 0], (0, 2, 3, 1)))
    assert np.allclose(np.array(depth), head["depth"].numpy(), atol=1e-3)
    assert np.allclose(np.array(sky), head["sky"].numpy(), atol=1e-3)


def check_da3_reference(model_name, builder, hidden_size):
    import torch
    from depth_anything_3.model.utils import transform
    from depth_anything_3.utils.geometry import affine_inverse
    from paz.models.foundation.depth_anything3 import port_weights
    net, state = load_reference_backbone(model_name)
    views, height, width = 2, 154, 154
    shape = 1, views, 3, height, width
    image = np.random.RandomState(5).randn(*shape).astype("float32")
    with torch.no_grad():
        feats, _ = net.backbone(torch.from_numpy(image))
        head = net.head(feats, height, width, patch_start_idx=0)
        pose = net.cam_dec(feats[-1][1])
        c2w, intrinsics = transform.pose_encoding_to_extri_intri(
            pose, (height, width))
        extrinsics = affine_inverse(c2w)
    model = builder(views, (height, width, 3))
    positions = (height // 14) * (width // 14) + 1
    port_weights.port_backbone_weights(model, state, 12, positions, hidden_size)
    port_weights.port_head_weights(model, state)
    port_weights.port_camera_decoder_weights(model, state)
    outputs = model(np.transpose(image, (0, 1, 3, 4, 2)))
    assert np.allclose(np.array(outputs[0]), head["depth"].numpy(), atol=1e-3)
    assert np.allclose(np.array(outputs[2]), extrinsics.numpy(), atol=1e-3)
    assert np.allclose(np.array(outputs[3]), intrinsics.numpy(), atol=1e-1)
    assert np.allclose(np.array(outputs[4]), head["ray"].numpy(), atol=1e-3)
