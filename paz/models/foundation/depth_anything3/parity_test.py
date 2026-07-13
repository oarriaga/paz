"""Optional bit-parity check against the official DA3-SMALL reference.

Runs only when DA3_REFERENCE is set and the reference package and Apache-2.0
checkpoint are importable. It ports the real checkpoint into the Keras
backbone and compares every collected feature and camera token.
"""
import os
import glob
import json

import numpy as np
import pytest

REFERENCE = os.environ.get("DA3_REFERENCE")


def load_reference_backbone():
    from depth_anything_3.cfg import import_item
    from safetensors.torch import load_file
    pattern = os.path.expanduser("~/.cache/huggingface/hub/"
                                 "models--depth-anything--DA3-SMALL/snapshots/*")
    snapshots = glob.glob(pattern)
    if not snapshots:
        pytest.skip("DA3-SMALL checkpoint not cached")
    config = json.load(open(os.path.join(snapshots[0], "config.json")))["config"]
    net = build_object(config, import_item).eval()
    state = load_file(os.path.join(snapshots[0], "model.safetensors"))
    net.load_state_dict({k[6:]: v for k, v in state.items()}, strict=False)
    return net, {k: v.numpy() for k, v in state.items()}


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
    from paz.models.foundation.depth_anything3.models import build_da3_small_backbone
    from paz.models.foundation.depth_anything3.port_weights import port_backbone_weights
    net, state = load_reference_backbone()
    views, size = 2, 518
    image = np.random.RandomState(0).randn(1, views, 3, size, size).astype("float32")
    with torch.no_grad():
        features = net.backbone(torch.from_numpy(image))[0]
    reference_features = [tensor[0].numpy() for tensor in features]
    reference_cameras = [tensor[1].numpy() for tensor in features]

    model = build_da3_small_backbone(views, (size, size, 3))
    port_backbone_weights(model, state, 12, 1370, 384)
    outputs = model(np.transpose(image, (0, 1, 3, 4, 2)))
    for index in range(4):
        assert np.allclose(np.array(outputs[index]),
                           reference_features[index], atol=2e-3)
        assert np.allclose(np.array(outputs[4 + index]),
                           reference_cameras[index], atol=1e-3)


@pytest.mark.skipif(not REFERENCE, reason="set DA3_REFERENCE to run")
def test_da3_small_matches_reference():
    import torch
    from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
    from depth_anything_3.utils.geometry import affine_inverse
    from paz.models.foundation.depth_anything3.models import build_da3_small
    from paz.models.foundation.depth_anything3 import port_weights
    net, state = load_reference_backbone()
    views, height, width = 2, 154, 154
    image = np.random.RandomState(5).randn(1, views, 3, height, width).astype("float32")
    tensor = torch.from_numpy(image)
    with torch.no_grad():
        feats, _ = net.backbone(tensor)
        head = net.head(feats, height, width, patch_start_idx=0)
        camera_to_world, intrinsics = pose_encoding_to_extri_intri(
            net.cam_dec(feats[-1][1]), (height, width))
        extrinsics = affine_inverse(camera_to_world)

    model = build_da3_small(views, (height, width, 3))
    positions = (height // 14) * (width // 14) + 1
    port_weights.port_backbone_weights(model, state, 12, positions, 384)
    port_weights.port_head_weights(model, state)
    port_weights.port_camera_decoder_weights(model, state)
    outputs = model(np.transpose(image, (0, 1, 3, 4, 2)))
    assert np.allclose(np.array(outputs[0]), head["depth"].numpy(), atol=1e-3)
    assert np.allclose(np.array(outputs[2]), extrinsics.numpy(), atol=1e-3)
    assert np.allclose(np.array(outputs[3]), intrinsics.numpy(), atol=1e-1)
    assert np.allclose(np.array(outputs[4]), head["ray"].numpy(), atol=1e-3)
