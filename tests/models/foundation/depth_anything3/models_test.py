import numpy as np
import jax

from paz.models.foundation.depth_anything3 import build_da3_small_backbone
from paz.models.foundation.depth_anything3.models import DepthAnything3Small
from paz.models.foundation.depth_anything3.models import DepthAnything3Base
from paz.models.foundation.depth_anything3.models import DepthAnything3MonoLarge
from paz.models.foundation.depth_anything3.models import grid_shape
from paz.models.foundation.depth_anything3 import port_weights

IMAGE_SHAPE = (70, 70, 3)
VIEWS = 2
HIDDEN = 384
DEPTH = 12


def make_input(batch=1):
    shape = (batch, VIEWS, *IMAGE_SHAPE)
    return np.random.RandomState(0).randn(*shape).astype("float32")


def test_backbone_returns_four_features_and_four_camera_tokens():
    model = build_da3_small_backbone(VIEWS, IMAGE_SHAPE)
    outputs = model(make_input())
    assert not isinstance(outputs, dict)
    assert not hasattr(outputs, "_fields")
    assert len(outputs) == 8


def test_backbone_feature_and_camera_shapes():
    model = build_da3_small_backbone(VIEWS, IMAGE_SHAPE)
    outputs = model(make_input())
    grid = grid_shape(IMAGE_SHAPE, 14)
    num_patches = grid[0] * grid[1]
    for feature_map in outputs[:4]:
        assert tuple(feature_map.shape) == (1, VIEWS, num_patches, 2 * HIDDEN)
    for camera_token in outputs[4:]:
        assert tuple(camera_token.shape) == (1, VIEWS, 2 * HIDDEN)


def test_backbone_jit_matches_eager():
    model = build_da3_small_backbone(VIEWS, IMAGE_SHAPE)
    data = make_input()
    eager = np.array(model(data)[0])
    jitted = np.array(jax.jit(lambda x: model(x))(data)[0])
    assert np.allclose(eager, jitted, atol=1e-5)


def test_da3_small_output_order_and_shapes():
    model = DepthAnything3Small(VIEWS, IMAGE_SHAPE)
    outputs = model(make_input())
    assert not isinstance(outputs, dict)
    assert not hasattr(outputs, "_fields")
    assert len(outputs) == 6
    depth, depth_conf, extrinsics, intrinsics, rays, ray_conf = outputs
    grid = grid_shape(IMAGE_SHAPE, 14)
    ray_size = (grid[0] * 8, grid[1] * 8)
    assert tuple(depth.shape) == (1, VIEWS, 70, 70)
    assert tuple(depth_conf.shape) == (1, VIEWS, 70, 70)
    assert tuple(extrinsics.shape) == (1, VIEWS, 3, 4)
    assert tuple(intrinsics.shape) == (1, VIEWS, 3, 3)
    assert tuple(rays.shape) == (1, VIEWS, ray_size[0], ray_size[1], 6)
    assert tuple(ray_conf.shape) == (1, VIEWS, ray_size[0], ray_size[1])


def test_da3_base_output_order_and_shapes():
    model = DepthAnything3Base(VIEWS, IMAGE_SHAPE)
    outputs = model(make_input())
    assert len(outputs) == 6
    depth, depth_conf, extrinsics, intrinsics, rays, ray_conf = outputs
    assert tuple(depth.shape) == (1, VIEWS, 70, 70)
    assert tuple(extrinsics.shape) == (1, VIEWS, 3, 4)
    assert tuple(rays.shape) == (1, VIEWS, 40, 40, 6)


def test_mono_large_returns_depth_and_sky():
    model = DepthAnything3MonoLarge(IMAGE_SHAPE)
    image = np.random.RandomState(0).randn(1, *IMAGE_SHAPE).astype("float32")
    outputs = model(image)
    assert not isinstance(outputs, dict)
    assert len(outputs) == 2
    depth, sky = outputs
    assert tuple(depth.shape) == (1, 70, 70)
    assert tuple(sky.shape) == (1, 70, 70)


def test_da3_small_jit_runs():
    model = DepthAnything3Small(VIEWS, IMAGE_SHAPE)
    data = make_input()
    eager = np.array(model(data)[0])
    jitted = np.array(jax.jit(lambda x: model(x))(data)[0])
    assert np.allclose(eager, jitted, atol=1e-5)


def build_backbone_state_dict(model):
    state = {}
    for layer in model.layers:
        add_parameters(state, layer)
    return state


def add_parameters(state, layer):
    name = layer.name
    weights = layer.get_weights()
    prefix = "model.backbone.pretrained."
    if name == "patch_embed_proj":
        kernel = np.transpose(weights[0], (3, 2, 0, 1))
        state[prefix + "patch_embed.proj.weight"] = kernel
        state[prefix + "patch_embed.proj.bias"] = weights[1]
    elif name == "cls_token":
        state[prefix + "cls_token"] = weights[0].reshape(1, 1, -1)
    elif name == "pos_embed":
        state[prefix + "pos_embed"] = weights[0].reshape(1, -1, HIDDEN)
    elif name == "camera_token":
        state[prefix + "camera_token"] = weights[0].reshape(1, 2, -1)
    elif name == "norm":
        state[prefix + "norm.weight"], state[prefix + "norm.bias"] = weights
    elif name.startswith("block_"):
        add_block_parameters(state, name, weights, prefix)


def add_block_parameters(state, name, weights, prefix):
    index = name.split("_")[1]
    suffix = name[len(f"block_{index}_"):]
    source = f"{prefix}blocks.{index}."
    linear = {"qkv": "attn.qkv", "proj": "attn.proj", "mlp_fc1": "mlp.fc1",
              "mlp_fc2": "mlp.fc2"}
    norms = {"norm1": "norm1", "norm2": "norm2", "q_norm": "attn.q_norm",
             "k_norm": "attn.k_norm"}
    if suffix in norms:
        state[source + norms[suffix] + ".weight"] = weights[0]
        state[source + norms[suffix] + ".bias"] = weights[1]
    elif suffix in linear:
        state[source + linear[suffix] + ".weight"] = weights[0].T
        state[source + linear[suffix] + ".bias"] = weights[1]
    elif suffix in ("ls1", "ls2"):
        state[source + suffix + ".gamma"] = weights[0]


def test_converter_reproduces_source_backbone():
    source = build_da3_small_backbone(VIEWS, IMAGE_SHAPE)
    state = build_backbone_state_dict(source)
    target = build_da3_small_backbone(VIEWS, IMAGE_SHAPE)
    num_positions = grid_shape(IMAGE_SHAPE, 14)
    num_positions = num_positions[0] * num_positions[1] + 1
    args = target, state, DEPTH, num_positions, HIDDEN
    port_weights.port_backbone_weights(*args)
    data = make_input()
    expected = np.array(source(data)[0])
    assert np.allclose(np.array(target(data)[0]), expected, atol=1e-5)
