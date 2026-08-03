import os
import sys

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest
import torch

from keras import Input, Model, ops

rfdetr_parent = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../../../../",
    "examples", "rf-detr_original_pytorch_implementation",
))
if rfdetr_parent not in sys.path:
    sys.path.insert(0, rfdetr_parent)

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../../")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from paz.models.foundation.dinov2.models.windowed_vision_transformer import (
    WindowedDinov2PatchEmbeddings,
    WindowedDinov2Layer,
    WindowedDinov2Encoder,
    WindowedDinov2Model,
    dinov2_windowed_small,
    dinov2_windowed_base,
    dinov2_windowed_large,
    dinov2_windowed_giant,
    interpolate_pos_encoding,
    EmbedArgs,
)
from paz.models.detection.dino_v2_object_detection.models.backbone.dinov2 import (  # fmt: skip
    DinoV2,
)

from rfdetr.models.backbone.dinov2_with_windowed_attn import (
    WindowedDinov2WithRegistersConfig,
    WindowedDinov2WithRegistersEmbeddings as PtEmbeddings,
    WindowedDinov2WithRegistersLayer as PtLayer,
    WindowedDinov2WithRegistersEncoder as PtEncoder,
)

from paz.models.detection.dino_v2_object_detection.models.backbone.backbone_weights_porting_utils import (  # noqa: E402  # fmt: skip
    assert_close,
    hwc_to_chw,
    transfer_patch_embeddings,
    transfer_layer,
    transfer_encoder,
)


def build_patch_model(image_size=56, patch_size=14, hidden_size=64, num_register_tokens=0, num_windows=1):  # fmt: skip
    pixels = Input((image_size, image_size, 3), name="pixels")
    tokens = WindowedDinov2PatchEmbeddings(
        pixels, image_size, patch_size, hidden_size,
        num_register_tokens, num_windows,
    )
    return Model(pixels, tokens, name="patch")


def build_layer_model(tokens_shape, hidden_size=64, num_attention_heads=4, num_windows=1, init_values=1.0, run_full_attention=False):  # fmt: skip
    x = Input(tokens_shape, name="tokens")
    y = WindowedDinov2Layer(
        x, hidden_size, num_attention_heads, mlp_ratio=4.0,
        num_windows=num_windows, init_values=init_values,
        run_full_attention=run_full_attention,
    )
    return Model(x, y, name="layer")


def build_encoder_model(tokens_shape, num_hidden_layers, num_windows=1, window_block_indexes=None):  # fmt: skip
    x = Input(tokens_shape, name="tokens")
    blocks = WindowedDinov2Encoder(
        x, hidden_size=64, num_hidden_layers=num_hidden_layers,
        num_attention_heads=4, mlp_ratio=4.0, num_windows=num_windows,
        window_block_indexes=window_block_indexes, init_values=1.0,
    )
    return Model(x, blocks, name="encoder")


def make_pt_config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, mlp_ratio=4, image_size=56, patch_size=14, num_register_tokens=0, num_windows=1, window_block_indexes=None, use_swiglu_ffn=False, layerscale_value=1.0, drop_path_rate=0.0):  # fmt: skip
    if window_block_indexes is None:
        window_block_indexes = list(range(num_hidden_layers))
    cfg = WindowedDinov2WithRegistersConfig(
        hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads, mlp_ratio=mlp_ratio,
        image_size=image_size, patch_size=patch_size,
        num_register_tokens=num_register_tokens, num_windows=num_windows,
        window_block_indexes=window_block_indexes,
        use_swiglu_ffn=use_swiglu_ffn,
        layerscale_value=layerscale_value, drop_path_rate=drop_path_rate,
        hidden_act="gelu", out_features=[f"stage{num_hidden_layers}"],
        out_indices=[num_hidden_layers],
    )
    cfg._attn_implementation = "eager"
    return cfg


def test_patch_embeddings_output_shape():
    model = build_patch_model(hidden_size=64, num_windows=1)
    x = np.random.randn(2, 56, 56, 3).astype(np.float32)
    out = model(x, training=False)
    assert ops.shape(out) == (2, 17, 64)


def test_patch_embeddings_parity():
    cfg = make_pt_config(hidden_size=64, image_size=56, patch_size=14)
    pt = PtEmbeddings(cfg).eval()
    model = build_patch_model(hidden_size=64, num_windows=1)
    transfer_patch_embeddings(pt, model, "embeddings")

    x_np = np.random.randn(1, 56, 56, 3).astype(np.float32)
    x_pt = torch.from_numpy(hwc_to_chw(x_np))
    with torch.no_grad():
        pt_out = pt(x_pt)
    assert_close(pt_out, model(x_np, training=False))


def test_register_tokens_insertion():
    kwargs = dict(hidden_size=64, num_register_tokens=4, num_windows=1)
    model = build_patch_model(**kwargs)
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    out = model(x, training=False)
    assert ops.shape(out)[1] == 21


def test_no_register_tokens():
    kwargs = dict(hidden_size=64, num_register_tokens=0, num_windows=1)
    model = build_patch_model(**kwargs)
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    out = model(x, training=False)
    assert ops.shape(out)[1] == 17


def test_register_tokens_parity():
    cfg = make_pt_config(
        hidden_size=64, image_size=56, patch_size=14, num_register_tokens=4
    )
    pt = PtEmbeddings(cfg).eval()
    kwargs = dict(hidden_size=64, num_register_tokens=4, num_windows=1)
    model = build_patch_model(**kwargs)
    transfer_patch_embeddings(pt, model, "embeddings")

    x_np = np.random.randn(1, 56, 56, 3).astype(np.float32)
    x_pt = torch.from_numpy(hwc_to_chw(x_np))
    with torch.no_grad():
        pt_out = pt(x_pt)
    assert_close(pt_out, model(x_np, training=False))


def test_interpolate_pos_encoding_same_size():
    embed = EmbedArgs(14, 64, 0, 1)
    table = Input((17, 64), name="table")
    tokens = Input((17, 64), name="tokens")
    result = interpolate_pos_encoding(table, tokens, embed, 56, 56)
    assert tuple(result.shape) == (None, 17, 64)


def test_interpolate_pos_encoding_different_size():
    embed = EmbedArgs(14, 64, 0, 1)
    table = Input((17, 64), name="table")
    tokens = Input((65, 64), name="tokens")
    result = interpolate_pos_encoding(table, tokens, embed, 112, 112)
    assert tuple(result.shape) == (None, 65, 64)


def test_windowed_embeddings_shape():
    kwargs = dict(hidden_size=64, num_windows=2, num_register_tokens=0)
    model = build_patch_model(**kwargs)
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    out = model(x, training=False)
    assert ops.shape(out) == (4, 5, 64)


def test_windowed_embeddings_parity():
    cfg = make_pt_config(
        hidden_size=64, image_size=56, patch_size=14,
        num_windows=2, num_register_tokens=0,
    )
    pt = PtEmbeddings(cfg).eval()
    kwargs = dict(hidden_size=64, num_windows=2, num_register_tokens=0)
    model = build_patch_model(**kwargs)
    transfer_patch_embeddings(pt, model, "embeddings")

    x_np = np.random.randn(1, 56, 56, 3).astype(np.float32)
    x_pt = torch.from_numpy(hwc_to_chw(x_np))
    with torch.no_grad():
        pt_out = pt(x_pt)
    assert_close(pt_out, model(x_np, training=False))


def test_windowed_with_registers_parity():
    cfg = make_pt_config(
        hidden_size=64, image_size=56, patch_size=14,
        num_windows=2, num_register_tokens=4,
    )
    pt = PtEmbeddings(cfg).eval()
    kwargs = dict(hidden_size=64, num_windows=2, num_register_tokens=4)
    model = build_patch_model(**kwargs)
    transfer_patch_embeddings(pt, model, "embeddings")

    x_np = np.random.randn(1, 56, 56, 3).astype(np.float32)
    x_pt = torch.from_numpy(hwc_to_chw(x_np))
    with torch.no_grad():
        pt_out = pt(x_pt)
    assert_close(pt_out, model(x_np, training=False))


def test_layer_no_windowing_parity():
    cfg = make_pt_config(hidden_size=64, num_attention_heads=4, num_windows=1)
    pt_l = PtLayer(cfg).eval()
    model = build_layer_model((17, 64), num_windows=1)
    transfer_layer(pt_l, model, "layer_0")

    x_np = np.random.randn(1, 17, 64).astype(np.float32)
    with torch.no_grad():
        pt_out = pt_l(torch.from_numpy(x_np), run_full_attention=False)[0]
    assert_close(pt_out, model(x_np, training=False))


def test_layer_with_full_attention_parity():
    cfg = make_pt_config(hidden_size=64, num_attention_heads=4, num_windows=2)
    pt_l = PtLayer(cfg).eval()
    model = build_layer_model((5, 64), num_windows=2, run_full_attention=True)
    transfer_layer(pt_l, model, "layer_0")

    x_np = np.random.randn(4, 5, 64).astype(np.float32)
    with torch.no_grad():
        pt_out = pt_l(torch.from_numpy(x_np), run_full_attention=True)[0]
    assert_close(pt_out, model(x_np, training=False))


def test_layer_windowed_attention_parity():
    cfg = make_pt_config(hidden_size=64, num_attention_heads=4, num_windows=2)
    pt_l = PtLayer(cfg).eval()
    model = build_layer_model((5, 64), num_windows=2, run_full_attention=False)
    transfer_layer(pt_l, model, "layer_0")

    x_np = np.random.randn(4, 5, 64).astype(np.float32)
    with torch.no_grad():
        pt_out = pt_l(torch.from_numpy(x_np), run_full_attention=False)[0]
    assert_close(pt_out, model(x_np, training=False))


def test_encoder_parity():
    cfg = make_pt_config(
        hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_windows=1, window_block_indexes=[0, 1, 2],
    )
    pt_enc = PtEncoder(cfg).eval()
    model = build_encoder_model(
        (17, 64), num_hidden_layers=3, num_windows=1,
        window_block_indexes=[0, 1, 2],
    )
    transfer_encoder(pt_enc, model, "encoder")

    x_np = np.random.randn(1, 17, 64).astype(np.float32)
    with torch.no_grad():
        pt_out = pt_enc(
            torch.from_numpy(x_np), output_hidden_states=True, return_dict=True
        )
    k_out = model(x_np, training=False)
    assert_close(pt_out.last_hidden_state, k_out[-1])
    for pt_h, k_h in zip(pt_out.hidden_states[1:], k_out):
        assert_close(pt_h, k_h)


def test_encoder_mixed_windowing_parity():
    cfg = make_pt_config(
        hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_windows=2, window_block_indexes=[0, 1],
    )
    pt_enc = PtEncoder(cfg).eval()
    model = build_encoder_model(
        (5, 64), num_hidden_layers=3, num_windows=2, window_block_indexes=[0, 1]
    )
    transfer_encoder(pt_enc, model, "encoder")

    x_np = np.random.randn(4, 5, 64).astype(np.float32)
    with torch.no_grad():
        pt_out = pt_enc(
            torch.from_numpy(x_np), output_hidden_states=True, return_dict=True
        )
    assert_close(pt_out.last_hidden_state, model(x_np, training=False)[-1])


def test_model_output_shape():
    model = WindowedDinov2Model(
        image_size=56, patch_size=14, hidden_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_windows=1, num_register_tokens=0,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    seq_out, *all_hidden = model(x, training=False)
    assert ops.shape(seq_out) == (1, 17, 64)
    assert len(all_hidden) == 2


def test_model_with_registers_shape():
    model = WindowedDinov2Model(
        image_size=56, patch_size=14, hidden_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_windows=1, num_register_tokens=4,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    seq_out = model(x, training=False)[0]
    assert ops.shape(seq_out) == (1, 21, 64)


def test_model_windowed_shape():
    model = WindowedDinov2Model(
        image_size=56, patch_size=14, hidden_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_windows=2, num_register_tokens=0,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    seq_out = model(x, training=False)[0]
    assert ops.shape(seq_out) == (4, 5, 64)


def test_builder_dinov2_windowed_small():
    model = dinov2_windowed_small(img_size=56, num_windows=1)
    layer = model.get_layer("encoder_layer_0_attention_qkv")
    assert layer.kernel.shape[0] == 384


def test_builder_dinov2_windowed_base():
    model = dinov2_windowed_base(img_size=56, num_windows=1)
    layer = model.get_layer("encoder_layer_0_attention_qkv")
    assert layer.kernel.shape[0] == 768


def test_builder_dinov2_windowed_large():
    model = dinov2_windowed_large(img_size=56, num_windows=1)
    layer = model.get_layer("encoder_layer_0_attention_qkv")
    assert layer.kernel.shape[0] == 1024


def test_builder_dinov2_windowed_giant():
    model = dinov2_windowed_giant(img_size=56, num_windows=1)
    layer = model.get_layer("encoder_layer_0_attention_qkv")
    assert layer.kernel.shape[0] == 1536


def test_dinov2_wrapper_output_shapes():
    wrapper = DinoV2(
        shape=(56, 56), out_feature_indexes=[0, 1], size="small",
        use_registers=False, patch_size=14, num_windows=1,
        positional_encoding_size=4,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    outputs = wrapper(x, training=False)
    assert len(outputs) == 2
    for out in outputs:
        assert ops.shape(out) == (1, 4, 4, 384)


def test_dinov2_wrapper_with_registers_shapes():
    wrapper = DinoV2(
        shape=(56, 56), out_feature_indexes=[0, 1], size="small",
        use_registers=True, patch_size=14, num_windows=1,
        positional_encoding_size=4,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    outputs = wrapper(x, training=False)
    assert len(outputs) == 2
    for out in outputs:
        assert ops.shape(out) == (1, 4, 4, 384)


def test_dinov2_wrapper_windowed_shapes():
    wrapper = DinoV2(
        shape=(56, 56), out_feature_indexes=[0, 1], size="small",
        use_registers=False, patch_size=14, num_windows=2,
        positional_encoding_size=4,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    outputs = wrapper(x, training=False)
    assert len(outputs) == 2
    for out in outputs:
        assert ops.shape(out) == (1, 4, 4, 384)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(batch_size):
    model = build_patch_model(hidden_size=64, num_windows=1)
    x = np.random.randn(batch_size, 56, 56, 3).astype(np.float32)
    out = model(x, training=False)
    assert ops.shape(out)[0] == batch_size


@pytest.mark.parametrize("batch_size", [1, 2])
def test_different_batch_sizes_windowed(batch_size):
    model = build_patch_model(hidden_size=64, num_windows=2)
    x = np.random.randn(batch_size, 56, 56, 3).astype(np.float32)
    out = model(x, training=False)
    assert ops.shape(out)[0] == batch_size * 4


def test_dinov2_small_config():
    wrapper = DinoV2(
        shape=(56, 56), out_feature_indexes=[0, 1], size="small",
        use_registers=True, patch_size=14, num_windows=1,
        positional_encoding_size=4,
    )
    assert wrapper.hidden_size == 384


def test_dinov2_base_config():
    wrapper = DinoV2(
        shape=(56, 56), out_feature_indexes=[0, 1], size="base",
        use_registers=True, patch_size=14, num_windows=1,
        positional_encoding_size=4,
    )
    assert wrapper.hidden_size == 768


def test_dinov2_large_config():
    wrapper = DinoV2(
        shape=(56, 56), out_feature_indexes=[0, 1], size="large",
        use_registers=True, patch_size=14, num_windows=1,
        positional_encoding_size=4,
    )
    assert wrapper.hidden_size == 1024


if __name__ == "__main__":
    pytest.main([__file__])
