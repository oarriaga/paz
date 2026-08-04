import os
import sys

import numpy as np
import pytest

_PT_DIR = "examples/rf-detr_original_pytorch_implementation"
_PT_ROOT = os.path.join(os.path.dirname(__file__), *([".."] * 6))
if not os.path.isdir(os.path.join(_PT_ROOT, _PT_DIR)):
    pytest.skip("RF-DETR reference unavailable", allow_module_level=True)
pytest.importorskip("torch")

import torch
import torch.nn as nn
from keras import Input, Model

current_dir = os.path.dirname(os.path.abspath(__file__))
rf_detr_root = os.path.abspath(os.path.join(
    current_dir,
    '../../../../../../examples/rf-detr_original_pytorch_implementation',
))
sys.path.append(rf_detr_root)

try:
    from rfdetr.models.transformer import MLP as TorchMLP
    from rfdetr.models.transformer import (
        gen_sineembed_for_position as torch_gen_sineembed,
    )
    from rfdetr.models.transformer import (
        gen_encoder_output_proposals as torch_gen_encoder_output_proposals,
    )
    from rfdetr.models.transformer import (
        TransformerDecoderLayer as TorchTransformerDecoderLayer,
    )
    from rfdetr.models.transformer import Transformer as TorchTransformer
except ImportError:
    pass

from transformer import mlp as keras_mlp
from transformer import apply_mlp
from transformer import embed_position_sine as keras_gen_sineembed
from transformer import (
    gen_encoder_output_proposals as keras_gen_encoder_output_proposals,
)
from transformer import materialize_decoder_layer, apply_decoder_layer
from transformer import Transformer as KerasTransformer
from transformer import apply_transformer

from transformer_weights_porting_utils import (
    to_numpy,
    to_torch,
    to_keras,
    transfer_mha,
    transfer_layernorm,
    transfer_dense,
    transfer_transformer_weights,
    build_parity_heads,
    transfer_parity_heads,
)


def build_transformer_inputs(bs, d_model, spatial_shapes):
    srcs_np = [np.random.randn(bs, d_model, h, w).astype(np.float32) for h, w in spatial_shapes]  # fmt: skip
    masks_np = [np.zeros((bs, h, w), dtype=bool) for h, w in spatial_shapes]
    pos_np = [np.random.randn(bs, d_model, h, w).astype(np.float32) for h, w in spatial_shapes]  # fmt: skip
    t_srcs = [to_torch(x) for x in srcs_np]
    t_masks = [torch.tensor(x, dtype=torch.bool) for x in masks_np]
    t_pos = [to_torch(x) for x in pos_np]
    k_srcs = [to_keras(np.transpose(x, (0, 2, 3, 1))) for x in srcs_np]
    k_masks = [to_keras(x) for x in masks_np]
    k_pos = [to_keras(np.transpose(x, (0, 2, 3, 1))) for x in pos_np]
    return (t_srcs, t_masks, t_pos), (k_srcs, k_masks, k_pos)


def test_mlp_parity():
    input_dim, hidden_dim, output_dim, num_layers = 16, 32, 16, 3
    bs = 2

    x_np = np.random.randn(bs, input_dim).astype(np.float32)

    torch_mlp = TorchMLP(input_dim, hidden_dim, output_dim, num_layers)
    torch_mlp.eval()

    x = Input(shape=(input_dim,), name="x")
    model = Model(
        x, keras_mlp(x, input_dim, hidden_dim, output_dim, num_layers, "mlp")
    )

    with torch.no_grad():
        for i, torch_layer in enumerate(torch_mlp.layers):
            transfer_dense(torch_layer, model.get_layer(f"mlp_dense_{i}"))

    out_torch = torch_mlp(to_torch(x_np))
    out_keras = apply_mlp(model, to_keras(x_np), num_layers, "mlp")

    assert np.allclose(to_numpy(out_torch), to_numpy(out_keras), atol=1e-5)


def test_sine_embed_parity():
    nq, bs = 5, 2
    pos_np = np.random.rand(nq, bs, 4).astype(np.float32)
    dim = 128

    out_torch = torch_gen_sineembed(to_torch(pos_np), dim)
    out_keras = keras_gen_sineembed(to_keras(pos_np), dim)

    diff = np.abs(to_numpy(out_torch) - to_numpy(out_keras))
    print(f"Max diff sine embed: {diff.max()}")
    assert np.allclose(to_numpy(out_torch), to_numpy(out_keras), atol=1e-5)


def test_gen_encoder_output_proposals():
    bs = 2
    spatial_shapes = [(4, 4), (2, 2)]
    d_model = 16

    total_len = sum([h * w for h, w in spatial_shapes])
    memory_np = np.random.randn(bs, total_len, d_model).astype(np.float32)

    mask_np = np.zeros((bs, total_len), dtype=bool)
    mask_np[:, -2:] = True

    t_memory = to_torch(memory_np)
    t_mask = torch.tensor(mask_np, dtype=torch.bool)

    out_mem_torch, out_prop_torch = torch_gen_encoder_output_proposals(
        t_memory, t_mask, spatial_shapes
    )

    out_mem_keras, out_prop_keras = keras_gen_encoder_output_proposals(
        to_keras(memory_np), to_keras(mask_np), spatial_shapes
    )

    assert np.allclose(
        to_numpy(out_mem_torch), to_numpy(out_mem_keras), atol=1e-5
    )

    valid_mask = ~np.isinf(to_numpy(out_prop_torch))
    assert np.allclose(
        to_numpy(out_prop_torch)[valid_mask],
        to_numpy(out_prop_keras)[valid_mask],
        atol=1e-5,
    )
    assert np.all(
        np.isinf(to_numpy(out_prop_torch)) == np.isinf(to_numpy(out_prop_keras))
    )


def test_decoder_layer_parity():
    d_model, sa_nhead, ca_nhead = 32, 4, 4
    n_levels, n_points = 2, 2
    bs, nq = 2, 5

    torch_layer = TorchTransformerDecoderLayer(
        d_model, sa_nhead, ca_nhead, num_feature_levels=n_levels,
        dec_n_points=n_points,
    )
    torch_layer.eval()

    query = Input(shape=(None, d_model), name="query")
    memory_in = Input(shape=(None, d_model), name="memory")
    outputs = materialize_decoder_layer(query, memory_in, d_model, sa_nhead, ca_nhead, 2048, 0.1, n_levels, n_points, "decoder_layer_0")  # fmt: skip
    model = Model([query, memory_in], outputs, name="decoder_layer")

    tgt = np.random.randn(bs, nq, d_model).astype(np.float32)
    memory = np.random.randn(bs, 20, d_model).astype(np.float32)
    query_pos = np.random.randn(bs, nq, d_model).astype(np.float32)
    ref_points = np.random.rand(bs, nq, n_levels, 4).astype(np.float32)

    spatial_shapes = [(4, 4), (2, 2)]
    np_spatial = np.array(spatial_shapes, dtype=np.int32)

    t_tgt = to_torch(tgt)
    t_memory = to_torch(memory)
    t_query_pos = to_torch(query_pos)
    t_ref_points = to_torch(ref_points)
    t_spatial = torch.tensor(spatial_shapes, dtype=torch.long)
    lens = t_spatial[:, 0] * t_spatial[:, 1]
    t_level_start = torch.cat(
        (torch.tensor([0]), torch.cumsum(lens, 0)[:-1])
    )

    name = "decoder_layer_0"
    with torch.no_grad():
        transfer_mha(torch_layer.self_attn, model.get_layer(f"{name}_self_attn"), d_model, sa_nhead)  # fmt: skip
        transfer_layernorm(torch_layer.norm1, model.get_layer(f"{name}_norm1"))
        transfer_layernorm(torch_layer.norm2, model.get_layer(f"{name}_norm2"))
        transfer_layernorm(torch_layer.norm3, model.get_layer(f"{name}_norm3"))
        transfer_dense(torch_layer.linear1, model.get_layer(f"{name}_linear1"))
        transfer_dense(torch_layer.linear2, model.get_layer(f"{name}_linear2"))
        transfer_dense(torch_layer.cross_attn.sampling_offsets, model.get_layer(f"{name}_cross_attn_sampling_offsets"))  # fmt: skip
        transfer_dense(torch_layer.cross_attn.attention_weights, model.get_layer(f"{name}_cross_attn_attention_weights"))  # fmt: skip
        transfer_dense(torch_layer.cross_attn.value_proj, model.get_layer(f"{name}_cross_attn_value_proj"))  # fmt: skip
        transfer_dense(torch_layer.cross_attn.output_proj, model.get_layer(f"{name}_cross_attn_output_proj"))  # fmt: skip

    out_torch = torch_layer.forward_post(
        t_tgt, t_memory,
        query_pos=t_query_pos,
        reference_points=t_ref_points,
        spatial_shapes=t_spatial,
        level_start_index=t_level_start
    )

    out_keras = apply_decoder_layer(model, to_keras(tgt), to_keras(memory), d_model, 1, n_levels, ca_nhead, n_points, "relu", 0.0, to_keras(query_pos), to_keras(ref_points), np_spatial, None, None, False, name)  # fmt: skip

    assert np.allclose(to_numpy(out_torch), to_numpy(out_keras), atol=1e-5)


def run_transformer_parity(torch_transformer, keras_transformer, two_stage, atol=1e-5):  # fmt: skip
    d_model = keras_transformer.d_model
    sa_nhead = torch_transformer.decoder.layers[0].self_attn.num_heads
    num_queries = keras_transformer.transformer_config["num_queries"]
    bs = 2
    spatial_shapes = [(4, 4), (2, 2)]

    bbox_embed, enc_cls, enc_bbox = build_parity_heads(
        torch_transformer, d_model
    )
    transfer_transformer_weights(torch_transformer, keras_transformer, d_model, sa_nhead)  # fmt: skip
    transfer_parity_heads(torch_transformer, bbox_embed, enc_cls, enc_bbox)

    (t_srcs, t_masks, t_pos), (k_srcs, k_masks, k_pos) = build_transformer_inputs(bs, d_model, spatial_shapes)  # fmt: skip

    query_feat_np = np.random.randn(num_queries, d_model).astype(np.float32)
    refpoint_embed_np = np.random.randn(num_queries, 4).astype(np.float32)
    t_query_feat = to_torch(query_feat_np)
    t_refpoint_embed = to_torch(refpoint_embed_np)

    with torch.no_grad():
        out_torch = torch_transformer(t_srcs, t_masks, t_pos, t_refpoint_embed, t_query_feat)  # fmt: skip

    out_keras = apply_transformer(keras_transformer, k_srcs, k_masks, k_pos, bbox_embed, enc_cls, enc_bbox, to_keras(query_feat_np), to_keras(refpoint_embed_np), training=False)  # fmt: skip

    names = ["HS", "Ref", "Mem TS", "Box TS"]
    for i, out_name in enumerate(names):
        if not two_stage and i >= 2:
            continue
        diff = np.abs(to_numpy(out_torch[i]) - to_numpy(out_keras[i]))
        print(f"{out_name} Max diff: {diff.max()}")
        assert np.allclose(
            to_numpy(out_torch[i]), to_numpy(out_keras[i]), atol=atol
        )


def test_transformer_full_parity():
    d_model, sa_nhead, ca_nhead = 32, 4, 4
    num_decoder_layers, dim_feedforward = 2, 64
    dropout, num_queries = 0.0, 5
    num_feature_levels, dec_n_points = 2, 2

    torch_transformer = TorchTransformer(
        d_model=d_model, sa_nhead=sa_nhead, ca_nhead=ca_nhead,
        num_queries=num_queries, num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward, dropout=dropout,
        return_intermediate_dec=True, two_stage=True,
        num_feature_levels=num_feature_levels, dec_n_points=dec_n_points
    )
    torch_transformer.eval()

    torch_transformer.enc_out_class_embed = nn.ModuleList(
        [nn.Linear(d_model, 91)]
    )
    torch_transformer.enc_out_bbox_embed = nn.ModuleList([TorchMLP(d_model, d_model, 4, 3)])  # fmt: skip
    torch_transformer.decoder.bbox_embed = TorchMLP(d_model, d_model, 4, 3)

    keras_transformer = KerasTransformer(
        d_model=d_model, sa_nhead=sa_nhead, ca_nhead=ca_nhead,
        num_queries=num_queries, num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward, dropout=dropout,
        return_intermediate_dec=True, two_stage=True,
        num_feature_levels=num_feature_levels, dec_n_points=dec_n_points
    )

    run_transformer_parity(torch_transformer, keras_transformer, two_stage=True)


@pytest.mark.parametrize("two_stage", [True, False])
@pytest.mark.parametrize("bbox_reparam", [True, False])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
@pytest.mark.parametrize("num_decoder_layers", [1, 2])
def test_transformer_configurations(
    two_stage, bbox_reparam, activation, num_decoder_layers
):
    d_model, sa_nhead, ca_nhead = 32, 4, 4
    dim_feedforward, dropout, num_queries = 64, 0.0, 5
    num_feature_levels, dec_n_points = 2, 2

    torch_transformer = TorchTransformer(
        d_model=d_model, sa_nhead=sa_nhead, ca_nhead=ca_nhead,
        num_queries=num_queries, num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
        return_intermediate_dec=True, two_stage=two_stage,
        num_feature_levels=num_feature_levels, dec_n_points=dec_n_points,
        bbox_reparam=bbox_reparam
    )
    torch_transformer.eval()

    if two_stage:
        torch_transformer.enc_out_class_embed = nn.ModuleList(
            [nn.Linear(d_model, 91)]
        )
        torch_transformer.enc_out_bbox_embed = nn.ModuleList([TorchMLP(d_model, d_model, 4, 3)])  # fmt: skip

    torch_transformer.decoder.bbox_embed = TorchMLP(d_model, d_model, 4, 3)

    keras_transformer = KerasTransformer(
        d_model=d_model, sa_nhead=sa_nhead, ca_nhead=ca_nhead,
        num_queries=num_queries, num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
        return_intermediate_dec=True, two_stage=two_stage,
        num_feature_levels=num_feature_levels, dec_n_points=dec_n_points,
        bbox_reparam=bbox_reparam
    )

    run_transformer_parity(
        torch_transformer, keras_transformer, two_stage=two_stage
    )


if __name__ == "__main__":
    pytest.main([__file__])
