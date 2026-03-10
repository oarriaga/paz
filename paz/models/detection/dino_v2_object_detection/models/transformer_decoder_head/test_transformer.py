import os
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn
import keras
from keras import layers
from keras import ops

current_dir = os.path.dirname(os.path.abspath(__file__))
rf_detr_root = os.path.abspath(os.path.join(current_dir, '../../../../../../examples/rf-detr_original_pytorch_implementation'))
sys.path.append(rf_detr_root)

try:
    from rfdetr.models.transformer import MLP as TorchMLP
    from rfdetr.models.transformer import gen_sineembed_for_position as torch_gen_sineembed
    from rfdetr.models.transformer import gen_encoder_output_proposals as torch_gen_encoder_output_proposals
    from rfdetr.models.transformer import TransformerDecoderLayer as TorchTransformerDecoderLayer
    from rfdetr.models.transformer import TransformerDecoder as TorchTransformerDecoder
    from rfdetr.models.transformer import Transformer as TorchTransformer
except ImportError:
    pass

from transformer import MLP as KerasMLP
from transformer import gen_sineembed_for_position as keras_gen_sineembed
from transformer import gen_encoder_output_proposals as keras_gen_encoder_output_proposals
from transformer import TransformerDecoderLayer as KerasTransformerDecoderLayer
from transformer import TransformerDecoder as KerasTransformerDecoder
from transformer import Transformer as KerasTransformer

from transformer_weights_porting_utils import (
    to_numpy,
    to_torch,
    to_keras
)

def test_mlp_parity():
    """Verify MLP forward-pass parity between reference and Keras."""
    input_dim = 16
    hidden_dim = 32
    output_dim = 16
    num_layers = 3
    bs = 2
    
    # Inputs
    x_np = np.random.randn(bs, input_dim).astype(np.float32)
    
    torch_mlp = TorchMLP(input_dim, hidden_dim, output_dim, num_layers)
    torch_mlp.eval()
    
    keras_mlp = KerasMLP(input_dim, hidden_dim, output_dim, num_layers)
    
    # Build
    _ = keras_mlp(to_keras(x_np))
    
    # Transfer weights: transpose Dense kernels for (in, out) convention
    with torch.no_grad():
        for i, (torch_layer, keras_layer) in enumerate(zip(torch_mlp.layers, keras_mlp.layers_list)):
            keras_layer.kernel.assign(to_keras(torch_layer.weight.T.numpy()))
            keras_layer.bias.assign(to_keras(torch_layer.bias.numpy()))
            
    out_torch = torch_mlp(to_torch(x_np))
    out_keras = keras_mlp(to_keras(x_np))
    
    assert np.allclose(to_numpy(out_torch), to_numpy(out_keras), atol=1e-5)

def test_sine_embed_parity():
    """Verify sine positional embedding parity."""
    nq, bs = 5, 2
    pos_np = np.random.rand(nq, bs, 4).astype(np.float32)
    dim = 128
    
    out_torch = torch_gen_sineembed(to_torch(pos_np), dim)
    out_keras = keras_gen_sineembed(to_keras(pos_np), dim)
    
    diff = np.abs(to_numpy(out_torch) - to_numpy(out_keras))
    print(f"Max diff sine embed: {diff.max()}")
    assert np.allclose(to_numpy(out_torch), to_numpy(out_keras), atol=1e-5)

def test_gen_encoder_output_proposals():
    """Verify encoder output proposal generation parity."""
    bs = 2
    spatial_shapes = [(4, 4), (2, 2)]
    d_model = 16
    
    # Create memory
    total_len = sum([h*w for h, w in spatial_shapes])
    memory_np = np.random.randn(bs, total_len, d_model).astype(np.float32)
    
    # Mask (some True)
    mask_np = np.zeros((bs, total_len), dtype=bool)
    mask_np[:, -2:] = True
    
    t_spatial = torch.tensor(spatial_shapes, dtype=torch.long)
    t_memory = to_torch(memory_np)
    t_mask = torch.tensor(mask_np, dtype=torch.bool)
    
    out_mem_torch, out_prop_torch = torch_gen_encoder_output_proposals(t_memory, t_mask, spatial_shapes)
    
    out_mem_keras, out_prop_keras = keras_gen_encoder_output_proposals(
        to_keras(memory_np), to_keras(mask_np), spatial_shapes
    )
    
    assert np.allclose(to_numpy(out_mem_torch), to_numpy(out_mem_keras), atol=1e-5)
    
    valid_mask = ~np.isinf(to_numpy(out_prop_torch))
    assert np.allclose(to_numpy(out_prop_torch)[valid_mask], to_numpy(out_prop_keras)[valid_mask], atol=1e-5)
    assert np.all(np.isinf(to_numpy(out_prop_torch)) == np.isinf(to_numpy(out_prop_keras)))

def test_decoder_layer_parity():
    """Verify TransformerDecoderLayer forward-pass parity."""
    d_model = 32
    sa_nhead = 4
    ca_nhead = 4
    n_levels = 2
    n_points = 2
    bs = 2
    nq = 5
    
    torch_layer = TorchTransformerDecoderLayer(d_model, sa_nhead, ca_nhead, num_feature_levels=n_levels, dec_n_points=n_points)
    torch_layer.eval()
    
    keras_layer = KerasTransformerDecoderLayer(d_model, sa_nhead, ca_nhead, num_feature_levels=n_levels, dec_n_points=n_points)
    
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
    t_level_start = torch.cat((torch.tensor([0]), torch.cumsum(t_spatial[:,0]*t_spatial[:,1], 0)[:-1]))
    
    keras_layer(to_keras(tgt), to_keras(memory), 
                query_pos=to_keras(query_pos), 
                reference_points=to_keras(ref_points),
                spatial_shapes=np_spatial,
                level_start_index=np_spatial)
    
    with torch.no_grad():
        q_w = to_keras(torch_layer.self_attn.in_proj_weight[:d_model, :].T.numpy())
        q_w = ops.reshape(q_w, (d_model, sa_nhead, d_model // sa_nhead))
        keras_layer.self_attn.query_dense.kernel.assign(q_w)
        q_b = to_keras(torch_layer.self_attn.in_proj_bias[:d_model].numpy())
        q_b = ops.reshape(q_b, (sa_nhead, d_model // sa_nhead))
        keras_layer.self_attn.query_dense.bias.assign(q_b)
        
        k_w = to_keras(torch_layer.self_attn.in_proj_weight[d_model:2*d_model, :].T.numpy())
        k_w = ops.reshape(k_w, (d_model, sa_nhead, d_model // sa_nhead))
        keras_layer.self_attn.key_dense.kernel.assign(k_w)
        k_b = to_keras(torch_layer.self_attn.in_proj_bias[d_model:2*d_model].numpy())
        k_b = ops.reshape(k_b, (sa_nhead, d_model // sa_nhead))
        keras_layer.self_attn.key_dense.bias.assign(k_b)
        
        v_w = to_keras(torch_layer.self_attn.in_proj_weight[2*d_model:, :].T.numpy())
        v_w = ops.reshape(v_w, (d_model, sa_nhead, d_model // sa_nhead))
        keras_layer.self_attn.value_dense.kernel.assign(v_w)
        v_b = to_keras(torch_layer.self_attn.in_proj_bias[2*d_model:].numpy())
        v_b = ops.reshape(v_b, (sa_nhead, d_model // sa_nhead))
        keras_layer.self_attn.value_dense.bias.assign(v_b)
        
        out_w = to_keras(torch_layer.self_attn.out_proj.weight.T.numpy())
        out_w = ops.reshape(out_w, (sa_nhead, d_model // sa_nhead, d_model))
        keras_layer.self_attn.output_dense.kernel.assign(out_w)
        keras_layer.self_attn.output_dense.bias.assign(to_keras(torch_layer.self_attn.out_proj.bias.numpy()))

        keras_layer.norm1.gamma.assign(to_keras(torch_layer.norm1.weight.numpy()))
        keras_layer.norm1.beta.assign(to_keras(torch_layer.norm1.bias.numpy()))
        keras_layer.norm2.gamma.assign(to_keras(torch_layer.norm2.weight.numpy()))
        keras_layer.norm2.beta.assign(to_keras(torch_layer.norm2.bias.numpy()))
        keras_layer.norm3.gamma.assign(to_keras(torch_layer.norm3.weight.numpy()))
        keras_layer.norm3.beta.assign(to_keras(torch_layer.norm3.bias.numpy()))
        
        keras_layer.linear1.kernel.assign(to_keras(torch_layer.linear1.weight.T.numpy()))
        keras_layer.linear1.bias.assign(to_keras(torch_layer.linear1.bias.numpy()))
        keras_layer.linear2.kernel.assign(to_keras(torch_layer.linear2.weight.T.numpy()))
        keras_layer.linear2.bias.assign(to_keras(torch_layer.linear2.bias.numpy()))
        
        keras_layer.cross_attn.sampling_offsets.kernel.assign(to_keras(torch_layer.cross_attn.sampling_offsets.weight.T.numpy()))
        keras_layer.cross_attn.sampling_offsets.bias.assign(to_keras(torch_layer.cross_attn.sampling_offsets.bias.numpy()))
        keras_layer.cross_attn.attention_weights.kernel.assign(to_keras(torch_layer.cross_attn.attention_weights.weight.T.numpy()))
        keras_layer.cross_attn.attention_weights.bias.assign(to_keras(torch_layer.cross_attn.attention_weights.bias.numpy()))
        keras_layer.cross_attn.value_proj.kernel.assign(to_keras(torch_layer.cross_attn.value_proj.weight.T.numpy()))
        keras_layer.cross_attn.value_proj.bias.assign(to_keras(torch_layer.cross_attn.value_proj.bias.numpy()))
        keras_layer.cross_attn.output_proj.kernel.assign(to_keras(torch_layer.cross_attn.output_proj.weight.T.numpy()))
        keras_layer.cross_attn.output_proj.bias.assign(to_keras(torch_layer.cross_attn.output_proj.bias.numpy()))

    out_torch = torch_layer.forward_post(
        t_tgt, t_memory, 
        query_pos=t_query_pos, 
        reference_points=t_ref_points,
        spatial_shapes=t_spatial,
        level_start_index=t_level_start
    )
    
    out_keras = keras_layer(
        to_keras(tgt), to_keras(memory),
        query_pos=to_keras(query_pos),
        reference_points=to_keras(ref_points),
        spatial_shapes=np_spatial,
        level_start_index=None
    )
    
    assert np.allclose(to_numpy(out_torch), to_numpy(out_keras), atol=1e-5)

def test_transformer_full_parity():
    """Verify full Transformer forward-pass parity with default config."""
    # Only verify default config here; exhaustive in test_transformer_configurations
    d_model = 32
    sa_nhead = 4
    ca_nhead = 4
    num_encoder_layers = 0 
    num_decoder_layers = 2
    dim_feedforward = 64
    dropout = 0.0
    bs = 2
    num_queries = 5
    num_feature_levels = 2
    dec_n_points = 2
    
    torch_transformer = TorchTransformer(
        d_model=d_model, sa_nhead=sa_nhead, ca_nhead=ca_nhead,
        num_queries=num_queries,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        return_intermediate_dec=True,
        two_stage=True,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points
    )
    torch_transformer.eval()
    
    torch_transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(d_model, 91)])
    torch_transformer.enc_out_bbox_embed = nn.ModuleList([TorchMLP(d_model, d_model, 4, 3)])
    torch_transformer.decoder.bbox_embed = TorchMLP(d_model, d_model, 4, 3)
    
    keras_transformer = KerasTransformer(
        d_model=d_model, sa_nhead=sa_nhead, ca_nhead=ca_nhead,
        num_queries=num_queries,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        return_intermediate_dec=True,
        two_stage=True,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points
    )
    
    keras_transformer.enc_out_class_embed = [layers.Dense(91)]
    keras_transformer.enc_out_bbox_embed = [KerasMLP(d_model, d_model, 4, 3)]
    keras_transformer.decoder.bbox_embed = KerasMLP(d_model, d_model, 4, 3)
    
    srcs_np = [np.random.randn(bs, d_model, 4, 4).astype(np.float32), 
               np.random.randn(bs, d_model, 2, 2).astype(np.float32)]
    masks_np = [np.zeros((bs, 4, 4), dtype=bool), np.zeros((bs, 2, 2), dtype=bool)]
    pos_embeds_np = [np.random.randn(bs, d_model, 4, 4).astype(np.float32), 
                     np.random.randn(bs, d_model, 2, 2).astype(np.float32)]
    
    query_feat_np = np.random.randn(num_queries, d_model).astype(np.float32)
    refpoint_embed_np = np.random.randn(num_queries, 4).astype(np.float32)
    
    t_srcs = [to_torch(x) for x in srcs_np]
    t_masks = [torch.tensor(x, dtype=torch.bool) for x in masks_np]
    t_pos_embeds = [to_torch(x) for x in pos_embeds_np]
    t_query_feat = to_torch(query_feat_np)
    t_refpoint_embed = to_torch(refpoint_embed_np)
    
    k_srcs = [to_keras(x) for x in srcs_np]
    k_masks = [to_keras(x) for x in masks_np]
    k_pos_embeds = [to_keras(x) for x in pos_embeds_np]
    k_query_feat = to_keras(query_feat_np)
    k_refpoint_embed = to_keras(refpoint_embed_np)
    
    keras_transformer(k_srcs, k_masks, k_pos_embeds, k_query_feat, k_refpoint_embed)
    
    with torch.no_grad():
        for i in range(1):
             keras_transformer.enc_output[i].kernel.assign(to_keras(torch_transformer.enc_output[i].weight.T.numpy()))
             keras_transformer.enc_output[i].bias.assign(to_keras(torch_transformer.enc_output[i].bias.numpy()))
             keras_transformer.enc_output_norm[i].gamma.assign(to_keras(torch_transformer.enc_output_norm[i].weight.numpy()))
             keras_transformer.enc_output_norm[i].beta.assign(to_keras(torch_transformer.enc_output_norm[i].bias.numpy()))
             keras_transformer.enc_out_class_embed[i].kernel.assign(to_keras(torch_transformer.enc_out_class_embed[i].weight.T.numpy()))
             keras_transformer.enc_out_class_embed[i].bias.assign(to_keras(torch_transformer.enc_out_class_embed[i].bias.numpy()))
             
             for j, (tk, klayer) in enumerate(zip(torch_transformer.enc_out_bbox_embed[i].layers, keras_transformer.enc_out_bbox_embed[i].layers_list)):
                  klayer.kernel.assign(to_keras(tk.weight.T.numpy()))
                  klayer.bias.assign(to_keras(tk.bias.numpy()))
        
        # Transfer decoder layer weights
        for i in range(num_decoder_layers):
            t_layer = torch_transformer.decoder.layers[i]
            k_layer = keras_transformer.decoder.layers_list[i]
            
            def transfer_mha(t_mha, k_mha):
                """Transfer MultiHeadAttention weights from reference to Keras."""
                q_w = to_keras(t_mha.in_proj_weight[:d_model, :].T.numpy())
                q_w = ops.reshape(q_w, (d_model, sa_nhead, d_model // sa_nhead))
                k_mha.query_dense.kernel.assign(q_w)
                q_b = to_keras(t_mha.in_proj_bias[:d_model].numpy())
                q_b = ops.reshape(q_b, (sa_nhead, d_model // sa_nhead))
                k_mha.query_dense.bias.assign(q_b)
                
                k_w = to_keras(t_mha.in_proj_weight[d_model:2*d_model, :].T.numpy())
                k_w = ops.reshape(k_w, (d_model, sa_nhead, d_model // sa_nhead))
                k_mha.key_dense.kernel.assign(k_w)
                k_b = to_keras(t_mha.in_proj_bias[d_model:2*d_model].numpy())
                k_b = ops.reshape(k_b, (sa_nhead, d_model // sa_nhead))
                k_mha.key_dense.bias.assign(k_b)
                
                v_w = to_keras(t_mha.in_proj_weight[2*d_model:, :].T.numpy())
                v_w = ops.reshape(v_w, (d_model, sa_nhead, d_model // sa_nhead))
                k_mha.value_dense.kernel.assign(v_w)
                v_b = to_keras(t_mha.in_proj_bias[2*d_model:].numpy())
                v_b = ops.reshape(v_b, (sa_nhead, d_model // sa_nhead))
                k_mha.value_dense.bias.assign(v_b)
                
                out_w = to_keras(t_mha.out_proj.weight.T.numpy())
                out_w = ops.reshape(out_w, (sa_nhead, d_model // sa_nhead, d_model))
                k_mha.output_dense.kernel.assign(out_w)
                k_mha.output_dense.bias.assign(to_keras(t_mha.out_proj.bias.numpy()))

            transfer_mha(t_layer.self_attn, k_layer.self_attn)
            k_layer.norm1.gamma.assign(to_keras(t_layer.norm1.weight.numpy()))
            k_layer.norm1.beta.assign(to_keras(t_layer.norm1.bias.numpy()))
            k_layer.cross_attn.sampling_offsets.kernel.assign(to_keras(t_layer.cross_attn.sampling_offsets.weight.T.numpy()))
            k_layer.cross_attn.sampling_offsets.bias.assign(to_keras(t_layer.cross_attn.sampling_offsets.bias.numpy()))
            k_layer.cross_attn.attention_weights.kernel.assign(to_keras(t_layer.cross_attn.attention_weights.weight.T.numpy()))
            k_layer.cross_attn.attention_weights.bias.assign(to_keras(t_layer.cross_attn.attention_weights.bias.numpy()))
            k_layer.cross_attn.value_proj.kernel.assign(to_keras(t_layer.cross_attn.value_proj.weight.T.numpy()))
            k_layer.cross_attn.value_proj.bias.assign(to_keras(t_layer.cross_attn.value_proj.bias.numpy()))
            k_layer.cross_attn.output_proj.kernel.assign(to_keras(t_layer.cross_attn.output_proj.weight.T.numpy()))
            k_layer.cross_attn.output_proj.bias.assign(to_keras(t_layer.cross_attn.output_proj.bias.numpy()))
            k_layer.norm2.gamma.assign(to_keras(t_layer.norm2.weight.numpy()))
            k_layer.norm2.beta.assign(to_keras(t_layer.norm2.bias.numpy()))
            k_layer.linear1.kernel.assign(to_keras(t_layer.linear1.weight.T.numpy()))
            k_layer.linear1.bias.assign(to_keras(t_layer.linear1.bias.numpy()))
            k_layer.linear2.kernel.assign(to_keras(t_layer.linear2.weight.T.numpy()))
            k_layer.linear2.bias.assign(to_keras(t_layer.linear2.bias.numpy()))
            k_layer.norm3.gamma.assign(to_keras(t_layer.norm3.weight.numpy()))
            k_layer.norm3.beta.assign(to_keras(t_layer.norm3.bias.numpy()))

        for j, (tk, klayer) in enumerate(zip(torch_transformer.decoder.ref_point_head.layers, keras_transformer.decoder.ref_point_head.layers_list)):
             klayer.kernel.assign(to_keras(tk.weight.T.numpy()))
             klayer.bias.assign(to_keras(tk.bias.numpy()))
             
        for j, (tk, klayer) in enumerate(zip(torch_transformer.decoder.bbox_embed.layers, keras_transformer.decoder.bbox_embed.layers_list)):
             klayer.kernel.assign(to_keras(tk.weight.T.numpy()))
             klayer.bias.assign(to_keras(tk.bias.numpy()))
             
        keras_transformer.decoder.norm.gamma.assign(to_keras(torch_transformer.decoder.norm.weight.numpy()))
        keras_transformer.decoder.norm.beta.assign(to_keras(torch_transformer.decoder.norm.bias.numpy()))
             
    out_hs_torch, out_ref_torch, out_mem_ts_torch, out_box_ts_torch = torch_transformer(
        t_srcs, t_masks, t_pos_embeds, t_refpoint_embed, t_query_feat
    )
    
    out_hs_keras, out_ref_keras, out_mem_ts_keras, out_box_ts_keras = keras_transformer(
        k_srcs, k_masks, k_pos_embeds, k_query_feat, k_refpoint_embed
    )
    
    diff_hs = np.abs(to_numpy(out_hs_torch) - to_numpy(out_hs_keras))
    print(f"HS Max diff: {diff_hs.max()}")
    assert np.allclose(to_numpy(out_hs_torch), to_numpy(out_hs_keras), atol=1e-5)
    
    diff_ref = np.abs(to_numpy(out_ref_torch) - to_numpy(out_ref_keras))
    print(f"Ref Max diff: {diff_ref.max()}")
    assert np.allclose(to_numpy(out_ref_torch), to_numpy(out_ref_keras), atol=1e-5)
    
    diff_mem = np.abs(to_numpy(out_mem_ts_torch) - to_numpy(out_mem_ts_keras))
    print(f"Mem TS Max diff: {diff_mem.max()}")
    assert np.allclose(to_numpy(out_mem_ts_torch), to_numpy(out_mem_ts_keras), atol=1e-5)
    
    diff_box = np.abs(to_numpy(out_box_ts_torch) - to_numpy(out_box_ts_keras))
    print(f"Box TS Max diff: {diff_box.max()}")
    assert np.allclose(to_numpy(out_box_ts_torch), to_numpy(out_box_ts_keras), atol=1e-5)

@pytest.mark.parametrize("two_stage", [True, False])
@pytest.mark.parametrize("bbox_reparam", [True, False])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
@pytest.mark.parametrize("num_decoder_layers", [1, 2])
def test_transformer_configurations(two_stage, bbox_reparam, activation, num_decoder_layers):
    """Verify full Transformer parity across configuration combinations."""
    d_model = 32
    sa_nhead = 4
    ca_nhead = 4
    dim_feedforward = 64
    dropout = 0.0
    bs = 2
    num_queries = 5
    num_feature_levels = 2
    dec_n_points = 2
    
    torch_transformer = TorchTransformer(
        d_model=d_model, sa_nhead=sa_nhead, ca_nhead=ca_nhead,
        num_queries=num_queries,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        return_intermediate_dec=True,
        two_stage=two_stage,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
        bbox_reparam=bbox_reparam
    )
    torch_transformer.eval()
    
    if two_stage:
        torch_transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(d_model, 91)])
        torch_transformer.enc_out_bbox_embed = nn.ModuleList([TorchMLP(d_model, d_model, 4, 3)])
    
    torch_transformer.decoder.bbox_embed = TorchMLP(d_model, d_model, 4, 3)
    
    keras_transformer = KerasTransformer(
        d_model=d_model, sa_nhead=sa_nhead, ca_nhead=ca_nhead,
        num_queries=num_queries,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        return_intermediate_dec=True,
        two_stage=two_stage,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
        bbox_reparam=bbox_reparam
    )
    
    if two_stage:
        keras_transformer.enc_out_class_embed = [layers.Dense(91)]
        keras_transformer.enc_out_bbox_embed = [KerasMLP(d_model, d_model, 4, 3)]
        
    keras_transformer.decoder.bbox_embed = KerasMLP(d_model, d_model, 4, 3)
    
    srcs_np = [np.random.randn(bs, d_model, 4, 4).astype(np.float32), 
               np.random.randn(bs, d_model, 2, 2).astype(np.float32)]
    masks_np = [np.zeros((bs, 4, 4), dtype=bool), np.zeros((bs, 2, 2), dtype=bool)]
    pos_embeds_np = [np.random.randn(bs, d_model, 4, 4).astype(np.float32), 
                     np.random.randn(bs, d_model, 2, 2).astype(np.float32)]
    
    query_feat_np = np.random.randn(num_queries, d_model).astype(np.float32)
    refpoint_embed_np = np.random.randn(num_queries, 4).astype(np.float32)
    
    t_srcs = [to_torch(x) for x in srcs_np]
    t_masks = [torch.tensor(x, dtype=torch.bool) for x in masks_np]
    t_pos_embeds = [to_torch(x) for x in pos_embeds_np]
    t_query_feat = to_torch(query_feat_np)
    t_refpoint_embed = to_torch(refpoint_embed_np)
    
    k_srcs = [to_keras(x) for x in srcs_np]
    k_masks = [to_keras(x) for x in masks_np]
    k_pos_embeds = [to_keras(x) for x in pos_embeds_np]
    k_query_feat = to_keras(query_feat_np)
    k_refpoint_embed = to_keras(refpoint_embed_np)
    
    # Build Keras
    keras_transformer(k_srcs, k_masks, k_pos_embeds, k_query_feat, k_refpoint_embed)
    
    with torch.no_grad():
        if two_stage:
            for i in range(1):
                 keras_transformer.enc_output[i].kernel.assign(to_keras(torch_transformer.enc_output[i].weight.T.numpy()))
                 keras_transformer.enc_output[i].bias.assign(to_keras(torch_transformer.enc_output[i].bias.numpy()))
                 keras_transformer.enc_output_norm[i].gamma.assign(to_keras(torch_transformer.enc_output_norm[i].weight.numpy()))
                 keras_transformer.enc_output_norm[i].beta.assign(to_keras(torch_transformer.enc_output_norm[i].bias.numpy()))
                 keras_transformer.enc_out_class_embed[i].kernel.assign(to_keras(torch_transformer.enc_out_class_embed[i].weight.T.numpy()))
                 keras_transformer.enc_out_class_embed[i].bias.assign(to_keras(torch_transformer.enc_out_class_embed[i].bias.numpy()))
                 
                 for j, (tk, klayer) in enumerate(zip(torch_transformer.enc_out_bbox_embed[i].layers, keras_transformer.enc_out_bbox_embed[i].layers_list)):
                      klayer.kernel.assign(to_keras(tk.weight.T.numpy()))
                      klayer.bias.assign(to_keras(tk.bias.numpy()))
        
        # Transfer decoder layer weights
        for i in range(num_decoder_layers):
            t_layer = torch_transformer.decoder.layers[i]
            k_layer = keras_transformer.decoder.layers_list[i]
            
            def transfer_mha(t_mha, k_mha):
                """Transfer MultiHeadAttention weights from reference to Keras."""
                q_w = to_keras(t_mha.in_proj_weight[:d_model, :].T.numpy())
                q_w = ops.reshape(q_w, (d_model, sa_nhead, d_model // sa_nhead))
                k_mha.query_dense.kernel.assign(q_w)
                q_b = to_keras(t_mha.in_proj_bias[:d_model].numpy())
                q_b = ops.reshape(q_b, (sa_nhead, d_model // sa_nhead))
                k_mha.query_dense.bias.assign(q_b)
                
                k_w = to_keras(t_mha.in_proj_weight[d_model:2*d_model, :].T.numpy())
                k_w = ops.reshape(k_w, (d_model, sa_nhead, d_model // sa_nhead))
                k_mha.key_dense.kernel.assign(k_w)
                k_b = to_keras(t_mha.in_proj_bias[d_model:2*d_model].numpy())
                k_b = ops.reshape(k_b, (sa_nhead, d_model // sa_nhead))
                k_mha.key_dense.bias.assign(k_b)
                
                v_w = to_keras(t_mha.in_proj_weight[2*d_model:, :].T.numpy())
                v_w = ops.reshape(v_w, (d_model, sa_nhead, d_model // sa_nhead))
                k_mha.value_dense.kernel.assign(v_w)
                v_b = to_keras(t_mha.in_proj_bias[2*d_model:].numpy())
                v_b = ops.reshape(v_b, (sa_nhead, d_model // sa_nhead))
                k_mha.value_dense.bias.assign(v_b)
                
                out_w = to_keras(t_mha.out_proj.weight.T.numpy())
                out_w = ops.reshape(out_w, (sa_nhead, d_model // sa_nhead, d_model))
                k_mha.output_dense.kernel.assign(out_w)
                k_mha.output_dense.bias.assign(to_keras(t_mha.out_proj.bias.numpy()))

            transfer_mha(t_layer.self_attn, k_layer.self_attn)
            k_layer.norm1.gamma.assign(to_keras(t_layer.norm1.weight.numpy()))
            k_layer.norm1.beta.assign(to_keras(t_layer.norm1.bias.numpy()))
            k_layer.cross_attn.sampling_offsets.kernel.assign(to_keras(t_layer.cross_attn.sampling_offsets.weight.T.numpy()))
            k_layer.cross_attn.sampling_offsets.bias.assign(to_keras(t_layer.cross_attn.sampling_offsets.bias.numpy()))
            k_layer.cross_attn.attention_weights.kernel.assign(to_keras(t_layer.cross_attn.attention_weights.weight.T.numpy()))
            k_layer.cross_attn.attention_weights.bias.assign(to_keras(t_layer.cross_attn.attention_weights.bias.numpy()))
            k_layer.cross_attn.value_proj.kernel.assign(to_keras(t_layer.cross_attn.value_proj.weight.T.numpy()))
            k_layer.cross_attn.value_proj.bias.assign(to_keras(t_layer.cross_attn.value_proj.bias.numpy()))
            k_layer.cross_attn.output_proj.kernel.assign(to_keras(t_layer.cross_attn.output_proj.weight.T.numpy()))
            k_layer.cross_attn.output_proj.bias.assign(to_keras(t_layer.cross_attn.output_proj.bias.numpy()))
            k_layer.norm2.gamma.assign(to_keras(t_layer.norm2.weight.numpy()))
            k_layer.norm2.beta.assign(to_keras(t_layer.norm2.bias.numpy()))
            k_layer.linear1.kernel.assign(to_keras(t_layer.linear1.weight.T.numpy()))
            k_layer.linear1.bias.assign(to_keras(t_layer.linear1.bias.numpy()))
            k_layer.linear2.kernel.assign(to_keras(t_layer.linear2.weight.T.numpy()))
            k_layer.linear2.bias.assign(to_keras(t_layer.linear2.bias.numpy()))
            k_layer.norm3.gamma.assign(to_keras(t_layer.norm3.weight.numpy()))
            k_layer.norm3.beta.assign(to_keras(t_layer.norm3.bias.numpy()))

        for j, (tk, klayer) in enumerate(zip(torch_transformer.decoder.ref_point_head.layers, keras_transformer.decoder.ref_point_head.layers_list)):
             klayer.kernel.assign(to_keras(tk.weight.T.numpy()))
             klayer.bias.assign(to_keras(tk.bias.numpy()))
             
        for j, (tk, klayer) in enumerate(zip(torch_transformer.decoder.bbox_embed.layers, keras_transformer.decoder.bbox_embed.layers_list)):
             klayer.kernel.assign(to_keras(tk.weight.T.numpy()))
             klayer.bias.assign(to_keras(tk.bias.numpy()))
             
        keras_transformer.decoder.norm.gamma.assign(to_keras(torch_transformer.decoder.norm.weight.numpy()))
        keras_transformer.decoder.norm.beta.assign(to_keras(torch_transformer.decoder.norm.bias.numpy()))
             
    out_hs_torch, out_ref_torch, out_mem_ts_torch, out_box_ts_torch = torch_transformer(
        t_srcs, t_masks, t_pos_embeds, t_refpoint_embed, t_query_feat
    )
    
    out_hs_keras, out_ref_keras, out_mem_ts_keras, out_box_ts_keras = keras_transformer(
        k_srcs, k_masks, k_pos_embeds, k_query_feat, k_refpoint_embed
    )
    
    diff_hs = np.abs(to_numpy(out_hs_torch) - to_numpy(out_hs_keras))
    print(f"HS Max diff: {diff_hs.max()}")
    assert np.allclose(to_numpy(out_hs_torch), to_numpy(out_hs_keras), atol=1e-5)
    
    diff_ref = np.abs(to_numpy(out_ref_torch) - to_numpy(out_ref_keras))
    print(f"Ref Max diff: {diff_ref.max()}")
    assert np.allclose(to_numpy(out_ref_torch), to_numpy(out_ref_keras), atol=1e-5)
    
    if two_stage:
        diff_mem = np.abs(to_numpy(out_mem_ts_torch) - to_numpy(out_mem_ts_keras))
        print(f"Mem TS Max diff: {diff_mem.max()}")
        assert np.allclose(to_numpy(out_mem_ts_torch), to_numpy(out_mem_ts_keras), atol=1e-5)
        
        diff_box = np.abs(to_numpy(out_box_ts_torch) - to_numpy(out_box_ts_keras))
        print(f"Box TS Max diff: {diff_box.max()}")
        assert np.allclose(to_numpy(out_box_ts_torch), to_numpy(out_box_ts_keras), atol=1e-5)
    else:
        pass

if __name__ == "__main__":
    pytest.main([__file__])
