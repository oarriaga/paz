import pytest
import numpy as np
import torch
import keras
import os
import sys

# -------------------------------------------------------------------------
# 0. Environment Setup
# -------------------------------------------------------------------------
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
sys.path.append(project_root)

print(f"Project Root: {project_root}")

# -------------------------------------------------------------------------
# 1. Imports
# -------------------------------------------------------------------------
# PyTorch Implementation
from examples.dino_object_detection.models.transformer_decoder_head.torch_transformer_for_testing import (
    TransformerDecoder as TransformerDecoder_PyTorch,
    TransformerDecoderLayer as TransformerDecoderLayer_PyTorch,
    MLP as MLP_PyTorch,
    MSDeformAttn as MSDeformAttn_PyTorch,
)

# Keras 3 Implementation
from examples.dino_object_detection.models.transformer_decoder_head.transformer_decoder import (
    TransformerDecoder as TransformerDecoder_Keras,
)
from examples.dino_object_detection.models.transformer_decoder_head.transformer_decoder_layer import (
    TransformerDecoderLayer as TransformerDecoderLayer_Keras,
)
from examples.dino_object_detection.models.transformer_decoder_head.MLP import (
    MLP as MLP_Keras,
)


# -------------------------------------------------------------------------
# 2. Helper Functions
# -------------------------------------------------------------------------


def to_keras_mha_input_kernel(w, n_head):
    """Reshapes PyTorch (Out, In) -> Keras (In, Heads, Head_Dim)"""
    w = w.T
    d_model = w.shape[0]
    head_dim = w.shape[1] // n_head
    return w.reshape(d_model, n_head, head_dim)


def to_keras_mha_bias(b, n_head):
    """Reshapes PyTorch (Out,) -> Keras (Heads, Head_Dim)"""
    head_dim = b.shape[0] // n_head
    return b.reshape(n_head, head_dim)


def to_keras_mha_output_kernel(w, n_head):
    """Reshapes PyTorch OutProj (Out, In) -> Keras (Heads, Head_Dim, Out)"""
    w = w.T
    d_model_out = w.shape[1]
    head_dim = w.shape[0] // n_head
    return w.reshape(n_head, head_dim, d_model_out)


def transfer_decoder_layer_weights(pt_layer, k_layer, d_model, sa_nhead, ca_nhead):
    """Transfers weights from a PyTorch TransformerDecoderLayer to a Keras one."""
    with torch.no_grad():
        # 1. Self Attention (MultiHeadAttention)
        pt_sa = pt_layer.self_attn
        qkv_w = pt_sa.in_proj_weight.detach().numpy()
        qkv_b = pt_sa.in_proj_bias.detach().numpy()
        q_w, k_w, v_w = np.split(qkv_w, 3, axis=0)
        q_b, k_b, v_b = np.split(qkv_b, 3, axis=0)
        out_w = pt_sa.out_proj.weight.detach().numpy()
        out_b = pt_sa.out_proj.bias.detach().numpy()

        q_w_k = to_keras_mha_input_kernel(q_w, sa_nhead)
        k_w_k = to_keras_mha_input_kernel(k_w, sa_nhead)
        v_w_k = to_keras_mha_input_kernel(v_w, sa_nhead)
        q_b_k = to_keras_mha_bias(q_b, sa_nhead)
        k_b_k = to_keras_mha_bias(k_b, sa_nhead)
        v_b_k = to_keras_mha_bias(v_b, sa_nhead)
        out_w_k = to_keras_mha_output_kernel(out_w, sa_nhead)
        out_b_k = out_b
        k_layer.self_attn.set_weights(
            [q_w_k, q_b_k, k_w_k, k_b_k, v_w_k, v_b_k, out_w_k, out_b_k]
        )

        # 2. Norms
        k_layer.norm1.set_weights(
            [
                pt_layer.norm1.weight.detach().numpy(),
                pt_layer.norm1.bias.detach().numpy(),
            ]
        )
        k_layer.norm2.set_weights(
            [
                pt_layer.norm2.weight.detach().numpy(),
                pt_layer.norm2.bias.detach().numpy(),
            ]
        )
        k_layer.norm3.set_weights(
            [
                pt_layer.norm3.weight.detach().numpy(),
                pt_layer.norm3.bias.detach().numpy(),
            ]
        )

        # 3. Cross Attention (MSDeformAttn)
        pt_ca = pt_layer.cross_attn
        k_ca = k_layer.cross_attn
        k_ca.value_proj.set_weights(
            [
                pt_ca.value_proj.weight.detach().numpy().T,
                pt_ca.value_proj.bias.detach().numpy(),
            ]
        )
        k_ca.output_proj.set_weights(
            [
                pt_ca.output_proj.weight.detach().numpy().T,
                pt_ca.output_proj.bias.detach().numpy(),
            ]
        )
        k_ca.sampling_offsets.set_weights(
            [
                pt_ca.sampling_offsets.weight.detach().numpy().T,
                pt_ca.sampling_offsets.bias.detach().numpy(),
            ]
        )
        k_ca.attention_weights.set_weights(
            [
                pt_ca.attention_weights.weight.detach().numpy().T,
                pt_ca.attention_weights.bias.detach().numpy(),
            ]
        )

        # 4. Feed Forward Network
        k_layer.linear1.set_weights(
            [
                pt_layer.linear1.weight.detach().numpy().T,
                pt_layer.linear1.bias.detach().numpy(),
            ]
        )
        k_layer.linear2.set_weights(
            [
                pt_layer.linear2.weight.detach().numpy().T,
                pt_layer.linear2.bias.detach().numpy(),
            ]
        )


def transfer_mlp_weights(pt_mlp, k_mlp):
    """Transfers weights from a PyTorch MLP to a Keras MLP."""
    with torch.no_grad():
        for i in range(pt_mlp.num_layers):
            pt_w = pt_mlp.layers[i].weight.detach().numpy()
            pt_b = pt_mlp.layers[i].bias.detach().numpy()
            k_mlp.mlp_layers.layers[i].set_weights([pt_w.T, pt_b])


# -------------------------------------------------------------------------
# 3. The PyTest Case
# -------------------------------------------------------------------------
def test_transformer_decoder_equivalence():
    # --- Configuration ---
    d_model = 256
    num_layers = 3
    sa_nhead = 8
    ca_nhead = 8
    dim_feedforward = 1024
    dropout = 0.0
    activation = "relu"
    num_feature_levels = 3
    dec_n_points = 4
    return_intermediate = True
    lite_refpoint_refine = False
    bbox_reparam = False

    batch_size = 2
    num_queries = 10

    # Define spatial shapes for memory
    spatial_shapes_list = [(8, 8), (4, 4), (2, 2)]
    total_len_in = sum(h * w for h, w in spatial_shapes_list)

    # --- Prepare Inputs ---
    tgt_np = np.random.randn(batch_size, num_queries, d_model).astype("float32")
    memory_np = np.random.randn(batch_size, total_len_in, d_model).astype("float32")
    refpoints_unsigmoid_np = np.random.randn(batch_size, num_queries, 4).astype(
        "float32"
    )

    spatial_shapes_np = np.array(spatial_shapes_list, dtype="int32")
    level_start_index_np = np.concatenate(
        [[0], np.cumsum([h * w for h, w in spatial_shapes_list])[:-1]]
    ).astype("int32")

    valid_ratios_np = (
        np.ones((batch_size, num_feature_levels, 2), dtype="float32") * 0.99
    )

    tgt_pt = torch.from_numpy(tgt_np)
    memory_pt = torch.from_numpy(memory_np)
    refpoints_unsigmoid_pt = torch.from_numpy(refpoints_unsigmoid_np)
    spatial_shapes_pt = torch.from_numpy(spatial_shapes_np).long()
    level_start_index_pt = torch.from_numpy(level_start_index_np).long()
    valid_ratios_pt = torch.from_numpy(valid_ratios_np)

    # --- Instantiate Layer Models (Templates) ---
    pt_layer = TransformerDecoderLayer_PyTorch(
        d_model=d_model,
        sa_nhead=sa_nhead,
        ca_nhead=ca_nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
    )
    k_layer = TransformerDecoderLayer_Keras(
        d_model=d_model,
        sa_nhead=sa_nhead,
        ca_nhead=ca_nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
        skip_self_attn=False,
    )

    # --- Assign bbox_embed weights BEFORE model creation ---
    if not lite_refpoint_refine:
        bbox_embed_pt = MLP_PyTorch(d_model, d_model, 4, 3)
        pt_layer.bbox_embed = bbox_embed_pt

        bbox_embed_k = MLP_Keras(d_model, d_model, 4, 3)
        k_layer.bbox_embed = bbox_embed_k

        # Build Keras bbox_embed to init variables
        bbox_embed_k.build((None, d_model))

    # --- Instantiate Decoder Models ---
    pt_model = TransformerDecoder_PyTorch(
        pt_layer,
        num_layers,
        norm=torch.nn.LayerNorm(d_model),
        return_intermediate=return_intermediate,
        d_model=d_model,
        lite_refpoint_refine=lite_refpoint_refine,
        bbox_reparam=bbox_reparam,
    )
    if not lite_refpoint_refine:
        pt_model.bbox_embed = bbox_embed_pt

    k_model = TransformerDecoder_Keras(
        k_layer,
        num_layers,
        norm=keras.layers.LayerNormalization(epsilon=1e-5),
        return_intermediate=return_intermediate,
        d_model=d_model,
        lite_refpoint_refine=lite_refpoint_refine,
        bbox_reparam=bbox_reparam,
    )
    pt_model.eval()

    # --- Build Keras Model (Dummy Call) ---
    _ = k_model(
        tgt=tgt_np,
        memory=memory_np,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        refpoints_unsigmoid=refpoints_unsigmoid_np,
        level_start_index=level_start_index_np,
        spatial_shapes=spatial_shapes_np,
        valid_ratios=valid_ratios_np,
        training=False,
    )

    # --- Weight Transfer (PyTorch -> Keras) ---

    # 1. Transfer MLP (ref_point_head)
    transfer_mlp_weights(pt_model.ref_point_head, k_model.ref_point_head)

    # 2. Transfer Decoder Layers (cloned modules)
    for i in range(num_layers):
        pt_dec_layer = pt_model.layers[i]
        k_dec_layer = k_model.decoder_layers[i]

        transfer_decoder_layer_weights(
            pt_dec_layer, k_dec_layer, d_model, sa_nhead, ca_nhead
        )

        # 3. Transfer Layer Norm (if present)
        if k_model.norm is not None:
            k_model.norm.set_weights(
                [
                    pt_model.norm.weight.detach().numpy(),
                    pt_model.norm.bias.detach().numpy(),
                ]
            )

        # 4. Transfer bbox_embed (Shared or per-layer?)
        if not lite_refpoint_refine and i == 0:
            transfer_mlp_weights(pt_model.bbox_embed, k_model.bbox_embed)

    print("Weights transferred successfully.")

    # --- Run Forward Pass ---

    # 1. PyTorch Output
    with torch.no_grad():
        pt_out_tuple = pt_model(
            tgt=tgt_pt,
            memory=memory_pt,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            pos=None,
            refpoints_unsigmoid=refpoints_unsigmoid_pt,
            level_start_index=level_start_index_pt,
            spatial_shapes=spatial_shapes_pt,
            valid_ratios=valid_ratios_pt,
        )
        pt_hs = pt_out_tuple[0].detach().numpy()
        pt_ref = pt_out_tuple[1].detach().numpy()

    # 2. Keras Output
    k_out_tuple = k_model(
        tgt=tgt_np,
        memory=memory_np,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        refpoints_unsigmoid=refpoints_unsigmoid_np,
        level_start_index=level_start_index_np,
        spatial_shapes=spatial_shapes_np,
        valid_ratios=valid_ratios_np,
        training=False,
    )
    k_hs = np.array(k_out_tuple[0])
    k_ref = np.array(k_out_tuple[1])

    # --- Compare ---
    print(f"\nComparing Hidden States (hs):")
    diff_hs = np.abs(pt_hs - k_hs)
    max_diff_hs = np.max(diff_hs)
    print(f"Max Absolute Difference (hs): {max_diff_hs:.8f}")

    np.testing.assert_allclose(
        pt_hs,
        k_hs,
        rtol=1e-4,
        atol=1e-5,
        err_msg="TransformerDecoder Hidden States (hs) do not match!",
    )

    print(f"\nComparing Reference Points (ref):")
    diff_ref = np.abs(pt_ref - k_ref)
    max_diff_ref = np.max(diff_ref)
    print(f"Max Absolute Difference (ref): {max_diff_ref:.8f}")

    np.testing.assert_allclose(
        pt_ref,
        k_ref,
        rtol=1e-4,
        atol=1e-5,
        err_msg="TransformerDecoder Reference Points (ref) do not match!",
    )

    print("\nFINAL SUCCESS: The Keras TransformerDecoder matches PyTorch.")


if __name__ == "__main__":
    # test_transformer_decoder_equivalence()
    pytest.main(["-v", __file__])
