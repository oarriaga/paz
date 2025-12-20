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
    TransformerDecoderLayer as TransformerDecoderLayer_PyTorch,
)

# Keras 3 Implementation
from examples.dino_object_detection.models.transformer_decoder_head.transformer_decoder_layer import (
    TransformerDecoderLayer as TransformerDecoderLayer_Keras,
)


# -------------------------------------------------------------------------
# 2. Test Logic
# -------------------------------------------------------------------------
def test_decoder_layer_equivalence():
    # --- Configuration ---
    d_model = 256
    sa_nhead = 8
    ca_nhead = 8
    dim_feedforward = 1024
    dropout = 0.0  # Disable dropout for deterministic testing
    activation = "relu"
    num_feature_levels = 4
    dec_n_points = 4

    batch_size = 2
    num_queries = 10

    # Total memory length must match sum of spatial_shapes areas
    num_memory = 100

    # --- Instantiate Models ---
    pt_model = TransformerDecoderLayer_PyTorch(
        d_model=d_model,
        sa_nhead=sa_nhead,
        ca_nhead=ca_nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
    )
    pt_model.eval()

    k_model = TransformerDecoderLayer_Keras(
        d_model=d_model,
        sa_nhead=sa_nhead,
        ca_nhead=ca_nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
    )

    # --- Prepare Dummy Inputs & Build Keras Model ---
    tgt_np = np.random.randn(batch_size, num_queries, d_model).astype("float32")
    memory_np = np.random.randn(batch_size, num_memory, d_model).astype("float32")
    query_pos_np = np.random.randn(batch_size, num_queries, d_model).astype("float32")

    # Reference points: (bs, nq, n_levels, 2)
    ref_points_np = np.random.rand(
        batch_size, num_queries, num_feature_levels, 2
    ).astype("float32")
    spatial_shapes_list = [(5, 5), (5, 5), (5, 5), (5, 5)]
    spatial_shapes_np = np.array(spatial_shapes_list, dtype="int32")

    level_start_index_np = np.concatenate(
        [[0], np.cumsum([h * w for h, w in spatial_shapes_list])[:-1]]
    ).astype("int32")

    # Run a dummy pass to build Keras model weights
    k_model(
        tgt=tgt_np,
        memory=memory_np,
        training=False,
        query_pos=query_pos_np,
        reference_points=ref_points_np,
        spatial_shapes=spatial_shapes_np,
        level_start_index=level_start_index_np,
    )

    # --- Weight Transfer (PyTorch -> Keras) ---

    with torch.no_grad():
        # 1. Self Attention (MultiHeadAttention)
        pt_sa = pt_model.self_attn

        # Retrieve PyTorch weights (Concatenated Q, K, V)
        qkv_w = pt_sa.in_proj_weight.detach().numpy()  # (3*d_model, d_model)
        qkv_b = pt_sa.in_proj_bias.detach().numpy()  # (3*d_model,)

        # Split into Q, K, V
        q_w, k_w, v_w = np.split(qkv_w, 3, axis=0)
        q_b, k_b, v_b = np.split(qkv_b, 3, axis=0)

        # Retrieve Output Projection
        out_w = pt_sa.out_proj.weight.detach().numpy()  # (d_model, d_model)
        out_b = pt_sa.out_proj.bias.detach().numpy()  # (d_model,)

        # --- Helper Functions for Reshaping ---

        def to_keras_mha_input_kernel(w, n_head):
            """Reshapes PyTorch (Out, In) -> Keras (In, Heads, Head_Dim)"""
            w = w.T  # Transpose to (In, Out) -> (256, 256)
            d_model = w.shape[0]
            head_dim = w.shape[1] // n_head
            # Reshape to (256, 8, 32)
            return w.reshape(d_model, n_head, head_dim)

        def to_keras_mha_bias(b, n_head):
            """Reshapes PyTorch (Out,) -> Keras (Heads, Head_Dim)"""
            head_dim = b.shape[0] // n_head
            # Reshape to (8, 32)
            return b.reshape(n_head, head_dim)

        def to_keras_mha_output_kernel(w, n_head):
            """Reshapes PyTorch OutProj (Out, In) -> Keras (Heads, Head_Dim, Out)"""
            w = w.T

            d_model_out = w.shape[1]
            head_dim = w.shape[0] // n_head
            return w.reshape(n_head, head_dim, d_model_out)

        # --- Apply Reshaping ---
        q_w_k = to_keras_mha_input_kernel(q_w, sa_nhead)
        k_w_k = to_keras_mha_input_kernel(k_w, sa_nhead)
        v_w_k = to_keras_mha_input_kernel(v_w, sa_nhead)

        q_b_k = to_keras_mha_bias(q_b, sa_nhead)
        k_b_k = to_keras_mha_bias(k_b, sa_nhead)
        v_b_k = to_keras_mha_bias(v_b, sa_nhead)

        out_w_k = to_keras_mha_output_kernel(out_w, sa_nhead)
        out_b_k = out_b  # Bias matches (256,)

        # Set Weights
        k_model.self_attn.set_weights(
            [q_w_k, q_b_k, k_w_k, k_b_k, v_w_k, v_b_k, out_w_k, out_b_k]
        )

        # 2. Norms
        k_model.norm1.set_weights(
            [
                pt_model.norm1.weight.detach().numpy(),
                pt_model.norm1.bias.detach().numpy(),
            ]
        )
        k_model.norm2.set_weights(
            [
                pt_model.norm2.weight.detach().numpy(),
                pt_model.norm2.bias.detach().numpy(),
            ]
        )
        k_model.norm3.set_weights(
            [
                pt_model.norm3.weight.detach().numpy(),
                pt_model.norm3.bias.detach().numpy(),
            ]
        )

        # 3. Cross Attention (MSDeformAttn)
        pt_ca = pt_model.cross_attn
        k_ca = k_model.cross_attn

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
        k_model.linear1.set_weights(
            [
                pt_model.linear1.weight.detach().numpy().T,
                pt_model.linear1.bias.detach().numpy(),
            ]
        )
        k_model.linear2.set_weights(
            [
                pt_model.linear2.weight.detach().numpy().T,
                pt_model.linear2.bias.detach().numpy(),
            ]
        )

    print("Weights transferred successfully.")

    # --- Run Comparison ---

    # Prepare Inputs (Convert to Tensor for PyTorch)
    tgt_pt = torch.from_numpy(tgt_np)
    memory_pt = torch.from_numpy(memory_np)
    query_pos_pt = torch.from_numpy(query_pos_np)
    ref_points_pt = torch.from_numpy(ref_points_np)
    spatial_shapes_pt = torch.from_numpy(spatial_shapes_np).long()
    level_start_index_pt = torch.from_numpy(level_start_index_np).long()

    # 1. PyTorch Forward
    pt_out = (
        pt_model(
            tgt=tgt_pt,
            memory=memory_pt,
            pos=None,
            query_pos=query_pos_pt,
            reference_points=ref_points_pt,
            spatial_shapes=spatial_shapes_pt,
            level_start_index=level_start_index_pt,
        )
        .detach()
        .numpy()
    )

    # 2. Keras Forward
    k_out = k_model(
        tgt=tgt_np,
        memory=memory_np,
        training=False,
        pos=None,
        query_pos=query_pos_np,
        reference_points=ref_points_np,
        spatial_shapes=spatial_shapes_np,
        level_start_index=level_start_index_np,
    )
    k_out = np.array(k_out)

    # --- Assertions ---
    print(f"\nPyTorch Output Mean: {np.mean(pt_out):.6f}")
    print(f"Keras Output Mean:   {np.mean(k_out):.6f}")

    diff = np.abs(pt_out - k_out)
    max_diff = np.max(diff)
    print(f"Max Absolute Difference: {max_diff:.8f}")

    np.testing.assert_allclose(
        pt_out,
        k_out,
        rtol=1e-4,
        atol=1e-5,
        err_msg="TransformerDecoderLayer outputs do not match!",
    )

    print("\nSUCCESS: The Keras TransformerDecoderLayer matches PyTorch.")


if __name__ == "__main__":
    # test_decoder_layer_equivalence()
    pytest.main(["-v", __file__])
