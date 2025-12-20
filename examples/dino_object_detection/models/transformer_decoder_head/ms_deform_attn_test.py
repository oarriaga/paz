import pytest
import numpy as np
import torch
import keras
import os
import sys

# -------------------------------------------------------------------------
# 0. Environment Setup (Same as MLP_test.py)
# -------------------------------------------------------------------------
os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
sys.path.append(project_root)

print(f"Project Root: {project_root}")

# -------------------------------------------------------------------------
# PyTorch Implementation
# -------------------------------------------------------------------------
from examples.dino_object_detection.models.transformer_decoder_head.torch_transformer_for_testing import (
    MSDeformAttn as MSDeformAttn_PyTorch,
)

# -------------------------------------------------------------------------
# Keras Implementation
# -------------------------------------------------------------------------
from examples.dino_object_detection.models.transformer_decoder_head.ms_deform_attn import (
    MSDeformAttn as MSDeformAttn_Keras,
)


# -------------------------------------------------------------------------
# 2. Test Logic
# -------------------------------------------------------------------------
def test_ms_deform_attn_equivalence():
    # --- Configuration ---
    N = 2  # Batch size
    d_model = 32  # Hidden dimension
    n_levels = 3  # Number of feature levels
    n_heads = 4  # Number of attention heads
    n_points = 4  # Number of sampling points
    Len_q = 5  # Query length

    # Define spatial shapes for the levels (H, W)
    spatial_shapes_list = [(8, 8), (4, 4), (2, 2)]

    # Calculate derived dimensions
    total_len_in = sum(h * w for h, w in spatial_shapes_list)

    # Prepare Spatial Shapes and Start Indices
    spatial_shapes_np = np.array(spatial_shapes_list, dtype="int32")
    level_start_index_np = np.concatenate(
        [[0], np.cumsum([h * w for h, w in spatial_shapes_list])[:-1]]
    )
    level_start_index_np = level_start_index_np.astype("int32")

    # Torch (for PyTorch model)
    spatial_shapes_pt = torch.from_numpy(spatial_shapes_np).long()
    level_start_index_pt = torch.from_numpy(level_start_index_np).long()

    # --- Instantiate Models ---
    pt_model = MSDeformAttn_PyTorch(d_model, n_levels, n_heads, n_points)
    k_model = MSDeformAttn_Keras(d_model, n_levels, n_heads, n_points)

    # --- Initialize Keras Model ---
    dummy_q = np.zeros((1, Len_q, d_model), dtype="float32")
    dummy_ref = np.zeros((1, Len_q, n_levels, 4), dtype="float32")  # 4D ref points
    dummy_in = np.zeros((1, total_len_in, d_model), dtype="float32")

    # Run one pass to initialize variables
    _ = k_model(dummy_q, dummy_ref, dummy_in, spatial_shapes_np, level_start_index_np)

    with torch.no_grad():
        # 1. Value Projection
        k_model.value_proj.set_weights(
            [
                pt_model.value_proj.weight.detach().numpy().T,
                pt_model.value_proj.bias.detach().numpy(),
            ]
        )

        # 2. Output Projection
        k_model.output_proj.set_weights(
            [
                pt_model.output_proj.weight.detach().numpy().T,
                pt_model.output_proj.bias.detach().numpy(),
            ]
        )

        # 3. Sampling Offsets
        k_model.sampling_offsets.set_weights(
            [
                pt_model.sampling_offsets.weight.detach().numpy().T,
                pt_model.sampling_offsets.bias.detach().numpy(),
            ]
        )

        # 4. Attention Weights
        k_model.attention_weights.set_weights(
            [
                pt_model.attention_weights.weight.detach().numpy().T,
                pt_model.attention_weights.bias.detach().numpy(),
            ]
        )

    print("Weights transferred successfully.")

    # --- Generate Random Inputs ---
    # Query
    query_np = np.random.randn(N, Len_q, d_model).astype("float32")

    # Reference Points
    ref_points_np = np.random.rand(N, Len_q, n_levels, 4).astype("float32")

    # Input Flatten (Memory)
    input_flatten_np = np.random.randn(N, total_len_in, d_model).astype("float32")

    # --- Run Forward Pass ---

    # 1. PyTorch Output
    pt_model.eval()
    pt_out_tensor = pt_model(
        query=torch.from_numpy(query_np),
        reference_points=torch.from_numpy(ref_points_np),
        input_flatten=torch.from_numpy(input_flatten_np),
        input_spatial_shapes=spatial_shapes_pt,
        input_level_start_index=level_start_index_pt,
        input_padding_mask=None,
    )
    pt_out = pt_out_tensor.detach().numpy()

    # 2. Keras Output
    k_out_tensor = k_model(
        query=query_np,
        reference_points=ref_points_np,
        input_flatten=input_flatten_np,
        input_spatial_shapes=spatial_shapes_np,
        input_level_start_index=level_start_index_np,
        input_padding_mask=None,
    )
    k_out = np.array(k_out_tensor)

    # --- Compare ---
    print(f"\nPyTorch Output Mean: {np.mean(pt_out):.6f}")
    print(f"Keras Output Mean:   {np.mean(k_out):.6f}")

    diff = np.abs(pt_out - k_out)
    print(f"Max Absolute Difference: {np.max(diff):.8f}")

    np.testing.assert_allclose(
        pt_out,
        k_out,
        rtol=1e-4,
        atol=1e-5,
        err_msg="MSDeformAttn Outputs do not match!",
    )

    print("\nSUCCESS: The Keras MSDeformAttn implementation matches PyTorch.")


if __name__ == "__main__":
    # test_ms_deform_attn_equivalence()
    pytest.main(["-v", __file__])
