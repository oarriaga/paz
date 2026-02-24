import os
import sys
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import keras
from keras import ops

# Add path for rfdetr to import original implementation
current_dir = os.path.dirname(os.path.abspath(__file__))
rf_detr_root = os.path.abspath(os.path.join(current_dir, '../../../rf-detr_original_pytorch_implementation'))
sys.path.append(rf_detr_root)

try:
    from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn as TorchMSDeformAttn
    from rfdetr.models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
except ImportError:
    pass

# Import our Keras implementation
from ms_deform_attn import MSDeformAttn as KerasMSDeformAttn
from ms_deform_attn import grid_sample as keras_grid_sample
from ms_deform_attn import ms_deform_attn_core as keras_ms_deform_attn_core

from transformer_weights_porting_utils import (
    to_numpy,
    to_torch,
    to_keras
)

@pytest.mark.parametrize("align_corners", [False, True])
def test_grid_samplepy(align_corners):
    N, C, H, W = 2, 4, 8, 8
    H_out, W_out = 4, 4
    
    input_np = np.random.randn(N, C, H, W).astype(np.float32)
    grid_np = np.random.uniform(-1.5, 1.5, size=(N, H_out, W_out, 2)).astype(np.float32)
    
    # Torch
    input_torch = torch.tensor(input_np)
    grid_torch = torch.tensor(grid_np)
    out_torch = F.grid_sample(input_torch, grid_torch, align_corners=align_corners, padding_mode="zeros", mode="bilinear")
    
    # Keras
    input_keras = ops.convert_to_tensor(input_np)
    grid_keras = ops.convert_to_tensor(grid_np)
    
    out_keras = keras_grid_sample(input_keras, grid_keras, align_corners=align_corners)
    
    # Check
    diff = np.abs(to_numpy(out_torch) - to_numpy(out_keras))
    print(f"Grid Sample Max diff (align_corners={align_corners}): {diff.max()}")
    assert np.allclose(to_numpy(out_torch), to_numpy(out_keras), atol=1e-5)

@pytest.mark.parametrize("ref_points_dim", [2, 4])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_ms_deform_attn_full_parity(ref_points_dim, batch_size):
    N = batch_size
    Len_q, n_heads, n_levels, n_points, d_model = 10, 4, 2, 4, 16
    Len_in_list = [20, 10]
    total_Len_in = sum(Len_in_list)
    spatial_shapes = [(5, 4), (2, 5)]
    
    query_np = np.random.randn(N, Len_q, d_model).astype(np.float32)
    if ref_points_dim == 2:
        ref_points_np = np.random.rand(N, Len_q, n_levels, 2).astype(np.float32)
    else:
        ref_points_np = np.random.rand(N, Len_q, n_levels, 4).astype(np.float32)
    
    input_flatten_np = np.random.randn(N, total_Len_in, d_model).astype(np.float32)
    input_spatial_shapes_np = np.array(spatial_shapes, dtype=np.int32)
    
    torch_model = TorchMSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
    torch_model.eval()
    
    keras_model = KerasMSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
    
    # Init Keras model
    _ = keras_model(to_keras(query_np), to_keras(ref_points_np), to_keras(input_flatten_np), input_spatial_shapes_np)
    
    # Transfer Weights
    with torch.no_grad():
        keras_model.sampling_offsets.kernel.assign(to_keras(torch_model.sampling_offsets.weight.T.numpy()))
        keras_model.sampling_offsets.bias.assign(to_keras(torch_model.sampling_offsets.bias.numpy()))
        
        keras_model.attention_weights.kernel.assign(to_keras(torch_model.attention_weights.weight.T.numpy()))
        keras_model.attention_weights.bias.assign(to_keras(torch_model.attention_weights.bias.numpy()))
        
        keras_model.value_proj.kernel.assign(to_keras(torch_model.value_proj.weight.T.numpy()))
        keras_model.value_proj.bias.assign(to_keras(torch_model.value_proj.bias.numpy()))
        
        keras_model.output_proj.kernel.assign(to_keras(torch_model.output_proj.weight.T.numpy()))
        keras_model.output_proj.bias.assign(to_keras(torch_model.output_proj.bias.numpy()))

    # Inputs
    t_query = to_torch(query_np)
    t_ref_points = to_torch(ref_points_np)
    t_input_flatten = to_torch(input_flatten_np)
    t_spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long)
    t_level_start_index = torch.cat((torch.tensor([0], dtype=torch.long), torch.cumsum(t_spatial_shapes[:, 0] * t_spatial_shapes[:, 1], 0)[:-1]))
    
    with torch.no_grad():
        out_torch = torch_model(t_query, t_ref_points, t_input_flatten, t_spatial_shapes, t_level_start_index)
    
    out_keras = keras_model(to_keras(query_np), to_keras(ref_points_np), to_keras(input_flatten_np), input_spatial_shapes_np)
    
    diff = np.abs(to_numpy(out_torch) - to_numpy(out_keras))
    print(f"Max diff: {diff.max()}")
    assert np.allclose(to_numpy(out_torch), to_numpy(out_keras), atol=1e-5)


@pytest.mark.parametrize("use_padding_mask", [False, True])
@pytest.mark.parametrize("n_levels", [1, 3])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("n_points", [2, 4])
@pytest.mark.parametrize("d_model", [64]) # Divisible by 4
def test_ms_deform_attn_enhanced(use_padding_mask, n_levels, n_heads, n_points, d_model):
    batch_size = 2
    Len_q = 8
    spatial_shapes = []
    total_Len_in = 0
    # Generate spatial shapes
    for i in range(n_levels):
        h, w = 4 * (i + 1), 4 * (i + 1)
        spatial_shapes.append((h, w))
        total_Len_in += h * w
    
    input_spatial_shapes_np = np.array(spatial_shapes, dtype=np.int32)
    
    query_np = np.random.randn(batch_size, Len_q, d_model).astype(np.float32)
    # Use 4D reference points for broader coverage
    ref_points_np = np.random.rand(batch_size, Len_q, n_levels, 4).astype(np.float32)
    input_flatten_np = np.random.randn(batch_size, total_Len_in, d_model).astype(np.float32)
    
    # Padding mask
    if use_padding_mask:
        # Create random boolean mask
        input_padding_mask_np = np.random.choice([False, True], size=(batch_size, total_Len_in), p=[0.9, 0.1])
        # PyTorch defaults: True for padding/ignored, False for valid
    else:
        input_padding_mask_np = None

    # Torch Model
    torch_model = TorchMSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
    torch_model.eval()
    
    # Keras Model
    keras_model = KerasMSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
    
    # Init Keras (call build implicitly)
    _ = keras_model(to_keras(query_np), to_keras(ref_points_np), to_keras(input_flatten_np), input_spatial_shapes_np, input_padding_mask=to_keras(input_padding_mask_np) if input_padding_mask_np is not None else None)
    
    # Transfer Weights
    with torch.no_grad():
        keras_model.sampling_offsets.kernel.assign(to_keras(torch_model.sampling_offsets.weight.T.numpy()))
        keras_model.sampling_offsets.bias.assign(to_keras(torch_model.sampling_offsets.bias.numpy()))
        keras_model.attention_weights.kernel.assign(to_keras(torch_model.attention_weights.weight.T.numpy()))
        keras_model.attention_weights.bias.assign(to_keras(torch_model.attention_weights.bias.numpy()))
        keras_model.value_proj.kernel.assign(to_keras(torch_model.value_proj.weight.T.numpy()))
        keras_model.value_proj.bias.assign(to_keras(torch_model.value_proj.bias.numpy()))
        keras_model.output_proj.kernel.assign(to_keras(torch_model.output_proj.weight.T.numpy()))
        keras_model.output_proj.bias.assign(to_keras(torch_model.output_proj.bias.numpy()))

    # Torch Forward
    t_query = to_torch(query_np)
    t_ref_points = to_torch(ref_points_np)
    t_input_flatten = to_torch(input_flatten_np)
    t_spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long)
    t_level_start_index = torch.cat((torch.tensor([0], dtype=torch.long), torch.cumsum(t_spatial_shapes[:, 0] * t_spatial_shapes[:, 1], 0)[:-1]))
    t_input_padding_mask = torch.tensor(input_padding_mask_np, dtype=torch.bool) if input_padding_mask_np is not None else None
    
    with torch.no_grad():
        out_torch = torch_model(t_query, t_ref_points, t_input_flatten, t_spatial_shapes, t_level_start_index, input_padding_mask=t_input_padding_mask)
        
    # Keras Forward
    k_input_padding_mask = to_keras(input_padding_mask_np) if input_padding_mask_np is not None else None
    out_keras = keras_model(to_keras(query_np), to_keras(ref_points_np), to_keras(input_flatten_np), input_spatial_shapes_np, input_padding_mask=k_input_padding_mask)
    
    diff = np.abs(to_numpy(out_torch) - to_numpy(out_keras))
    print(f"Max diff: {diff.max()}")
    assert np.allclose(to_numpy(out_torch), to_numpy(out_keras), atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])
