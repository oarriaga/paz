import pytest
import torch
import numpy as np
import keras
from keras import ops
import sys
import os

# Import Keras implementation
# Assumes ms_deform_attn.py is in the same directory
try:
    from ms_deform_attn import MSDeformAttn as KerasMSDeformAttn
except ImportError:
    from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.ms_deform_attn import MSDeformAttn as KerasMSDeformAttn

# Import RFDETR variants
# We assume the environment has rfdetr installed or path is set correctly
try:
    from rfdetr import (
        RFDETRSmall,
        RFDETRMedium,
        RFDETRNano,
        RFDETRLarge,
    )
except ImportError:
    # Fallback to local path if package not installed
    # Calculate relevant path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # d:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\paz\examples\dino_v2_object_detection\models\transformer_decoder_head
    rfdetr_root = os.path.abspath(os.path.join(current_dir, '../../../../../../examples/rf-detr_original_pytorch_implementation'))
    sys.path.append(rfdetr_root)
    from rfdetr import (
        RFDETRSmall,
        RFDETRMedium,
        RFDETRNano,
        RFDETRLarge,
    )


from transformer_weights_porting_utils import (
    to_numpy,
    to_keras,
    get_ms_deform_attn_from_model
)

@pytest.mark.parametrize("model_class", [RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge])
def test_rfdetr_ms_deform_attn_real_weights(model_class):
    print(f"\nTesting MSDeformAttn parity for {model_class.__name__}...")
    
    # 1. Instantiate PyTorch model with pretrained weights and handle wrapper
    try:
        rfdetr_wrapper = model_class(pretrained=True)
        # RFDETR wrapper has .model (Model class) which has .model (nn.Module)
        if hasattr(rfdetr_wrapper, 'model') and hasattr(rfdetr_wrapper.model, 'model'):
            torch_full_model = rfdetr_wrapper.model.model
        else:
            torch_full_model = rfdetr_wrapper
            
        torch_full_model.eval()
    except Exception as e:
        pytest.fail(f"Failed to instantiate {model_class.__name__}: {e}")

    # 2. Locate MSDeformAttn module
    torch_attn = get_ms_deform_attn_from_model(torch_full_model)
    if torch_attn is None:
        pytest.fail(f"Could not locate MSDeformAttn in {model_class.__name__}")

    # 3. Extract Configuration
    d_model = torch_attn.d_model
    n_levels = torch_attn.n_levels
    n_heads = torch_attn.n_heads
    n_points = torch_attn.n_points
    
    print(f"Config: d_model={d_model}, n_levels={n_levels}, n_heads={n_heads}, n_points={n_points}")

    # 4. Instantiate Keras Layer
    keras_attn = KerasMSDeformAttn(
        d_model=d_model,
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points
    )

    # 5. Build Keras Layer (by running a dummy pass or calling build)
    # We need dummy input shapes
    batch_size = 1
    Len_q = 10
    Len_in = 20
    
    # Create dummy inputs
    query_np = np.random.randn(batch_size, Len_q, d_model).astype(np.float32)
    # Reference points (N, Lq, n_levels, 4) - typically 4D in detection
    ref_points_np = np.random.rand(batch_size, Len_q, n_levels, 4).astype(np.float32) 
    input_flatten_np = np.random.randn(batch_size, Len_in, d_model).astype(np.float32)
    
    # Mock spatial shapes for Len_in=20
    # e.g. 5x4 = 20
    input_spatial_shapes_np = np.array([[5, 4]], dtype=np.int32)
    # If using multiple levels, this needs to match n_levels
    if n_levels > 1:
        # We need total length to match Len_in
        # Let's start with a simpler test or adjust Len_in
        # For n_levels=4, we need 4 shapes summing to Len_in
        # Let's adjust Len_in dynamically
        shapes = []
        for i in range(n_levels):
            shapes.append([5, 4]) # 20 per level
        input_spatial_shapes_np = np.array(shapes, dtype=np.int32)
        Len_in = 20 * n_levels
        input_flatten_np = np.random.randn(batch_size, Len_in, d_model).astype(np.float32)
        
    print(f"Test Input Sizes: Len_q={Len_q}, Len_in={Len_in}")

    # Build layer
    _ = keras_attn(to_keras(query_np), to_keras(ref_points_np), to_keras(input_flatten_np), input_spatial_shapes_np)

    # 6. Port Weights
    with torch.no_grad():
        # sampling_offsets: Linear (in, out) -> Keras (in, out) - Transpose!
        keras_attn.sampling_offsets.kernel.assign(to_keras(torch_attn.sampling_offsets.weight.T.numpy()))
        keras_attn.sampling_offsets.bias.assign(to_keras(torch_attn.sampling_offsets.bias.numpy()))
        
        # attention_weights
        keras_attn.attention_weights.kernel.assign(to_keras(torch_attn.attention_weights.weight.T.numpy()))
        keras_attn.attention_weights.bias.assign(to_keras(torch_attn.attention_weights.bias.numpy()))
        
        # value_proj
        keras_attn.value_proj.kernel.assign(to_keras(torch_attn.value_proj.weight.T.numpy()))
        keras_attn.value_proj.bias.assign(to_keras(torch_attn.value_proj.bias.numpy()))
        
        # output_proj
        keras_attn.output_proj.kernel.assign(to_keras(torch_attn.output_proj.weight.T.numpy()))
        keras_attn.output_proj.bias.assign(to_keras(torch_attn.output_proj.bias.numpy()))

    # 7. Run Forward Comparison (CPU)
    torch_attn = torch_attn.cpu()
    t_query = torch.from_numpy(query_np)
    t_ref_points = torch.from_numpy(ref_points_np)
    t_input_flatten = torch.from_numpy(input_flatten_np)
    t_spatial_shapes = torch.from_numpy(input_spatial_shapes_np).long()
    
    # Calculate level_start_index for PyTorch
    # [0, H0*W0, H0*W0+H1*W1, ...]
    lens = t_spatial_shapes[:, 0] * t_spatial_shapes[:, 1]
    level_start_index = torch.cat((torch.tensor([0]), torch.cumsum(lens, 0)[:-1]))
    t_level_start_index = level_start_index.long()
    
    with torch.no_grad():
        out_torch = torch_attn(t_query, t_ref_points, t_input_flatten, t_spatial_shapes, t_level_start_index)
        
    out_keras = keras_attn(to_keras(query_np), to_keras(ref_points_np), to_keras(input_flatten_np), input_spatial_shapes_np)
    
    # Check parity
    diff = np.abs(to_numpy(out_torch) - to_numpy(out_keras))
    print(f"Max diff for {model_class.__name__}: {diff.max()}")
    
    # Assertion
    assert np.allclose(to_numpy(out_torch), to_numpy(out_keras), atol=1e-5, rtol=1e-5), \
        f"Mismatch for {model_class.__name__}! Max diff: {diff.max()}"

if __name__ == "__main__":
    pytest.main([__file__])
