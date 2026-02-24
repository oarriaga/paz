import os
import pytest
import torch
import numpy as np
import keras
from keras import ops

# Import PyTorch models
import sys
rfdetr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../rf-detr_original_pytorch_implementation'))
sys.path.insert(0, rfdetr_path)

try:
    from rfdetr.models.segmentation_head import SegmentationHead as PTSegmentationHead
    from rfdetr.models.segmentation_head import DepthwiseConvBlock as PTDepthwiseConvBlock
    from rfdetr.models.segmentation_head import MLPBlock as PTMLPBlock
    from rfdetr.models.segmentation_head import point_sample as pt_point_sample
    from rfdetr.models.segmentation_head import calculate_uncertainty as pt_calculate_uncertainty
    from rfdetr.models.segmentation_head import get_uncertain_point_coords_with_randomness as pt_get_uncertain_point_coords
except ImportError as e:
    print(f"Error importing rfdetr: {e}")
    # Continue only if testing Keras parts that don't depend on PT comparison immediately
    # but here we need PT for parity.
    pass

# Import Keras models
from examples.dino_v2_object_detection.models.segmentation_head.segmentation_head_keras import (
    SegmentationHead as KerasSegmentationHead,
    DepthwiseConvBlock as KerasDepthwiseConvBlock,
    MLPBlock as KerasMLPBlock,
    point_sample as keras_point_sample,
    calculate_uncertainty as keras_calculate_uncertainty,
    get_uncertain_point_coords_with_randomness as keras_get_uncertain_point_coords
)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    keras.utils.set_random_seed(seed)


from examples.dino_v2_object_detection.models.segmentation_head.segmentation_head_weights_porting_utils import (
    to_numpy,
    assert_allclose,
    copy_depthwise_conv_block,
    copy_mlp_block,
    copy_segmentation_head
)



# --- Tests ---

def test_depthwise_conv_block():
    set_seed()
    dim = 64
    pt_block = PTDepthwiseConvBlock(dim, layer_scale_init_value=1e-6)
    pt_block.eval()
    
    keras_block = KerasDepthwiseConvBlock(dim, layer_scale_init_value=1e-6)
    
    # Helper to build weights
    dummy_in = ops.ones((1, dim, 32, 32))
    keras_block(dummy_in)
    
    copy_depthwise_conv_block(pt_block, keras_block)
    
    # Input
    x = np.random.randn(2, dim, 32, 32).astype(np.float32)
    
    # Forward
    with torch.no_grad():
        pt_out = pt_block(torch.from_numpy(x))
    
    keras_out = keras_block(x)
    
    assert_allclose(pt_out, keras_out, atol=1e-5)
    print("DepthwiseConvBlock test passed!")

def test_mlp_block():
    set_seed()
    dim = 64
    pt_block = PTMLPBlock(dim, layer_scale_init_value=1e-6)
    pt_block.eval()
    
    keras_block = KerasMLPBlock(dim, layer_scale_init_value=1e-6)
    
    # Build
    dummy_in = ops.ones((1, 10, dim))
    keras_block(dummy_in)
    
    copy_mlp_block(pt_block, keras_block)
    
    x = np.random.randn(2, 5, dim).astype(np.float32)
    
    with torch.no_grad():
        pt_out = pt_block(torch.from_numpy(x))
        
    keras_out = keras_block(x)
    
    assert_allclose(pt_out, keras_out, atol=1e-5)
    print("MLPBlock test passed!")

def test_segmentation_head():
    set_seed()
    in_dim = 64
    num_blocks = 2
    
    pt_head = PTSegmentationHead(in_dim, num_blocks)
    pt_head.eval()
    
    keras_head = KerasSegmentationHead(in_dim, num_blocks)
    
    # Build
    spatial = ops.ones((1, in_dim, 32, 32))
    qf = ops.ones((1, 10, in_dim))
    keras_head(spatial, [qf]*num_blocks, image_size=[128, 128])
    
    copy_segmentation_head(pt_head, keras_head)
    
    # Inputs
    image_size = (128, 128)
    spatial_np = np.random.randn(2, in_dim, 32, 32).astype(np.float32)
    qf_np_list = [np.random.randn(2, 10, in_dim).astype(np.float32) for _ in range(num_blocks)]
    
    # PT Forward
    with torch.no_grad():
        pt_qfs = [torch.from_numpy(q) for q in qf_np_list]
        pt_out = pt_head(torch.from_numpy(spatial_np), pt_qfs, image_size)
    
    # Keras Forward
    keras_out = keras_head(spatial_np, qf_np_list, image_size=image_size)
    
    # Compare each output in list
    for p, k in zip(pt_out, keras_out):
        assert_allclose(p, k, atol=1e-4)
        
    print("SegmentationHead forward test passed!")

def test_point_sample():
    set_seed()
    N, C, H, W = 2, 3, 32, 32
    P = 100
    
    input_np = np.random.randn(N, C, H, W).astype(np.float32)
    point_coords_np = np.random.rand(N, P, 2).astype(np.float32) # [0, 1]
    
    with torch.no_grad():
        pt_out = pt_point_sample(torch.from_numpy(input_np), torch.from_numpy(point_coords_np), align_corners=False)
        
    keras_out = keras_point_sample(input_np, point_coords_np, align_corners=False)
    
    assert_allclose(pt_out, keras_out, atol=1e-5)
    print("point_sample test passed!")

def test_uncertainty():
    logits = np.random.randn(10, 1, 32, 32).astype(np.float32)
    
    pt_u = pt_calculate_uncertainty(torch.from_numpy(logits))
    keras_u = keras_calculate_uncertainty(logits)
    
    assert_allclose(pt_u, keras_u)
    print("calculate_uncertainty test passed!")

def test_get_uncertain_points():
    set_seed()
    N, C, H, W = 2, 1, 32, 32
    num_points = 50
    coarse_logits = np.random.randn(N, C, H, W).astype(np.float32)
    
    pts = keras_get_uncertain_point_coords(
        coarse_logits, 
        keras_calculate_uncertainty, 
        num_points
    )
    
    assert pts.shape == (N, num_points, 2)
    assert np.all(pts >= 0.0) and np.all(pts <= 1.0)
    print("get_uncertain_point_coords test passed (shape check)!")


def test_segmentation_head_bottleneck_none():
    print("Starting test_segmentation_head_bottleneck_none...")
    set_seed()
    in_dim = 32
    num_blocks = 1
    # bottleneck_ratio=None
    
    pt_head = PTSegmentationHead(in_dim, num_blocks, bottleneck_ratio=None)
    pt_head.eval()
    
    keras_head = KerasSegmentationHead(in_dim, num_blocks, bottleneck_ratio=None)
    
    # Build
    spatial = ops.ones((1, in_dim, 16, 16))
    qf = ops.ones((1, 1, in_dim))
    keras_head(spatial, [qf], image_size=[64, 64])
    
    copy_segmentation_head(pt_head, keras_head)
    
    image_size = (64, 64)
    spatial_np = np.random.randn(2, in_dim, 16, 16).astype(np.float32)
    qf_np = np.random.randn(2, 1, in_dim).astype(np.float32)
    
    with torch.no_grad():
        pt_out = pt_head(torch.from_numpy(spatial_np), [torch.from_numpy(qf_np)], image_size)
        
    keras_out = keras_head(spatial_np, [qf_np], image_size=image_size)
    
    for p, k in zip(pt_out, keras_out):
        assert_allclose(p, k, atol=1e-4)
    print("SegmentationHead bottleneck=None test passed!")

def test_segmentation_head_skip_blocks():
    print("Starting test_segmentation_head_skip_blocks...")
    set_seed()
    in_dim = 32
    num_blocks = 2
    
    pt_head = PTSegmentationHead(in_dim, num_blocks)
    pt_head.eval()
    
    keras_head = KerasSegmentationHead(in_dim, num_blocks)
    
    # Build with skip_blocks=False first to ensure blocks are built
    spatial = ops.ones((1, in_dim, 16, 16))
    qf = ops.ones((1, 1, in_dim))
    # We must call it once without skip_blocks to build the blocks
    keras_head(spatial, [qf]*num_blocks, image_size=[64, 64], skip_blocks=False)
    
    copy_segmentation_head(pt_head, keras_head)
    
    image_size = (64, 64)
    spatial_np = np.random.randn(2, in_dim, 16, 16).astype(np.float32)
    qf_np = np.random.randn(2, 1, in_dim).astype(np.float32)
    
    # skip_blocks=True requires len(query_features) == 1
    with torch.no_grad():
        pt_out = pt_head(torch.from_numpy(spatial_np), [torch.from_numpy(qf_np)], image_size, skip_blocks=True)
        
    keras_out = keras_head(spatial_np, [qf_np], image_size=image_size, skip_blocks=True)
    
    for p, k in zip(pt_out, keras_out):
        assert_allclose(p, k, atol=1e-4)
    print("SegmentationHead skip_blocks=True test passed!")

def test_segmentation_head_sparse():
    print("Starting test_segmentation_head_sparse...")
    set_seed()
    in_dim = 32
    num_blocks = 1
    
    pt_head = PTSegmentationHead(in_dim, num_blocks)
    pt_head.eval()
    
    keras_head = KerasSegmentationHead(in_dim, num_blocks)
    
    # Build
    spatial = ops.ones((1, in_dim, 16, 16))
    qf = ops.ones((1, 1, in_dim))
    keras_head.sparse_call(spatial, [qf], image_size=[64, 64])
    
    copy_segmentation_head(pt_head, keras_head)
    
    image_size = (64, 64)
    spatial_np = np.random.randn(2, in_dim, 16, 16).astype(np.float32)
    qf_np = np.random.randn(2, 1, in_dim).astype(np.float32)
    
    with torch.no_grad():
        pt_out_dicts = pt_head.sparse_forward(torch.from_numpy(spatial_np), [torch.from_numpy(qf_np)], image_size)
        
    keras_out_dicts = keras_head.sparse_call(spatial_np, [qf_np], image_size=image_size)
    
    assert len(pt_out_dicts) == len(keras_out_dicts)
    for pt_d, keras_d in zip(pt_out_dicts, keras_out_dicts):
        assert_allclose(pt_d["spatial_features"], keras_d["spatial_features"], atol=1e-4)
        assert_allclose(pt_d["query_features"], keras_d["query_features"], atol=1e-4)
        # bias is scalar or (1,)
        assert_allclose(pt_d["bias"], keras_d["bias"], atol=1e-6)
        
    print("SegmentationHead sparse_call test passed!")

if __name__ == "__main__":
    test_depthwise_conv_block()
    test_mlp_block()
    test_segmentation_head()
    test_segmentation_head_bottleneck_none()
    test_segmentation_head_skip_blocks()
    test_segmentation_head_sparse()
    test_point_sample()
    test_uncertainty()
    test_get_uncertain_points()

