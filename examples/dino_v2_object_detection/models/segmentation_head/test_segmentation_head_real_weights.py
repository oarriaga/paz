import os
import sys
import numpy as np
import pytest
import torch
import keras
from keras import ops

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# RFDETR imports
try:
    from rfdetr.detr import (
        RFDETRSegNano,
        RFDETRSegSmall,
        RFDETRSegMedium,
        RFDETRSegLarge,
        RFDETRSegXLarge,
        RFDETRSeg2XLarge,
        RFDETRSegPreview,
    )
except ImportError:
    rfdetr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../rf-detr_original_pytorch_implementation'))
    if rfdetr_path not in sys.path:
        sys.path.insert(0, rfdetr_path)
    from rfdetr.detr import (
        RFDETRSegNano,
        RFDETRSegSmall,
        RFDETRSegMedium,
        RFDETRSegLarge,
        RFDETRSegXLarge,
        RFDETRSeg2XLarge,
        RFDETRSegPreview,
    )

from examples.dino_v2_object_detection.models.segmentation_head.segmentation_head_keras import SegmentationHead as KerasSegmentationHead
from examples.dino_v2_object_detection.models.segmentation_head.segmentation_head_weights_porting_utils import (
    copy_segmentation_head,
    assert_allclose,
    to_numpy,
)

MODEL_VARIANTS = {
    "Nano": RFDETRSegNano,
    "Small": RFDETRSegSmall,
    # "Medium": RFDETRSegMedium,
    # "Large": RFDETRSegLarge,
    # "XLarge": RFDETRSegXLarge, # Commented out to save time/memory if needed, but requested by user
    # "2XLarge": RFDETRSeg2XLarge, # Commented out to save time/memory if needed, but requested by user
    "Preview": RFDETRSegPreview,
}

# Add larger models if environment permits
MODEL_VARIANTS["XLarge"] = RFDETRSegXLarge
MODEL_VARIANTS["2XLarge"] = RFDETRSeg2XLarge

def extract_pt_segmentation_head(model_class):
    print(f"Loading pretrained {model_class.__name__}...")
    # Instantiate PyTorch model with pretrained weights and eval mode
    model = model_class(pretrained=True) 
    
    pt_wrapper = model
    pt_model = pt_wrapper.model.model # Wrapper -> Model -> LWDETR
    
    pt_head = pt_model.segmentation_head
    pt_head.eval()
    return pt_head, pt_wrapper.model_config

@pytest.mark.parametrize("variant_name", list(MODEL_VARIANTS.keys()))
def test_segmentation_head_real_weights(variant_name):
    print(f"\n{'='*60}")
    print(f"Testing SegmentationHead parity for RFDETR {variant_name}")
    print(f"{'='*60}")
    
    model_cls = MODEL_VARIANTS[variant_name]
    
    # 1. Load PyTorch Head
    pt_head, config = extract_pt_segmentation_head(model_cls)
    pt_head = pt_head.double() # Run PyTorch in float64
    
    # 2. Configure Keras Head
    # Extract config from PyTorch config object
    hidden_dim = config.hidden_dim
    dec_layers = config.dec_layers
    mask_downsample_ratio = config.mask_downsample_ratio
    
    bottleneck_ratio = 1 
    
    print(f"Configuration: hidden_dim={hidden_dim}, dec_layers={dec_layers}, downsample={mask_downsample_ratio}")
    
    keras_head = KerasSegmentationHead(
        hidden_dim, 
        dec_layers, 
        bottleneck_ratio=bottleneck_ratio, 
        downsample_ratio=mask_downsample_ratio
    )
    
    # 3. Build Keras Head
    image_size = (config.resolution, config.resolution)
    spatial_shape = (1, hidden_dim, image_size[0] // 32, image_size[1] // 32) # Approx feature map size (P5)

    
    # For build, precise shape doesn't matter as long as channel dim matches.
    spatial_dummy = ops.ones(spatial_shape)
    qf_dummy = [ops.ones((1, 10, hidden_dim)) for _ in range(dec_layers)]
    
    keras_head(spatial_dummy, qf_dummy, image_size=image_size)
    
    # 4. Copy Weights
    print("Copying weights...")
    copy_segmentation_head(pt_head, keras_head)
    
    # Define inputs for verification
    spatial_np = np.random.randn(*spatial_shape).astype(np.float32)
    qf_np = [np.random.randn(1, 10, hidden_dim).astype(np.float32) for _ in range(dec_layers)]

    print("\n--- Verifying Full Head ---")

    with torch.no_grad():
        pt_qfs = [torch.from_numpy(q).double() for q in qf_np]
        pt_out = pt_head(torch.from_numpy(spatial_np).double(), pt_qfs, image_size)
        
    keras_out = keras_head(spatial_np, qf_np, image_size=image_size)
    
    assert_allclose(pt_out, keras_out, atol=5e-4, rtol=1e-4)
    print(f"RFDETR {variant_name} SegmentationHead Verification PASSED!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Allow running specific variant: python test... Large
        variant = sys.argv[1]
        if variant in MODEL_VARIANTS:
            test_segmentation_head_real_weights(variant)
        else:
            print(f"Unknown variant {variant}. Available: {list(MODEL_VARIANTS.keys())}")
    else:
        # Run all via pytest main if called directly without args, or let pytest handle discovery
        pytest.main([__file__, "-v", "-s"])
