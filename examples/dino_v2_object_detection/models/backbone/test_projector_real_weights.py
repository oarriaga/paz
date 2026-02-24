import pytest
import torch
import numpy as np
import keras
import sys
import os

# Import Keras implementation
try:
    from projector import MultiScaleProjector as KerasMultiScaleProjector
    from projector_weights_porting_utils import port_weights_multiscale_projector
except ImportError:
    from examples.dino_v2_object_detection.models.backbone.projector import MultiScaleProjector as KerasMultiScaleProjector
    from examples.dino_v2_object_detection.models.backbone.projector_weights_porting_utils import port_weights_multiscale_projector

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
    sys.path.append(os.path.abspath(r"examples\rf-detr_original_pytorch_implementation"))
    from rfdetr import (
        RFDETRSmall,
        RFDETRMedium,
        RFDETRNano,
        RFDETRLarge,
    )

def to_numpy(t):
    return t.detach().cpu().numpy()

@pytest.mark.parametrize("model_class", [RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge])
def test_rfdetr_projector_parity(model_class):
    print(f"\nTesting parity for {model_class.__name__}...")
    
    # Instantiate PyTorch model with pretrained weights
    try:
        torch_full_model = model_class(pretrained=True)
    except Exception as e:
        pytest.fail(f"Failed to instantiate {model_class.__name__}: {e}")
        
    # RFDETR wrappers usually have the core model at .model.model
    # But let's be robust if structure varies (e.g. Nano might be different?)
    if hasattr(torch_full_model, 'model') and hasattr(torch_full_model.model, 'model'):
        inner_model = torch_full_model.model.model
    else:
        inner_model = torch_full_model
        
    inner_model.eval()
    
    # Locate projector
    if hasattr(inner_model, 'backbone'):
        backbone = inner_model.backbone
        # Handle Joiner
        if isinstance(backbone, (torch.nn.Sequential, torch.nn.ModuleList)) or (hasattr(backbone, '__getitem__') and not isinstance(backbone, torch.Tensor)):
            try:
                base_backbone = backbone[0]
            except:
                base_backbone = backbone
        else:
            base_backbone = backbone
            
        if hasattr(base_backbone, 'projector'):
            torch_projector = base_backbone.projector
        else:
             pytest.fail(f"Projector not found on base_backbone of {model_class.__name__}")
    else:
        pytest.fail(f"Backbone not found on inner model of {model_class.__name__}")

    print(f"Located Projector for {model_class.__name__}")

    # Deduce Configuration
    if hasattr(torch_projector, 'stages_sampling'):
        num_scales = len(torch_projector.stages_sampling)
        num_inputs = len(torch_projector.stages_sampling[0])
    elif hasattr(torch_projector, 'stages'): # Fallback inspection
        num_scales = len(torch_projector.stages)
        # Assume num_inputs based on standard backbone
        # Warning: this path might be brittle
        num_inputs = 3 # default guess
    else:
        pytest.fail("Unknown projector structure")

    scale_factors = list(torch_projector.scale_factors)

    # Deduce in_channels
    in_channels = []
    # Try scale 0
    if len(torch_projector.stages_sampling) > 0:
        for j in range(num_inputs):
            sampler = torch_projector.stages_sampling[0][j]
            if len(sampler) > 0:
                first_layer = sampler[0]
                if isinstance(first_layer, torch.nn.ConvTranspose2d):
                    in_channels.append(first_layer.in_channels)
                elif hasattr(first_layer, 'conv'): # ConvX
                    in_channels.append(first_layer.conv.in_channels)
                elif isinstance(first_layer, torch.nn.Conv2d):
                    in_channels.append(first_layer.in_channels)
                else:
                    in_channels.append(None)
            else:
                in_channels.append(None)
    
    # Try to deduce from C2f weights (stages) if Nones remain
    if any(c is None for c in in_channels):
        # Config logic: in_dim_combined = sum(in_channel // max(1, scale))
        # For scale 1.0, this is sum(in_channels).
        # We can look at stages[0].
        # torch_projector.stages[0] is often [C2f, Norm]
        # C2f.cv1.weight shape is [2*c, In, 1, 1]
        try:
            stage_modal = torch_projector.stages[0]
            # Handle if it is Sequential or ModuleList
            if isinstance(stage_modal, (torch.nn.Sequential, torch.nn.ModuleList)):
                 c2f = stage_modal[0]
            else:
                 c2f = stage_modal
            
            if hasattr(c2f, 'cv1') and hasattr(c2f.cv1, 'conv'):
                 total_in = c2f.cv1.conv.in_channels
                 # Assuming equal distribution across inputs and Scale 1.0 logic dominating
                 # logic: in_dim_combined = sum(ch / scale)
                 # If scale is 1.0 (index 0 usually?)
                 # Let's check scale factor for this stage
                 scale = scale_factors[0]
                 # If scale is 1.0, total_in = sum(in_channels)
                 # If all inputs are equal (common in DINO)
                 per_channel = int(total_in * max(1, scale) / num_inputs)
                 
                 # Fill Nones
                 in_channels = [c if c is not None else per_channel for c in in_channels]
                 print(f"Deduced in_channels from C2f weights: {per_channel} (Total {total_in})")
        except Exception as e:
            print(f"Failed to deduce from C2f: {e}")

    # Ultimate fallback based on model name if still None
    if any(c is None for c in in_channels):
        if "Small" in model_class.__name__:
            fallback = 384
        elif "Medium" in model_class.__name__: # DINOv2 Base
            fallback = 768
        elif "Large" in model_class.__name__: # DINOv2 Large
             fallback = 1024
        elif "Nano" in model_class.__name__:
             fallback = 384 
        else:
             fallback = 256 # Generic guess

        in_channels = [c if c is not None else fallback for c in in_channels]
        print(f"Used fallback {fallback} for inferred None channels.")
        print(f"Used fallback {fallback} for inferred None channels.")

    # Deduce out_channels
    if hasattr(torch_projector.stages[0][0], 'cv2'):
        out_channels = torch_projector.stages[0][0].cv2.conv.out_channels
    else:
        out_channels = 256 # standard
        
    # scale_factors = list(torch_projector.scale_factors)
    
    print(f"Config: In={in_channels}, Out={out_channels}, Scales={scale_factors}")

    # Build Keras Model
    keras_model = KerasMultiScaleProjector(
        in_channels=in_channels,
        out_channels=out_channels,
        scale_factors=scale_factors,
        input_scales=[1.0] * len(in_channels),
        layer_norm=True
    )
    
    # Input
    size = 32
    # Ensure size is divisible if needed, but 32 is usually fine
    x_np = [np.random.randn(1, size, size, c).astype('float32') for c in in_channels]
    
    keras_model(x_np)
    
    # Port Weights
    port_weights_multiscale_projector(torch_projector, keras_model)
    
    # Compare — move projector to CPU so both frameworks compute on the same
    # device, avoiding GPU-vs-CPU float32 numerical divergence.
    torch_projector = torch_projector.cpu()
    x_torch = [torch.from_numpy(x.transpose(0, 3, 1, 2)) for x in x_np]
    
    with torch.no_grad():
        out_torch = torch_projector(x_torch)
        
    out_keras = keras_model(x_np, training=False)
    
    for i, (o_t, o_k) in enumerate(zip(out_torch, out_keras)):
        o_t_np = to_numpy(o_t).transpose(0, 2, 3, 1)
        # Using 1e-4 tolerance as verified earlier for Small
        np.testing.assert_allclose(o_t_np, o_k, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
    pytest.main([__file__])
