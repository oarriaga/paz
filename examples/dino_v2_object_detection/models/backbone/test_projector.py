import pytest
import torch
import numpy as np
import keras
from keras import ops

# Import PyTorch implementation using importlib to avoid name collision
import sys
import os
import importlib.util

# Import PyTorch implementation using importlib to avoid name collision
torch_proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../", "rf-detr_original_pytorch_implementation", "rfdetr", "models", "backbone", "projector.py"))
spec = importlib.util.spec_from_file_location("torch_projector", torch_proj_path)
torch_projector = importlib.util.module_from_spec(spec)
sys.modules["torch_projector"] = torch_projector
spec.loader.exec_module(torch_projector)

from torch_projector import MultiScaleProjector as TorchMultiScaleProjector, SimpleProjector as TorchSimpleProjector, ConvX as TorchConvX, C2f as TorchC2f
# Ensure project root is in path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Keras implementation
try:
    from examples.dino_v2_object_detection.models.backbone.projector import (
        MultiScaleProjector as KerasMultiScaleProjector,
        SimpleProjector as KerasSimpleProjector,
        ConvX as KerasConvX,
        C2f as KerasC2f
    )
    from examples.dino_v2_object_detection.models.backbone.projector_weights_porting_utils import (
        copy_conv2d,
        copy_bn,
        copy_ln,
        copy_weights_convx,
        copy_weights_c2f
    )
except ImportError:
    # Fallback: add local dir to path (only if absolute fails)
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from projector import (
        MultiScaleProjector as KerasMultiScaleProjector,
        SimpleProjector as KerasSimpleProjector,
        ConvX as KerasConvX,
        C2f as KerasC2f
    )
    from projector_weights_porting_utils import (
        copy_conv2d,
        copy_bn,
        copy_ln,
        copy_weights_convx,
        copy_weights_c2f
    )

def set_seed():
    np.random.seed(42)
    torch.manual_seed(42)

def to_numpy(t):
    return t.detach().cpu().numpy()



def test_convx_parity():
    set_seed()
    C_in, C_out = 32, 64
    t_mod = TorchConvX(C_in, C_out, kernel=3, stride=1, act='silu', layer_norm=False)
    k_mod = KerasConvX(C_in, C_out, kernel=3, stride=1, act='silu', layer_norm=False)
    
    # Set PyTorch to eval mode (important for BN)
    t_mod.eval()
    
    # Build Keras
    x_np = np.random.randn(1, 32, 32, C_in).astype('float32') # NHWC
    k_mod(x_np)
    
    copy_weights_convx(t_mod, k_mod)
    
    device = next(t_mod.parameters()).device
    x_torch = torch.from_numpy(x_np.transpose(0, 3, 1, 2)).to(device)
    
    with torch.no_grad():
        out_torch = t_mod(x_torch)
    out_keras = k_mod(x_np, training=False)
    
    out_torch_np = to_numpy(out_torch).transpose(0, 2, 3, 1) # to NHWC
    
    np.testing.assert_allclose(out_torch_np, out_keras, rtol=1e-5, atol=1e-5)

def test_c2f_parity():
    set_seed()
    C_in, C_out = 64, 64
    t_mod = TorchC2f(C_in, C_out, n=2, shortcut=True)
    k_mod = KerasC2f(C_in, C_out, n=2, shortcut=True)
    
    t_mod.eval()
    
    x_np = np.random.randn(1, 32, 32, C_in).astype('float32')
    k_mod(x_np)
    
    copy_weights_c2f(t_mod, k_mod)
    
    device = next(t_mod.parameters()).device
    x_torch = torch.from_numpy(x_np.transpose(0, 3, 1, 2)).to(device)
    
    with torch.no_grad():
        out_torch = t_mod(x_torch)
    out_keras = k_mod(x_np, training=False)
    
    out_torch_np = to_numpy(out_torch).transpose(0, 2, 3, 1)
    np.testing.assert_allclose(out_torch_np, out_keras, rtol=1e-5, atol=1e-5)

def test_multiscale_projector_parity():
    set_seed()
    # Setup
    in_channels = [64, 128, 256]
    out_channels = 64
    scale_factors = [4.0, 2.0, 1.0, 0.5]
    
    t_mod = TorchMultiScaleProjector(in_channels, out_channels, scale_factors)
    k_mod = KerasMultiScaleProjector(in_channels, out_channels, scale_factors, input_scales=[1.0]*len(in_channels))
    
    t_mod.eval()
    
    # Inputs (NHWC for Keras)
    # Using SAME spatial size because MultiScaleProjector in PyTorch (as analyzed)
    # likely treats inputs as coming from a ViT (same resolution) or requires explicit handling not visible here.
    # To verify WEIGHT PORTING, we use compatible constraints (same resolution).
    size = 32
    x_np = [
        np.random.randn(1, size, size, 64).astype('float32'),
        np.random.randn(1, size, size, 128).astype('float32'),
        np.random.randn(1, size, size, 256).astype('float32')
    ]
    
    # Build Keras model
    k_mod(x_np)
    
    # Copy Weights
    # 1. stages_sampling (ModuleList of ModuleList of Sequential/Identity)
    # 2. stages (ModuleList of Sequential)
    
    # Copy stages_sampling
    for i in range(len(t_mod.stages_sampling)): # scale index
        for j in range(len(t_mod.stages_sampling[i])): # in_channel index
            t_sub = t_mod.stages_sampling[i][j]
            k_sub = k_mod.stages_sampling_blocks[i][j]
            
            # t_sub is Sequential or Identity
            # If identity, nothing to copy
            if isinstance(k_sub, keras.layers.Identity):
                continue
                
            # Iterate over layers in sequential
            # Keras Sequential: k_sub.layers
            # PyTorch Sequential: t_sub
            
            k_idx = 0
            for t_layer in t_sub:
                if isinstance(t_layer, torch.nn.ConvTranspose2d):
                    # Keras Conv2DTranspose
                    k_layer = k_sub.layers[k_idx]
                    w = t_layer.weight.data.cpu().numpy() # [In, Out, K, K] for ConvTranspose2d in PyTorch? 
                    # Wait, PyTorch ConvTranspose2d weight is [In, Out, K, K] usually? Check docs.
                    # PyTorch ConvTranspose2d: (in_channels, out_channels/groups, kernel_size[0], kernel_size[1])
                    # Keras Conv2DTranspose: (kernel_size[0], kernel_size[1], out_channels, in_channels) ? No.
                    # Keras Conv2DTranspose: (kernel_size[0], kernel_size[1], out_channels, in_channels) ??
                    # Actually standard tf/keras conv_transpose weights are (K, K, Out, In) usually.
                    
                    # Correction: PyTorch ConvTranspose2d weight: [In, Out/groups, K, K]
                    # Keras Conv2DTranspose weight: [K, K, Out, In]
                    
                    # Let's verify standard Keras Conv2DTranspose layout.
                    # It's usually [k, k, out, in].
                    
                    if k_layer.use_bias and t_layer.bias is not None:
                        b = t_layer.bias.data.cpu().numpy()
                        k_layer.set_weights([w.transpose(2, 3, 1, 0), b])
                    else:
                        k_layer.set_weights([w.transpose(2, 3, 1, 0)])
                    k_idx += 1
                    
                elif isinstance(t_layer, torch.nn.GELU):
                    k_idx += 1
                    continue # Activation, no weights
                    
                elif hasattr(t_layer, 'weight') and hasattr(t_layer, 'normalized_shape'): 
                     # LayerNorm (custom or standard)
                    k_layer = k_sub.layers[k_idx]
                    copy_ln(t_layer, k_layer)
                    k_idx += 1
                
                elif isinstance(t_layer, TorchConvX):
                    k_layer = k_sub.layers[k_idx] 
                    copy_weights_convx(t_layer, k_layer)
                    k_idx += 1

    # Copy stages
    for i in range(len(t_mod.stages)):
        t_seq = t_mod.stages[i]
        k_seq = k_mod.stages_blocks[i]
        
        # t_seq[0] is C2f
        copy_weights_c2f(t_seq[0], k_seq.layers[0])
        # t_seq[1] is Norm
        copy_ln(t_seq[1], k_seq.layers[1])

    # Run Verify
    device = next(t_mod.parameters()).device
    x_torch = [torch.from_numpy(x.transpose(0, 3, 1, 2)).to(device) for x in x_np]
    
    with torch.no_grad():
        out_torch = t_mod(x_torch)
        
    out_keras = k_mod(x_np)
    
    
    for i, (o_t, o_k) in enumerate(zip(out_torch, out_keras)):
        o_t_np = to_numpy(o_t).transpose(0, 2, 3, 1)
        np.testing.assert_allclose(o_t_np, o_k, rtol=1e-4, atol=1e-4)

def test_simple_projector_parity():
    set_seed()
    t_mod = TorchSimpleProjector(64, 64)
    k_mod = KerasSimpleProjector(64, 64)
    
    t_mod.eval()
    
    x_np = [np.random.randn(1, 32, 32, 64).astype('float32')]
    k_mod(x_np)
    
    copy_weights_convx(t_mod.convx1, k_mod.convx1)
    copy_weights_convx(t_mod.convx2, k_mod.convx2)
    copy_ln(t_mod.ln, k_mod.ln)
    
    device = next(t_mod.parameters()).device
    x_torch = [torch.from_numpy(x_np[0].transpose(0, 3, 1, 2)).to(device)]
    
    with torch.no_grad():
        out_torch = t_mod(x_torch)
        
    out_keras = k_mod(x_np, training=False)
    np.testing.assert_allclose(to_numpy(out_torch[0]).transpose(0, 2, 3, 1), out_keras[0], rtol=1e-5, atol=1e-5)

def test_conv_transpose_parity():
    set_seed()
    C_in, C_out = 64, 32
    # PyTorch: In, Out, K, K. 
    t_mod = torch.nn.ConvTranspose2d(C_in, C_out, kernel_size=2, stride=2, bias=True)
    
    # Keras: K, K, Out, In
    k_mod = keras.layers.Conv2DTranspose(C_out, kernel_size=2, strides=2, padding="valid")
    
    # Build
    x_np = np.random.randn(1, 32, 32, C_in).astype('float32')
    k_mod(x_np)
    
    # Copy weights
    # PyTorch weight: (In, Out, K, K)
    # Keras weight: (K, K, Out, In)
    w = t_mod.weight.data.cpu().numpy()
    b = t_mod.bias.data.cpu().numpy()
    
    # Transpose (2, 3, 1, 0)
    # 2->K, 3->K, 1->Out, 0->In
    k_mod.set_weights([w.transpose(2, 3, 1, 0), b])
    
    # Verify
    device = next(t_mod.parameters()).device
    x_torch = torch.from_numpy(x_np.transpose(0, 3, 1, 2)).to(device)
    
    with torch.no_grad():
        out_torch = t_mod(x_torch)
    
    out_keras = k_mod(x_np, training=False)
    
    np.testing.assert_allclose(to_numpy(out_torch).transpose(0, 2, 3, 1), out_keras, rtol=1e-5, atol=1e-5)

    np.testing.assert_allclose(to_numpy(out_torch).transpose(0, 2, 3, 1), out_keras, rtol=1e-5, atol=1e-5)

def test_multiscale_projector_configurations():
    set_seed()
    
    # Configurations to test based on backbone.py usage
    # level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
    configs = [
        {
            "name": "Standard (P3, P4, P5)",
            "in_channels": [64, 128, 256],
            "out_channels": 64,
            "scale_factors": [2.0, 1.0, 0.5], 
            "layer_norm": False
        },
        {
            "name": "Single Scale (P4)",
            "in_channels": [128], 
            "out_channels": 128,
            "scale_factors": [1.0], 
            "layer_norm": False
        },
         {
            "name": "Upsample Only (P3)",
            "in_channels": [64, 128], 
            "out_channels": 64,
            "scale_factors": [4.0, 2.0], # Testing 4.0 explicitly
            "layer_norm": False
        }
    ]

    for conf in configs:
        print(f"Testing configuration: {conf['name']}")
        in_channels = conf["in_channels"]
        out_channels = conf["out_channels"]
        scale_factors = conf["scale_factors"]
        layer_norm = conf["layer_norm"]

        t_mod = TorchMultiScaleProjector(in_channels, out_channels, scale_factors, layer_norm=layer_norm)
        k_mod = KerasMultiScaleProjector(in_channels, out_channels, scale_factors, input_scales=[1.0]*len(in_channels), layer_norm=layer_norm)
        
        t_mod.eval()
        
        size = 32
        x_np = [np.random.randn(1, size, size, c).astype('float32') for c in in_channels]
        
        k_mod(x_np)
        
        for i in range(len(t_mod.stages_sampling)):
            for j in range(len(t_mod.stages_sampling[i])):
                t_sub = t_mod.stages_sampling[i][j]
                k_sub = k_mod.stages_sampling_blocks[i][j]
                # Check for Identity
                if isinstance(k_sub, keras.layers.Identity):
                     # PyTorch might be Identity too? 
                     # PyTorch doesn't have Identity layer in sampling block, checks loop condition.
                     # But my code puts Identity in Keras if scale=1.0? 
                     # Wait, checking implementation logic.
                     # PyTorch: if 1.0: pass. stages_sampling[-1] appended empty Sequential? 
                     # No, layers=[] then Sequential(*layers). Sequential([]) identifies as identity? No.
                     # It is an empty Sequential.
                     continue
                
                k_idx = 0
                for t_layer in t_sub:
                    if isinstance(t_layer, torch.nn.ConvTranspose2d):
                        k_layer = k_sub.layers[k_idx]
                        w = t_layer.weight.data.cpu().numpy()
                        if k_layer.use_bias and t_layer.bias is not None:
                            b = t_layer.bias.data.cpu().numpy()
                            k_layer.set_weights([w.transpose(2, 3, 1, 0), b])
                        else:
                            k_layer.set_weights([w.transpose(2, 3, 1, 0)])
                        k_idx += 1
                    elif isinstance(t_layer, torch.nn.GELU):
                         k_idx += 1
                         continue
                    elif hasattr(t_layer, 'weight') and hasattr(t_layer, 'normalized_shape'):
                        k_layer = k_sub.layers[k_idx]
                        copy_ln(t_layer, k_layer)
                        k_idx += 1
                    elif isinstance(t_layer, TorchConvX):
                        k_layer = k_sub.layers[k_idx]
                        copy_weights_convx(t_layer, k_layer)
                        k_idx += 1
        
        for i in range(len(t_mod.stages)):
            t_seq = t_mod.stages[i]
            k_seq = k_mod.stages_blocks[i]
            copy_weights_c2f(t_seq[0], k_seq.layers[0])
            copy_ln(t_seq[1], k_seq.layers[1])
            
        device = next(t_mod.parameters()).device
        x_torch = [torch.from_numpy(x.transpose(0, 3, 1, 2)).to(device) for x in x_np]
        with torch.no_grad():
            out_torch = t_mod(x_torch)
        out_keras = k_mod(x_np, training=False)
        
        for i_out, (o_t, o_k) in enumerate(zip(out_torch, out_keras)):
            o_t_np = to_numpy(o_t).transpose(0, 2, 3, 1)
            try:
                np.testing.assert_allclose(o_t_np, o_k, rtol=1e-4, atol=1e-4)
            except AssertionError as e:
                print(f"FAILED: {conf['name']} at Scale {scale_factors[i_out]}")
                raise e

def test_multiscale_projector_extra_pool():
    # Keras-specific test for P6 (scale 0.25) which crashes in provided PyTorch code
    set_seed()
    in_channels = [64]
    out_channels = 32
    scale_factors = [2.0, 1.0, 0.5, 0.25] # P6 included
    
    k_mod = KerasMultiScaleProjector(in_channels, out_channels, scale_factors, input_scales=[1.0])
    
    size = 32
    x_np = [np.random.randn(1, size, size, 64).astype('float32')]
    
    # Run
    out_keras = k_mod(x_np, training=False)
    
    # Verify outputs
    # Scales: 2.0 (Up 2x -> 64), 1.0 (Same -> 32), 0.5 (Down 2x -> 16), 0.25 (Extra pool on last -> 8?)
    # Wait, 0.25 just sets use_extra_pool=True.
    # It appends MaxPool2D(result[-1]) at the END.
    # result[-1] is the result of the LAST STAGE.
    # The last stage processed covers up to scale 0.5 (since 0.25 is skipped).
    # scale 0.5 output size: size // 2 = 16.
    # So P6 output size: 16 // 2 = 8.
    
    expected_shapes = [
        (1, 64, 64, 32), # Scale 2.0
        (1, 32, 32, 32), # Scale 1.0
        (1, 16, 16, 32), # Scale 0.5
        (1, 8, 8, 32)    # P6 (Pool of Scale 0.5)
    ]
    
    assert len(out_keras) == 4
    for i, out in enumerate(out_keras):
        assert out.shape == expected_shapes[i]


if __name__ == "__main__":
    pytest.main([__file__])