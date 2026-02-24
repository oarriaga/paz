import os
import sys
import torch
import numpy as np
import keras

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.insert(0, project_root)
rfdetr_path = os.path.join(current_dir, "../../../rf-detr_original_pytorch_implementation")
sys.path.insert(0, rfdetr_path)

from rfdetr import RFDETRNano
from rfdetr.util.misc import NestedTensor

from examples.dino_v2_object_detection.models.backbone import build_backbone as build_keras_backbone
from examples.dino_v2_object_detection.models.backbone.backbone_weights_porting_utils import (
    transfer_encoder as transfer_backbone_encoder,
    transfer_patch_embeddings,
    transfer_layernorm
)

def debug_backbone():
    print("Instantiating PT model...")
    pt_wrapper = RFDETRNano()
    pt_backbone = pt_wrapper.model.model.backbone[0] # The WindowedDinov2WithRegistersBackbone
    pt_backbone.eval()

    print("Building Keras backbone...")
    res = 384
    keras_backbone_joiner = build_keras_backbone(
        encoder="dinov2_windowed_small",
        hidden_dim=256,
        patch_size=16,
        num_windows=2,
        out_feature_indexes=[2, 5, 8, 11],
        projector_scale=["P4"],
        layer_norm=True
    )
    keras_backbone = keras_backbone_joiner.backbone
    keras_dinov2 = keras_backbone.encoder
    
    # 1. Build Keras models
    dummy = np.random.randn(1, res, res, 3).astype(np.float32)
    keras_backbone_joiner(dummy)
    
    # 2. Transfer Weights (Manually to be sure)
    print("Transferring manual weights...")
    # Encoder
    transfer_backbone_encoder(pt_backbone.encoder, keras_dinov2.encoder)
    # Patch Embed
    transfer_patch_embeddings(pt_backbone.embeddings, keras_dinov2.patch_embed)
    # Final Norm
    transfer_layernorm(pt_backbone.layernorm, keras_dinov2.layernorm)
    
    # DEBUG WEIGHTS
    print(f"PT Block 0 Norm1 Weight Max: {pt_backbone.encoder.layer[0].norm1.weight.abs().max().item():.6e}")
    print(f"Keras Block 0 Norm1 Gamma Max: {np.abs(keras_dinov2.encoder.encoder_layers[0].norm1.gamma.numpy()).max():.6e}")
    
    print(f"PT Block 0 LS1 Lambda Max: {pt_backbone.encoder.layer[0].layer_scale1.lambda1.abs().max().item():.6e}")
    print(f"Keras Block 0 LS1 Gamma Max: {np.abs(keras_dinov2.encoder.encoder_layers[0].layer_scale1.gamma.numpy()).max():.6e}")

    # 3. Test Patch Embeddings
    print("Testing Patch Embeddings...")
    img = np.random.randn(1, res, res, 3).astype(np.float32)
    img_pt = torch.from_numpy(img.transpose(0, 3, 1, 2)).float()
    
    with torch.no_grad():
        pt_embed = pt_backbone.embeddings(img_pt)
    
    k_embed = keras_dinov2.patch_embed(img)
    
    diff = np.abs(pt_embed.numpy() - k_embed.numpy()).max()
    print(f"Patch Embed Max Diff: {diff:.6e}")
    
    # 3.5 Test Block 0
    print("Testing Block 0...")
    with torch.no_grad():
        pt_block0_out = pt_backbone.encoder.layer[0](pt_embed, run_full_attention=False)[0]
    
    k_block0_out = keras_dinov2.encoder.encoder_layers[0](k_embed, training=False)
    
    diff = np.abs(pt_block0_out.numpy() - k_block0_out.numpy()).max()
    print(f"Block 0 Max Diff: {diff:.6e}")

    # 4. Test Full DinoV2 (no projector)
    print("Testing Full DinoV2 (no projector)...")
    with torch.no_grad():
        # In PyTorch, Backbone(DinoV2).encoder is WindowedDinov2WithRegistersBackbone
        # WindowedDinov2WithRegistersBackbone.forward returns list of features
        pt_features = pt_backbone(img_pt) # This might return (feats, pos) depending on which object it is
        # Wait, pt_backbone above is the 0-th element of Joiner.
        # So it's the Backbone object.
        # Let's check what it returns.
        pass
    
    # Let's call pt_backbone.encoder directly if needed
    with torch.no_grad():
        # WindowedDinov2WithRegistersBackbone.forward
        pt_out = pt_backbone(img_pt)
        # RF-DETR Backbone.forward returns (feats, masks) or list(feats)?
        # According to dinov2.py: return list(x[0])
        # BUT pt_backbone.forward in backbone.py returns:
        # if self.projector is not None: x = self.projector(x)
    
    # Let's test only the encoder part
    with torch.no_grad():
        pt_emb_out = pt_backbone.embeddings(img_pt)
        pt_enc_out = pt_backbone.encoder(pt_emb_out, output_hidden_states=True)
        # pt_enc_out has .hidden_states
        pt_last_hidden = pt_enc_out.hidden_states[-1]
        pt_last_hidden = pt_backbone.layernorm(pt_last_hidden)
    
    # Keras DinoV2.call returns list of features
    k_features = keras_dinov2(img)
    k_last_hidden = k_features[-1]
    
    # Check shapes
    print(f"PT Last Hidden Shape: {pt_last_hidden.shape}")
    print(f"Keras Last Hidden Shape: {k_last_hidden.shape}")
    
    # Compare
    # Keras returns (B, H, W, C)
    # PyTorch last_hidden is (B, N, C) where N = 1 + num_reg + HW
    # We need to slice it
    num_reg = 0 # Nano uses registers? Check config.
    start_idx = 1 + num_reg
    pt_feat = pt_last_hidden[:, start_idx:]
    # Reshape pt_feat to (1, 24, 24, 256)
    pt_feat = pt_feat.view(1, 24, 24, -1).numpy()
    
    diff = np.abs(pt_feat - k_last_hidden.numpy()).max()
    print(f"DinoV2 Last Hidden Max Diff: {diff:.6e}")

if __name__ == "__main__":
    debug_backbone()
