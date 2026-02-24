import os
import sys
import torch
import numpy as np

# Add rf-detr path
current_dir = os.path.dirname(os.path.abspath(__file__))
rfdetr_path = os.path.join(current_dir, "../rf-detr_original_pytorch_implementation")
sys.path.insert(0, rfdetr_path)

from rfdetr import RFDETRNano
from rfdetr.util.misc import NestedTensor

def debug_shapes():
    print("Instantiating RFDETRNano...")
    model = RFDETRNano()
    model = model.model.model # Get the inner nn.Module
    model.eval()

    img = torch.randn(1, 3, 384, 384)
    mask = torch.zeros((1, 384, 384), dtype=torch.bool)
    samples = NestedTensor(img, mask)
    print(f"Input image shape: {img.shape}")

    # Trace backbone
    with torch.no_grad():
        features, pos = model.backbone(samples)
        
        # Actually Joiner returns (feats, pos)
        # where feats is list of NestedTensor
        print(f"Number of feature levels: {len(features)}")
        for i, feat in enumerate(features):
             print(f"Level {i} tensor shape: {feat.tensors.shape}")
             print(f"Level {i} mask shape: {feat.mask.shape}")
             print(f"Level {i} pos shape: {pos[i].shape}")

    # Trace transformer inputs
    srcs = []
    masks = []
    for feat in features:
        srcs.append(feat.tensors)
        masks.append(feat.mask)
    
    print(f"Transformer srcs lengths: {len(srcs)}")
    print(f"Transformer poss lengths: {len(pos)}")

if __name__ == "__main__":
    debug_shapes()
