import os
import sys
import torch
import torch.nn as nn

# Add rf-detr path
current_dir = os.path.dirname(os.path.abspath(__file__))
rfdetr_path = os.path.join(current_dir, "../rf-detr_original_pytorch_implementation")
sys.path.insert(0, rfdetr_path)

from rfdetr import RFDETRNano

def inspect_model():
    print("Instantiating RFDETRNano...")
    pt_model = RFDETRNano()
    inner_pt = pt_model.model.model
    
    backbone = inner_pt.backbone[0]
    print(f"Backbone type: {type(backbone)}")
    
    def print_attrs(obj, depth=0, max_depth=5):
        if depth > max_depth:
            return
        prefix = "  " * depth
        for name, module in obj.named_children():
            print(f"{prefix}{name}: {type(module)}")
            print_attrs(module, depth + 1, max_depth)

    print("\nBackbone tree:")
    print_attrs(backbone)

if __name__ == "__main__":
    inspect_model()
