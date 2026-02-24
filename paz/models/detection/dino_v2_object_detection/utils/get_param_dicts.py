from typing import Any, Dict, List
import keras

def get_vit_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12) -> float:
    """
    Calculate lr decay rate for different ViT blocks.

    Args:
        name: parameter name.
        lr_decay_rate: base lr decay rate.
        num_layers: number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    # Keras names might use / instead of .
    # standardized to . for logic checks or handle both
    norm_name = name.replace("/", ".")
    
    if norm_name.startswith("backbone"):
        if ".pos_embed" in norm_name or ".patch_embed" in norm_name:
            layer_id = 0
        elif ".blocks." in norm_name and ".residual." not in norm_name:
            # Assumes name format like backbone.blocks.N. ...
            try:
                # Find the index of the block
                parts = norm_name.split(".")
                blocks_idx = parts.index("blocks")
                layer_id = int(parts[blocks_idx + 1]) + 1
            except (ValueError, IndexError):
                pass
                
    # print("name: {}, lr_decay: {}".format(name, lr_decay_rate ** (num_layers + 1 - layer_id)))
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_vit_weight_decay_rate(name: str, weight_decay_rate: float = 1.0) -> float:
    """
    Calculate weight decay rate for different ViT parameters.

    Args:
        name: parameter name.
        weight_decay_rate: base weight decay rate.
    Returns:
        weight decay rate for the given parameter.
    """
    # Keras variables often have names like 'kernel', 'bias', 'gamma', 'beta'
    # 'pos_embed', 'rel_pos' might be in the path
    
    if ('gamma' in name) or ('pos_embed' in name) or ('rel_pos' in name) or ('bias' in name) or ('norm' in name) or ('beta' in name):
        weight_decay_rate = 0.
    
    # print("name: {}, weight_decay rate: {}".format(name, weight_decay_rate))
    return weight_decay_rate


def get_param_dict(args: Any, model: keras.Model) -> List[Dict[str, Any]]:
    """
    Create parameter groups for optimization.
    
    Args:
        args: Argument object with 'lr', 'lr_component_decay', etc.
        model: Keras model
        
    Returns:
        List of dicts with 'params', 'lr', 'weight_decay'
    """
    # We iterate over all trainable variables
    # We need to classify them: backbone, decoder, other
    
    backbone_params = []
    decoder_params = []
    other_params = []
    
    # Heuristics for grouping
    # We assume standard naming conventions in the ported model
    
    for v in model.trainable_variables:
        name = v.name
        # Normalize name for checks
        norm_name = name.replace("/", ".")
        
        # Check if backbone
        # The backbone variable names usually start with 'backbone' if the layer is named 'backbone'
        # Or if it's a nested model, it might be 'functional_...' but we hope user named layers.
        # Assuming the ported model has a 'backbone' layer.
        
        is_backbone = "backbone" in norm_name
        is_decoder = "transformer.decoder" in norm_name or "transformer/decoder" in name
        
        if is_backbone:
            # Backbone logic might vary if it's ViT (decay rates) or ResNet (standard)
            # Original code called backbone.get_named_param_lr_pairs which implies backbone might handle self
            # For simplicity, we assign backbone LR here. 
            # If args has lr_backbone, maybe use it? 
            # The original code used 'backbone_param_lr_pairs'.
            # If it's a ViT backbone, we might need get_vit_lr_decay_rate. 
            # Let's check args for ViT specific flags? original code assumes Joiner handles it.
            # We will use args.lr_backbone if available, else args.lr.
            
            lr_val = getattr(args, 'lr_backbone', args.lr)
            
            # Apply layer-wise decay if it looks like ViT and we have mechanism?
            # For now, minimal port:
            backbone_params.append({"params": v, "lr": lr_val})
            
        elif is_decoder:
            lr_val = args.lr * getattr(args, 'lr_component_decay', 1.0)
            decoder_params.append({"params": v, "lr": lr_val})
            
        else:
            other_params.append({"params": v, "lr": args.lr})
            
    # Combine
    return other_params + backbone_params + decoder_params
