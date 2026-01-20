import re
import keras
from keras import Model

from examples.dino_object_detection.models.backbone.dinov2_backbone_wrapper import (
    Joiner,
)


def get_vit_lr_decay_rate(var_path, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    """
    # Normalized path
    path = var_path.replace("/", ".")

    layer_id = num_layers + 1

    # 1. Early Layers (Embeddings / Mask) -> Layer 0
    if any(
        x in path
        for x in [
            "pos_embed",
            "patch_embed",
            "mask_token",
            "position_embedding",
        ]
    ):
        layer_id = 0

    # 2. Transformer Blocks
    match = re.search(r"blocks[._](\d+)([._]|$)", path)
    if match:
        try:
            layer_id = int(match.group(1)) + 1
        except ValueError:
            pass

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_weight_decay_rate(var, default_wd=0.0):
    path = var.path if hasattr(var, "path") else var.name
    name = var.name.lower()

    if "bias" in name:
        return 0.0

    # Combined check for normalization patterns
    if "norm" in path.lower() or any(
        x in name for x in ["gamma", "beta", "moving_mean", "moving_variance", "scale"]
    ):
        return 0.0

    if any(
        x in path for x in ["pos_embed", "rel_pos", "cls_token", "position_embedding"]
    ):
        return 0.0

    return default_wd


def get_param_dict(args, model: Model):
    """
    Generates parameter groups for the optimizer matching RF-DETR/LW-DETR logic.
    """
    base_lr = args.lr
    backbone_base_lr = getattr(args, "lr_backbone", base_lr)
    lr_decay_rate = getattr(args, "lr_vit_layer_decay", 1.0)
    weight_decay = getattr(args, "weight_decay", 0.0)
    component_decay = getattr(args, "lr_component_decay", 1.0)
    num_layers = getattr(args, "num_layers", 12)

    backbone_layer = None
    if isinstance(model, Joiner):
        backbone_layer = model.backbone
    else:
        for layer in model.layers:
            if isinstance(layer, Joiner):
                backbone_layer = layer.backbone
                break
            if hasattr(layer, "layers"):
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, Joiner):
                        backbone_layer = sub_layer.backbone
                        break

    backbone_var_ids = set()
    if backbone_layer:
        backbone_var_ids = set(id(v) for v in backbone_layer.trainable_variables)
    else:
        print("[Warning] Could not strictly identify 'Joiner.backbone'. Falling back.")

    param_groups = []

    for var in model.trainable_variables:
        v_id = id(var)
        # Use path if available (standard in Keras 3), else fallback to name
        v_path = var.path if hasattr(var, "path") else var.name
        norm_path = v_path.replace("/", ".")

        wd = get_weight_decay_rate(var, default_wd=weight_decay)

        # Logic Branching
        is_backbone = v_id in backbone_var_ids
        if not backbone_layer and ("backbone" in norm_path or "dinov2" in norm_path):
            is_backbone = True

        if is_backbone:
            decay_factor = get_vit_lr_decay_rate(norm_path, lr_decay_rate, num_layers)
            current_lr = backbone_base_lr * decay_factor

            param_groups.append(
                {
                    "params": var,
                    "lr": current_lr,
                    "weight_decay": wd,
                    # FIX: Use norm_path instead of var.name to include layer names
                    "name": f"backbone_{norm_path}",
                }
            )

        elif (
            "transformer.decoder" in norm_path
            or "transformer_decoder" in norm_path
            or "transformer/decoder" in v_path
        ):
            current_lr = base_lr * component_decay

            param_groups.append(
                {
                    "params": var,
                    "lr": current_lr,
                    "weight_decay": wd,
                    # FIX: Use norm_path instead of var.name
                    "name": f"decoder_{norm_path}",
                }
            )

        else:
            param_groups.append(
                {
                    "params": var,
                    "lr": base_lr,
                    "weight_decay": wd,
                    "name": f"other_{norm_path}",
                }
            )

    return param_groups
