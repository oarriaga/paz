from typing import Any, Dict, List
import math


def get_vit_lr_decay_rate(name: str, lr_decay_rate: float = 1.0,
                          num_layers: int = 12) -> float:
    """Per-block LR decay for ViT / DinoV2 backbones."""
    layer_id = num_layers + 1
    norm_name = name.replace("/", ".")

    if norm_name.startswith("backbone"):
        if "embeddings" in norm_name:
            layer_id = 0
        elif ".layer." in norm_name and ".residual." not in norm_name:
            layer_id = int(
                norm_name[norm_name.find(".layer."):].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_vit_weight_decay_rate(name: str,
                              weight_decay_rate: float = 1.0) -> float:
    """Return 0 for bias / norm / positional-embedding parameters."""
    if (("gamma" in name) or ("pos_embed" in name)
            or ("rel_pos" in name) or ("bias" in name)
            or ("norm" in name) or ("embeddings" in name)):
        return 0.0
    return weight_decay_rate


# ---------------------------------------------------------------------------
# Classify a variable name into a parameter group
# ---------------------------------------------------------------------------


def classify_variable(name: str):
    """Return ``'backbone'``, ``'decoder'``, or ``'other'``."""
    norm = name.replace("/", ".")
    if "backbone" in norm:
        return "backbone"
    if "transformer.decoder" in norm or "transformer/decoder" in name:
        return "decoder"
    return "other"


def compute_backbone_lr(name, *, lr_encoder, lr_vit_layer_decay,
                        lr_component_decay, num_layers):
    """LR for a single backbone variable.

    Formula (from PyTorch ``Backbone.get_named_param_lr_pairs``):
        lr = lr_encoder × layer_decay(name) × lr_component_decay²
    """
    layer_decay = get_vit_lr_decay_rate(
        name, lr_decay_rate=lr_vit_layer_decay, num_layers=num_layers)
    return lr_encoder * layer_decay * (lr_component_decay ** 2)


# ---------------------------------------------------------------------------
# Build LR-scale map (model-level)
# ---------------------------------------------------------------------------


def build_lr_scale_map(model, *, lr, lr_encoder, lr_vit_layer_decay,
                       lr_component_decay, weight_decay,
                       num_layers) -> Dict[str, Dict[str, float]]:
    """Return ``{var.name: {"lr_scale": s, "wd": w}}`` for every trainable var.

    ``lr_scale`` is the multiplier so that
        effective_lr(var) = base_lr_schedule(step) × lr_scale
    ``wd`` is the per-variable weight decay (0 for bias/norm/embed).
    """
    result = {}
    for v in model.trainable_variables:
        name = v.name
        group = classify_variable(name)

        if group == "backbone":
            var_lr = compute_backbone_lr(
                name,
                lr_encoder=lr_encoder,
                lr_vit_layer_decay=lr_vit_layer_decay,
                lr_component_decay=lr_component_decay,
                num_layers=num_layers,
            )
            wd = weight_decay * get_vit_weight_decay_rate(name)
        elif group == "decoder":
            var_lr = lr * lr_component_decay
            wd = weight_decay
        else:  # heads, query embeds, projector, enc_out, …
            var_lr = lr
            wd = weight_decay

        # Scale relative to the base LR the optimizer will use.
        # The optimizer schedule outputs ``base_lr * lr_lambda(step)``.
        # We want the effective LR for this var to be
        #   ``var_lr * lr_lambda(step)``.
        # So the scale factor is ``var_lr / base_lr``.
        lr_scale = var_lr / lr if lr > 0 else 1.0
        result[name] = {"lr_scale": lr_scale, "wd": wd}

    return result


def scale_gradients_by_lr(grads, trainable_variables, lr_scale_map):
    """Multiply each gradient by its ``lr_scale``."""
    scaled = []
    for g, v in zip(grads, trainable_variables):
        if g is None:
            scaled.append(None)
            continue
        info = lr_scale_map.get(v.name)
        if info is not None and info["lr_scale"] != 1.0:
            g = g * info["lr_scale"]
        scaled.append(g)
    return scaled
