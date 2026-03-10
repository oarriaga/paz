from typing import Any, Dict, List
import math


def get_vit_lr_decay_rate(name: str, lr_decay_rate: float = 1.0,
                          num_layers: int = 12) -> float:
    """Compute per-layer LR decay multiplier for ViT / DINOv2 backbones.

    Assigns a layer index based on the variable name and returns
    ``lr_decay_rate ** (num_layers + 1 - layer_id)``.

    Args:
        name (str): Variable name (may use ``/`` or ``.`` separators).
        lr_decay_rate (float): Base decay rate per layer.
        num_layers (int): Total number of transformer layers in the
            backbone.

    Returns:
        float: LR multiplier for this variable.
    """
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
    """Return weight decay for a variable, zeroing out bias/norm/embed.

    Args:
        name (str): Variable name.
        weight_decay_rate (float): Default weight-decay value returned
            for parameters that are *not* exempted.

    Returns:
        float: ``0.0`` for bias, norm, positional-embedding, and
            other embedding parameters; *weight_decay_rate* otherwise.
    """
    if (("gamma" in name) or ("pos_embed" in name)
            or ("rel_pos" in name) or ("bias" in name)
            or ("norm" in name) or ("embeddings" in name)):
        return 0.0
    return weight_decay_rate


# ---------------------------------------------------------------------------
# Classify a variable name into a parameter group
# ---------------------------------------------------------------------------


def classify_variable(name: str):
    """Classify a variable into a parameter group by name.

    Args:
        name (str): Variable name.

    Returns:
        str: One of ``'backbone'``, ``'decoder'``, or ``'other'``.
    """
    norm = name.replace("/", ".")
    if "backbone" in norm:
        return "backbone"
    if "transformer.decoder" in norm or "transformer/decoder" in name:
        return "decoder"
    return "other"


def compute_backbone_lr(name, *, lr_encoder, lr_vit_layer_decay,
                        lr_component_decay, num_layers):
    """Compute the effective LR for a single backbone variable.

    Formula::

        lr = lr_encoder * layer_decay(name) * lr_component_decay ** 2

    Args:
        name (str): Variable name.
        lr_encoder (float): Base backbone learning rate.
        lr_vit_layer_decay (float): Per-layer LR decay rate.
        lr_component_decay (float): Component-level LR decay multiplier.
        num_layers (int): Number of transformer layers in the backbone.

    Returns:
        float: Effective learning rate for this variable.
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
    """Build a per-variable LR-scale and weight-decay map.

    Returns a dictionary keyed by variable name, where each value
    contains:

    - ``lr_scale``: multiplier so that
      ``effective_lr(var) = base_lr_schedule(step) * lr_scale``.
    - ``wd``: per-variable weight decay (``0`` for bias/norm/embed).

    Args:
        model: Keras model.
        lr (float): Base learning rate for heads and query embeddings.
        lr_encoder (float): Base backbone learning rate.
        lr_vit_layer_decay (float): Per-layer LR decay in the backbone.
        lr_component_decay (float): Component-level LR multiplier.
        weight_decay (float): Default weight decay.
        num_layers (int): Number of transformer layers in the backbone.

    Returns:
        dict: ``{var_name: {"lr_scale": float, "wd": float}}``.
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
    """Scale each gradient by its variable's ``lr_scale``.

    Args:
        grads: List of gradient tensors (may contain ``None``).
        trainable_variables: Corresponding list of model variables.
        lr_scale_map (dict): Output of :func:`build_lr_scale_map`.

    Returns:
        list: Scaled gradient tensors.
    """
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
