def read_vit_layer_id(name, num_layers):
    layer_id = num_layers + 1
    normalized = name.replace("/", ".")
    inside = ".layer." in normalized and ".residual." not in normalized
    if normalized.startswith("backbone") and "embeddings" in normalized:
        layer_id = 0
    elif normalized.startswith("backbone") and inside:
        tail = normalized[normalized.find(".layer."):]
        layer_id = int(tail.split(".")[2]) + 1
    return layer_id


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    layer_id = read_vit_layer_id(name, num_layers)
    return lr_decay_rate ** (num_layers + 1 - layer_id)


NO_DECAY_TOKENS = ("gamma", "pos_embed", "rel_pos", "bias", "norm", "embeddings")  # fmt: skip


def get_vit_weight_decay_rate(name, weight_decay_rate=1.0):
    if any(token in name for token in NO_DECAY_TOKENS):
        weight_decay_rate = 0.0
    return weight_decay_rate




def classify_variable(name):
    norm = name.replace("/", ".")
    if "backbone" in norm:
        group = "backbone"
    elif "transformer.decoder" in norm or "transformer/decoder" in name:
        group = "decoder"
    else:
        group = "other"
    return group


def compute_backbone_lr(name, *, lr_encoder, lr_vit_layer_decay, lr_component_decay, num_layers):  # fmt: skip
    layer_decay = get_vit_lr_decay_rate(
        name, lr_decay_rate=lr_vit_layer_decay, num_layers=num_layers)
    return lr_encoder * layer_decay * (lr_component_decay ** 2)




def compute_variable_rates(name, *, lr, lr_encoder, lr_vit_layer_decay, lr_component_decay, weight_decay, num_layers):  # fmt: skip
    group = classify_variable(name)
    # Anything not backbone or decoder (heads, query embeds, projector,
    # enc_out) trains at the base learning rate.
    variable_lr = lr
    decay = weight_decay
    if group == "backbone":
        keys = ("lr_encoder", "lr_vit_layer_decay", "lr_component_decay", "num_layers")  # fmt: skip
        values = (lr_encoder, lr_vit_layer_decay, lr_component_decay, num_layers)  # fmt: skip
        variable_lr = compute_backbone_lr(name, **dict(zip(keys, values)))
        decay = weight_decay * get_vit_weight_decay_rate(name)
    if group == "decoder":
        variable_lr = lr * lr_component_decay
    return variable_lr, decay


def build_lr_scale_map(model, *, lr, lr_encoder, lr_vit_layer_decay, lr_component_decay, weight_decay, num_layers):  # fmt: skip
    keys = ("lr", "lr_encoder", "lr_vit_layer_decay", "lr_component_decay", "weight_decay", "num_layers")  # fmt: skip
    values = (lr, lr_encoder, lr_vit_layer_decay, lr_component_decay, weight_decay, num_layers)  # fmt: skip
    kwargs = dict(zip(keys, values))
    result = {}
    for variable in model.trainable_variables:
        variable_lr, decay = compute_variable_rates(variable.name, **kwargs)
        # The optimizer schedule outputs ``base_lr * lr_lambda(step)``, so a
        # per-variable factor of ``variable_lr / base_lr`` yields the wanted
        # effective rate ``variable_lr * lr_lambda(step)``.
        lr_scale = variable_lr / lr if lr > 0 else 1.0
        result[variable.name] = {"lr_scale": lr_scale, "wd": decay}
    return result


def scale_gradient(gradient, variable, lr_scale_map):
    info = lr_scale_map.get(variable.name)
    if gradient is not None and info is not None:
        gradient = gradient * info["lr_scale"]
    return gradient


def scale_gradients_by_lr(grads, trainable_variables, lr_scale_map):
    paired = zip(grads, trainable_variables)
    return [scale_gradient(g, v, lr_scale_map) for g, v in paired]
