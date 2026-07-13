"""Development-only converter from the official DA3 checkpoint to Keras.

Torch, numpy, and safetensors are imported lazily; nothing here runs at
model call time. Covers the backbone; head and camera converters are added
with their modules.
"""
import numpy as np

BACKBONE = "backbone.pretrained."


def port_backbone_weights(model, state_dict, depth, num_positions,
                          hidden_size):
    state_dict = strip_model_prefix(state_dict)
    assign(model, "patch_embed_proj", patch_embedding(state_dict))
    assign(model, "cls_token", [reshape(state_dict, "cls_token", hidden_size)])
    assign(model, "pos_embed", [positions(state_dict, num_positions, hidden_size)])
    assign(model, "camera_token", [take(state_dict, "camera_token").reshape(2, hidden_size)])
    assign(model, "norm", norm(state_dict, "norm"))
    for index in range(depth):
        assign_block(model, state_dict, index)
    return model


def assign_block(model, state_dict, index):
    source = f"blocks.{index}."
    name = f"block_{index}"
    assign(model, f"{name}_norm1", norm(state_dict, source + "norm1"))
    assign(model, f"{name}_norm2", norm(state_dict, source + "norm2"))
    assign(model, f"{name}_qkv", dense(state_dict, source + "attn.qkv"))
    assign(model, f"{name}_proj", dense(state_dict, source + "attn.proj"))
    assign(model, f"{name}_mlp_fc1", dense(state_dict, source + "mlp.fc1"))
    assign(model, f"{name}_mlp_fc2", dense(state_dict, source + "mlp.fc2"))
    assign(model, f"{name}_ls1", [take(state_dict, source + "ls1.gamma")])
    assign(model, f"{name}_ls2", [take(state_dict, source + "ls2.gamma")])
    if index >= 4:
        assign(model, f"{name}_q_norm", norm(state_dict, source + "attn.q_norm"))
        assign(model, f"{name}_k_norm", norm(state_dict, source + "attn.k_norm"))


def patch_embedding(state_dict):
    kernel = take(state_dict, "patch_embed.proj.weight")
    bias = take(state_dict, "patch_embed.proj.bias")
    return [np.transpose(kernel, (2, 3, 1, 0)), bias]


def positions(state_dict, num_positions, hidden_size):
    source = take(state_dict, "pos_embed").reshape(-1, hidden_size)
    if source.shape[0] == num_positions:
        return source
    return interpolate_positions(source, num_positions, hidden_size)


def dense(state_dict, source):
    return [take(state_dict, source + ".weight").T,
            take(state_dict, source + ".bias")]


def norm(state_dict, source):
    return [take(state_dict, source + ".weight"),
            take(state_dict, source + ".bias")]


def reshape(state_dict, key, hidden_size):
    return take(state_dict, key).reshape(1, hidden_size)


def take(state_dict, key):
    return np.asarray(state_dict[BACKBONE + key])


def assign(model, name, arrays):
    layer = model.get_layer(name)
    current = [weight.shape for weight in layer.get_weights()]
    target = [tuple(array.shape) for array in arrays]
    if current != target:
        raise ValueError(f"{name} shape {target} does not fit {current}")
    layer.set_weights(arrays)


def strip_model_prefix(state_dict):
    if any(key.startswith("model.") for key in state_dict):
        return {key[len("model."):]: value for key, value in state_dict.items()}
    return state_dict


def interpolate_positions(source, num_positions, hidden_size):
    import torch
    import torch.nn.functional as functional
    class_position, patch_positions = source[:1], source[1:]
    source_grid = int(round(np.sqrt(patch_positions.shape[0])))
    target_grid = int(round(np.sqrt(num_positions - 1)))
    grid = patch_positions.reshape(1, source_grid, source_grid, hidden_size)
    grid = torch.from_numpy(grid).permute(0, 3, 1, 2)
    scale = (target_grid + 0.1) / source_grid
    grid = functional.interpolate(grid, scale_factor=(scale, scale),
                                  mode="bicubic", antialias=False)
    grid = grid.permute(0, 2, 3, 1).reshape(-1, hidden_size).numpy()
    return np.concatenate([class_position, grid], axis=0)
