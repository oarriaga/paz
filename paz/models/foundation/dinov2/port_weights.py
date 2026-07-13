"""Development-only converter from official torch DINOv2 to canonical Keras.

Maps every source parameter explicitly, transposes Dense and Conv2D kernels,
preserves fused QKV order, and interpolates positional embeddings to the
constructed grid. Not imported at runtime; torch/numpy are imported lazily.
"""
import numpy as np


def port_weights(model, state_dict, num_positions, hidden_size, depth):
    expected = set()
    assign_patch_embedding(model, state_dict, expected)
    assign_class_token(model, state_dict, hidden_size, expected)
    assign_position_embedding(model, state_dict, num_positions, hidden_size,
                              expected)
    for index in range(depth):
        assign_block(model, state_dict, index, expected)
    assign_final_norm(model, state_dict, expected)
    reject_unmapped_keys(state_dict, expected)
    return model


def assign_patch_embedding(model, state_dict, expected):
    kernel = take(state_dict, "patch_embed.proj.weight", expected)
    bias = take(state_dict, "patch_embed.proj.bias", expected)
    kernel = np.transpose(kernel, (2, 3, 1, 0))
    set_layer(model, "patch_embed_proj", [kernel, bias])


def assign_class_token(model, state_dict, hidden_size, expected):
    token = take(state_dict, "cls_token", expected).reshape(1, hidden_size)
    set_layer(model, "cls_token", [token])


def assign_position_embedding(model, state_dict, num_positions, hidden_size,
                              expected):
    source = take(state_dict, "pos_embed", expected).reshape(-1, hidden_size)
    resized = interpolate_positions(source, num_positions, hidden_size)
    set_layer(model, "pos_embed", [resized])


def assign_block(model, state_dict, index, expected):
    prefix = f"blocks.{index}"
    name = f"block_{index}"
    set_norm(model, state_dict, f"{prefix}.norm1", f"{name}_norm1", expected)
    set_norm(model, state_dict, f"{prefix}.norm2", f"{name}_norm2", expected)
    set_dense(model, state_dict, f"{prefix}.attn.qkv", f"{name}_qkv", expected)
    set_dense(model, state_dict, f"{prefix}.attn.proj", f"{name}_proj", expected)
    set_dense(model, state_dict, f"{prefix}.mlp.fc1", f"{name}_mlp_fc1", expected)
    set_dense(model, state_dict, f"{prefix}.mlp.fc2", f"{name}_mlp_fc2", expected)
    scale1 = take(state_dict, f"{prefix}.ls1.gamma", expected)
    scale2 = take(state_dict, f"{prefix}.ls2.gamma", expected)
    set_layer(model, f"{name}_ls1", [scale1])
    set_layer(model, f"{name}_ls2", [scale2])


def assign_final_norm(model, state_dict, expected):
    set_norm(model, state_dict, "norm", "norm", expected)


def set_dense(model, state_dict, source, target, expected):
    kernel = take(state_dict, f"{source}.weight", expected).T
    bias = take(state_dict, f"{source}.bias", expected)
    set_layer(model, target, [kernel, bias])


def set_norm(model, state_dict, source, target, expected):
    gamma = take(state_dict, f"{source}.weight", expected)
    beta = take(state_dict, f"{source}.bias", expected)
    set_layer(model, target, [gamma, beta])


def set_layer(model, name, arrays):
    layer = model.get_layer(name)
    current = [weight.shape for weight in layer.get_weights()]
    target = [tuple(array.shape) for array in arrays]
    if current != target:
        raise ValueError(f"{name} shape {target} does not fit {current}")
    layer.set_weights(arrays)


def take(state_dict, key, expected):
    if key not in state_dict:
        raise KeyError(f"missing source key: {key}")
    expected.add(key)
    return np.asarray(state_dict[key])


def reject_unmapped_keys(state_dict, expected):
    unused = set(state_dict) - expected
    if unused:
        raise KeyError(f"unexpected source keys: {sorted(unused)}")


def interpolate_positions(source, num_positions, hidden_size):
    if source.shape[0] == num_positions:
        return source
    class_position, patch_positions = source[:1], source[1:]
    source_grid = int(round(np.sqrt(patch_positions.shape[0])))
    target_grid = int(round(np.sqrt(num_positions - 1)))
    resized = resize_grid(patch_positions, source_grid, target_grid)
    return np.concatenate([class_position, resized], axis=0)


def resize_grid(positions, source_grid, target_grid):
    import torch
    import torch.nn.functional as functional
    hidden_size = positions.shape[-1]
    grid = positions.reshape(1, source_grid, source_grid, hidden_size)
    grid = torch.from_numpy(grid).permute(0, 3, 1, 2)
    size = (target_grid, target_grid)
    grid = functional.interpolate(grid, size=size, mode="bicubic",
                                  align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(-1, hidden_size)
    return grid.numpy()
