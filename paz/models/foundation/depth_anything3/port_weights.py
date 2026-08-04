"""Development-only converter from the official DA3 checkpoint to Keras.

Torch, numpy, and safetensors are imported lazily; nothing here runs at
model call time. Covers the backbone, the DualDPT head, the standard DPT
head, and the camera decoder.
"""
import numpy as np

BACKBONE = "backbone.pretrained."


def port_backbone_weights(model, state_dict, depth, num_positions,
                          hidden_size, use_camera=True, use_qk_norm=True):
    state_dict = strip_model_prefix(state_dict)
    assign(model, "patch_embed_proj", patch_embedding(state_dict))
    assign(model, "cls_token", [reshape(state_dict, "cls_token", hidden_size)])
    pos = positions(state_dict, num_positions, hidden_size)
    assign(model, "pos_embed", [pos])
    if use_camera:
        camera = take(state_dict, "camera_token").reshape(2, hidden_size)
        assign(model, "camera_token", [camera])
    assign(model, "norm", norm(state_dict, "norm"))
    for index in range(depth):
        assign_block(model, state_dict, index, use_qk_norm)
    return model


def port_dpt_head_weights(model, state_dict):
    state_dict = strip_model_prefix(state_dict)
    port_projections(model, state_dict)
    port_refinenets(model, state_dict, "")
    head_conv(model, state_dict, "head_out1", "output_conv1")
    head_conv(model, state_dict, "head_out2_conv0", "output_conv2.0")
    head_conv(model, state_dict, "head_out2_conv1", "output_conv2.2")
    head_conv(model, state_dict, "head_sky_conv0", "sky_output_conv2.0")
    head_conv(model, state_dict, "head_sky_conv1", "sky_output_conv2.2")
    return model


def port_head_weights(model, state_dict):
    state_dict = strip_model_prefix(state_dict)
    assign(model, "head_norm", named_norm(state_dict, "head.norm"))
    port_projections(model, state_dict)
    for suffix in ("", "_aux"):
        port_refinenets(model, state_dict, suffix)
    head_conv(model, state_dict, "head_out1", "output_conv1")
    head_conv(model, state_dict, "head_out2_conv0", "output_conv2.0")
    head_conv(model, state_dict, "head_out2_conv1", "output_conv2.2")
    port_aux_outputs(model, state_dict)
    return model


def port_projections(model, state_dict):
    for stage in range(4):
        source = f"head.projects.{stage}"
        assign(model, f"head_project_{stage}", conv_bias(state_dict, source))
    for stage in (0, 1, 3):
        source = f"head.resize_layers.{stage}"
        assign(model, f"head_resize_{stage}", conv_bias(state_dict, source))
    for stage in range(1, 5):
        source = f"head.scratch.layer{stage}_rn"
        kernel = [conv_kernel(state_dict, source)]
        assign(model, f"head_layer{stage}_rn", kernel)


def port_aux_outputs(model, state_dict):
    for index in range(5):
        source = f"head.scratch.output_conv1_aux.3.{index}"
        assign(model, f"head_out1_aux_{index}", conv_bias(state_dict, source))
    head_conv(model, state_dict, "head_out2_aux_conv0", "output_conv2_aux.3.0")
    ln = named_norm(state_dict, "head.scratch.output_conv2_aux.0.2")
    assign(model, "head_out2_aux_ln", ln)
    head_conv(model, state_dict, "head_out2_aux_conv1", "output_conv2_aux.3.5")


def port_refinenets(model, state_dict, suffix):
    for index in (1, 2, 3):
        source = f"head.scratch.refinenet{index}{suffix}.resConfUnit1"
        assign(model, f"head_refine{index}{suffix}_unit1_conv1",
               conv_bias(state_dict, source + ".conv1"))
        assign(model, f"head_refine{index}{suffix}_unit1_conv2",
               conv_bias(state_dict, source + ".conv2"))
    for index in (1, 2, 3, 4):
        block = f"head.scratch.refinenet{index}{suffix}"
        assign(model, f"head_refine{index}{suffix}_unit2_conv1",
               conv_bias(state_dict, block + ".resConfUnit2.conv1"))
        assign(model, f"head_refine{index}{suffix}_unit2_conv2",
               conv_bias(state_dict, block + ".resConfUnit2.conv2"))
        assign(model, f"head_refine{index}{suffix}_out",
               conv_bias(state_dict, block + ".out_conv"))


def port_camera_decoder_weights(model, state_dict):
    state_dict = strip_model_prefix(state_dict)
    assign(model, "cam_dec_fc1", named_dense(state_dict, "cam_dec.backbone.0"))
    assign(model, "cam_dec_fc2", named_dense(state_dict, "cam_dec.backbone.2"))
    assign(model, "cam_dec_t", named_dense(state_dict, "cam_dec.fc_t"))
    assign(model, "cam_dec_qvec", named_dense(state_dict, "cam_dec.fc_qvec"))
    assign(model, "cam_dec_fov", named_dense(state_dict, "cam_dec.fc_fov.0"))
    return model


def assign_block(model, state_dict, index, use_qk_norm=True):
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
    if use_qk_norm and index >= 4:
        assign_qk_norm(model, state_dict, source, name)


def assign_qk_norm(model, state_dict, source, name):
    assign(model, f"{name}_q_norm", norm(state_dict, source + "attn.q_norm"))
    assign(model, f"{name}_k_norm", norm(state_dict, source + "attn.k_norm"))


def head_conv(model, state_dict, target, source):
    assign(model, target, conv_bias(state_dict, "head.scratch." + source))


def conv_bias(state_dict, source):
    return [conv_kernel(state_dict, source),
            np.asarray(state_dict[source + ".bias"])]


def conv_kernel(state_dict, source):
    kernel = np.asarray(state_dict[source + ".weight"])
    return np.transpose(kernel, (2, 3, 1, 0))


def named_dense(state_dict, source):
    return [np.asarray(state_dict[source + ".weight"]).T,
            np.asarray(state_dict[source + ".bias"])]


def named_norm(state_dict, source):
    return [np.asarray(state_dict[source + ".weight"]),
            np.asarray(state_dict[source + ".bias"])]


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
