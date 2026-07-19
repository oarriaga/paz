"""Convert an official SAM 2 checkpoint into the PAZ image-inference models.

Maps every image-required parameter explicitly, transposes convolution and
dense kernels, and fails on any unmapped image-prefix key. Video-memory
parameters (memory_attention/memory_encoder, maskmem*, obj_ptr*, no_mem_pos*,
no_obj*, mask_downsample) are deferred and tolerated, never silently mapped.
Takes a plain ``{key: ndarray}`` state dict so the runtime needs no torch.
"""
import numpy as np

from paz.models.foundation.sam2 import hiera

TRUNK = "image_encoder.trunk."
NECK = "image_encoder.neck."
PROMPT = "sam_prompt_encoder."
DECODER = "sam_mask_decoder."
TRANSFORMER = "sam_mask_decoder.transformer."

# Only keys under these prefixes must be mapped; everything else is deferred.
IMAGE = "image_encoder sam_prompt_encoder sam_mask_decoder no_mem_embed".split()


def convert(models, state_dict, used=None):
    used = set() if used is None else used
    convert_image_encoder(models.image_encoder, state_dict, models.config, used)
    convert_point_encoder(models.point_encoder, state_dict, used)
    convert_mask_downscaling(models.mask_downscaling, state_dict, used)
    convert_mask_decoder(models.mask_decoder, state_dict, used)
    reject_unmapped_image_keys(state_dict, used)
    return models


def convert_image_encoder(model, state_dict, config, used):
    def dense(name, source):
        set_dense(model, name, source, state_dict, used)

    def conv(name, source):
        set_conv(model, name, source, state_dict, used)

    def norm(name, source):
        set_norm(model, name, source, state_dict, used)

    conv("patch_embed_proj", TRUNK + "patch_embed.proj")
    background = take(state_dict, TRUNK + "pos_embed", used)
    window = take(state_dict, TRUNK + "pos_embed_window", used)
    set_layer(model, "trunk_pos_embed", [to_last(background), to_last(window)])
    specifications, _ = hiera.build_block_specifications(config)
    for index, dim, dim_out, _, _, _, name in specifications:
        block = f"{TRUNK}blocks.{index}"
        norm(f"{name}_norm1", f"{block}.norm1")
        norm(f"{name}_norm2", f"{block}.norm2")
        dense(f"{name}_attn_qkv", f"{block}.attn.qkv")
        dense(f"{name}_attn_proj", f"{block}.attn.proj")
        dense(f"{name}_mlp_fc1", f"{block}.mlp.layers.0")
        dense(f"{name}_mlp_fc2", f"{block}.mlp.layers.1")
        if dim != dim_out:
            dense(f"{name}_proj", f"{block}.proj")
    for index in range(4):
        conv(f"neck_conv_{index}", f"{NECK}convs.{index}.conv")
    conv("sam_mask_decoder_conv_s0", DECODER + "conv_s0")
    conv("sam_mask_decoder_conv_s1", DECODER + "conv_s1")
    vector = take(state_dict, "no_mem_embed", used).reshape(-1)
    set_layer(model, "no_mem_embed", [vector])


def convert_point_encoder(model, state_dict, used):
    key = PROMPT + "pe_layer.positional_encoding_gaussian_matrix"
    set_layer(model, "prompt_pe", [take(state_dict, key, used)])
    corners = []
    for index in range(4):
        source = f"{PROMPT}point_embeddings.{index}.weight"
        corners.append(take(state_dict, source, used)[0])
    not_a_point = take(state_dict, PROMPT + "not_a_point_embed.weight", used)
    set_layer(model, "point_label_embed", [np.stack(corners), not_a_point])


def convert_mask_downscaling(model, state_dict, used):
    def conv(name, source):
        set_conv(model, name, source, state_dict, used)

    def norm(name, source):
        set_norm(model, name, source, state_dict, used)

    conv("mask_down_0", PROMPT + "mask_downscaling.0")
    norm("mask_down_ln0", PROMPT + "mask_downscaling.1")
    conv("mask_down_3", PROMPT + "mask_downscaling.3")
    norm("mask_down_ln3", PROMPT + "mask_downscaling.4")
    conv("mask_down_6", PROMPT + "mask_downscaling.6")
    no_mask = take(state_dict, PROMPT + "no_mask_embed.weight", used)
    set_layer(model, "no_mask_embed", [no_mask.reshape(-1)])


def convert_mask_decoder(model, state_dict, used):
    def dense(name, source):
        set_dense(model, name, source, state_dict, used)

    def norm(name, source):
        set_norm(model, name, source, state_dict, used)

    def conv_t(name, source):
        set_conv_transpose(model, name, source, state_dict, used)

    for token in ("obj_score_token", "iou_token", "mask_tokens"):
        weight = take(state_dict, f"{DECODER}{token}.weight", used)
        set_layer(model, token, [weight])
    for index in range(2):
        convert_two_way_block(dense, norm, index)
    final = TRANSFORMER + "final_attn_token_to_image"
    convert_attention(dense, "twoway_final_attn", final)
    norm("twoway_norm_final", TRANSFORMER + "norm_final_attn")
    conv_t("output_upscaling_0", DECODER + "output_upscaling.0")
    norm("output_upscaling_1", DECODER + "output_upscaling.1")
    conv_t("output_upscaling_3", DECODER + "output_upscaling.3")
    for index in range(4):
        name = f"output_hypernetworks_mlps_{index}"
        source = f"{DECODER}output_hypernetworks_mlps.{index}"
        convert_mlp(dense, name, source, 3)
    iou_head = DECODER + "iou_prediction_head"
    obj_head = DECODER + "pred_obj_score_head"
    convert_mlp(dense, "iou_prediction_head", iou_head, 3)
    convert_mlp(dense, "pred_obj_score_head", obj_head, 3)


def convert_two_way_block(dense, norm, index):
    block = f"{TRANSFORMER}layers.{index}"
    name = f"twoway_{index}"
    convert_attention(dense, f"{name}_self", f"{block}.self_attn")
    t2i = f"{block}.cross_attn_token_to_image"
    i2t = f"{block}.cross_attn_image_to_token"
    convert_attention(dense, f"{name}_cross_t2i", t2i)
    convert_attention(dense, f"{name}_cross_i2t", i2t)
    convert_mlp(dense, f"{name}_mlp", f"{block}.mlp", 2)
    for number in range(1, 5):
        norm(f"{name}_norm{number}", f"{block}.norm{number}")


def convert_attention(dense, name, source):
    for part in ("q", "k", "v", "out"):
        dense(f"{name}_{part}", f"{source}.{part}_proj")


def convert_mlp(dense, name, source, layers):
    for index in range(layers):
        dense(f"{name}_layers_{index}", f"{source}.layers.{index}")


def set_dense(model, name, source, state_dict, used):
    kernel = take(state_dict, f"{source}.weight", used).T
    bias = take(state_dict, f"{source}.bias", used)
    set_layer(model, name, [kernel, bias])


def set_conv(model, name, source, state_dict, used):
    kernel = take(state_dict, f"{source}.weight", used)
    bias = take(state_dict, f"{source}.bias", used)
    set_layer(model, name, [np.transpose(kernel, (2, 3, 1, 0)), bias])


def set_conv_transpose(model, name, source, state_dict, used):
    kernel = take(state_dict, f"{source}.weight", used)
    bias = take(state_dict, f"{source}.bias", used)
    set_layer(model, name, [np.transpose(kernel, (2, 3, 1, 0)), bias])


def set_norm(model, name, source, state_dict, used):
    gamma = take(state_dict, f"{source}.weight", used)
    beta = take(state_dict, f"{source}.bias", used)
    set_layer(model, name, [gamma, beta])


def to_last(parameter):
    return parameter[0].transpose(1, 2, 0)


def set_layer(model, name, arrays):
    layer = model.get_layer(name)
    current = [tuple(weight.shape) for weight in layer.get_weights()]
    target = [tuple(array.shape) for array in arrays]
    if current != target:
        raise ValueError(f"{name} shapes {target} do not fit {current}")
    layer.set_weights(arrays)


def take(state_dict, key, used):
    if key not in state_dict:
        raise KeyError(f"missing source key: {key}")
    used.add(key)
    return np.asarray(state_dict[key])


def reject_unmapped_image_keys(state_dict, used):
    unused = [key for key in state_dict if key not in used]
    missed = [k for k in unused if any(k.startswith(p) for p in IMAGE)]
    if missed:
        raise KeyError(f"unmapped image keys: {sorted(missed)}")
