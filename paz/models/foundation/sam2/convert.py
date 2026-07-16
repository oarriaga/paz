"""Convert an official SAM 2 checkpoint into the PAZ image-inference models.

Maps every image-required parameter explicitly, transposes convolution and
dense kernels, and fails on missing, mismatched, or unexpected image keys.
Video-memory parameters are listed as deferred and reported, never silently
ignored. Takes a plain ``{key: ndarray}`` state dict so the runtime env needs
no torch, mirroring the dinov2 converter.
"""
import numpy as np

from paz.models.foundation.sam2 import hiera

TRUNK = "image_encoder.trunk."
NECK = "image_encoder.neck."
PROMPT = "sam_prompt_encoder."
DECODER = "sam_mask_decoder."
TRANSFORMER = "sam_mask_decoder.transformer."

DEFERRED = ("memory_attention", "memory_encoder", "maskmem_tpos_enc",
            "no_mem_pos_enc", "no_obj_ptr", "no_obj_embed_spatial",
            "mask_downsample", "obj_ptr_proj", "obj_ptr_tpos_proj")


def convert(models, state_dict, used=None):
    used = set() if used is None else used
    convert_image_encoder(models.image_encoder, state_dict, models.config, used)
    convert_point_encoder(models.point_encoder, state_dict, used)
    convert_mask_downscaling(models.mask_downscaling, state_dict, used)
    convert_mask_decoder(models.mask_decoder, state_dict, used)
    reject_unused_image_keys(state_dict, used)
    return models


def convert_image_encoder(model, state_dict, config, used):
    set_conv(model, "patch_embed_proj", TRUNK + "patch_embed.proj",
             state_dict, used)
    background = take(state_dict, TRUNK + "pos_embed", used)
    window = take(state_dict, TRUNK + "pos_embed_window", used)
    set_layer(model, "trunk_pos_embed", [to_last(background), to_last(window)])
    specifications, _ = hiera.build_block_specifications(config)
    for index, dim, dim_out, _, _, _, name in specifications:
        block = f"{TRUNK}blocks.{index}"
        set_norm(model, f"{name}_norm1", f"{block}.norm1", state_dict, used)
        set_norm(model, f"{name}_norm2", f"{block}.norm2", state_dict, used)
        set_dense(model, f"{name}_attn_qkv", f"{block}.attn.qkv", state_dict,
                  used)
        set_dense(model, f"{name}_attn_proj", f"{block}.attn.proj", state_dict,
                  used)
        set_dense(model, f"{name}_mlp_fc1", f"{block}.mlp.layers.0",
                  state_dict, used)
        set_dense(model, f"{name}_mlp_fc2", f"{block}.mlp.layers.1",
                  state_dict, used)
        if dim != dim_out:
            set_dense(model, f"{name}_proj", f"{block}.proj", state_dict, used)
    for index in range(4):
        set_conv(model, f"neck_conv_{index}",
                 f"{NECK}convs.{index}.conv", state_dict, used)
    set_conv(model, "sam_mask_decoder_conv_s0", DECODER + "conv_s0",
             state_dict, used)
    set_conv(model, "sam_mask_decoder_conv_s1", DECODER + "conv_s1",
             state_dict, used)
    vector = take(state_dict, "no_mem_embed", used).reshape(-1)
    set_layer(model, "no_mem_embed", [vector])


def convert_point_encoder(model, state_dict, used):
    matrix = take(state_dict, PROMPT + "pe_layer."
                  "positional_encoding_gaussian_matrix", used)
    set_layer(model, "prompt_pe", [matrix])
    corners = [take(state_dict, f"{PROMPT}point_embeddings.{index}.weight",
                    used)[0] for index in range(4)]
    not_a_point = take(state_dict, PROMPT + "not_a_point_embed.weight", used)
    set_layer(model, "point_label_embed", [np.stack(corners), not_a_point])


def convert_mask_downscaling(model, state_dict, used):
    set_conv(model, "mask_down_0", PROMPT + "mask_downscaling.0", state_dict,
             used)
    set_norm(model, "mask_down_ln0", PROMPT + "mask_downscaling.1", state_dict,
             used)
    set_conv(model, "mask_down_3", PROMPT + "mask_downscaling.3", state_dict,
             used)
    set_norm(model, "mask_down_ln3", PROMPT + "mask_downscaling.4", state_dict,
             used)
    set_conv(model, "mask_down_6", PROMPT + "mask_downscaling.6", state_dict,
             used)
    no_mask = take(state_dict, PROMPT + "no_mask_embed.weight", used)
    set_layer(model, "no_mask_embed", [no_mask.reshape(-1)])


def convert_mask_decoder(model, state_dict, used):
    set_layer(model, "obj_score_token",
              [take(state_dict, DECODER + "obj_score_token.weight", used)])
    set_layer(model, "iou_token",
              [take(state_dict, DECODER + "iou_token.weight", used)])
    set_layer(model, "mask_tokens",
              [take(state_dict, DECODER + "mask_tokens.weight", used)])
    for index in range(2):
        convert_two_way_block(model, state_dict, index, used)
    convert_attention(model, "twoway_final_attn",
                      TRANSFORMER + "final_attn_token_to_image", state_dict,
                      used)
    set_norm(model, "twoway_norm_final", TRANSFORMER + "norm_final_attn",
             state_dict, used)
    set_conv_transpose(model, "output_upscaling_0",
                       DECODER + "output_upscaling.0", state_dict, used)
    set_norm(model, "output_upscaling_1", DECODER + "output_upscaling.1",
             state_dict, used)
    set_conv_transpose(model, "output_upscaling_3",
                       DECODER + "output_upscaling.3", state_dict, used)
    for index in range(4):
        convert_mlp(model, f"output_hypernetworks_mlps_{index}",
                    f"{DECODER}output_hypernetworks_mlps.{index}", 3,
                    state_dict, used)
    convert_mlp(model, "iou_prediction_head",
                DECODER + "iou_prediction_head", 3, state_dict, used)
    convert_mlp(model, "pred_obj_score_head",
                DECODER + "pred_obj_score_head", 3, state_dict, used)


def convert_two_way_block(model, state_dict, index, used):
    block = f"{TRANSFORMER}layers.{index}"
    name = f"twoway_{index}"
    convert_attention(model, f"{name}_self", f"{block}.self_attn", state_dict,
                      used)
    convert_attention(model, f"{name}_cross_t2i",
                      f"{block}.cross_attn_token_to_image", state_dict, used)
    convert_attention(model, f"{name}_cross_i2t",
                      f"{block}.cross_attn_image_to_token", state_dict, used)
    convert_mlp(model, f"{name}_mlp", f"{block}.mlp", 2, state_dict, used)
    for norm in range(1, 5):
        set_norm(model, f"{name}_norm{norm}", f"{block}.norm{norm}",
                 state_dict, used)


def convert_attention(model, name, source, state_dict, used):
    for part in ("q", "k", "v", "out"):
        target = f"{name}_{part}"
        set_dense(model, target, f"{source}.{part}_proj", state_dict, used)


def convert_mlp(model, name, source, layers, state_dict, used):
    for index in range(layers):
        set_dense(model, f"{name}_layers_{index}", f"{source}.layers.{index}",
                  state_dict, used)


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


def reject_unused_image_keys(state_dict, used):
    unused = [key for key in state_dict if key not in used]
    unexpected = [key for key in unused if not key.startswith(DEFERRED)]
    if unexpected:
        raise KeyError(f"unexpected image keys: {sorted(unexpected)}")
