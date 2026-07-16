"""Converter coverage and strictness without torch.

Builds a torch-style state dict from a constructed bundle (the inverse of the
converter), converts it into a fresh bundle, and checks the mapping is
complete and reproduces the outputs. Also checks missing and unexpected keys
are rejected and deferred video keys are tolerated.
"""
import numpy as np
import pytest

from paz.models.foundation.sam2 import model as sam2_model, convert, hiera
from paz.models.foundation.sam2.configuration import TINY

TRUNK = convert.TRUNK
NECK = convert.NECK
PROMPT = convert.PROMPT
DECODER = convert.DECODER
TRANSFORMER = convert.TRANSFORMER


def dense(model, name, source, state):
    kernel, bias = model.get_layer(name).get_weights()
    state[f"{source}.weight"] = kernel.T
    state[f"{source}.bias"] = bias


def conv(model, name, source, state):
    kernel, bias = model.get_layer(name).get_weights()
    state[f"{source}.weight"] = np.transpose(kernel, (3, 2, 0, 1))
    state[f"{source}.bias"] = bias


def norm(model, name, source, state):
    gamma, beta = model.get_layer(name).get_weights()
    state[f"{source}.weight"] = gamma
    state[f"{source}.bias"] = beta


def to_torch_state_dict(bundle):
    state = {}
    encode_image_encoder(bundle.image_encoder, state)
    encode_point_encoder(bundle.point_encoder, state)
    encode_mask_downscaling(bundle.mask_downscaling, state)
    encode_mask_decoder(bundle.mask_decoder, state)
    return state


def encode_image_encoder(model, state):
    conv(model, "patch_embed_proj", TRUNK + "patch_embed.proj", state)
    background, window = model.get_layer("trunk_pos_embed").get_weights()
    state[TRUNK + "pos_embed"] = background.transpose(2, 0, 1)[None]
    state[TRUNK + "pos_embed_window"] = window.transpose(2, 0, 1)[None]
    specifications, _ = hiera.build_block_specifications(TINY)
    for index, dim, dim_out, _, _, _, name in specifications:
        block = f"{TRUNK}blocks.{index}"
        norm(model, f"{name}_norm1", f"{block}.norm1", state)
        norm(model, f"{name}_norm2", f"{block}.norm2", state)
        dense(model, f"{name}_attn_qkv", f"{block}.attn.qkv", state)
        dense(model, f"{name}_attn_proj", f"{block}.attn.proj", state)
        dense(model, f"{name}_mlp_fc1", f"{block}.mlp.layers.0", state)
        dense(model, f"{name}_mlp_fc2", f"{block}.mlp.layers.1", state)
        if dim != dim_out:
            dense(model, f"{name}_proj", f"{block}.proj", state)
    for index in range(4):
        conv(model, f"neck_conv_{index}", f"{NECK}convs.{index}.conv", state)
    conv(model, "sam_mask_decoder_conv_s0", DECODER + "conv_s0", state)
    conv(model, "sam_mask_decoder_conv_s1", DECODER + "conv_s1", state)
    vector = model.get_layer("no_mem_embed").get_weights()[0]
    state["no_mem_embed"] = vector.reshape(1, 1, -1)


def encode_point_encoder(model, state):
    matrix = model.get_layer("prompt_pe").get_weights()[0]
    state[PROMPT + "pe_layer.positional_encoding_gaussian_matrix"] = matrix
    corners, not_a_point = model.get_layer("point_label_embed").get_weights()
    for index in range(4):
        state[f"{PROMPT}point_embeddings.{index}.weight"] = corners[index:
                                                                     index + 1]
    state[PROMPT + "not_a_point_embed.weight"] = not_a_point


def encode_mask_downscaling(model, state):
    conv(model, "mask_down_0", PROMPT + "mask_downscaling.0", state)
    norm(model, "mask_down_ln0", PROMPT + "mask_downscaling.1", state)
    conv(model, "mask_down_3", PROMPT + "mask_downscaling.3", state)
    norm(model, "mask_down_ln3", PROMPT + "mask_downscaling.4", state)
    conv(model, "mask_down_6", PROMPT + "mask_downscaling.6", state)
    no_mask = model.get_layer("no_mask_embed").get_weights()[0]
    state[PROMPT + "no_mask_embed.weight"] = no_mask.reshape(1, -1)


def encode_mask_decoder(model, state):
    for name, source in (("obj_score_token", "obj_score_token"),
                         ("iou_token", "iou_token"),
                         ("mask_tokens", "mask_tokens")):
        state[DECODER + f"{source}.weight"] = \
            model.get_layer(name).get_weights()[0]
    for index in range(2):
        encode_two_way_block(model, index, state)
    encode_attention(model, "twoway_final_attn",
                     TRANSFORMER + "final_attn_token_to_image", state)
    norm(model, "twoway_norm_final", TRANSFORMER + "norm_final_attn", state)
    encode_conv_transpose(model, "output_upscaling_0",
                          DECODER + "output_upscaling.0", state)
    norm(model, "output_upscaling_1", DECODER + "output_upscaling.1", state)
    encode_conv_transpose(model, "output_upscaling_3",
                          DECODER + "output_upscaling.3", state)
    for index in range(4):
        encode_mlp(model, f"output_hypernetworks_mlps_{index}",
                   f"{DECODER}output_hypernetworks_mlps.{index}", 3, state)
    encode_mlp(model, "iou_prediction_head",
               DECODER + "iou_prediction_head", 3, state)
    encode_mlp(model, "pred_obj_score_head",
               DECODER + "pred_obj_score_head", 3, state)


def encode_two_way_block(model, index, state):
    block = f"{TRANSFORMER}layers.{index}"
    name = f"twoway_{index}"
    encode_attention(model, f"{name}_self", f"{block}.self_attn", state)
    encode_attention(model, f"{name}_cross_t2i",
                     f"{block}.cross_attn_token_to_image", state)
    encode_attention(model, f"{name}_cross_i2t",
                     f"{block}.cross_attn_image_to_token", state)
    encode_mlp(model, f"{name}_mlp", f"{block}.mlp", 2, state)
    for number in range(1, 5):
        norm(model, f"{name}_norm{number}", f"{block}.norm{number}", state)


def encode_attention(model, name, source, state):
    for part in ("q", "k", "v", "out"):
        dense(model, f"{name}_{part}", f"{source}.{part}_proj", state)


def encode_mlp(model, name, source, layers, state):
    for index in range(layers):
        dense(model, f"{name}_layers_{index}", f"{source}.layers.{index}",
              state)


def encode_conv_transpose(model, name, source, state):
    kernel, bias = model.get_layer(name).get_weights()
    state[f"{source}.weight"] = np.transpose(kernel, (3, 2, 0, 1))
    state[f"{source}.bias"] = bias


def randomize(bundle):
    generator = np.random.RandomState(1)
    for model in bundle[:4]:
        model.set_weights([generator.randn(*w.shape).astype("float32") * 0.05
                           for w in model.get_weights()])


def test_converter_reproduces_source():
    source = sam2_model.build(TINY)
    randomize(source)
    state = to_torch_state_dict(source)
    target = sam2_model.build(TINY)
    used = set()
    convert.convert(target, state, used)
    assert used == set(state)
    inputs = [np.zeros((1, 64, 64, 256), np.float32),
              np.zeros((1, 256, 256, 32), np.float32),
              np.zeros((1, 128, 128, 64), np.float32),
              np.zeros((1, 2, 256), np.float32),
              np.zeros((1, 64, 64, 256), np.float32),
              np.zeros((1, 64, 64, 256), np.float32)]
    expected = np.array(source.mask_decoder(inputs)[0])
    result = np.array(target.mask_decoder(inputs)[0])
    assert np.allclose(expected, result, atol=1e-5)


def test_converter_rejects_missing_key():
    bundle = sam2_model.build(TINY)
    state = to_torch_state_dict(bundle)
    del state[DECODER + "iou_token.weight"]
    with pytest.raises(KeyError):
        convert.convert(sam2_model.build(TINY), state)


def test_converter_rejects_unexpected_key():
    bundle = sam2_model.build(TINY)
    state = to_torch_state_dict(bundle)
    state["sam_mask_decoder.surprise"] = np.zeros((2,), np.float32)
    with pytest.raises(KeyError):
        convert.convert(sam2_model.build(TINY), state)


def test_converter_tolerates_deferred_keys():
    bundle = sam2_model.build(TINY)
    state = to_torch_state_dict(bundle)
    state["memory_encoder.deferred"] = np.zeros((2,), np.float32)
    convert.convert(sam2_model.build(TINY), state)
