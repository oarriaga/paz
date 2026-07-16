"""Converter coverage and strictness without torch.

Builds a torch-style state dict from a constructed bundle (the inverse of the
converter), converts it into a fresh bundle, and checks the mapping is
complete and reproduces the outputs. Also checks unmapped image keys are
rejected and deferred video keys are tolerated.
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


def setters(model, state):
    def dense(name, source):
        kernel, bias = model.get_layer(name).get_weights()
        state[f"{source}.weight"] = kernel.T
        state[f"{source}.bias"] = bias

    def conv(name, source):
        kernel, bias = model.get_layer(name).get_weights()
        state[f"{source}.weight"] = np.transpose(kernel, (3, 2, 0, 1))
        state[f"{source}.bias"] = bias

    def norm(name, source):
        gamma, beta = model.get_layer(name).get_weights()
        state[f"{source}.weight"] = gamma
        state[f"{source}.bias"] = beta

    return dense, conv, norm


def to_torch_state_dict(bundle):
    state = {}
    encode_image_encoder(bundle.image_encoder, state)
    encode_point_encoder(bundle.point_encoder, state)
    encode_mask_downscaling(bundle.mask_downscaling, state)
    encode_mask_decoder(bundle.mask_decoder, state)
    return state


def encode_image_encoder(model, state):
    dense, conv, norm = setters(model, state)
    conv("patch_embed_proj", TRUNK + "patch_embed.proj")
    background, window = model.get_layer("trunk_pos_embed").get_weights()
    state[TRUNK + "pos_embed"] = background.transpose(2, 0, 1)[None]
    state[TRUNK + "pos_embed_window"] = window.transpose(2, 0, 1)[None]
    specifications, _ = hiera.build_block_specifications(TINY)
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
    vector = model.get_layer("no_mem_embed").get_weights()[0]
    state["no_mem_embed"] = vector.reshape(1, 1, -1)


def encode_point_encoder(model, state):
    matrix = model.get_layer("prompt_pe").get_weights()[0]
    state[PROMPT + "pe_layer.positional_encoding_gaussian_matrix"] = matrix
    corners, not_a_point = model.get_layer("point_label_embed").get_weights()
    for index in range(4):
        source = f"{PROMPT}point_embeddings.{index}.weight"
        state[source] = corners[index:index + 1]
    state[PROMPT + "not_a_point_embed.weight"] = not_a_point


def encode_mask_downscaling(model, state):
    _, conv, norm = setters(model, state)
    conv("mask_down_0", PROMPT + "mask_downscaling.0")
    norm("mask_down_ln0", PROMPT + "mask_downscaling.1")
    conv("mask_down_3", PROMPT + "mask_downscaling.3")
    norm("mask_down_ln3", PROMPT + "mask_downscaling.4")
    conv("mask_down_6", PROMPT + "mask_downscaling.6")
    no_mask = model.get_layer("no_mask_embed").get_weights()[0]
    state[PROMPT + "no_mask_embed.weight"] = no_mask.reshape(1, -1)


def encode_mask_decoder(model, state):
    dense, conv, norm = setters(model, state)
    for token in ("obj_score_token", "iou_token", "mask_tokens"):
        state[f"{DECODER}{token}.weight"] = layer_weight(model, token)
    for index in range(2):
        encode_two_way_block(dense, norm, index, state)
    encode_attention(dense, "twoway_final_attn", final_source())
    norm("twoway_norm_final", TRANSFORMER + "norm_final_attn")
    conv("output_upscaling_0", DECODER + "output_upscaling.0")
    norm("output_upscaling_1", DECODER + "output_upscaling.1")
    conv("output_upscaling_3", DECODER + "output_upscaling.3")
    for index in range(4):
        name = f"output_hypernetworks_mlps_{index}"
        source = f"{DECODER}output_hypernetworks_mlps.{index}"
        encode_mlp(dense, name, source, 3)
    encode_mlp(dense, "iou_prediction_head", DECODER + "iou_prediction_head", 3)
    encode_mlp(dense, "pred_obj_score_head", DECODER + "pred_obj_score_head", 3)


def encode_two_way_block(dense, norm, index, state):
    block = f"{TRANSFORMER}layers.{index}"
    name = f"twoway_{index}"
    encode_attention(dense, f"{name}_self", f"{block}.self_attn")
    t2i = f"{block}.cross_attn_token_to_image"
    i2t = f"{block}.cross_attn_image_to_token"
    encode_attention(dense, f"{name}_cross_t2i", t2i)
    encode_attention(dense, f"{name}_cross_i2t", i2t)
    encode_mlp(dense, f"{name}_mlp", f"{block}.mlp", 2)
    for number in range(1, 5):
        norm(f"{name}_norm{number}", f"{block}.norm{number}")


def encode_attention(dense, name, source):
    for part in ("q", "k", "v", "out"):
        dense(f"{name}_{part}", f"{source}.{part}_proj")


def encode_mlp(dense, name, source, layers):
    for index in range(layers):
        dense(f"{name}_layers_{index}", f"{source}.layers.{index}")


def final_source():
    return TRANSFORMER + "final_attn_token_to_image"


def layer_weight(model, name):
    return model.get_layer(name).get_weights()[0]


def randomize(bundle):
    generator = np.random.RandomState(1)
    for model in bundle[:4]:
        weights = [generator.randn(*w.shape) for w in model.get_weights()]
        model.set_weights([w.astype("float32") * 0.05 for w in weights])


def decoder_inputs():
    embed = np.zeros((1, 64, 64, 256), np.float32)
    high_res_0 = np.zeros((1, 256, 256, 32), np.float32)
    high_res_1 = np.zeros((1, 128, 128, 64), np.float32)
    sparse = np.zeros((1, 2, 256), np.float32)
    dense = np.zeros((1, 64, 64, 256), np.float32)
    return [embed, high_res_0, high_res_1, sparse, dense, embed]


def test_converter_reproduces_source():
    source = sam2_model.build(TINY)
    randomize(source)
    state = to_torch_state_dict(source)
    target = sam2_model.build(TINY)
    used = set()
    convert.convert(target, state, used)
    assert used == set(state)
    expected = np.array(source.mask_decoder(decoder_inputs())[0])
    result = np.array(target.mask_decoder(decoder_inputs())[0])
    assert np.allclose(expected, result, atol=1e-5)


def test_converter_rejects_missing_key():
    bundle = sam2_model.build(TINY)
    state = to_torch_state_dict(bundle)
    del state[DECODER + "iou_token.weight"]
    with pytest.raises(KeyError):
        convert.convert(sam2_model.build(TINY), state)


def test_converter_rejects_unmapped_image_key():
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
