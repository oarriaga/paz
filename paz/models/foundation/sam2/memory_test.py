"""Video memory-module tests without torch, internet, or checkpoints."""
import numpy as np
import pytest

from paz.models.foundation.sam2 import memory_encoder as me
from paz.models.foundation.sam2 import memory_attention as ma
from paz.models.foundation.sam2 import model as sam2_model, convert

ENCODER = convert.MEMORY_ENCODER
ATTENTION = convert.MEMORY_ATTENTION


def test_memory_encoder_output_shape():
    model = me.build()
    pix_feat = np.zeros((1, 64, 64, 256), np.float32)
    mask = np.zeros((1, 1024, 1024, 1), np.float32)
    features = np.array(model((pix_feat, mask)))
    assert features.shape == (1, 64, 64, 64)


def test_sine_position_encoding_shape():
    encoding = me.sine_position_encoding(64, 64, 64)
    assert encoding.shape == (1, 64, 64, 64)


def test_rotary_tables_shape():
    cos, sin = ma.rotary_tables(4, 4)
    assert cos.shape == (16, ma.ROPE_DIM)
    assert sin.shape == (16, ma.ROPE_DIM)


def test_identity_tables_are_neutral():
    cos, sin = ma.identity_tables(5)
    assert np.allclose(cos, 1.0)
    assert np.allclose(sin, 0.0)


def memory_attention_inputs(spatial, pointers):
    total = spatial * spatial + pointers
    curr = np.zeros((1, spatial * spatial, 256), np.float32)
    memory = np.zeros((1, total, 64), np.float32)
    cos, sin = ma.rotary_tables(spatial, spatial)
    icos, isin = ma.identity_tables(pointers)
    memory_cos = np.concatenate([cos, icos], axis=0)[None]
    memory_sin = np.concatenate([sin, isin], axis=0)[None]
    rope = [cos[None], sin[None], memory_cos, memory_sin]
    return [curr, curr, memory, memory] + rope


def test_memory_attention_output_shape():
    tokens = np.array(ma.build()(memory_attention_inputs(4, 2)))
    assert tokens.shape == (1, 16, 256)


def encoder_state(model):
    state = {}
    conv(model, "mem_pix_proj", ENCODER + "pix_feat_proj", state)
    conv(model, "mem_out_proj", ENCODER + "out_proj", state)
    grid = ENCODER + "mask_downsampler.encoder."
    for index in (0, 3, 6, 9):
        conv(model, f"mask_ds_conv_{index}", f"{grid}{index}", state)
        norm(model, f"mask_ds_ln_{index + 1}", f"{grid}{index + 1}", state)
    conv(model, "mask_ds_final", f"{grid}12", state)
    for index in range(2):
        block = f"{ENCODER}fuser.layers.{index}"
        depthwise(model, f"fuser_{index}_dw", f"{block}.dwconv", state)
        norm(model, f"fuser_{index}_norm", f"{block}.norm", state)
        dense(model, f"fuser_{index}_pw1", f"{block}.pwconv1", state)
        dense(model, f"fuser_{index}_pw2", f"{block}.pwconv2", state)
        scale = model.get_layer(f"fuser_{index}_gamma").get_weights()[0]
        state[f"{block}.gamma"] = scale
    return state


def attention_state(model):
    state = {}
    for index in range(4):
        layer = f"{ATTENTION}layers.{index}"
        for part in ("q", "k", "v", "out"):
            self_source = f"{layer}.self_attn.{part}_proj"
            dense(model, f"mematt_{index}_self_{part}", self_source, state)
            cross = f"{layer}.cross_attn_image.{part}_proj"
            dense(model, f"mematt_{index}_cross_{part}", cross, state)
        for number in (1, 2, 3):
            norm_source = f"{layer}.norm{number}"
            norm(model, f"mematt_{index}_norm{number}", norm_source, state)
        dense(model, f"mematt_{index}_mlp1", f"{layer}.linear1", state)
        dense(model, f"mematt_{index}_mlp2", f"{layer}.linear2", state)
    norm(model, "mematt_norm", ATTENTION + "norm", state)
    return state


def conv(model, name, source, state):
    kernel, bias = model.get_layer(name).get_weights()
    state[f"{source}.weight"] = np.transpose(kernel, (3, 2, 0, 1))
    state[f"{source}.bias"] = bias


def depthwise(model, name, source, state):
    kernel, bias = model.get_layer(name).get_weights()
    state[f"{source}.weight"] = np.transpose(kernel, (2, 3, 0, 1))
    state[f"{source}.bias"] = bias


def dense(model, name, source, state):
    kernel, bias = model.get_layer(name).get_weights()
    state[f"{source}.weight"] = kernel.T
    state[f"{source}.bias"] = bias


def norm(model, name, source, state):
    gamma, beta = model.get_layer(name).get_weights()
    state[f"{source}.weight"] = gamma
    state[f"{source}.bias"] = beta


def randomize(model):
    generator = np.random.RandomState(2)
    weights = [generator.randn(*w.shape) for w in model.get_weights()]
    model.set_weights([w.astype("float32") * 0.05 for w in weights])


def test_memory_converter_round_trip():
    source = sam2_model.build_memory()
    randomize(source.encoder)
    randomize(source.attention)
    state = encoder_state(source.encoder)
    state.update(attention_state(source.attention))
    target = sam2_model.build_memory()
    used = set()
    convert.convert_memory(target.encoder, target.attention, state, used)
    assert used == set(state)
    pix = np.zeros((1, 64, 64, 256), np.float32)
    mask = np.zeros((1, 1024, 1024, 1), np.float32)
    expected = np.array(source.encoder((pix, mask)))
    result = np.array(target.encoder((pix, mask)))
    assert np.allclose(expected, result, atol=1e-5)
    inputs = memory_attention_inputs(4, 2)
    expected = np.array(source.attention(inputs))
    result = np.array(target.attention(inputs))
    assert np.allclose(expected, result, atol=1e-5)


def test_memory_converter_rejects_missing_key():
    bundle = sam2_model.build_memory()
    state = encoder_state(bundle.encoder)
    state.update(attention_state(bundle.attention))
    del state[ATTENTION + "norm.weight"]
    fresh = sam2_model.build_memory()
    with pytest.raises(KeyError):
        convert.convert_memory(fresh.encoder, fresh.attention, state)
