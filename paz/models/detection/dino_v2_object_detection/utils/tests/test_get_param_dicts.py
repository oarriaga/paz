import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from get_param_dicts import (
    get_vit_lr_decay_rate,
    get_vit_weight_decay_rate,
    classify_variable,
    build_lr_scale_map,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeVar:
    def __init__(self, name):
        self.name = name

class _FakeModel:
    def __init__(self, variables):
        self.trainable_variables = variables


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_classify_backbone():
    assert classify_variable("backbone/encoder/layer/0/kernel:0") == "backbone"

def test_classify_decoder():
    assert classify_variable("transformer/decoder/layers/0/kernel:0") == "decoder"  # fmt: skip

def test_classify_head():
    assert classify_variable("class_embed/layers/0/kernel:0") == "other"


def test_build_lr_scale_map_three_groups():
    vars_ = [
        _FakeVar("backbone.0.encoder.layer.5.attention.weight"),
        _FakeVar("transformer.decoder.layers.0.self_attn.weight"),
        _FakeVar("class_embed.layers.0.kernel"),
    ]
    model = _FakeModel(vars_)
    lr_map = build_lr_scale_map(
        model, lr=1e-4, lr_encoder=1.5e-4, lr_vit_layer_decay=0.8,
        lr_component_decay=0.7, weight_decay=1e-4, num_layers=12,
    )

    # Backbone scale != 1.0
    bb = lr_map["backbone.0.encoder.layer.5.attention.weight"]
    assert bb["lr_scale"] != 1.0

    # Decoder scale == lr_component_decay = 0.7
    dec = lr_map["transformer.decoder.layers.0.self_attn.weight"]
    assert dec["lr_scale"] == pytest.approx(0.7)

    # Head scale == 1.0
    head = lr_map["class_embed.layers.0.kernel"]
    assert head["lr_scale"] == pytest.approx(1.0)


def test_vit_lr_decay_embeddings_is_strongest():
    embed = get_vit_lr_decay_rate(
        "backbone.0.encoder.embeddings.weight", 0.8, 12)
    last = get_vit_lr_decay_rate(
        "backbone.0.encoder.layer.11.output.weight", 0.8, 12)
    assert embed < last


def test_vit_weight_decay_bias_is_zero():
    assert get_vit_weight_decay_rate("backbone.layer.0.attention.bias") == 0.0

def test_vit_weight_decay_kernel_is_nonzero():
    assert get_vit_weight_decay_rate("backbone.layer.0.attention.weight") == 1.0


def test_vit_decay_rates():
    rate = get_vit_lr_decay_rate(
        "backbone.0.encoder.layer.5.mlp.weight", lr_decay_rate=0.9,
        num_layers=12)
    # layer_id = 6, exponent = 12 + 1 - 6 = 7
    assert abs(rate - (0.9 ** 7)) < 1e-6

    rate = get_vit_weight_decay_rate("backbone.norm.weight",
                                     weight_decay_rate=0.1)
    assert rate == 0.0

