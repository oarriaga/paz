import os
import sys
import pytest
import keras
import numpy as np

# Dynamic import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import get_param_dicts

class MockVariable:
    def __init__(self, name):
        self.name = name

class MockModel:
    def __init__(self, variables):
        self.trainable_variables = variables

class MockArgs:
    def __init__(self, lr, lr_backbone=None, lr_component_decay=1.0):
        self.lr = lr
        if lr_backbone is not None:
            self.lr_backbone = lr_backbone
        self.lr_component_decay = lr_component_decay

def test_get_param_dict_grouping():
    # Setup variables
    vars = [
        MockVariable("backbone/layer1/kernel:0"),
        MockVariable("backbone/layer1/bias:0"),
        MockVariable("transformer/decoder/layer1/kernel:0"),
        MockVariable("transformer/encoder/layer1/kernel:0"),
        MockVariable("class_embed/kernel:0")
    ]
    model = MockModel(vars)
    args = MockArgs(lr=1e-4, lr_backbone=1e-5, lr_component_decay=0.1)
    
    # Run
    param_dicts = get_param_dicts.get_param_dict(args, model)
    
    # Analyze results
    # We expect 3 groups (or merged list). The function returns a list of dicts.
    # Each dict has 'params' (single variable) and 'lr'.
    
    # Check backbone
    backbone_params = [p for p in param_dicts if p['params'].name.startswith("backbone")]
    assert len(backbone_params) == 2
    for p in backbone_params:
        assert p['lr'] == 1e-5
        
    # Check decoder
    decoder_params = [p for p in param_dicts if "decoder" in p['params'].name]
    assert len(decoder_params) == 1
    assert decoder_params[0]['lr'] == 1e-4 * 0.1
    
    # Check others
    other_params = [p for p in param_dicts if "backbone" not in p['params'].name and "decoder" not in p['params'].name]
    assert len(other_params) == 2 # encoder + class_embed
    for p in other_params:
        assert p['lr'] == 1e-4

def test_vit_decay_rates():
    # Test helper functions
    rate = get_param_dicts.get_vit_lr_decay_rate("backbone.blocks.5.mlp", lr_decay_rate=0.9, num_layers=12)
    # layer_id = 6
    # exponent = 12 + 1 - 6 = 7
    # rate = 0.9 ** 7
    assert abs(rate - (0.9 ** 7)) < 1e-6
    
    rate = get_param_dicts.get_vit_weight_decay_rate("backbone.norm.weight", weight_decay_rate=0.1)
    assert rate == 0.0
