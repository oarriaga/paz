import sys
import os

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import numpy as np
import keras
from keras import ops

# Adjust imports
from paz.models.detection.dino_v2_object_detection.models.lwdetr.lwdetr import (
    LWDETR, SetCriterion, PostProcess, MLP, 
    sigmoid_focal_loss, sigmoid_varifocal_loss, position_supervised_loss, dice_loss, sigmoid_ce_loss
)
from paz.models.detection.dino_v2_object_detection.models.matcher.matcher import HungarianMatcher
from paz.models.detection.dino_v2_object_detection.utils.misc import NestedTensor

class MockBackbone(keras.layers.Layer):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def call(self, inputs, mask=None):
        # inputs: tensor (B, 3, H, W)
        # return features, pos
        B, H, W, _ = inputs.shape # Keras uses channels-last by default? 
        # But DINO usually uses (B, C, H, W) in Logic or (B, H, W, C).
        # Let's assume (B, 3, H, W) based on valid_W/H logic in transformer.
        
        # Return 3 feature maps
        feats = []
        poss = []
        for i in range(3):
            shape = (B, self.hidden_dim, H // (2**(i+1)), W // (2**(i+1)))
            f = keras.random.normal(shape)
            m = ops.zeros((B, H // (2**(i+1)), W // (2**(i+1))), dtype="bool")
            p = keras.random.normal((B, self.hidden_dim, H // (2**(i+1)), W // (2**(i+1))))
            
            # Using NestedTensor-like structure or just tensors
            # LWDETR expects list of features.
            # We can return simple tensors since lwdetr handles it.
            feats.append(f)
            poss.append(p)
            
        return feats, poss

class MockTransformer(keras.layers.Layer):
    def __init__(self, d_model=256, num_queries=100):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.decoder = type('obj', (object,), {'bbox_embed': None}) # Mock decoder object
        self.enc_out_bbox_embed = None
        self.enc_out_class_embed = None
        # For transfer_transformer_weights with two_stage=True
        self.enc_output = [keras.layers.Dense(d_model) for _ in range(1)] 
        for layer in self.enc_output:
            layer.build((None, d_model))

    def call(self, srcs, masks, pos_embeds, query_feat=None, refpoint_embed=None, training=None):
        B = ops.shape(srcs[0])[0]
        # hs: (L, B, Q, C)
        hs = ops.ones((6, B, self.num_queries, self.d_model))
        # ref_unsigmoid: (B, Q, 4)
        ref_unsigmoid = ops.ones((B, self.num_queries, 4))
        # hs_enc: (B, Q, C)
        hs_enc = ops.ones((B, self.num_queries, self.d_model))
        # ref_enc: (B, Q, 4)
        ref_enc = ops.ones((B, self.num_queries, 4))
        
        return hs, ref_unsigmoid, hs_enc, ref_enc

def test_mlp():
    input_dim = 16
    hidden_dim = 32
    output_dim = 4
    num_layers = 3
    
    mlp = MLP(input_dim, hidden_dim, output_dim, num_layers)
    x = keras.random.normal((2, 10, input_dim))
    y = mlp(x)
    
    assert y.shape == (2, 10, output_dim)

def test_loss_functions():
    B, Q, C = 2, 10, 4
    inputs = keras.random.normal((B, Q, C))
    targets = keras.random.uniform((B, Q, C), minval=0, maxval=2)
    targets = ops.cast(targets, "float32")
    
    loss = sigmoid_focal_loss(inputs, targets, num_boxes=Q)
    assert ops.ndim(loss) == 0
    assert loss > 0
    
    loss = sigmoid_varifocal_loss(inputs, targets, num_boxes=Q)
    assert ops.ndim(loss) == 0
    
    loss = position_supervised_loss(inputs, targets, num_boxes=Q)
    assert ops.ndim(loss) == 0

def test_lwdetr_instantiation():
    d_model = 256
    num_classes = 91
    num_queries = 100
    
    backbone = MockBackbone(hidden_dim=d_model)
    transformer = MockTransformer(d_model=d_model, num_queries=num_queries)
    
    model = LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=None,
        num_classes=num_classes,
        num_queries=num_queries
    )
    
    assert model.num_queries == num_queries
    assert model.backbone == backbone

@pytest.mark.parametrize("group_detr", [1, 3])
@pytest.mark.parametrize("two_stage", [True, False])
@pytest.mark.parametrize("lite_refpoint_refine", [True, False])
@pytest.mark.parametrize("aux_loss", [True, False])
def test_lwdetr_forward(group_detr, two_stage, lite_refpoint_refine, aux_loss):
    d_model = 256
    num_classes = 91
    num_queries = 100
    
    backbone = MockBackbone(hidden_dim=d_model)
    transformer = MockTransformer(d_model=d_model, num_queries=num_queries)
    
    model = LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=None,
        num_classes=num_classes,
        num_queries=num_queries,
        aux_loss=aux_loss,
        group_detr=group_detr,
        two_stage=two_stage,
        lite_refpoint_refine=lite_refpoint_refine
    )
    
    # Input: (B, C, H, W)
    batch_size = 2
    x = keras.random.normal((batch_size, 3, 64, 64))
    
    out = model(x, training=False)
    
    assert "pred_logits" in out
    assert "pred_boxes" in out
    
    # Check shapes
    assert out["pred_logits"].shape == (batch_size, num_queries, num_classes)
    assert out["pred_boxes"].shape == (batch_size, num_queries, 4)
    
    if aux_loss:
        assert "aux_outputs" in out
        # Mock transformer returns 6 layers (hs has shape (6, ...))
        # So aux_outputs should have 5 items
        assert len(out["aux_outputs"]) == 5
    else:
        assert "aux_outputs" not in out
        
    if two_stage:
        assert "enc_outputs" in out
    else:
        assert "enc_outputs" not in out

def test_lwdetr_various_inputs():
    d_model = 128
    num_classes = 10
    num_queries = 50
    
    backbone = MockBackbone(hidden_dim=d_model)
    transformer = MockTransformer(d_model=d_model, num_queries=num_queries)
    
    model = LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=None,
        num_classes=num_classes,
        num_queries=num_queries,
        two_stage=True
    )
    
    # Test batch size 1 and non-square image
    x = keras.random.normal((1, 3, 128, 64))
    out = model(x, training=True)
    
    assert out["pred_logits"].shape == (1, num_queries * 1, num_classes) # group_detr=1
    assert out["pred_boxes"].shape == (1, num_queries * 1, 4)
    
    # Test larger batch size
    x = keras.random.normal((4, 3, 32, 32))
    out = model(x, training=False)
    assert out["pred_logits"].shape == (4, num_queries, num_classes)

def test_postprocess():
    B, Q, C = 2, 100, 91
    num_select = 10
    postprocessor = PostProcess(num_select=num_select)
    
    outputs = {
        'pred_logits': keras.random.normal((B, Q, C)),
        'pred_boxes': keras.random.uniform((B, Q, 4), minval=0.0, maxval=1.0)
    }
    # target_sizes is (H, W) per image
    target_sizes = ops.convert_to_tensor([[480, 640], [800, 600]]) 
    
    scores, labels, boxes = postprocessor(outputs, target_sizes)
    assert scores.shape == (B, 10)
    assert labels.shape == (B, 10)
    assert boxes.shape == (B, 10, 4)
    # Check that boxes are scaled (at least one box should have coordinate > 1 if scaled)
    assert ops.any(boxes > 1.0)

def test_criterion():
    num_classes = 91
    matcher = HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1}
    losses = ['labels', 'boxes']
    
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=0.25,
        loss_types=losses
    )
    
    B, Q = 2, 10
    outputs = {
        'pred_logits': keras.random.normal((B, Q, num_classes)),
        'pred_boxes': keras.random.uniform((B, Q, 4), minval=0.0, maxval=1.0)
    }
    
    targets = [
        {
            'labels': ops.convert_to_tensor([1, 2], dtype="int64"),
            'boxes': ops.convert_to_tensor([[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]], dtype="float32")
        },
        {
            'labels': ops.convert_to_tensor([3], dtype="int64"),
            'boxes': ops.convert_to_tensor([[0.8, 0.8, 0.1, 0.1]], dtype="float32")
        }
    ]
    
    loss_dict = criterion(outputs, targets)
    
    assert 'loss_ce' in loss_dict
    assert 'loss_bbox' in loss_dict
    assert 'loss_giou' in loss_dict
