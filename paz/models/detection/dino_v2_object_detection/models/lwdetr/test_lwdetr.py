import sys
import os
import functools

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import numpy as np
import keras
from keras import Input, Model, layers, ops

from paz.models.detection.dino_v2_object_detection.models.lwdetr import (
    lwdetr as lwdetr_module,
)
from paz.models.detection.dino_v2_object_detection.models.lwdetr.lwdetr import (
    LWDETR,
    apply_lwdetr,
    mlp,
    CriterionArgs,
    set_criterion,
    post_process,
    sigmoid_focal_loss,
    sigmoid_varifocal_loss,
    position_supervised_loss,
)
from paz.models.detection.dino_v2_object_detection.models.matcher.matcher import (  # fmt: skip
    hungarian_matcher,
)


def mock_apply_transformer(transformer, srcs, masks, pos_embeds, bbox_embed, enc_out_class_embed, enc_out_bbox_embed, query_feat, refpoint_embed, training):  # fmt: skip
    B = ops.shape(srcs[0])[0]
    d_model = transformer.d_model
    num_queries = transformer.num_queries
    hs = ops.ones((6, B, num_queries, d_model))
    ref_unsigmoid = ops.ones((B, num_queries, 4))
    hs_enc = ops.ones((B, num_queries, d_model))
    ref_enc = ops.ones((B, num_queries, 4))
    return hs, ref_unsigmoid, hs_enc, ref_enc


@pytest.fixture(autouse=True)
def patch_apply_transformer(monkeypatch):
    attr = "apply_transformer"
    monkeypatch.setattr(lwdetr_module, attr, mock_apply_transformer)


def build_mock_backbone(hidden_dim=256, levels=3):
    image = Input((None, None, 3), name="mock_backbone_image")
    mask = Input((None, None), dtype="bool", name="mock_backbone_mask")
    features = []
    positions = []
    for level in range(levels):
        stride = 2 ** (level + 1)
        proj_name = f"mock_proj_{level}"
        projection = layers.Conv2D(hidden_dim, 1, name=proj_name)(image)
        feature = layers.AveragePooling2D(stride, name=f"mock_pool_{level}")(projection)  # fmt: skip
        # Consumes mask so every declared Input reaches an output; Keras
        # 3.10 rejects a Functional whose Input is unconnected. Value is
        # unchanged: still an all-False mask shaped like the feature map.
        level_mask = layers.Lambda(
            lambda tensors: ops.cast(ops.zeros_like(tensors[0][..., 0]), "bool"),  # fmt: skip
            name=f"mock_mask_{level}",
        )([feature, mask])
        features.append([feature, level_mask])
        positions.append(feature)
    return Model([image, mask], [features, positions], name="mock_backbone")


def build_mock_transformer(d_model=256, num_queries=100):
    query = Input((None, d_model), name="mock_query")
    memory = Input((None, d_model), name="mock_memory")
    sine = Input((None, 2 * d_model), name="mock_sine")
    # Consumes memory and sine so every declared Input reaches an output;
    # Keras 3.10 rejects a Functional whose Input is unconnected. Value is
    # unchanged: the Dense still sees exactly query.
    anchored = layers.Lambda(
        lambda tensors: tensors[0], name="mock_transformer_anchor"
    )([query, memory, sine])
    output = layers.Dense(d_model, name="mock_transformer_dense")(anchored)
    model = Model([query, memory, sine], [output], name="mock_transformer")
    model.d_model = d_model
    model.num_queries = num_queries
    return model


def test_mlp():
    input_dim, hidden_dim, output_dim, num_layers = 16, 32, 4, 3
    inputs = Input((10, input_dim))
    outputs = mlp(inputs, input_dim, hidden_dim, output_dim, num_layers, "mlp")
    model = Model(inputs, outputs)
    y = model(keras.random.normal((2, 10, input_dim)))
    assert y.shape == (2, 10, output_dim)

def test_loss_functions():
    B, Q, C = 2, 10, 4
    np.random.seed(42)
    inputs = ops.convert_to_tensor(
        np.random.randn(B, Q, C).astype("float32")
    )
    # Targets must be in [0, 1] for sigmoid-based losses (BCE)
    targets = ops.convert_to_tensor(
        np.random.uniform(0, 1, (B, Q, C)).astype("float32")
    )
    
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
    
    backbone = build_mock_backbone(hidden_dim=d_model)
    transformer = build_mock_transformer(d_model, num_queries)

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
    
    backbone = build_mock_backbone(hidden_dim=d_model)
    transformer = build_mock_transformer(d_model, num_queries)

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
    batch_size = 2
    x = keras.random.normal((batch_size, 64, 64, 3))

    out = apply_lwdetr(model, x, training=False)

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
    
    backbone = build_mock_backbone(hidden_dim=d_model)
    transformer = build_mock_transformer(d_model, num_queries)

    model = LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=None,
        num_classes=num_classes,
        num_queries=num_queries,
        two_stage=True
    )

    # Test batch size 1 and non-square image
    x = keras.random.normal((1, 128, 64, 3))
    out = apply_lwdetr(model, x, training=True)

    assert out["pred_logits"].shape == (1, num_queries * 1, num_classes)
    assert out["pred_boxes"].shape == (1, num_queries * 1, 4)

    # Test larger batch size
    x = keras.random.normal((4, 32, 32, 3))
    out = apply_lwdetr(model, x, training=False)
    assert out["pred_logits"].shape == (4, num_queries, num_classes)

def test_postprocess():
    B, Q, C = 2, 100, 91
    num_select = 10
    postprocessor = functools.partial(post_process, num_select=num_select)
    
    outputs = {
        'pred_logits': keras.random.normal((B, Q, C)),
        'pred_boxes': keras.random.uniform((B, Q, 4), minval=0.0, maxval=1.0)
    }
    target_sizes = ops.convert_to_tensor([[480, 640], [800, 600]])
    
    scores, labels, boxes = postprocessor(outputs, target_sizes)
    assert scores.shape == (B, 10)
    assert labels.shape == (B, 10)
    assert boxes.shape == (B, 10, 4)
    # Check that boxes are scaled (at least one box should have
    # coordinate > 1 if scaled)
    assert ops.any(boxes > 1.0)

def test_criterion():
    num_classes = 91
    matcher = functools.partial(
        hungarian_matcher,
        cost_class=1,
        cost_bbox=1,
        cost_giou=1,
        focal_alpha=0.25,
    )
    weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1}
    losses = ['labels', 'boxes']
    
    criterion = CriterionArgs(
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
            'boxes': ops.convert_to_tensor(
                [[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
                dtype="float32",
            )
        },
        {
            'labels': ops.convert_to_tensor([3], dtype="int64"),
            'boxes': ops.convert_to_tensor(
                [[0.8, 0.8, 0.1, 0.1]], dtype="float32"
            )
        }
    ]
    
    loss_dict = set_criterion(outputs, targets, criterion)
    
    assert 'loss_ce' in loss_dict
    assert 'loss_bbox' in loss_dict
    assert 'loss_giou' in loss_dict
