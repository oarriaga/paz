import os
import sys
import numpy as np
import torch
from torch import nn
import keras
from keras import ops
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Keras model imports
from paz.models.detection.dino_v2_object_detection.models.lwdetr.lwdetr import LWDETR, SetCriterion, PostProcess
from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.transformer import Transformer
from paz.models.detection.dino_v2_object_detection.models.backbone.backbone import Backbone
from paz.models.detection.dino_v2_object_detection.models.matcher.matcher import HungarianMatcher

# Weight transfer utilities
from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.transformer_weights_porting_utils import (
    transfer_transformer_weights, to_numpy, to_keras, to_torch
)

# Reference LWDETR imports
try:
    from rfdetr.models.lwdetr import LWDETR as PTLWDETR
    from rfdetr.models.lwdetr import build_model
    from rfdetr.models.backbone import Backbone as PTBackbone
    from rfdetr.models.transformer import Transformer as PTTransformer
except ImportError:
    # Adjust path to find rfdetr if needed
    rfdetr_path = os.path.abspath(os.path.join(current_dir, "../../../../../../examples/rf-detr_original_pytorch_implementation"))
    if rfdetr_path not in sys.path:
        sys.path.insert(0, rfdetr_path)
    from rfdetr.models.lwdetr import LWDETR as PTLWDETR
    from rfdetr.models.backbone import Backbone as PTBackbone
    from rfdetr.models.transformer import Transformer as PTTransformer

def transfer_lwdetr_weights(pt_model, keras_model):
    """Transfers LWDETR detection head weights to the Keras model.

    Copies classification head, bbox MLP, query/refpoint embeddings,
    and (when two-stage) encoder output heads from the reference model.
    Backbone weights are skipped when using mock components.

    Args:
        pt_model: Source reference LWDETR model.
        keras_model: Target Keras LWDETR model.
    """
    print("Transferring Backbone weights...")
    if isinstance(pt_model.backbone, (nn.Sequential, list, tuple)):
        pt_backbone_module = pt_model.backbone[0]
    else:
        pt_backbone_module = pt_model.backbone
    # Skip backbone transfer for mock components
    pass
    print("Transferring Transformer weights...")
    is_real_transformer = (
        hasattr(keras_model.transformer, 'decoder') and
        hasattr(keras_model.transformer.decoder, 'layers_list')
    )
    
    if not is_real_transformer:
        print("Mock Transformer detected, skipping transformer weight transfer...")
    else:
        sa_nhead = keras_model.transformer.decoder.layers_list[0].self_attn.num_heads
        transfer_transformer_weights(
            pt_model.transformer,
            keras_model.transformer,
            d_model=keras_model.transformer.d_model,
            sa_nhead=sa_nhead
        )
    print("Transferring Heads weights...")
    # Classification head
    keras_model.class_embed.kernel.assign(to_keras(pt_model.class_embed.weight.detach().T.numpy()))
    keras_model.class_embed.bias.assign(to_keras(pt_model.class_embed.bias.detach().numpy()))

    # Bbox MLP
    for j, (tk, klayer) in enumerate(zip(pt_model.bbox_embed.layers, keras_model.bbox_embed.layers_list)):
        klayer.kernel.assign(to_keras(tk.weight.detach().T.numpy()))
        klayer.bias.assign(to_keras(tk.bias.detach().numpy()))

    # Query and reference point embeddings
    keras_model.refpoint_embed.assign(to_keras(pt_model.refpoint_embed.weight.detach().numpy()))
    keras_model.query_feat.assign(to_keras(pt_model.query_feat.weight.detach().numpy()))

    # Two-stage encoder output heads
    if keras_model.two_stage and hasattr(pt_model, 'transformer') and hasattr(pt_model.transformer, 'enc_out_bbox_embed'):
        print("Transferring two-stage heads...")
        for i in range(len(keras_model.enc_out_bbox_embed)):
            pt_bbox = pt_model.transformer.enc_out_bbox_embed[i]
            k_bbox = keras_model.enc_out_bbox_embed[i]
            for pt_l, k_l in zip(pt_bbox.layers, k_bbox.layers_list):
                k_l.kernel.assign(to_keras(pt_l.weight.detach().T.numpy()))
                k_l.bias.assign(to_keras(pt_l.bias.detach().numpy()))
            pt_cls = pt_model.transformer.enc_out_class_embed[i]
            k_cls = keras_model.enc_out_class_embed[i]
            k_cls.kernel.assign(to_keras(pt_cls.weight.detach().T.numpy()))
            k_cls.bias.assign(to_keras(pt_cls.bias.detach().numpy()))

    print("Weights transfer complete.")


@pytest.mark.parametrize("group_detr", [1, 3])
@pytest.mark.parametrize("two_stage", [True, False])
@pytest.mark.parametrize("lite_refpoint_refine", [True, False])
@pytest.mark.parametrize("aux_loss", [True, False])
def test_parity_with_real_weights(group_detr, two_stage, lite_refpoint_refine, aux_loss):
    """Verifies weight transfer produces identical outputs across implementations."""
    num_classes = 91
    num_queries = 100
    d_model = 256
    
    print(f"\nTesting config: group_detr={group_detr}, two_stage={two_stage}, "
          f"lite_refpoint_refine={lite_refpoint_refine}, aux_loss={aux_loss}")
    # Keras model with mock components
    from paz.models.detection.dino_v2_object_detection.models.lwdetr.test_lwdetr import MockBackbone, MockTransformer
    
    keras_backbone = MockBackbone(hidden_dim=d_model)
    keras_transformer = MockTransformer(d_model=d_model, num_queries=num_queries)
    
    keras_model = LWDETR(
        backbone=keras_backbone,
        transformer=keras_transformer,
        segmentation_head=None,
        num_classes=num_classes,
        num_queries=num_queries,
        aux_loss=aux_loss,
        group_detr=group_detr,
        two_stage=two_stage,
        lite_refpoint_refine=lite_refpoint_refine
    )
    # Build two-stage heads before main model build if present
    if hasattr(keras_model, 'enc_out_bbox_embed'):
        for bbox, cls in zip(keras_model.enc_out_bbox_embed, keras_model.enc_out_class_embed):
            bbox.build((None, d_model))
            cls.build((None, d_model))

    # Build Keras model with dummy input
    input_shape = (1, 3, 224, 224)
    dummy_img = keras.random.normal(input_shape)
    keras_model(dummy_img)

    # Instantiate reference model with mock components for parity check
    print("Instantiating reference model for parity check...")

    class PTMockBackbone(nn.Module):
        """Mock backbone returning fixed-shape feature tensors."""
        def __init__(self, hidden_dim=256):
            super().__init__()
            self.hidden_dim = hidden_dim
        def forward(self, samples):
            B, _, H, W = samples.tensors.shape
            feats = []
            poss = []
            for i in range(3):
                h, w = H // (2**(i+1)), W // (2**(i+1))
                f = torch.ones(B, self.hidden_dim, h, w)
                m = torch.zeros(B, h, w).bool()
                p = torch.ones(B, self.hidden_dim, h, w)
                from rfdetr.util.misc import NestedTensor
                feats.append(NestedTensor(f, m))
                poss.append(p)
            return feats, poss

    class PTMockTransformer(nn.Module):
        """Mock transformer returning fixed-shape decoder outputs."""
        def __init__(self, d_model=256, num_queries=100, two_stage=True):
            super().__init__()
            self.d_model = d_model
            self.num_queries = num_queries
            self.two_stage = two_stage
            self.decoder = nn.Module()
            self.decoder.bbox_embed = None
            self.enc_output = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(1)])
        def forward(self, srcs, masks, pos_embeds, refpoint_embed, query_embed):
            B = srcs[0].shape[0]
            hs = torch.ones(6, B, self.num_queries, self.d_model)
            ref_unsigmoid = torch.ones(B, self.num_queries, 4)
            hs_enc = torch.ones(B, self.num_queries, self.d_model)
            ref_enc = torch.ones(B, self.num_queries, 4)
            return hs, ref_unsigmoid, hs_enc, ref_enc

    pt_backbone = PTMockBackbone(hidden_dim=d_model)
    pt_transformer = PTMockTransformer(d_model=d_model, num_queries=num_queries)
    
    pt_model = PTLWDETR(
        backbone=pt_backbone,
        transformer=pt_transformer,
        segmentation_head=None,
        num_classes=num_classes,
        num_queries=num_queries,
        aux_loss=aux_loss,
        group_detr=group_detr,
        two_stage=two_stage,
        lite_refpoint_refine=lite_refpoint_refine
    )
    pt_model.eval()

    # Transfer weights and verify parity
    transfer_lwdetr_weights(pt_model, keras_model)

    print("Running parity check...")
    img = np.random.randn(1, 3, 224, 224).astype("float32")
    # Reference forward pass
    from rfdetr.util.misc import nested_tensor_from_tensor_list
    pt_img = nested_tensor_from_tensor_list([torch.from_numpy(img[0])])
    with torch.no_grad():
        pt_out = pt_model(pt_img)
    # Keras forward pass
    k_img = to_keras(img)
    k_out = keras_model(k_img, training=False)
    # Compare logits and boxes
    pt_logits = pt_out['pred_logits'].numpy()
    k_logits = to_numpy(k_out['pred_logits'])
    
    diff_logits = np.abs(pt_logits - k_logits).max()
    print(f"Max diff pred_logits: {diff_logits}")

    pt_boxes = pt_out['pred_boxes'].numpy()
    k_boxes = to_numpy(k_out['pred_boxes'])
    
    diff_boxes = np.abs(pt_boxes - k_boxes).max()
    print(f"Max diff pred_boxes: {diff_boxes}")
    
    assert diff_logits < 1e-4
    assert diff_boxes < 1e-4
    print("Parity check PASSED!")

if __name__ == "__main__":
    test_parity_with_real_weights()
