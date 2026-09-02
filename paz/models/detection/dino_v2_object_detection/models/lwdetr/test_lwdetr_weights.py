import os
import sys

import numpy as np
import pytest

_PT_DIR = "examples/rf-detr_original_pytorch_implementation"
_PT_ROOT = os.path.join(os.path.dirname(__file__), *([".."] * 6))
if not os.path.isdir(os.path.join(_PT_ROOT, _PT_DIR)):
    pytest.skip("RF-DETR reference unavailable", allow_module_level=True)
pytest.importorskip("torch")

import torch
from torch import nn
import keras

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Keras model imports
from paz.models.detection.dino_v2_object_detection.models.lwdetr import (
    lwdetr as lwdetr_module,
)
from paz.models.detection.dino_v2_object_detection.models.lwdetr.lwdetr import (
    LWDETR,
    apply_lwdetr,
)


@pytest.fixture(autouse=True)
def patch_apply_transformer(monkeypatch):
    from paz.models.detection.dino_v2_object_detection.models.lwdetr.test_lwdetr import (  # fmt: skip
        mock_apply_transformer,
    )
    attr = "apply_transformer"
    monkeypatch.setattr(lwdetr_module, attr, mock_apply_transformer)

# Weight transfer utilities
from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.transformer_weights_porting_utils import (  # fmt: skip
    transfer_transformer_weights,
    to_numpy,
    to_keras,
)

# Reference LWDETR imports
try:
    from rfdetr.models.lwdetr import LWDETR as PTLWDETR
except ImportError:
    # Adjust path to find rfdetr if needed
    rfdetr_path = os.path.abspath(
        os.path.join(
            current_dir,
            "../../../../../../examples/"
            "rf-detr_original_pytorch_implementation",
        )
    )
    if rfdetr_path not in sys.path:
        sys.path.insert(0, rfdetr_path)
    from rfdetr.models.lwdetr import LWDETR as PTLWDETR

def transfer_lwdetr_weights(pt_model, keras_model):
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
        print(
            "Mock Transformer detected, skipping transformer "
            "weight transfer..."
        )
    else:
        decoder = keras_model.transformer.decoder
        sa_nhead = decoder.layers_list[0].self_attn.num_heads
        transfer_transformer_weights(
            pt_model.transformer,
            keras_model.transformer,
            d_model=keras_model.transformer.d_model,
            sa_nhead=sa_nhead
        )
    print("Transferring Heads weights...")
    # Classification head
    class_head = keras_model.get_layer("class_embed")
    class_weight = pt_model.class_embed.weight.detach().T.numpy()
    class_head.kernel.assign(to_keras(class_weight))
    class_head.bias.assign(to_keras(pt_model.class_embed.bias.detach().numpy()))

    # Bbox MLP
    for j, tk in enumerate(pt_model.bbox_embed.layers):
        klayer = keras_model.get_layer(f"bbox_embed_dense_{j}")
        klayer.kernel.assign(to_keras(tk.weight.detach().T.numpy()))
        klayer.bias.assign(to_keras(tk.bias.detach().numpy()))

    # Query and reference point embeddings
    keras_model.get_layer("refpoint_embed").embeddings.assign(
        to_keras(pt_model.refpoint_embed.weight.detach().numpy())
    )
    keras_model.get_layer("query_feat").embeddings.assign(
        to_keras(pt_model.query_feat.weight.detach().numpy())
    )

    # Two-stage encoder output heads
    has_tf = hasattr(pt_model, 'transformer')
    has_bbox = has_tf and hasattr(pt_model.transformer, 'enc_out_bbox_embed')
    if keras_model.two_stage and has_bbox:
        print("Transferring two-stage heads...")
        for i in range(keras_model.group_detr):
            pt_bbox = pt_model.transformer.enc_out_bbox_embed[i]
            for j, pt_l in enumerate(pt_bbox.layers):
                k_l = keras_model.get_layer(f"enc_out_bbox_embed_{i}_dense_{j}")
                k_l.kernel.assign(to_keras(pt_l.weight.detach().T.numpy()))
                k_l.bias.assign(to_keras(pt_l.bias.detach().numpy()))
            pt_cls = pt_model.transformer.enc_out_class_embed[i]
            k_cls = keras_model.get_layer(f"enc_out_class_embed_{i}")
            k_cls.kernel.assign(to_keras(pt_cls.weight.detach().T.numpy()))
            k_cls.bias.assign(to_keras(pt_cls.bias.detach().numpy()))

    print("Weights transfer complete.")


D_MODEL = 256
NUM_CLASSES = 91
NUM_QUERIES = 100
LWDETR_KEYS = ("backbone", "transformer", "segmentation_head", "num_classes", "num_queries", "aux_loss", "group_detr", "two_stage", "lite_refpoint_refine")  # fmt: skip


class PTMockBackbone(nn.Module):
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
    def __init__(self, d_model=256, num_queries=100, two_stage=True):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.two_stage = two_stage
        self.decoder = nn.Module()
        self.decoder.bbox_embed = None
        linears = [nn.Linear(d_model, d_model) for _ in range(1)]
        self.enc_output = nn.ModuleList(linears)

    def forward(self, srcs, masks, pos_embeds, refpoint_embed, query_embed):
        B = srcs[0].shape[0]
        hs = torch.ones(6, B, self.num_queries, self.d_model)
        ref_unsigmoid = torch.ones(B, self.num_queries, 4)
        hs_enc = torch.ones(B, self.num_queries, self.d_model)
        ref_enc = torch.ones(B, self.num_queries, 4)
        return hs, ref_unsigmoid, hs_enc, ref_enc


def build_keras_parity_model(aux_loss, group_detr, two_stage, lite_refpoint_refine):  # fmt: skip
    # Keras model with mock components
    from paz.models.detection.dino_v2_object_detection.models.lwdetr.test_lwdetr import (  # fmt: skip
        build_mock_backbone, build_mock_transformer,
    )
    backbone = build_mock_backbone(hidden_dim=D_MODEL)
    transformer = build_mock_transformer(d_model=D_MODEL, num_queries=NUM_QUERIES)  # fmt: skip
    values = (backbone, transformer, None, NUM_CLASSES, NUM_QUERIES, aux_loss, group_detr, two_stage, lite_refpoint_refine)  # fmt: skip
    keras_model = LWDETR(**dict(zip(LWDETR_KEYS, values)))
    # Exercise the functional model once with a dummy NHWC input.
    dummy_img = keras.random.normal((1, 224, 224, 3))
    apply_lwdetr(keras_model, dummy_img)
    return keras_model


def build_torch_parity_model(aux_loss, group_detr, two_stage, lite_refpoint_refine):  # fmt: skip
    backbone = PTMockBackbone(hidden_dim=D_MODEL)
    transformer = PTMockTransformer(d_model=D_MODEL, num_queries=NUM_QUERIES)
    values = (backbone, transformer, None, NUM_CLASSES, NUM_QUERIES, aux_loss, group_detr, two_stage, lite_refpoint_refine)  # fmt: skip
    pt_model = PTLWDETR(**dict(zip(LWDETR_KEYS, values)))
    pt_model.eval()
    return pt_model


def run_parity_forwards(pt_model, keras_model):
    img = np.random.randn(1, 3, 224, 224).astype("float32")
    # Reference forward pass
    from rfdetr.util.misc import nested_tensor_from_tensor_list
    pt_img = nested_tensor_from_tensor_list([torch.from_numpy(img[0])])
    with torch.no_grad():
        pt_out = pt_model(pt_img)
    # Keras forward pass (mock backbone expects NHWC)
    k_img = to_keras(np.transpose(img, (0, 2, 3, 1)))
    k_out = apply_lwdetr(keras_model, k_img, training=False)
    return pt_out, k_out


def assert_parity_outputs(pt_out, k_out):
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


@pytest.mark.parametrize("group_detr", [1, 3])
@pytest.mark.parametrize("two_stage", [True, False])
@pytest.mark.parametrize("lite_refpoint_refine", [True, False])
@pytest.mark.parametrize("aux_loss", [True, False])
def test_parity_with_real_weights(group_detr, two_stage, lite_refpoint_refine, aux_loss):  # fmt: skip
    print(f"\nTesting config: group_detr={group_detr}, two_stage={two_stage}, "
          f"lite_refpoint_refine={lite_refpoint_refine}, aux_loss={aux_loss}")
    config = (aux_loss, group_detr, two_stage, lite_refpoint_refine)
    keras_model = build_keras_parity_model(*config)
    # Instantiate reference model with mock components for parity check
    print("Instantiating reference model for parity check...")
    pt_model = build_torch_parity_model(*config)
    # Transfer weights and verify parity
    transfer_lwdetr_weights(pt_model, keras_model)
    print("Running parity check...")
    pt_out, k_out = run_parity_forwards(pt_model, keras_model)
    assert_parity_outputs(pt_out, k_out)

if __name__ == "__main__":
    test_parity_with_real_weights()
