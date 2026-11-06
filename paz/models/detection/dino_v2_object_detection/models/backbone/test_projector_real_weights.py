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

from paz.models.detection.dino_v2_object_detection.models.backbone.projector import (  # noqa: E501  # fmt: skip
    MultiScaleProjector as KerasMultiScaleProjector,
)
from paz.models.detection.dino_v2_object_detection.models.backbone.projector_weights_porting_utils import (  # noqa: E501  # fmt: skip
    port_weights_multiscale_projector,
)

try:
    from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRNano, RFDETRLarge
except ImportError:
    here = os.path.dirname(__file__)
    sys.path.append(os.path.abspath(os.path.join(
        here, "../../../../../../",
        "examples", "rf-detr_original_pytorch_implementation",
    )))
    from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRNano, RFDETRLarge


def to_numpy(t):
    return t.detach().cpu().numpy()


def as_list(outputs):
    return outputs if isinstance(outputs, list) else [outputs]


def locate_projector(model_class):
    torch_full_model = model_class(pretrained=True)
    if hasattr(torch_full_model, "model") and hasattr(torch_full_model.model, "model"):  # fmt: skip
        inner_model = torch_full_model.model.model
    else:
        inner_model = torch_full_model
    inner_model.eval()
    backbone = inner_model.backbone
    if hasattr(backbone, "__getitem__") and not isinstance(backbone, torch.Tensor):  # fmt: skip
        base_backbone = backbone[0]
    else:
        base_backbone = backbone
    return base_backbone.projector


def deduce_in_channels(torch_projector, num_inputs, scale_factors, model_class):
    in_channels = []
    for index in range(num_inputs):
        sampler = torch_projector.stages_sampling[0][index]
        in_channels.append(first_layer_in_channels(sampler))
    if any(c is None for c in in_channels):
        in_channels = fill_from_c2f(in_channels, torch_projector, num_inputs, scale_factors)  # fmt: skip
    if any(c is None for c in in_channels):
        in_channels = fill_from_variant(in_channels, model_class)
    return in_channels


def first_layer_in_channels(sampler):
    if len(sampler) == 0:
        return None
    first_layer = sampler[0]
    if isinstance(first_layer, torch.nn.ConvTranspose2d):
        return first_layer.in_channels
    if hasattr(first_layer, "conv"):
        return first_layer.conv.in_channels
    if isinstance(first_layer, torch.nn.Conv2d):
        return first_layer.in_channels
    return None


def fill_from_c2f(in_channels, torch_projector, num_inputs, scale_factors):
    c2f = torch_projector.stages[0][0]
    if not (hasattr(c2f, "cv1") and hasattr(c2f.cv1, "conv")):
        return in_channels
    total_in = c2f.cv1.conv.in_channels
    per_channel = int(total_in * max(1, scale_factors[0]) / num_inputs)
    return [c if c is not None else per_channel for c in in_channels]


def fill_from_variant(in_channels, model_class):
    fallback = {"Nano": 384, "Small": 384, "Medium": 768, "Large": 1024}
    default = 256
    for name, channels in fallback.items():
        if name in model_class.__name__:
            default = channels
    return [c if c is not None else default for c in in_channels]


@pytest.mark.parametrize(
    "model_class", [RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge]
)
def test_rfdetr_projector_parity(model_class):
    torch_projector = locate_projector(model_class)
    scale_factors = list(torch_projector.scale_factors)
    num_inputs = len(torch_projector.stages_sampling[0])
    in_channels = deduce_in_channels(torch_projector, num_inputs, scale_factors, model_class)  # fmt: skip
    out_channels = torch_projector.stages[0][0].cv2.conv.out_channels
    input_scales = [1.0] * len(in_channels)
    keras_model = KerasMultiScaleProjector(in_channels, out_channels, scale_factors, input_scales=input_scales, layer_norm=True)  # fmt: skip
    port_weights_multiscale_projector(torch_projector, keras_model)
    size = 32
    x_np = [np.random.randn(1, size, size, c).astype("float32") for c in in_channels]  # fmt: skip
    torch_projector = torch_projector.cpu()
    x_torch = [torch.from_numpy(x.transpose(0, 3, 1, 2)) for x in x_np]
    with torch.no_grad():
        out_torch = torch_projector(x_torch)
    out_keras = as_list(keras_model(x_np, training=False))
    for o_t, o_k in zip(out_torch, out_keras):
        o_t_np = to_numpy(o_t).transpose(0, 2, 3, 1)
        np.testing.assert_allclose(o_t_np, o_k, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
