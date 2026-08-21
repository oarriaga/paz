"""Opt-in parity check against the published PyTorch RF-DETR.

Needs ``torch``, the ``rfdetr`` package and the official checkpoints, so it is
skipped unless ``RFDETR_CHECKPOINTS`` points at a directory holding them. The
recorded differences are float32 accumulation noise over twelve windowed
transformer blocks and the decoder stack.

    RFDETR_CHECKPOINTS=~/rfdetr_checkpoints pytest paz/models/detection/rf_detr
"""
import os

import jax
import numpy as np
import pytest

# Reduced-precision float32 matmuls are the default on some CPUs and drift far
# enough over twelve transformer blocks to fail the tolerances below. Pin the
# precision here so the command in the docstring works on its own.
jax.config.update("jax_default_matmul_precision", "highest")

CHECKPOINTS = os.environ.get("RFDETR_CHECKPOINTS")
VARIANTS = [
    ("RFDETRNano", "rf-detr-nano.pth", 384),
    ("RFDETRSmall", "rf-detr-small.pth", 512),
    ("RFDETRMedium", "rf-detr-medium.pth", 576),
    ("RFDETRBase", "rf-detr-base.pth", 560),
    ("RFDETRLarge", "rf-detr-large-2026.pth", 704),
]

pytestmark = pytest.mark.skipif(
    CHECKPOINTS is None, reason="set RFDETR_CHECKPOINTS to run parity checks")


def build_reference(name, checkpoint):
    import rfdetr
    detector = getattr(rfdetr, name)(pretrain_weights=checkpoint)
    return detector.model.model.eval()


def run_reference(module, images):
    import torch
    tensor = torch.from_numpy(np.transpose(images, (0, 3, 1, 2)))
    with torch.no_grad():
        outputs = module(tensor)
    return outputs["pred_logits"].numpy(), outputs["pred_boxes"].numpy()


@pytest.mark.parametrize("name,filename,resolution", VARIANTS)
def test_matches_reference(name, filename, resolution):
    import paz
    from paz.models.detection.rf_detr.port_weights import port_weights
    checkpoint = os.path.join(os.path.expanduser(CHECKPOINTS), filename)
    module = build_reference(name, checkpoint)
    model = port_weights(getattr(paz.models, name)(), checkpoint)
    random = np.random.RandomState(0)
    images = random.randn(1, resolution, resolution, 3).astype("float32")
    logits, boxes = [np.array(tensor) for tensor in model(images)]
    expected_logits, expected_boxes = run_reference(module, images)
    assert np.allclose(logits, expected_logits, atol=1e-3)
    assert np.allclose(boxes, expected_boxes, atol=1e-3)
