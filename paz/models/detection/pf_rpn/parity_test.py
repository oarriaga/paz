"""Opt-in parity check against the published PyTorch PF-RPN.

The reference runs on MMDetection, which does not share an environment with
paz, so its tensors are read from a file. The check is skipped unless
``PFRPN_REFERENCE`` points at a directory holding the published checkpoint
and ``reference.npz``, the tensors of one upstream forward pass:

- ``images``: the standardized input, ``(1, 3, H, W)``
- ``neck_3``: the coarsest neck level, ``(1, 256, H / 64, W / 64)``
- ``encoder_memory``: the encoder output, ``(1, cells, 256)``
- ``class_logits``, ``confidence`` and ``boxes``: the head outputs of the
  last decoder layer, before ranking

    PFRPN_REFERENCE=~/pfrpn-ref/reference pytest paz/models/detection/pf_rpn
"""
import os

import jax
import numpy as np
import pytest
from keras import Model

REFERENCE = os.environ.get("PFRPN_REFERENCE")
reason = "set PFRPN_REFERENCE to run parity checks"
pytestmark = pytest.mark.skipif(REFERENCE is None, reason=reason)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_matches_reference():
    from paz.models.detection.pf_rpn import PFRPN
    from paz.models.detection.pf_rpn.port_weights import port_weights
    directory = os.path.expanduser(REFERENCE)
    reference = np.load(os.path.join(directory, "reference.npz"))
    images = np.transpose(reference["images"], (0, 2, 3, 1))
    checkpoint = os.path.join(directory, "pf_rpn_swinb_5p_coco_imagenet.pth")
    model = port_weights(PFRPN(images.shape[1:]), checkpoint)
    names = "neck_extra_convs_0_gn", "encoder_layers_5_norms_1"
    taps = [model.get_layer(name).output for name in names]
    tapped = Model(model.input, taps + model.outputs)
    # Jitted, because an eager call keeps every intermediate tensor in memory.
    # Reduced-precision float32 matmuls are the default on some CPUs and
    # drift far enough over the backbone to fail the tolerances below.
    with jax.default_matmul_precision("highest"):
        outputs = jax.jit(lambda x: tapped(x))(images)
    neck, memory, scores, boxes = [np.array(output) for output in outputs]
    similarity = sigmoid(reference["class_logits"]).max(axis=-1)
    # Upstream squashes the confidence once more when it ranks the boxes.
    confidence = sigmoid(reference["confidence"][..., 0])
    expected_neck = np.transpose(reference["neck_3"], (0, 2, 3, 1))
    assert np.allclose(neck, expected_neck, atol=1e-3)
    assert np.allclose(memory, reference["encoder_memory"], atol=1e-3)
    assert np.allclose(scores, similarity + confidence, atol=1e-3)
    assert np.allclose(boxes, reference["boxes"], atol=1e-3)
