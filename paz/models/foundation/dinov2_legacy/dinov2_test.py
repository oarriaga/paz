import os

import numpy as np
import pytest

from paz.models.foundation.dinov2_legacy.models.vision_transformer import (
    DINOV2Small,
    DINOV2Base,
    DINOV2Large,
)

IMAGE_SIZE = 518
TOLERANCE = 1e-4
GROUND_TRUTH_PATH = os.path.join("weights", "dinov2_ground_truth.npz")
WEIGHTS = {
    "dinov2_vits14": os.path.join("weights", "dinov2_vits14_ported.keras"),
    "dinov2_vitb14": os.path.join("weights", "dinov2_vitb14_ported.keras"),
    "dinov2_vitl14": os.path.join("weights", "dinov2_vitl14_ported.keras"),
}
VARIANTS = list(WEIGHTS)
BUILDERS = {
    "dinov2_vits14": DINOV2Small,
    "dinov2_vitb14": DINOV2Base,
    "dinov2_vitl14": DINOV2Large,
}


def load_ground_truth(variant):
    if not os.path.exists(GROUND_TRUTH_PATH):
        pytest.skip(
            f"ground truth not found: {GROUND_TRUTH_PATH} "
            "(regenerate via weights/dinov2_test.py with --run-heavy)"
        )
    return np.load(GROUND_TRUTH_PATH)[variant]


def make_ramp_input(size=IMAGE_SIZE):
    count = 1 * 3 * size * size
    ramp = (np.arange(count) % 255 / 255.0).astype(np.float32)
    torch_layout = ramp.reshape(1, 3, size, size)
    return np.transpose(torch_layout, (0, 2, 3, 1)).copy()


def load_ported_model(variant):
    path = WEIGHTS[variant]
    if not os.path.exists(path):
        pytest.skip(f"ported weights not found: {path}")
    model = BUILDERS[variant](img_size=IMAGE_SIZE)
    model.load_weights(path)
    return model


@pytest.mark.parametrize("variant", VARIANTS)
def test_ported_output_matches_pytorch_ground_truth(variant):
    expected = load_ground_truth(variant)
    model = load_ported_model(variant)
    output = np.array(model(make_ramp_input(), training=False)).reshape(-1)
    assert output.shape == expected.shape
    assert np.allclose(output, expected, atol=TOLERANCE)


@pytest.mark.parametrize("variant", VARIANTS)
def test_ported_output_shape(variant):
    model = load_ported_model(variant)
    output = np.array(model(make_ramp_input(), training=False))
    expected_dim = {"dinov2_vits14": 384, "dinov2_vitb14": 768,
                    "dinov2_vitl14": 1024}[variant]
    assert output.shape == (1, expected_dim)
