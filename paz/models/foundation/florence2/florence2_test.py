import os

import numpy as np
import pytest
from keras import Input, Model

from paz.models.foundation.florence2.configuration import CONFIGS
from paz.models.foundation.florence2.encoder import encoder_block
from paz.models.foundation.florence2.model import build
from paz.models.foundation.florence2.vision import ImageEncoder
from paz.models.foundation.florence2.configuration import to_vision_args

TINY = dict(CONFIGS["florence2_large_flower"])
TINY.update(image_size=32, stage_dims=(8, 16, 24, 32),
            stage_depths=(1, 1, 2, 1), stage_heads=(2, 2, 2, 2),
            stage_groups=(2, 2, 2, 2), window_size=2, projection_dim=16,
            vocabulary_size=64, hidden_dim=16, num_layers=2, num_heads=2,
            ffn_dim=32, max_positions=128)


def test_image_encoder_token_shape():
    encoder = ImageEncoder(to_vision_args(TINY))
    images = np.zeros((2, 32, 32, 3), "float32")
    tokens = encoder.predict(images, verbose=0)
    assert tokens.shape == (2, 2, 16)


def test_model_output_shape():
    model = build(TINY)
    images = np.zeros((2, 32, 32, 3), "float32")
    token_ids = np.zeros((2, 5), "int32")
    context = model.predict([images, images, token_ids], verbose=0)
    assert context.shape == (2, 9, 16)


def test_encoder_block_mask_hides_keys():
    x = Input((4, 16), name="block_x")
    mask = Input((4,), name="block_mask")
    y = encoder_block(x, mask, 2, 32, "block_mask_test")
    model = Model([x, mask], y)
    values = np.random.default_rng(0).normal(size=(1, 4, 16))
    values = values.astype("float32")
    altered = np.copy(values)
    altered[0, 3] = 7.0
    mask_values = np.array([[1.0, 1.0, 1.0, 0.0]], "float32")
    kept = model.predict([values, mask_values], verbose=0)
    changed = model.predict([altered, mask_values], verbose=0)
    assert np.allclose(kept[:, :3], changed[:, :3], atol=1e-5)


@pytest.mark.skipif(os.environ.get("FLOWER_WEIGHTS_TEST") != "1",
                    reason="needs converted FLOWER weights and fixtures")
def test_encoder_hidden_states_match_torch_fixture():
    weights = os.environ["FLORENCE2_WEIGHTS"]
    fixtures = np.load(os.environ["FLORENCE2_FIXTURES"])
    model = build(CONFIGS["florence2_large_flower"])
    model.load_weights(weights)
    static = fixtures["static_image_preprocessed"][:, 0]
    static = static.transpose(0, 2, 3, 1).astype("float32")
    gripper = fixtures["gripper_image_preprocessed"][:, 0]
    gripper = gripper.transpose(0, 2, 3, 1).astype("float32")
    flow_and_text = np.concatenate(
        [[[51289]], fixtures["text_input_ids"]], axis=1).astype("int32")
    hidden = model.predict([static, gripper, flow_and_text], verbose=0)
    reference = fixtures["encoder_hidden_states"]
    assert np.abs(hidden - reference).max() < 1e-4
