import os

import numpy as np
import pytest
from keras import Input, Model

from paz.models.foundation.florence2.encoder import encoder_block
from paz.models.foundation.florence2.model import build
from paz.models.foundation.florence2.vision import ImageEncoder

TINY = {"image_size": 32, "stage_dims": (8, 16, 24, 32),
        "stage_depths": (1, 1, 2, 1), "stage_heads": (2, 2, 2, 2),
        "stage_groups": (2, 2, 2, 2), "window_size": 2,
        "vocabulary_size": 64, "hidden_dim": 16, "num_layers": 2,
        "num_heads": 2, "ffn_dim": 32, "max_positions": 128}


def test_image_encoder_token_shape():
    args = (TINY["image_size"], TINY["hidden_dim"], TINY["stage_dims"],
            TINY["stage_depths"], TINY["stage_heads"],
            TINY["stage_groups"], TINY["window_size"])
    encoder = ImageEncoder(*args)
    images = np.zeros((2, 32, 32, 3), "float32")
    tokens = encoder.predict(images, verbose=0)
    assert tokens.shape == (2, 2, 16)


def test_model_output_shape():
    model = build(**TINY)
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


WEIGHTS_OFF = os.environ.get("FLOWER_WEIGHTS_TEST") != "1"


@pytest.mark.skipif(WEIGHTS_OFF, reason="needs converted FLOWER weights")
def test_encoder_hidden_states_match_torch_fixture():
    weights = os.environ["FLORENCE2_WEIGHTS"]
    fixtures = np.load(os.environ["FLORENCE2_FIXTURES"])
    model = build()
    model.load_weights(weights)
    static = fixtures["static_image_preprocessed"][:, 0]
    static = static.transpose(0, 2, 3, 1).astype("float32")
    gripper = fixtures["gripper_image_preprocessed"][:, 0]
    gripper = gripper.transpose(0, 2, 3, 1).astype("float32")
    text_ids = fixtures["text_input_ids"]
    flow_and_text = np.concatenate([[[51289]], text_ids], axis=1)
    flow_and_text = flow_and_text.astype("int32")
    hidden = model.predict([static, gripper, flow_and_text], verbose=0)
    reference = fixtures["encoder_hidden_states"]
    assert np.abs(hidden - reference).max() < 1e-4


def test_default_attends_padding():
    # The default has to behave like the reference, which attends every token,
    # so appending padding must reach the real tokens as well.
    model = build(**TINY)
    images = np.random.default_rng(2).normal(size=(1, 32, 32, 3))
    images = images.astype("float32")
    short = np.array([[5, 6, 7]], "int32")
    padded = np.array([[5, 6, 7, 1, 1]], "int32")
    kept = model.predict([images, images, short], verbose=0)
    grown = model.predict([images, images, padded], verbose=0)
    assert np.abs(kept - grown[:, :kept.shape[1]]).max() > 1e-6


def test_pad_id_hides_padding_from_real_tokens():
    # With pad_id set, appending padding must not change the real tokens.
    model = build(pad_id=1, **TINY)
    images = np.random.default_rng(0).normal(size=(1, 32, 32, 3))
    images = images.astype("float32")
    short = np.array([[5, 6, 7]], "int32")
    padded = np.array([[5, 6, 7, 1, 1]], "int32")
    kept = model.predict([images, images, short], verbose=0)
    grown = model.predict([images, images, padded], verbose=0)
    assert np.abs(kept - grown[:, :kept.shape[1]]).max() < 1e-4


def test_pad_id_changes_the_result_when_padding_is_present():
    # Guard against the mask silently doing nothing.
    images = np.random.default_rng(1).normal(size=(1, 32, 32, 3))
    images = images.astype("float32")
    padded = np.array([[5, 6, 7, 1, 1]], "int32")
    masked = build(pad_id=1, **TINY)
    unmasked = build(**TINY)
    unmasked.set_weights(masked.get_weights())
    with_mask = masked.predict([images, images, padded], verbose=0)
    without = unmasked.predict([images, images, padded], verbose=0)
    assert np.abs(with_mask - without).max() > 1e-6
