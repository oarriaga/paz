import numpy as np
from keras import ops

from examples.gemma3.functional.gemma3 import build_vision_encoder
from examples.gemma3.functional.test_utils import copy_vision_encoder_weights

from keras_hub.src.models.gemma3 import gemma3_vision_encoder


def _build_encoder():
    return build_vision_encoder(
        image_size=16,
        patch_size=4,
        num_heads=2,
        hidden_dim=8,
        num_layers=2,
        intermediate_dim=16,
        output_dim=12,
        pool_size=2,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        dtype="float32",
        name_prefix="clean",
    )


def _build_hub_encoder(name):
    return gemma3_vision_encoder.Gemma3VisionEncoder(
        image_size=16,
        patch_size=4,
        num_heads=2,
        hidden_dim=8,
        num_layers=2,
        intermediate_dim=16,
        output_dim=12,
        pool_size=2,
        layer_norm_epsilon=1e-6,
        dtype="float32",
        name=name,
    )


def test_vision_encoder_output_matches_hub():
    rng = np.random.default_rng(3)
    shape = (2, 1, 16, 16, 3)
    values = rng.standard_normal(shape).astype("float32")
    images = ops.convert_to_tensor(values)

    hub_model = _build_hub_encoder("hub_vision")
    hub_model(images)

    apply_vision, layers = _build_encoder()
    apply_vision(images, None, False)

    copy_vision_encoder_weights(layers, hub_model)

    hub_output = hub_model(images)
    clean_output = apply_vision(images, None, False)
    clean_np = ops.convert_to_numpy(clean_output)
    hub_np = ops.convert_to_numpy(hub_output)
    np.testing.assert_allclose(clean_np, hub_np, rtol=1e-5, atol=1e-5)
