import numpy as np
import keras

from paz.models.foundation.dinov2.models.windowed_vision_transformer import (
    WindowedDinov2Model,
)


def build_tiny_feature_model(name):
    return WindowedDinov2Model(
        image_size=56, patch_size=14, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_windows=1, num_register_tokens=0,
        out_feature_indexes=[0, 1], name=name,
    )


def copy_weights_by_name(src, dst):
    for layer in src.layers:
        weights = layer.get_weights()
        if weights:
            dst.get_layer(layer.name).set_weights(weights)


def test_roundtrip_name_based_weight_copy_matches():
    keras.utils.set_random_seed(0)
    src = build_tiny_feature_model("src")
    dst = build_tiny_feature_model("dst")
    x = np.random.RandomState(0).randn(1, 56, 56, 3).astype(np.float32)
    src(x)
    dst(x)
    copy_weights_by_name(src, dst)
    for src_out, dst_out in zip(src(x), dst(x)):
        np.testing.assert_array_equal(np.array(src_out), np.array(dst_out))


if __name__ == "__main__":
    test_roundtrip_name_based_weight_copy_matches()
