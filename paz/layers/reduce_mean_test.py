import keras
import numpy as np
import paz


def test_reduce_mean_matches_numpy():
    x = np.arange(24, dtype="float32").reshape(2, 3, 4)
    output = paz.layers.ReduceMean(axes=[1], keepdims=False)(x)
    np.testing.assert_allclose(np.asarray(output), x.mean(axis=1))


def test_reduce_mean_config_round_trip():
    layer = paz.layers.ReduceMean(axes=[1], keepdims=False)
    restored = paz.layers.ReduceMean.from_config(layer.get_config())
    assert restored.axes == [1]
    assert restored.keepdims is False


def test_reduce_mean_serialization_round_trip():
    x = np.arange(24, dtype="float32").reshape(2, 3, 4)
    layer = paz.layers.ReduceMean(axes=[1], keepdims=False)
    serialized = keras.saving.serialize_keras_object(layer)
    restored = keras.saving.deserialize_keras_object(serialized)
    np.testing.assert_allclose(np.asarray(restored(x)), x.mean(axis=1))
