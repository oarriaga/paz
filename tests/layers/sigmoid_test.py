import keras
import numpy as np
import paz


def test_sigmoid_matches_numpy():
    x = np.linspace(-3, 3, 7, dtype="float32")
    output = paz.layers.Sigmoid()(x)
    expected = 1 / (1 + np.exp(-x))
    np.testing.assert_allclose(np.asarray(output), expected, rtol=1e-6)


def test_sigmoid_serialization_round_trip():
    x = np.linspace(-3, 3, 7, dtype="float32")
    serialized = keras.saving.serialize_keras_object(paz.layers.Sigmoid())
    restored = keras.saving.deserialize_keras_object(serialized)
    expected = 1 / (1 + np.exp(-x))
    np.testing.assert_allclose(np.asarray(restored(x)), expected, rtol=1e-6)
