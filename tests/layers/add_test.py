import keras
import numpy as np
import paz


def test_add_matches_numpy():
    x = np.arange(6, dtype="float32").reshape(2, 3)
    y = np.ones((2, 3), dtype="float32")
    output = paz.layers.Add()(x, y)
    np.testing.assert_allclose(np.asarray(output), x + y)


def test_add_serialization_round_trip():
    x = np.arange(6, dtype="float32").reshape(2, 3)
    serialized = keras.saving.serialize_keras_object(paz.layers.Add())
    restored = keras.saving.deserialize_keras_object(serialized)
    np.testing.assert_allclose(np.asarray(restored(x, x)), 2 * x)
