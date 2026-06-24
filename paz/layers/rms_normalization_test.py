import numpy as np
import paz


def test_rms_normalization_matches_reference():
    layer = paz.layers.RMSNormalization(epsilon=1e-6)
    x = np.random.RandomState(0).rand(2, 5).astype("float32")
    output = np.asarray(layer(x))
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    expected = x / np.sqrt(variance + 1e-6)   # scale initializes to ones
    np.testing.assert_allclose(output, expected, atol=1e-5)


def test_rms_normalization_preserves_shape():
    layer = paz.layers.RMSNormalization()
    x = np.zeros((3, 4, 8), "float32")
    assert tuple(np.asarray(layer(x)).shape) == (3, 4, 8)
