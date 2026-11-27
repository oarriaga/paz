import numpy as np
import paz


def test_compute_pairwise_distances_is_mean_squared_difference():
    x = np.array([[0.0, 0.0], [3.0, 4.0]], dtype="float32")
    y = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]], dtype="float32")
    output = paz.layers.ComputePairwiseDistances()(x, y)
    expected = np.array([[0.0, 4.5, 8.0], [12.5, 8.0, 4.5]])
    np.testing.assert_allclose(np.asarray(output), expected)
