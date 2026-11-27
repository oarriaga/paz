import numpy as np
import paz


def test_merge_dims_merges_adjacent_axes():
    x = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype("float32")
    output = np.asarray(paz.layers.MergeDims(axis=-2)(x))
    assert output.shape == (2, 3, 20)
    np.testing.assert_array_equal(output, x.reshape(2, 3, 20))
