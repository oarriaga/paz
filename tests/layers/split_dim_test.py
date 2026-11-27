import numpy as np
import paz


def test_split_dim_splits_axis_into_sizes():
    x = np.arange(2 * 20 * 5).reshape(2, 20, 5).astype("float32")
    output = np.asarray(paz.layers.SplitDim(axis=1, sizes=(4, 5))(x))
    assert output.shape == (2, 4, 5, 5)
    np.testing.assert_array_equal(output, x.reshape(2, 4, 5, 5))
