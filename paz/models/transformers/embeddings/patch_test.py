import numpy as np

from paz.models.transformers.embeddings.patch import build_patch_positions


def test_build_patch_positions_is_row_major_cartesian():
    positions = np.array(build_patch_positions(2, 3))
    expected = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    assert positions.tolist() == expected
