import numpy as np

from paz.models.transformers import windowing


def build_patches(batch=2, height=4, width=6, hidden_size=3):
    size = batch * height * width * hidden_size
    values = np.arange(size, dtype="float32")
    return values.reshape(batch, height * width, hidden_size)


def test_partition_folds_windows_into_batch():
    windows = windowing.partition(build_patches(), (4, 6), 2)
    assert tuple(windows.shape) == (8, 6, 3)


def test_unpartition_inverts_partition():
    patches = build_patches()
    windows = windowing.partition(patches, (4, 6), 2)
    grid = windowing.unpartition(windows, (4, 6), 2)
    restored = np.reshape(np.array(grid), patches.shape)
    assert np.allclose(restored, patches)


def test_partition_keeps_each_window_contiguous():
    patches = build_patches(batch=1, height=2, width=2, hidden_size=1)
    windows = np.array(windowing.partition(patches, (2, 2), 2))
    assert np.allclose(windows[:, 0, 0], [0.0, 1.0, 2.0, 3.0])


def test_split_inverts_merge():
    windows = windowing.partition(build_patches(), (4, 6), 2)
    merged = windowing.merge(windows, 2)
    assert tuple(merged.shape) == (2, 24, 3)
    assert np.allclose(np.array(windowing.split(merged, 2)), np.array(windows))
