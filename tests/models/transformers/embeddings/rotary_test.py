import numpy as np

from paz.models.transformers.embeddings import rotary


def test_apply_at_position_zero_is_identity():
    x = np.random.RandomState(0).rand(1, 3, 8).astype("float32")
    positions = np.zeros((3, 1), "float32")   # angle 0 -> cos 1, sin 0
    output = np.asarray(rotary.apply(x, 10000.0, 1.0, 8, positions))
    np.testing.assert_allclose(output, x, atol=1e-5)


def test_apply_rotates_by_sequence_position_and_keeps_the_norm():
    x = np.random.RandomState(1).rand(2, 4, 6).astype("float32")
    output = np.asarray(rotary.apply(x, 10000.0, 1.0, 6))
    np.testing.assert_allclose(output[:, 0], x[:, 0], atol=1e-6)
    assert not np.allclose(output[:, 1:], x[:, 1:], atol=1e-3)
    source = np.linalg.norm(x, axis=-1)
    rotated = np.linalg.norm(output, axis=-1)
    np.testing.assert_allclose(rotated, source, atol=1e-5)


def test_apply_partial_rotates_only_the_rotary_fraction():
    # A 0.5 factor on 8 dims rotates the first 2 dims of each half.
    x = np.random.RandomState(2).rand(1, 3, 8).astype("float32")
    output = np.asarray(rotary.apply_partial(x, 10000.0, 1.0, 0.5))
    static, rotated = [2, 3, 6, 7], [0, 1, 4, 5]
    np.testing.assert_array_equal(output[..., static], x[..., static])
    changed = np.abs(output[:, 1:, rotated] - x[:, 1:, rotated])
    assert changed.max() > 1e-3


def test_apply_2d_at_position_zero_is_identity():
    tokens = np.random.RandomState(0).randn(1, 2, 1, 8).astype("float32")
    positions = np.zeros((1, 1, 2), "int32")
    output = np.asarray(rotary.apply_2D(tokens, positions, 100.0))
    np.testing.assert_allclose(output, tokens, atol=1e-6)


def test_apply_2d_preserves_norm_of_each_axis_half():
    from paz.models.transformers.embeddings.patch import build_patch_positions
    tokens = np.random.RandomState(1).randn(2, 3, 5, 8).astype("float32")
    grid = np.array(build_patch_positions(1, 5))
    positions = np.broadcast_to(grid, (2, 5, 2))
    output = np.asarray(rotary.apply_2D(tokens, positions, 100.0))
    for start in (0, 4):
        half = slice(start, start + 4)
        source = np.linalg.norm(tokens[..., half], axis=-1)
        rotated = np.linalg.norm(output[..., half], axis=-1)
        np.testing.assert_allclose(source, rotated, atol=1e-5)
