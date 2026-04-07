import tempfile
from pathlib import Path

import jax
import jax.numpy as jp

import paz


def legacy_forward_differences(image):
    H, W, C = image.shape
    dy = image[1:, :, :] - image[:-1, :, :]
    dx = image[:, 1:, :] - image[:, :-1, :]
    dy = jp.concatenate([dy, jp.zeros((1, W, C))], axis=0)
    dx = jp.concatenate([dx, jp.zeros((H, 1, C))], axis=1)
    return dy, dx


def test_write_accepts_path_object():
    image = jp.full((4, 4, 3), 128, dtype=jp.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "image.png"
        paz.image.write(filepath, image)
        assert filepath.is_file()


def test_load_accepts_path_object():
    image = jp.full((4, 4, 3), 200, dtype=jp.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "image.png"
        paz.image.write(str(filepath), image)
        loaded = paz.image.load(filepath)
        assert loaded.shape == image.shape


def test_forward_differences_matches_example_formula():
    image = jp.array([[[1.0], [2.0]], [[4.0], [8.0]]])
    dy, dx = paz.image.forward_differences(image)
    expected_dy, expected_dx = legacy_forward_differences(image)
    assert jp.allclose(dy, expected_dy)
    assert jp.allclose(dx, expected_dx)
    assert jp.allclose(dy[-1], jp.zeros_like(dy[-1]))
    assert jp.allclose(dx[:, -1], jp.zeros_like(dx[:, -1]))


def test_forward_differences_batches_images():
    image = jp.array([[[1.0], [2.0]], [[4.0], [8.0]]])
    batch = jp.stack((image, image * 2.0))
    dy, dx = paz.image.forward_differences(batch)
    expected = jax.vmap(legacy_forward_differences)(batch)
    assert jp.allclose(dy, expected[0])
    assert jp.allclose(dx, expected[1])
