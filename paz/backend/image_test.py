import tempfile
from pathlib import Path

import cv2
import numpy as np
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


def test_make_random_plain_image_is_uniform_color():
    image = paz.image.make_random_plain_image(jax.random.PRNGKey(0), (8, 8, 3))
    assert image.shape == (8, 8, 3)
    assert jp.all(image[0, 0] == image[5, 4])


def test_blend_background_respects_mask():
    foreground = jp.full((4, 4, 3), 200, jp.uint8)
    background = jp.zeros((4, 4, 3), jp.uint8)
    mask = jp.zeros((4, 4)).at[0, 0].set(1.0)
    blended = paz.image.blend_background(foreground, background, mask)
    assert int(blended[0, 0, 0]) == 200
    assert int(blended[1, 1, 0]) == 0


def test_add_occlusion_changes_image_and_keeps_shape():
    image = jp.full((32, 32, 3), 200, jp.uint8)
    occluded = paz.image.add_occlusion(jax.random.PRNGKey(1), image)
    assert occluded.shape == (32, 32, 3)
    assert not bool(jp.all(occluded == 200))


def test_randomize_rendered_image_keeps_shape():
    image = jp.full((32, 32, 3), 128, jp.uint8)
    mask = jp.ones((32, 32))
    key = jax.random.PRNGKey(2)
    output = paz.image.randomize_rendered_image(key, image, mask)
    assert output.shape == (32, 32, 3)


def test_fill_polygon_matches_cv2():
    polygon = jp.array([[12, 10], [50, 16], [55, 48], [28, 58], [10, 38]],
                       jp.float32)
    image = jp.full((64, 64, 3), 200, jp.uint8)
    filled = paz.image.fill_polygon(image, polygon, jp.zeros(3, jp.uint8))
    reference = np.full((64, 64, 3), 200, np.uint8)
    cv2.fillPoly(reference, [np.asarray(polygon).astype(np.int32)], (0, 0, 0))
    jax_mask = np.asarray((filled == 0).all(-1))
    reference_mask = (reference == 0).all(-1)
    assert (jax_mask == reference_mask).mean() > 0.95


def test_apply_gaussian_blur_matches_cv2_interior():
    image = np.random.default_rng(0).integers(0, 255, (64, 64, 3), np.uint8)
    blurred = paz.image.apply_gaussian_blur(jp.array(image), 9, 2.0)
    blurred = np.asarray(blurred).astype(np.int16)
    reference = cv2.GaussianBlur(image, (9, 9), 2.0).astype(np.int16)
    interior = np.abs(blurred[8:-8, 8:-8] - reference[8:-8, 8:-8]).mean()
    assert interior < 2.0


def test_randomize_rendered_image_jits_without_recompilation():
    randomize = jax.jit(paz.image.randomize_rendered_image)
    image = jp.full((32, 32, 3), 128, jp.uint8)
    mask = jp.ones((32, 32))
    randomize(jax.random.PRNGKey(0), image, mask).block_until_ready()
    randomize(jax.random.PRNGKey(1), image, mask).block_until_ready()
    assert randomize._cache_size() == 1
