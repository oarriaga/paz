import tempfile
from pathlib import Path

import cv2
import numpy as np
import jax
import jax.numpy as jp

import paz


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
        assert np.array_equal(loaded, image)


def test_decode_JPEG_recovers_encoded_image():
    image = jp.full((8, 8, 3), jp.array([200, 100, 50]), dtype=jp.uint8)
    decoded = paz.image.decode_JPEG(paz.image.encode_JPEG(image))
    assert decoded.shape == image.shape
    assert jp.abs(decoded.astype(int) - image.astype(int)).max() <= 3


def test_forward_differences_pads_last_row_and_column_with_zeros():
    image = jp.array([[[1.0], [2.0]], [[4.0], [8.0]]])
    dy, dx = paz.image.forward_differences(image)
    assert jp.allclose(dy, jp.array([[[3.0], [6.0]], [[0.0], [0.0]]]))
    assert jp.allclose(dx, jp.array([[[1.0], [0.0]], [[4.0], [0.0]]]))


def test_forward_differences_batches_images():
    image = jp.array([[[1.0], [2.0]], [[4.0], [8.0]]])
    batch = jp.stack((image, image * 2.0))
    dy, dx = paz.image.forward_differences(batch)
    expected_dy, expected_dx = paz.image.forward_differences(image)
    assert jp.allclose(dy, jp.stack((expected_dy, expected_dy * 2.0)))
    assert jp.allclose(dx, jp.stack((expected_dx, expected_dx * 2.0)))


def test_make_random_plain_image_is_uniform_color():
    image = paz.image.make_random_plain_image(jax.random.PRNGKey(0), (8, 8, 3))
    assert image.shape == (8, 8, 3)
    assert jp.all(image[0, 0] == image[5, 4])


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


def test_random_photometric_changes_image_and_keeps_uint8():
    key = jax.random.PRNGKey(1)
    image = jax.random.randint(key, (64, 64, 3), 0, 256).astype(jp.uint8)
    changed = [image]
    for seed in range(4):
        key = jax.random.PRNGKey(seed)
        changed.append(paz.image.random_photometric(key, image))
    assert all(output.dtype == jp.uint8 for output in changed)
    assert all(output.shape == image.shape for output in changed)
    assert any(not np.array_equal(output, image) for output in changed)


def test_scale_with_aspect_ratio_returns_scaled_image():
    image = jp.ones((10, 20, 3))
    scaled = paz.image.scale_with_aspect_ratio(image, (2.0, 0.5))
    assert scaled.shape == (20, 10, 3)


def test_denormalize_rounds_instead_of_truncating():
    image = jax.random.randint(jax.random.PRNGKey(0), (16, 16, 3), 0, 256)
    image = image.astype(jp.uint8)
    same = paz.image.random_hue(jax.random.PRNGKey(0), image, max_delta=0.0)
    assert np.array_equal(same, image)
    white = jp.full((2, 2, 3), 255, jp.uint8)
    assert np.all(np.asarray(paz.image.RGB_to_GRAY(white)) == 255)


def test_resize_with_aspect_ratio_fills_a_square_target():
    image = jp.ones((49, 49, 3))
    resized = paz.image.resize_with_aspect_ratio(image, 512, 512)
    assert resized.shape == (512, 512, 3)
    assert np.all(np.asarray(resized) > 0)


def test_random_blur_keeps_shape_of_2D_image():
    image = jax.random.randint(jax.random.PRNGKey(0), (20, 20), 0, 256)
    image = image.astype(jp.uint8)
    blurred = paz.image.apply_gaussian_blur(image)[..., 0].astype(jp.uint8)
    for seed in range(4):
        output = paz.image.random_blur(jax.random.PRNGKey(seed), image)
        assert output.shape == image.shape
        is_blurred = np.array_equal(output, blurred)
        assert is_blurred or np.array_equal(output, image)


def test_place_in_canvas_identity():
    image = jax.random.randint(jax.random.PRNGKey(0), (32, 32, 3), 0, 256)
    scale, translation = jp.array([1.0, 1.0]), jp.array([0.0, 0.0])
    args = image, (32, 32), scale, translation, jp.zeros(3)
    same = paz.image.place_in_canvas(*args)
    assert jp.allclose(same, image.astype(jp.float32), atol=1e-4)


def test_place_in_canvas_fills_uncovered():
    image = jp.full((32, 32, 3), 200.0)
    scale, translation = jp.array([0.5, 0.5]), jp.array([0.0, 0.0])
    fill = jp.array([10.0, 20.0, 30.0])
    args = image, (32, 32), scale, translation, fill
    canvas = paz.image.place_in_canvas(*args)
    assert jp.allclose(canvas[-1, -1], fill, atol=1e-4)
    assert jp.allclose(canvas[0, 0], jp.full(3, 200.0), atol=1e-4)
