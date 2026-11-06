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


def test_randomize_rendered_image_keeps_shape():
    image = jp.full((32, 32, 3), 128, jp.uint8)
    mask = jp.ones((32, 32))
    key = jax.random.PRNGKey(2)
    output = paz.image.randomize_rendered_image(key, image, mask)
    assert output.shape == (32, 32, 3)


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


def test_random_photometric_stays_in_uint8_range():
    key = jax.random.PRNGKey(1)
    image = jax.random.randint(key, (64, 64, 3), 0, 256).astype(jp.uint8)
    out = paz.image.random_photometric(jax.random.PRNGKey(2), image)
    assert out.dtype == jp.uint8
    assert int(out.min()) >= 0 and int(out.max()) <= 255


def build_enhance_image():
    return jp.array([[[217, 163, 130], [0, 4, 168], [19, 4, 44]]], jp.uint8)


def build_texture_image():
    values = (jp.arange(8 * 10) * 37) % 256
    values = jp.reshape(values, (8, 10, 1)).astype(jp.uint8)
    return jp.repeat(values, 3, axis=-1)


def assert_crop_matches_pillow(box, expected):
    H, W = len(expected), len(expected[0])
    window = jp.array(box) / jp.array([10, 8, 10, 8])
    resized = paz.image.crop_and_resize(build_texture_image(), window, H, W)
    assert resized.shape == (H, W, 3) and resized.dtype == jp.float32
    assert np.abs(resized[..., 0] - np.array(expected)).max() <= 1.0


def test_scale_with_aspect_ratio_returns_resized_image():
    image = jax.random.uniform(jax.random.PRNGKey(0), (10, 20, 3))
    scale_with_aspect_ratio = paz.image.scale_with_aspect_ratio
    scaled = scale_with_aspect_ratio(image, (0.35, 0.5))
    assert scaled.shape == (3, 10, 3)
    assert jp.allclose(scaled, paz.image.resize(image, (3, 10)))
    for method, antialias in (("nearest", False), ("linear", True)):
        scaled = scale_with_aspect_ratio(image, (0.35, 0.5), method, antialias)
        expected = paz.image.resize(image, (3, 10), method, antialias)
        assert jp.allclose(scaled, expected)


def test_compute_luminance_matches_pillow():
    luminance = paz.image.compute_luminance(build_enhance_image())
    # PIL convert("L"); 0.299 / 0.587 / 0.114 weights give 22 for (0, 4, 168)
    assert luminance.shape == (1, 3, 1) and luminance.dtype == jp.uint8
    assert luminance.ravel().tolist() == [175, 21, 13]
    # PIL rounds (+32768 >> 16); floor or float weights give 151 and 35
    pixels = jp.array([[[69, 211, 65], [21, 7, 221]]], jp.uint8)
    assert paz.image.compute_luminance(pixels).ravel().tolist() == [152, 36]


def test_adjust_functions_match_pillow():
    # PIL ImageEnhance.Brightness / Contrast / Color(image).enhance(factor)
    cases = [("brightness", 0.65, [141, 105, 84, 0, 2, 109, 12, 2, 28]),
             ("brightness", 1.35, [255, 220, 175, 0, 5, 226, 25, 5, 59]),
             ("contrast", 0.65, [165, 130, 109, 24, 27, 133, 36, 27, 53]),
             ("contrast", 1.35, [255, 195, 151, 0, 0, 202, 1, 0, 34]),
             ("color", 0.65, [202, 167, 145, 7, 9, 116, 16, 7, 33]),
             ("color", 1.35, [231, 158, 114, 0, 0, 219, 21, 0, 54])]
    for name, factor, expected in cases:
        adjust = getattr(paz.image, "adjust_" + name)
        adjusted = adjust(build_enhance_image(), factor)
        assert adjusted.dtype == jp.uint8
        assert adjusted.ravel().tolist() == expected


def test_adjust_contrast_rounds_mean_half_up_and_jits():
    image = jp.array([[[100, 100, 100], [101, 101, 101]]], jp.uint8)
    adjust = jax.jit(paz.image.adjust_contrast)
    # PIL ImageEnhance.Contrast(image).enhance(0.0) fills with 101
    assert bool(jp.all(adjust(image, jp.float32(0.0)) == 101))
    assert jp.array_equal(adjust(image, jp.float32(1.0)), image)
    assert adjust._cache_size() == 1
    pixels = jp.array([[[217, 163, 130], [69, 78, 10]]], jp.uint8)
    # PIL mean L is 122; float 0.299 / 0.587 / 0.114 weights give 121
    flat = paz.image.adjust_contrast(pixels, 0.0)
    assert bool(jp.all(flat == 122))


def test_adjust_contrast_keeps_exact_mean_on_large_images():
    image = jp.full((3000, 3000, 3), 250, jp.uint8)
    # 9 MP overflows a plain uint32 sum; PIL Contrast(0.0) fills with 250
    assert bool(jp.all(paz.image.adjust_contrast(image, 0.0) == 250))


def test_approximate_gaussian_blur_matches_pillow_impulse():
    image = jp.zeros((7, 7, 3), jp.uint8).at[3, 3].set(255)
    blur = jax.jit(paz.image.approximate_gaussian_blur)
    blurred = blur(image, jp.float32(0.8))
    # PIL image.filter(ImageFilter.GaussianBlur(0.8)), first channel
    expected = [[0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 4, 2, 0, 0],
                [0, 1, 11, 28, 11, 1, 0],
                [0, 4, 29, 75, 29, 4, 0],
                [0, 1, 11, 28, 11, 1, 0],
                [0, 0, 2, 4, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]]
    assert blurred.dtype == jp.uint8
    assert blurred[..., 0].tolist() == expected
    # PIL GaussianBlur(0.2), centre row
    assert blur(image, jp.float32(0.2))[3, 2:5, 0].tolist() == [6, 237, 6]
    assert blur._cache_size() == 1


def test_approximate_gaussian_blur_replicates_edges():
    image = jp.full((5, 6, 3), 200, jp.uint8).at[:, 3:].set(40)
    blurred = paz.image.approximate_gaussian_blur(image, 0.8)
    # PIL image.filter(ImageFilter.GaussianBlur(0.8)), every row
    assert blurred[..., 0].tolist() == [[200, 195, 163, 77, 45, 40]] * 5
    values = jp.array([246, 36, 172, 176, 169, 63], jp.uint8)
    row = jp.repeat(values[None, :, None], 3, axis=-1)
    blurred = paz.image.approximate_gaussian_blur(row, 0.5)
    # PIL GaussianBlur(0.5); integer passes round like BoxBlur.c
    assert blurred[0, :, 0].tolist() == [222, 74, 158, 174, 158, 75]


def test_crop_and_resize_matches_pillow_on_every_pixel():
    # torchvision F.resize(PIL image.crop(box), (H, W)), first channel
    shrink = [[93, 121, 121, 100], [120, 126, 122, 117],
              [138, 129, 129, 139], [127, 131, 150, 140]]
    assert_crop_matches_pillow((0, 0, 10, 8), shrink)
    enlarge = [[188, 207, 188, 43, 25, 49, 74, 99, 117],
               [117, 136, 139, 79, 82, 106, 131, 156, 174],
               [65, 84, 108, 133, 137, 139, 164, 189, 207],
               [141, 160, 184, 209, 127, 45, 70, 95, 113],
               [89, 108, 132, 157, 118, 78, 103, 128, 146],
               [18, 37, 61, 86, 111, 135, 160, 185, 203]]
    assert_crop_matches_pillow((2, 1, 8, 5), enlarge)
    shrink_crop = [[101, 125, 118, 114, 103], [114, 137, 111, 133, 116],
                   [153, 116, 142, 127, 158]]
    assert_crop_matches_pillow((1, 0, 10, 7), shrink_crop)


def test_crop_and_resize_jits_and_vmaps_over_windows():
    crop = paz.lock(paz.image.crop_and_resize, 5, 7)
    crop = jax.jit(jax.vmap(crop, (None, 0)))
    windows = jp.array([[0.0, 0.0, 1.0, 1.0], [0.1, 0.2, 0.6, 0.9]])
    assert crop(build_texture_image(), windows).shape == (2, 5, 7, 3)
    crop(build_texture_image(), windows * 0.9)
    assert crop._cache_size() == 1


def test_compute_short_side_size_matches_reference():
    # rfdetr transforms.resize(image, None, short_side) output (H, W)
    cases = [(480, 640, 400, (400, 533)), (640, 480, 500, (666, 500)),
             (333, 500, 600, (600, 900)), (500, 500, 400, (400, 400)),
             (97, 1000, 600, (600, 6185))]
    for H, W, short_side, expected in cases:
        size = paz.image.compute_short_side_size(H, W, short_side)
        assert tuple(int(side) for side in size) == expected
    compute = jax.jit(paz.image.compute_short_side_size, static_argnums=(0, 1))
    for short_side, expected in ((500, (500, 666)), (600, (600, 800))):
        size = compute(480, 640, jp.array(short_side))
        assert tuple(int(side) for side in size) == expected
    assert compute._cache_size() == 1


def test_compute_limited_short_side_size_matches_reference():
    # rfdetr transforms.resize(image, None, short_side, max_size) (H, W)
    cases = [(480, 640, 704, 1333, (704, 938)),
             (300, 1000, 800, 1333, (400, 1333)),
             (1, 2, 1000, 1333, (666, 1332)),
             (200, 1000, 400, 1333, (267, 1335)),
             (1000, 100, 800, 1333, (1330, 133))]
    for H, W, short_side, max_long_side, expected in cases:
        args = (H, W, short_side, max_long_side)
        size = paz.image.compute_limited_short_side_size(*args)
        assert tuple(int(side) for side in size) == expected


def test_sample_enhance_factor_matches_reference_range():
    keys = jax.random.split(jax.random.PRNGKey(0), 256)
    sample = jax.vmap(paz.image.sample_enhance_factor, (0, None))
    factors = sample(keys, 0.35)
    assert factors.shape == (256,) and factors.dtype == jp.float32
    assert 0.65 - 1e-6 <= float(factors.min()) < 0.7
    assert 1.3 < float(factors.max()) <= 1.35 + 1e-6
    # fork: max(0.1, 1 + uniform(-strength, strength))
    clamped = sample(keys, 2.0)
    assert abs(float(clamped.min()) - 0.1) < 1e-6
    assert 0.15 < float(jp.mean(clamped < 0.1 + 1e-6)) < 0.40
