import pytest
import numpy as np
from paz import processors as pr


def test_DrawBoxes2D_with_invalid_class_names_type():
    with pytest.raises(TypeError):
        class_names = 'Face'
        colors = [[255, 0, 0]]
        pr.DrawBoxes2D(class_names, colors)


def test_DrawBoxes2D_with_invalid_colors_type():
    with pytest.raises(TypeError):
        class_names = ['Face']
        colors = [255, 0, 0]
        pr.DrawBoxes2D(class_names, colors)


def get_test_images(image_W, image_H, batch_size=1):
    """Generates a simple mock image.

    # Arguments
        image_size: Int, integer value for H x W image shape.
        batch_size: Int, batch size for the input array.

    # Returns
        image: Zeros of shape (batch_size, H, W, C)
    """
    return np.zeros((batch_size, image_W, image_H, 3), dtype=np.float32)


@pytest.mark.parametrize(('size, target_resizing_shape, target_scale'),
                         [
                            (32, (24, 32), 0.05),
                            (64, (48, 64), 0.1),
                            (128, (96, 128), 0.2),
                            (256, (192, 256), 0.4),
                            (512, (384, 512), 0.8),
                         ])
def test_ComputeResizingShape(size, target_resizing_shape, target_scale):
    image = get_test_images(640, 480, batch_size=1)[0]
    compute_shape = pr.ComputeResizingShape(size)
    resizing_shape, scale = compute_shape(image)
    assert resizing_shape == target_resizing_shape
    assert scale == target_scale


@pytest.mark.parametrize(('size'), [32, 64, 128, 256, 512])
def test_PadImage(size):
    image = get_test_images(16, 12, batch_size=1)[0]
    pad_image = pr.PadImage(size, mode='constant')
    padded_image = pad_image(image)
    assert np.sum(padded_image) == 0.0
    assert padded_image.shape == (size, size, 3)


def test_EqualizeHistogram():
    image = np.uint8(np.random.rand(640, 480, 3) * 255)
    equalize_histogram = pr.EqualizeHistogram()
    image_histogram_equalized = equalize_histogram(image)
    assert image_histogram_equalized.shape == image.shape
    assert image_histogram_equalized.dtype == image.dtype
    assert np.std(image_histogram_equalized) >= np.std(image)


def test_InvertColors():
    image = np.uint8(np.random.rand(640, 480, 3) * 255)
    invert_colors = pr.InvertColors(probability=1.0)
    image_inverted = invert_colors(image)
    assert np.all(image_inverted == 255 - image)
    assert image_inverted.shape == image.shape
    assert image_inverted.dtype == image.dtype


def test_Posterize():
    image = np.uint8(np.random.rand(640, 480, 3) * 255)
    posterize = pr.Posterize(probability=1.0, num_bits=4)
    image_posterized = posterize(image)
    assert np.std(image_posterized) <= np.std(image)
    assert image_posterized.shape == image.shape
    assert image_posterized.dtype == image.dtype


@pytest.mark.parametrize(('threshold'), [75, 100, 125, 150, 175])
def test_Solarize(threshold):
    image = np.uint8(np.random.rand(640, 480, 3) * 255)
    solarize = pr.Solarize(probability=1.0, threshold=threshold)
    image_solarized = solarize(image)
    assert (np.all(image_solarized[image >= threshold] ==
                   255 - image[image >= threshold]))
    assert (np.all(image_solarized[image < threshold] ==
                   image[image < threshold]))
    assert image_solarized.shape == image.shape
    assert image_solarized.dtype == image.dtype


def test_SharpenImage():
    image = np.uint8(np.random.rand(640, 480, 3) * 255)
    sharpen_image = pr.SharpenImage(probability=1.0)
    image_sharpened = sharpen_image(image)
    assert np.std(image_sharpened) >= np.std(image)
    assert image_sharpened.shape == image.shape
    assert image_sharpened.dtype == image.dtype


def test_Cutout():
    image = np.uint8(np.random.rand(640, 480, 3) * 255)
    cutout = pr.Cutout(probability=1.0)
    image_cutout = cutout(image)
    assert image_cutout.shape == image.shape
    assert image_cutout.dtype == image.dtype


@pytest.mark.parametrize(('mean, scale'),
                         [(10, 1),
                          (20, 2),
                          (30, 3),
                          (40, 4),
                          (50, 5)])
def test_AddGaussianNoise(mean, scale):
    image = np.uint8(np.random.rand(640, 480, 3) * 255)
    add_gaussian_noise = pr.AddGaussianNoise(1.0, mean, scale)
    image_noisy = add_gaussian_noise(image)
    assert image_noisy.shape == image.shape
    assert image_noisy.dtype == image.dtype
