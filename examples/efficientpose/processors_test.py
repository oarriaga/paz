import pytest
import numpy as np
from pose import LINEMOD_CAMERA_MATRIX as camera_matrix
from efficientpose import (EfficientPosePhi0, EfficientPosePhi1,
                           EfficientPosePhi2, EfficientPosePhi3, 
                           EfficientPosePhi4, EfficientPosePhi5, 
                           EfficientPosePhi6, EfficientPosePhi7)
from processors import (ComputeResizingShape, PadImage, RegressTranslation,
                        ComputeCameraParameter, ComputeTxTyTz,
                        TransformRotation, ConcatenatePoses, ConcatenateScale,
                        ScaleBoxes2D, Augment6DOF, AutoContrast,
                        EqualizeHistogram, InvertColors, Posterize, Solarize,
                        SharpenImage, Cutout, AddGaussianNoise)


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
    compute_shape = ComputeResizingShape(size)    
    resizing_shape, scale = compute_shape(image)
    assert resizing_shape == target_resizing_shape
    assert scale == target_scale


@pytest.mark.parametrize(('size'), [32, 64, 128, 256, 512])
def test_PadImage(size):
    image = get_test_images(16, 12, batch_size=1)[0]
    pad_image = PadImage(size, mode='constant')
    padded_image = pad_image(image)
    assert np.sum(padded_image) == 0.0
    assert padded_image.shape == (size, size, 3)


@pytest.mark.parametrize(
        ('model'), [EfficientPosePhi0, EfficientPosePhi1, EfficientPosePhi2,
                    EfficientPosePhi3, EfficientPosePhi4, EfficientPosePhi5,
                    EfficientPosePhi6, EfficientPosePhi7])
def test_RegressTranslation(model):
    model = model(num_classes=2, base_weights='COCO', head_weights=None)
    regress_translation = RegressTranslation(model.translation_priors)
    translation_raw = np.zeros_like(model.translation_priors)
    translation_raw = np.expand_dims(translation_raw, axis=0)
    translation = regress_translation(translation_raw)
    assert translation[:, 0].sum() == model.translation_priors[:, 0].sum()
    assert translation[:, 1].sum() == model.translation_priors[:, 1].sum()
    assert translation[:, 2].sum() == translation_raw[:, :, 2].sum()
    del model


@pytest.mark.parametrize(('image_scale'), [0.6, 0.7, 0.8, 0.9, 1.0])
def test_ComputeCameraParameter(image_scale):
    translation_scale_norm = 1000.0
    compute_camera_parameter = ComputeCameraParameter(camera_matrix,
                                                      translation_scale_norm)
    camera_parameter = compute_camera_parameter(image_scale)
    assert camera_parameter.shape == (6, )
    assert np.all(camera_parameter == np.array([camera_matrix[0, 0],
                                                camera_matrix[1, 1],
                                                camera_matrix[0, 2],
                                                camera_matrix[1, 2],
                                                translation_scale_norm,
                                                image_scale]))
