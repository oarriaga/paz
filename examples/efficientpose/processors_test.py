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
                        Augment6DOF, AutoContrast, EqualizeHistogram,
                        InvertColors, Posterize, Solarize, SharpenImage,
                        Cutout, AddGaussianNoise)


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


# @pytest.mark.parametrize(
#         ('model'), [EfficientPosePhi0, EfficientPosePhi1, EfficientPosePhi2,
#                     EfficientPosePhi3, EfficientPosePhi4, EfficientPosePhi5,
#                     EfficientPosePhi6, EfficientPosePhi7])
# def test_RegressTranslation(model):
#     model = model(num_classes=2, base_weights='COCO', head_weights=None)
#     regress_translation = RegressTranslation(model.translation_priors)
#     translation_raw = np.zeros_like(model.translation_priors)
#     translation_raw = np.expand_dims(translation_raw, axis=0)
#     translation = regress_translation(translation_raw)
#     assert translation[:, 0].sum() == model.translation_priors[:, 0].sum()
#     assert translation[:, 1].sum() == model.translation_priors[:, 1].sum()
#     assert translation[:, 2].sum() == translation_raw[:, :, 2].sum()
#     del model


# @pytest.mark.parametrize(('image_scale'), [0.6, 0.7, 0.8, 0.9, 1.0])
# def test_ComputeCameraParameter(image_scale):
#     translation_scale_norm = 1000.0
#     compute_camera_parameter = ComputeCameraParameter(camera_matrix,
#                                                       translation_scale_norm)
#     camera_parameter = compute_camera_parameter(image_scale)
#     assert camera_parameter.shape == (6, )
#     assert np.all(camera_parameter == np.array([camera_matrix[0, 0],
#                                                 camera_matrix[1, 1],
#                                                 camera_matrix[0, 2],
#                                                 camera_matrix[1, 2],
#                                                 translation_scale_norm,
#                                                 image_scale]))


# @pytest.mark.parametrize(
#         ('model'), [EfficientPosePhi0, EfficientPosePhi1, EfficientPosePhi2,
#                     EfficientPosePhi3, EfficientPosePhi4, EfficientPosePhi5,
#                     EfficientPosePhi6, EfficientPosePhi7])
# def test_ComputeTxTyTz(model):
#     model = model(num_classes=2, base_weights='COCO', head_weights=None)
#     translation_raw = np.zeros_like(model.translation_priors)
#     compute_camera_parameter = np.ones(shape=(6, ))
#     compute_tx_ty_tz = ComputeTxTyTz()
#     tx_ty_tz = compute_tx_ty_tz(translation_raw, compute_camera_parameter)
#     assert tx_ty_tz.shape == model.translation_priors.shape
#     del model


@pytest.mark.parametrize(('num_rotations'), [1, 2, 3, 4, 5])
def test_TransformRotation(num_rotations):
    rotations = np.random.rand(num_rotations, 9)
    transform_rotation = TransformRotation(num_pose_dims=3)
    rotations_transformed = transform_rotation(rotations)
    assert rotations_transformed.shape == (num_rotations, 5)


@pytest.mark.parametrize(('rotation_size, translation_size'),
                         [(10, 50),
                          (20, 40),
                          (30, 30),
                          (40, 20),
                          (50, 10)])
def test_ConcatenatePoses(rotation_size, translation_size):
    num_rows = 10
    rotations = np.random.rand(num_rows, rotation_size)
    translations = np.random.rand(num_rows, translation_size)
    concatenate_poses = ConcatenatePoses()
    poses = concatenate_poses(rotations, translations)
    assert np.all(poses[:, :rotation_size] == rotations)
    assert np.all(poses[:, rotation_size:] == translations)
    assert (poses.shape == (num_rows, rotation_size + translation_size))


@pytest.mark.parametrize(('pose_size, scale'),
                         [(10, 0.1),
                          (20, 0.2),
                          (30, 0.3),
                          (40, 0.4),
                          (50, 0.5)])
def test_ConcatenateScale(pose_size, scale):
    num_rows = 10
    poses = np.random.rand(num_rows, pose_size)
    concatenate_scale = ConcatenateScale()
    poses_concatenated = concatenate_scale(poses, scale)
    assert np.all(poses_concatenated[:, :-1] == poses)
    assert np.all(poses_concatenated[:, -1] == scale)
    assert (poses_concatenated.shape == (num_rows, pose_size + 1))


@pytest.mark.parametrize(('num_annotations, image_W, image_H'),
                         [(10, 100, 500),
                          (20, 200, 400),
                          (30, 300, 300),
                          (40, 400, 200),
                          (50, 500, 100)])
def test_Augment6DOF(num_annotations, image_W, image_H):
    augment_6DOF = Augment6DOF()
    image = np.zeros((image_W, image_H, 3))
    mask = np.ones((image_W, image_H, 1))
    boxes = np.random.rand(num_annotations, 5)
    rotation = np.random.rand(num_annotations, 9)
    translation_raw = np.random.rand(num_annotations, 3)
    augmentations = augment_6DOF(image, boxes, rotation, translation_raw, mask)
    (augmented_image, augmented_boxes, augmented_rotation,
     augmented_translation, augmented_mask) = augmentations
    assert augmented_image.shape == (image_W, image_H, 3)
    assert augmented_mask.shape == (image_W, image_H, 1)
    assert augmented_boxes.shape == (num_annotations, 5)
    assert augmented_rotation.shape == (num_annotations, 9)
    assert augmented_translation.shape == (num_annotations, 3)
    assert augmented_image.mean() == 0.0
    assert augmented_mask.mean() == 1.0
