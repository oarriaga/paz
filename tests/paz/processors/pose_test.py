import pytest
import numpy as np
import paz.processors as pr
from paz.datasets.linemod import LINEMOD_CAMERA_MATRIX as camera_matrix


@pytest.fixture()
def get_camera_matrix():
    return camera_matrix


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
    concatenate_poses = pr.ConcatenatePoses()
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
    concatenate_scale = pr.ConcatenateScale()
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
def test_AugmentPose6D(num_annotations, image_W, image_H, get_camera_matrix):
    augment_pose_6D = pr.AugmentPose6D()
    image = np.zeros((image_W, image_H, 3))
    mask = np.ones((image_W, image_H, 1))
    boxes = np.random.rand(num_annotations, 5)
    rotation = np.random.rand(num_annotations, 9)
    translation_raw = np.random.rand(num_annotations, 3)
    augmentations = augment_pose_6D(get_camera_matrix, image, boxes, rotation,
                                    translation_raw, mask)
    (augmented_image, augmented_boxes, augmented_rotation,
     augmented_translation, augmented_mask) = augmentations
    assert augmented_image.dtype == image.dtype
    assert augmented_image.shape == (image_W, image_H, 3)
    assert augmented_mask.dtype == mask.dtype
    assert augmented_mask.shape == (image_W, image_H, 1)
    assert augmented_boxes.shape == (num_annotations, 5)
    assert augmented_rotation.shape == (num_annotations, 9)
    assert augmented_translation.shape == (num_annotations, 3)
    assert augmented_image.mean() == 0.0
    assert augmented_mask.mean() == 1.0
