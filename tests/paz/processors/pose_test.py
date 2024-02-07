import pytest
import numpy as np
import paz.processors as pr


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
