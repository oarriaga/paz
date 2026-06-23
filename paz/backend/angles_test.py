import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

from paz.backend import angles
from paz.datasets.hands import MPIIHandJoints


def z_rotation_quaternion(theta):
    return np.array([0.0, 0.0, np.sin(theta / 2), np.cos(theta / 2)])


def test_quaternion_to_rotation_matrix_z_rotation():
    theta = 0.7
    matrix = angles.quaternion_to_rotation_matrix(z_rotation_quaternion(theta))
    expected = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(matrix, expected, atol=1e-6)


def test_compact_axis_angle_recovers_z_rotation():
    theta = 0.7
    matrix = angles.quaternion_to_rotation_matrix(z_rotation_quaternion(theta))
    axis_angle = angles.rotation_matrix_to_compact_axis_angle(matrix)
    assert np.allclose(axis_angle, [0.0, 0.0, theta], atol=1e-6)


def test_to_affine_matrix_structure():
    rotation = np.eye(3)
    affine = angles.to_affine_matrix(rotation, np.array([1.0, 2.0, 3.0]))
    assert affine.shape == (4, 4)
    assert np.allclose(affine[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(affine[3], [0.0, 0.0, 0.0, 1.0])


def test_compute_orientation_vector_root_is_zero():
    keypoints = MPIIHandJoints.links_origin
    delta = angles.compute_orientation_vector(keypoints, MPIIHandJoints.parents)
    assert delta.shape == (21, 3)
    assert np.allclose(delta[0], 0.0)
    assert np.allclose(delta[2], keypoints[2] - keypoints[1])


def test_change_link_order_roundtrip():
    joints = np.arange(len(MPIIHandJoints.labels) * 3).reshape(-1, 3)
    mano = angles.change_link_order(
        joints, MPIIHandJoints.labels, angles.MANOHandJoints.labels)
    back = angles.change_link_order(
        mano, angles.MANOHandJoints.labels, MPIIHandJoints.labels)
    assert np.allclose(back, joints)


def random_unit_quaternions(seed=0):
    random_state = np.random.default_rng(seed)
    quaternions = random_state.normal(size=(21, 4))
    return quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)


def test_compute_relative_angles_shape_and_finite():
    relative = angles.compute_relative_angles(random_unit_quaternions())
    assert relative.shape == (21, 3)
    assert np.isfinite(relative).all()


def test_compute_relative_angles_finger_base_joints_are_zero():
    relative = angles.compute_relative_angles(random_unit_quaternions())
    for finger_base_joint in [1, 5, 9, 13, 17]:
        assert np.allclose(relative[finger_base_joint], 0.0)
