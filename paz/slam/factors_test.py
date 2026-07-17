import os
import sys

import jax
import jax.numpy as jp
import numpy as np

from paz.backend.lie import SE3
from paz.slam import factors

TESTS_SLAM = os.path.join(os.path.dirname(__file__), "..", "..", "tests",
                          "slam")
sys.path.insert(0, os.path.abspath(TESTS_SLAM))

import synthetic


def build_problem(scene, poses, landmarks, capacity):
    visibility = np.asarray(scene.visibility)
    pose_index, landmark_index = np.nonzero(visibility)
    num_observations = len(pose_index)
    observed = np.asarray(scene.observations)[pose_index, landmark_index]
    uv = np.zeros((capacity, 2))
    uv[:num_observations] = observed
    pose_slots = np.zeros(capacity, dtype=np.int32)
    pose_slots[:num_observations] = pose_index
    landmark_slots = np.zeros(capacity, dtype=np.int32)
    landmark_slots[:num_observations] = landmark_index
    active = np.zeros(capacity, dtype=bool)
    active[:num_observations] = True
    fields = (jp.asarray(poses), jp.ones(len(poses), dtype=bool),
              jp.asarray(landmarks), jp.ones(len(landmarks), dtype=bool),
              jp.asarray(scene.intrinsics)[None], jp.eye(4)[None],
              jp.asarray(uv), jp.asarray(pose_slots),
              jp.asarray(landmark_slots), jp.zeros(capacity, jp.int32),
              jp.ones(capacity), jp.asarray(active))
    return factors.BundleProblem(*fields)


def tight_capacity(scene):
    return int(np.sum(np.asarray(scene.visibility)))


def residual_at(problem, observation):
    residuals = factors.compute_observation_residuals(problem)
    return np.asarray(residuals[observation])


def replace_pose(problem, pose_index, pose):
    return problem._replace(poses=problem.poses.at[pose_index].set(pose))


def replace_landmark(problem, landmark_index, landmark):
    landmarks = problem.landmarks.at[landmark_index].set(landmark)
    return problem._replace(landmarks=landmarks)


def pose_block_differences(problem, observation, epsilon):
    pose_index = int(problem.observation_pose[observation])
    pose = problem.poses[pose_index]
    columns = []
    for axis in range(6):
        delta = jp.zeros(6).at[axis].set(epsilon)
        plus = replace_pose(problem, pose_index, SE3.retract(pose, delta))
        minus = replace_pose(problem, pose_index, SE3.retract(pose, -delta))
        residual_plus = residual_at(plus, observation)
        residual_minus = residual_at(minus, observation)
        columns.append((residual_plus - residual_minus) / (2.0 * epsilon))
    return np.stack(columns, axis=1)


def landmark_block_differences(problem, observation, epsilon):
    landmark_index = int(problem.observation_landmark[observation])
    landmark = problem.landmarks[landmark_index]
    columns = []
    for axis in range(3):
        delta = jp.zeros(3).at[axis].set(epsilon)
        plus = replace_landmark(problem, landmark_index, landmark + delta)
        minus = replace_landmark(problem, landmark_index, landmark - delta)
        residual_plus = residual_at(plus, observation)
        residual_minus = residual_at(minus, observation)
        columns.append((residual_plus - residual_minus) / (2.0 * epsilon))
    return np.stack(columns, axis=1)


@synthetic.in_double_precision
def test_residuals_zero_at_ground_truth():
    args = (jax.random.PRNGKey(0), 3, 20, 0.0, 0.0, 0.0)
    scene = synthetic.build_bundle_adjustment_scene(*args)
    capacity = tight_capacity(scene)
    problem = build_problem(scene, scene.poses, scene.points3D, capacity)
    residuals = factors.compute_observation_residuals(problem)
    np.testing.assert_allclose(np.asarray(residuals), 0.0, atol=1e-6)


@synthetic.in_double_precision
def test_jacobians_match_finite_differences():
    args = (jax.random.PRNGKey(1), 3, 20, 0.5, 0.03, 0.05)
    scene = synthetic.build_bundle_adjustment_scene(*args)
    capacity = tight_capacity(scene)
    problem = build_problem(scene, scene.noisy_poses,
                            scene.noisy_points3D, capacity)
    jacobians = factors.compute_observation_jacobians(problem)
    pose_blocks, landmark_blocks = jacobians
    for observation in range(0, capacity, 7):
        fd_pose = pose_block_differences(problem, observation, 1e-3)
        fd_args = (problem, observation, 1e-3)
        fd_landmark = landmark_block_differences(*fd_args)
        np.testing.assert_allclose(np.asarray(pose_blocks[observation]),
                                   fd_pose, rtol=1e-3, atol=1e-3)
        landmark_block = np.asarray(landmark_blocks[observation])
        np.testing.assert_allclose(landmark_block, fd_landmark,
                                   rtol=1e-3, atol=1e-3)


def test_inactive_observations_are_exactly_zero():
    args = (jax.random.PRNGKey(2), 3, 20, 0.5, 0.03, 0.05)
    scene = synthetic.build_bundle_adjustment_scene(*args)
    capacity = tight_capacity(scene) + 10
    problem = build_problem(scene, scene.noisy_poses,
                            scene.noisy_points3D, capacity)
    deactivated = problem.observation_active.at[3].set(False)
    problem = problem._replace(observation_active=deactivated)
    residuals = np.asarray(factors.compute_observation_residuals(problem))
    jacobians = factors.compute_observation_jacobians(problem)
    pose_blocks, landmark_blocks = jacobians
    inactive = ~np.asarray(problem.observation_active)
    assert np.any(inactive)
    assert np.all(residuals[inactive] == 0.0)
    assert np.all(np.asarray(pose_blocks)[inactive] == 0.0)
    assert np.all(np.asarray(landmark_blocks)[inactive] == 0.0)
