import os
import sys

import jax
import jax.numpy as jp
import numpy as np

from paz.slam import bundle_adjustment
from paz.slam import factors

TESTS_SLAM = os.path.join(os.path.dirname(__file__), "..", "..", "tests",
                          "slam")
sys.path.insert(0, os.path.abspath(TESTS_SLAM))

import metrics
import reference
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


def clamp_first_pose(scene):
    poses = np.asarray(scene.noisy_poses).copy()
    poses[0] = np.asarray(scene.poses[0])
    return jp.asarray(poses)


def build_fixture_D():
    args = (jax.random.PRNGKey(7), 6, 200, 0.5, 0.03, 0.05)
    return synthetic.build_bundle_adjustment_scene(*args)


def build_small_scene(key):
    return synthetic.build_bundle_adjustment_scene(key, 3, 30, 0.5,
                                                   0.03, 0.05)


def solve_dense_reference(problem, huber_scale, damping):
    residuals = factors.compute_observation_residuals(problem)
    jacobians = factors.compute_observation_jacobians(problem)
    args = (problem, residuals, huber_scale)
    weights = np.asarray(bundle_adjustment.compute_irls_weights(*args))
    residuals = np.asarray(residuals)
    pose_blocks = np.asarray(jacobians[0])
    landmark_blocks = np.asarray(jacobians[1])
    num_poses, num_landmarks = len(problem.poses), len(problem.landmarks)
    size = 6 * num_poses + 3 * num_landmarks
    hessian = np.zeros((size, size))
    gradient = np.zeros(size)
    pose_index = np.asarray(problem.observation_pose)
    landmark_index = np.asarray(problem.observation_landmark)
    for observation in range(len(residuals)):
        rows = np.zeros((2, size))
        start = 6 * pose_index[observation]
        rows[:, start:start + 6] = pose_blocks[observation]
        start = 6 * num_poses + 3 * landmark_index[observation]
        rows[:, start:start + 3] = landmark_blocks[observation]
        hessian += weights[observation] * rows.T @ rows
        gradient += weights[observation] * rows.T @ residuals[observation]
    hessian += damping * np.diag(np.diag(hessian)) + 1e-12 * np.eye(size)
    hessian[:6, :] = 0.0
    hessian[:, :6] = 0.0
    hessian[:6, :6] = np.eye(6)
    gradient[:6] = 0.0
    return np.linalg.solve(hessian, -gradient)


def pose_errors(poses, true_poses):
    rotation_errors, translation_errors = [], []
    for pose, true_pose in zip(np.asarray(poses), np.asarray(true_poses)):
        error = metrics.compute_rotation_error(pose[:3, :3],
                                               true_pose[:3, :3])
        rotation_errors.append(error)
        offset = pose[:3, 3] - true_pose[:3, 3]
        translation_errors.append(np.linalg.norm(offset))
    return max(rotation_errors), max(translation_errors)


@synthetic.in_double_precision
def test_schur_step_matches_dense_solve():
    scene = build_small_scene(jax.random.PRNGKey(1))
    initial_poses = clamp_first_pose(scene)
    args = (scene, initial_poses, scene.noisy_points3D)
    problem = build_problem(*args, tight_capacity(scene))
    huber_scale, damping = 3.0, 1e-3
    step_args = (problem, huber_scale, damping)
    delta_poses, delta_landmarks = bundle_adjustment.compute_schur_step(
        *step_args)
    full_delta = solve_dense_reference(problem, huber_scale, damping)
    num_poses = len(problem.poses)
    dense_poses = full_delta[:6 * num_poses].reshape(num_poses, 6)
    dense_landmarks = full_delta[6 * num_poses:].reshape(-1, 3)
    np.testing.assert_allclose(np.asarray(delta_poses), dense_poses,
                               rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(np.asarray(delta_landmarks),
                               dense_landmarks, rtol=1e-5, atol=1e-9)


def test_bundle_adjust_matches_scipy_reference():
    scene = build_fixture_D()
    initial_poses = clamp_first_pose(scene)
    args = (scene, initial_poses, scene.noisy_points3D)
    problem = build_problem(*args, tight_capacity(scene))
    adjust = jax.jit(bundle_adjustment.bundle_adjust, static_argnums=1)
    result = adjust(problem, 30, 3.0, 1e-3)
    assert bool(result.valid)
    assert float(result.final_cost) < float(result.initial_cost)
    assert float(result.final_rmse) < 0.75
    assert int(result.num_accepted) > 0
    trace = np.asarray(result.cost_trace)
    assert np.all(np.isfinite(trace))
    assert np.all(np.diff(trace) <= 1e-9)
    assert np.all(np.isfinite(np.asarray(result.damping_trace)))
    reference_args = (scene.intrinsics, initial_poses,
                      scene.noisy_points3D, scene.observations,
                      scene.visibility)
    oracle = reference.bundle_adjust_reference(*reference_args)
    assert float(result.final_rmse) <= 1.1 * oracle.final_rmse + 0.05
    paz_rotation, paz_translation = pose_errors(result.poses, scene.poses)
    oracle_rotation, oracle_translation = pose_errors(oracle.poses,
                                                      scene.poses)
    assert paz_rotation <= 1.1 * oracle_rotation + 0.05
    assert paz_translation <= 1.1 * oracle_translation + 0.01


def test_padding_matches_tight_capacity():
    scene = build_small_scene(jax.random.PRNGKey(2))
    initial_poses = clamp_first_pose(scene)
    capacity = tight_capacity(scene)
    args = (scene, initial_poses, scene.noisy_points3D)
    tight = build_problem(*args, capacity)
    padded = build_problem(*args, capacity + 33)
    result_tight = bundle_adjustment.bundle_adjust(tight, 10, 3.0, 1e-3)
    result_padded = bundle_adjustment.bundle_adjust(padded, 10, 3.0, 1e-3)
    np.testing.assert_allclose(np.asarray(result_padded.poses),
                               np.asarray(result_tight.poses),
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(result_padded.landmarks),
                               np.asarray(result_tight.landmarks),
                               rtol=1e-9, atol=1e-12)


def test_jit_compiles_once():
    scene_a = build_small_scene(jax.random.PRNGKey(3))
    scene_b = build_small_scene(jax.random.PRNGKey(4))
    capacity = max(tight_capacity(scene_a), tight_capacity(scene_b))
    args_a = (scene_a, clamp_first_pose(scene_a), scene_a.noisy_points3D)
    args_b = (scene_b, clamp_first_pose(scene_b), scene_b.noisy_points3D)
    problem_a = build_problem(*args_a, capacity)
    problem_b = build_problem(*args_b, capacity)

    def adjust(problem):
        return bundle_adjustment.bundle_adjust(problem, 5, 3.0, 1e-3)

    jitted = jax.jit(adjust)
    result_a = jitted(problem_a)
    result_b = jitted(problem_b)
    assert bool(result_a.valid) and bool(result_b.valid)
    assert jitted._cache_size() == 1
