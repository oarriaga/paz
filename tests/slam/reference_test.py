import jax
import jax.numpy as jp
import numpy as np

import metrics
import paz
import reference
import synthetic


def build_fixture_A():
    return synthetic.build_two_view_scene(jax.random.PRNGKey(1), 120,
                                          0.0, 0.0)


def build_fixture_B():
    return synthetic.build_two_view_scene(jax.random.PRNGKey(2), 150,
                                          0.5, 0.15)


def compute_relative_pose(scene):
    relative = synthetic.compute_relative_transform(scene.pose_A,
                                                    scene.pose_B)
    relative = np.asarray(relative)
    return relative[:3, :3], relative[:3, 3]


def recover_relative_pose(scene, fundamental):
    K_A = np.asarray(scene.intrinsics_A)
    K_B = np.asarray(scene.intrinsics_B)
    essential = K_B.T @ fundamental @ K_A
    args = (essential, scene.points_A, scene.points_B, K_A, K_B)
    return reference.recover_pose_reference(*args)


def test_reference_versions():
    versions = reference.get_reference_versions()
    for version in versions:
        assert isinstance(version, str) and len(version) > 0


def test_eight_point_matches_true_fundamental():
    scene = build_fixture_A()
    fundamental = reference.estimate_fundamental_reference(scene.points_A,
                                                           scene.points_B)
    gt_args = (scene.intrinsics_A, scene.intrinsics_B, scene.pose_A,
               scene.pose_B)
    fundamental_true = synthetic.compute_fundamental(*gt_args)
    difference = metrics.matrix_scale_invariant_difference(
        fundamental, np.asarray(fundamental_true))
    assert difference < 1e-6


def test_eight_point_recovers_pose():
    scene = build_fixture_A()
    fundamental = reference.estimate_fundamental_reference(scene.points_A,
                                                           scene.points_B)
    rotation, translation, mask = recover_relative_pose(scene,
                                                        fundamental)
    rotation_true, translation_true = compute_relative_pose(scene)
    rotation_error = metrics.compute_rotation_error(rotation,
                                                    rotation_true)
    direction_args = (translation, translation_true)
    direction_error = metrics.compute_translation_direction_error(
        *direction_args)
    assert rotation_error < 0.05
    assert direction_error < 0.1


def test_fundamental_ransac_identifies_inliers():
    scene = build_fixture_B()
    args = (scene.points_A, scene.points_B, 3.0)
    fundamental, mask = reference.estimate_fundamental_ransac_reference(
        *args)
    precision, recall = metrics.compute_inlier_precision_recall(
        mask, scene.inlier_mask)
    assert precision > 0.9
    assert recall > 0.9


def test_ransac_pose_on_noisy_fixture():
    scene = build_fixture_B()
    args = (scene.points_A, scene.points_B, 3.0)
    _, mask = reference.estimate_fundamental_ransac_reference(*args)
    points_A = np.asarray(scene.points_A)[mask]
    points_B = np.asarray(scene.points_B)[mask]
    fundamental = reference.estimate_fundamental_reference(points_A,
                                                           points_B)
    K_A = np.asarray(scene.intrinsics_A)
    K_B = np.asarray(scene.intrinsics_B)
    essential = K_B.T @ fundamental @ K_A
    pose_args = (essential, points_A, points_B, K_A, K_B)
    rotation, translation, _ = reference.recover_pose_reference(
        *pose_args)
    rotation_true, translation_true = compute_relative_pose(scene)
    rotation_error = metrics.compute_rotation_error(rotation,
                                                    rotation_true)
    direction_error = metrics.compute_translation_direction_error(
        translation, translation_true)
    assert rotation_error < 0.5
    assert direction_error < 2.0


def test_triangulate_reference_reproduces_points():
    scene = build_fixture_A()
    args = (scene.intrinsics_A, scene.pose_A, scene.intrinsics_B,
            scene.pose_B, scene.points_A, scene.points_B)
    points3D = reference.triangulate_reference(*args)
    np.testing.assert_allclose(points3D, np.asarray(scene.points3D),
                               atol=1e-6)


def test_solve_pnp_reference():
    scene = synthetic.build_pnp_scene(jax.random.PRNGKey(3), 150, 0.5,
                                      0.15)
    args = (scene.intrinsics, scene.points3D, scene.points2D)
    pose, inlier_mask = reference.solve_pnp_reference(*args)
    pose_true = np.asarray(scene.pose)
    rotation_error = metrics.compute_rotation_error(pose[:3, :3],
                                                    pose_true[:3, :3])
    translation_error = np.linalg.norm(pose[:3, 3] - pose_true[:3, 3])
    assert rotation_error < 0.1
    assert translation_error < 0.01
    precision, recall = metrics.compute_inlier_precision_recall(
        inlier_mask, scene.inlier_mask)
    assert precision > 0.95
    assert recall > 0.95


def test_refine_pose_reference():
    scene = synthetic.build_pnp_scene(jax.random.PRNGKey(6), 120, 0.0,
                                      0.0)
    tangent = jp.array([0.03, -0.02, 0.04, 0.05, -0.04, 0.06])
    perturbation = paz.SE3.exp(paz.SE3.hat(tangent))
    initial_pose = np.asarray(perturbation @ scene.pose)
    args = (scene.intrinsics, scene.points3D, scene.points2D,
            initial_pose)
    refined_pose = reference.refine_pose_reference(*args)
    pose_true = np.asarray(scene.pose)
    rotation_error = metrics.compute_rotation_error(refined_pose[:3, :3],
                                                    pose_true[:3, :3])
    translation_error = np.linalg.norm(refined_pose[:3, 3]
                                       - pose_true[:3, 3])
    assert rotation_error < 1e-4
    assert translation_error < 1e-6


def test_bundle_adjust_reference():
    args = (jax.random.PRNGKey(7), 6, 200, 0.5, 0.03, 0.05)
    scene = synthetic.build_bundle_adjustment_scene(*args)
    ba_args = (scene.intrinsics, scene.noisy_poses, scene.noisy_points3D,
               scene.observations, scene.visibility)
    result = reference.bundle_adjust_reference(*ba_args)
    assert result.final_rmse < 0.75
    assert result.final_rmse < result.initial_rmse
    assert result.initial_rmse > 1.0


def test_stereo_sequence_pnp_trajectory():
    args = (jax.random.PRNGKey(4), 40, 500, 0.5, 0.1)
    sequence = synthetic.build_stereo_sequence(*args)
    points3D = np.asarray(sequence.points3D)
    estimated_poses = []
    for frame in range(40):
        visible = np.asarray(sequence.visibility_left[frame])
        observed = np.asarray(sequence.observations_left[frame])
        pnp_args = (sequence.intrinsics, points3D[visible],
                    observed[visible])
        pose, _ = reference.solve_pnp_reference(*pnp_args)
        estimated_poses.append(pose)
    estimated_poses = np.stack(estimated_poses)
    ate = metrics.compute_ATE(estimated_poses, sequence.poses)
    assert ate < 0.03
