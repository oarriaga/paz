import jax
import numpy as np

import metrics
import synthetic


def assert_scenes_equal(scene_a, scene_b):
    for field_a, field_b in zip(scene_a, scene_b):
        np.testing.assert_array_equal(np.asarray(field_a),
                                      np.asarray(field_b))


def compute_center(pose):
    pose = np.asarray(pose)
    return -pose[:3, :3].T @ pose[:3, 3]


def test_two_view_determinism():
    key = jax.random.PRNGKey(0)
    scene_a = synthetic.build_two_view_scene(key, 120, 0.5, 0.15)
    scene_b = synthetic.build_two_view_scene(key, 120, 0.5, 0.15)
    assert_scenes_equal(scene_a, scene_b)


def test_pnp_determinism():
    key = jax.random.PRNGKey(0)
    scene_a = synthetic.build_pnp_scene(key, 150, 0.5, 0.15)
    scene_b = synthetic.build_pnp_scene(key, 150, 0.5, 0.15)
    assert_scenes_equal(scene_a, scene_b)


def test_bundle_determinism():
    args = (jax.random.PRNGKey(0), 6, 200, 0.5, 0.03, 0.05)
    scene_a = synthetic.build_bundle_adjustment_scene(*args)
    scene_b = synthetic.build_bundle_adjustment_scene(*args)
    assert_scenes_equal(scene_a, scene_b)


def test_stereo_determinism():
    args = (jax.random.PRNGKey(0), 40, 500, 0.5, 0.1)
    sequence_a = synthetic.build_stereo_sequence(*args)
    sequence_b = synthetic.build_stereo_sequence(*args)
    assert_scenes_equal(sequence_a, sequence_b)


def test_degenerate_determinism():
    scenes_a = synthetic.build_degenerate_scenes(jax.random.PRNGKey(0))
    scenes_b = synthetic.build_degenerate_scenes(jax.random.PRNGKey(0))
    for scene_a, scene_b in zip(scenes_a, scenes_b):
        assert_scenes_equal(scene_a, scene_b)


def test_two_view_noise_free_reprojection():
    key = jax.random.PRNGKey(1)
    scene = synthetic.build_two_view_scene(key, 120, 0.0, 0.0)
    args_A = (scene.pose_A, scene.intrinsics_A, scene.points3D,
              scene.points_A)
    args_B = (scene.pose_B, scene.intrinsics_B, scene.points3D,
              scene.points_B)
    assert np.max(metrics.compute_reprojection_errors(*args_A)) < 1e-9
    assert np.max(metrics.compute_reprojection_errors(*args_B)) < 1e-9
    assert np.all(np.asarray(scene.inlier_mask))
    assert np.all(np.asarray(scene.valid_mask))
    depths_A = synthetic.compute_depths(scene.pose_A, scene.points3D)
    depths_B = synthetic.compute_depths(scene.pose_B, scene.points3D)
    assert np.all(np.asarray(depths_A) > 0.0)
    assert np.all(np.asarray(depths_B) > 0.0)


def test_two_view_meaningful_relative_motion():
    key = jax.random.PRNGKey(1)
    scene = synthetic.build_two_view_scene(key, 120, 0.0, 0.0)
    relative = synthetic.compute_relative_transform(scene.pose_A,
                                                    scene.pose_B)
    relative = np.asarray(relative)
    angle = metrics.compute_rotation_error(np.eye(3), relative[:3, :3])
    assert angle > 5.0
    assert np.linalg.norm(relative[:3, 3]) > 0.5


def test_two_view_outliers_and_noise():
    key = jax.random.PRNGKey(2)
    scene = synthetic.build_two_view_scene(key, 120, 0.5, 0.15)
    inliers = np.asarray(scene.inlier_mask)
    assert np.sum(~inliers) == 18
    args = (scene.pose_B, scene.intrinsics_B, scene.points3D,
            scene.points_B)
    errors = metrics.compute_reprojection_errors(*args)
    assert np.median(errors[inliers]) < 2.0
    assert np.max(errors[inliers]) < 5.0


def test_pnp_noise_free_reprojection():
    key = jax.random.PRNGKey(3)
    scene = synthetic.build_pnp_scene(key, 150, 0.0, 0.0)
    args = (scene.pose, scene.intrinsics, scene.points3D, scene.points2D)
    assert np.max(metrics.compute_reprojection_errors(*args)) < 1e-9
    depths = synthetic.compute_depths(scene.pose, scene.points3D)
    assert np.all(np.asarray(depths) > 0.0)


def test_bundle_scene_structure():
    args = (jax.random.PRNGKey(3), 6, 200, 0.5, 0.03, 0.05)
    scene = synthetic.build_bundle_adjustment_scene(*args)
    assert scene.poses.shape == (6, 4, 4)
    assert scene.noisy_poses.shape == (6, 4, 4)
    assert scene.observations.shape == (6, 200, 2)
    assert scene.visibility.shape == (6, 200)
    visibility = np.asarray(scene.visibility)
    assert 0.0 < visibility.mean() < 1.0
    assert not np.allclose(scene.noisy_poses, scene.poses)
    assert not np.allclose(scene.noisy_points3D, scene.points3D)


def test_bundle_scene_noise_free_reprojection():
    args = (jax.random.PRNGKey(3), 6, 200, 0.0, 0.03, 0.05)
    scene = synthetic.build_bundle_adjustment_scene(*args)
    for pose_index in range(6):
        visible = np.asarray(scene.visibility[pose_index])
        pose_args = (scene.poses[pose_index], scene.intrinsics,
                     scene.points3D, scene.observations[pose_index])
        errors = metrics.compute_reprojection_errors(*pose_args)
        assert np.max(errors[visible]) < 1e-9


def test_stereo_sequence_structure():
    args = (jax.random.PRNGKey(4), 40, 500, 0.5, 0.1)
    sequence = synthetic.build_stereo_sequence(*args)
    assert sequence.poses.shape == (40, 4, 4)
    assert sequence.observations_left.shape == (40, 500, 2)
    assert sequence.observations_right.shape == (40, 500, 2)
    assert sequence.visibility_left.shape == (40, 500)
    assert sequence.outlier_mask_left.shape == (40, 500)
    baseline = np.linalg.norm(np.asarray(sequence.left_to_right)[:3, 3])
    assert abs(baseline - 0.15) < 1e-12
    timestamps = np.asarray(sequence.timestamps)
    assert np.all(np.diff(timestamps) > 0.0)
    visible_counts = np.sum(np.asarray(sequence.visibility_left), axis=1)
    assert np.median(visible_counts) >= 100


def test_stereo_sequence_reentry():
    args = (jax.random.PRNGKey(4), 40, 500, 0.5, 0.1)
    sequence = synthetic.build_stereo_sequence(*args)
    visibility = np.asarray(sequence.visibility_left)
    transitions = np.sum(np.abs(np.diff(visibility.astype(int),
                                        axis=0)), axis=0)
    assert np.any(transitions >= 3)


def test_stereo_noise_free_reprojection():
    args = (jax.random.PRNGKey(4), 40, 500, 0.0, 0.0)
    sequence = synthetic.build_stereo_sequence(*args)
    left_to_right = np.asarray(sequence.left_to_right)
    for frame in (0, 20, 39):
        pose_left = np.asarray(sequence.poses[frame])
        visible_left = np.asarray(sequence.visibility_left[frame])
        left_args = (pose_left, sequence.intrinsics, sequence.points3D,
                     sequence.observations_left[frame])
        errors_left = metrics.compute_reprojection_errors(*left_args)
        assert np.max(errors_left[visible_left]) < 1e-9
        pose_right = left_to_right @ pose_left
        visible_right = np.asarray(sequence.visibility_right[frame])
        right_args = (pose_right, sequence.intrinsics, sequence.points3D,
                      sequence.observations_right[frame])
        errors_right = metrics.compute_reprojection_errors(*right_args)
        assert np.max(errors_right[visible_right]) < 1e-9


def test_degenerate_planar():
    scenes = synthetic.build_degenerate_scenes(jax.random.PRNGKey(5))
    points3D = np.asarray(scenes.planar.points3D)
    spread = points3D - points3D.mean(axis=0)
    smallest = np.linalg.svd(spread, compute_uv=False)[2]
    assert smallest < 1e-3


def test_degenerate_pure_rotation():
    scenes = synthetic.build_degenerate_scenes(jax.random.PRNGKey(5))
    center_A = compute_center(scenes.pure_rotation.pose_A)
    center_B = compute_center(scenes.pure_rotation.pose_B)
    assert np.linalg.norm(center_A - center_B) < 1e-12
    rotation_A = np.asarray(scenes.pure_rotation.pose_A)[:3, :3]
    rotation_B = np.asarray(scenes.pure_rotation.pose_B)[:3, :3]
    assert metrics.compute_rotation_error(rotation_A, rotation_B) > 5.0


def test_degenerate_low_parallax():
    scenes = synthetic.build_degenerate_scenes(jax.random.PRNGKey(5))
    center_A = compute_center(scenes.low_parallax.pose_A)
    center_B = compute_center(scenes.low_parallax.pose_B)
    baseline = np.linalg.norm(center_A - center_B)
    assert 0.0 < baseline < 1e-3


def test_degenerate_behind_camera():
    scenes = synthetic.build_degenerate_scenes(jax.random.PRNGKey(5))
    scene = scenes.behind_camera
    depths_B = synthetic.compute_depths(scene.pose_B, scene.points3D)
    assert np.any(np.asarray(depths_B) < 0.0)
    assert not np.all(np.asarray(scene.inlier_mask))


def test_degenerate_collinear():
    scenes = synthetic.build_degenerate_scenes(jax.random.PRNGKey(5))
    points3D = np.asarray(scenes.collinear.points3D)
    spread = points3D - points3D.mean(axis=0)
    singular_values = np.linalg.svd(spread, compute_uv=False)
    assert singular_values[1] < 1e-9


def test_degenerate_insufficient():
    scenes = synthetic.build_degenerate_scenes(jax.random.PRNGKey(5))
    assert np.sum(np.asarray(scenes.insufficient.valid_mask)) < 8
