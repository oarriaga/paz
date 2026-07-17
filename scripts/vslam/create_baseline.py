import json
import os
import subprocess

import numpy as np

import fixtures
import metrics
import reference
import synthetic

TOLERANCES = {
    "clean_sampson_median": 1e-5,
    "clean_rotation_error": 0.05,
    "clean_direction_error": 0.1,
    "clean_triangulation_rmse": 1e-4,
    "clean_pnp_rotation_error": 0.05,
    "clean_pnp_translation_error": 1e-4,
    "relative_factor": 1.1,
    "rotation_slack": 0.05,
    "direction_slack": 0.1,
    "translation_slack": 1e-3,
    "precision_recall_slack": 0.03,
    "refine_rmse_factor": 1.05,
    "bundle_rmse_factor": 1.1,
    "bundle_rmse_slack": 0.05,
    "bundle_final_rmse": 0.75,
    "stereo_ate_rmse": 0.03,
    "stereo_median_reprojection": 1.0,
}


def compute_two_view_reference(scene):
    points_A = np.asarray(scene.points_A)
    points_B = np.asarray(scene.points_B)
    args = (points_A, points_B, 3.0)
    _, inliers = reference.estimate_fundamental_ransac_reference(*args)
    precision, recall = metrics.compute_inlier_precision_recall(
        inliers, scene.inlier_mask)
    refit = reference.estimate_fundamental_reference(
        points_A[inliers], points_B[inliers])
    intrinsics_A = np.asarray(scene.intrinsics_A)
    intrinsics_B = np.asarray(scene.intrinsics_B)
    essential = intrinsics_B.T @ refit @ intrinsics_A
    pose_args = (essential, points_A[inliers], points_B[inliers],
                 intrinsics_A, intrinsics_B)
    rotation, translation, _ = reference.recover_pose_reference(*pose_args)
    relative = np.asarray(
        synthetic.compute_relative_transform(scene.pose_A, scene.pose_B))
    return {
        "precision": precision,
        "recall": recall,
        "rotation_error": float(metrics.compute_rotation_error(
            rotation, relative[:3, :3])),
        "direction_error": float(metrics.compute_translation_direction_error(
            translation, relative[:3, 3])),
    }


def compute_pnp_reference(scene):
    intrinsics = np.asarray(scene.intrinsics)
    points3D = np.asarray(scene.points3D)
    points2D = np.asarray(scene.points2D)
    pose, inliers = reference.solve_pnp_reference(
        intrinsics, points3D, points2D)
    precision, recall = metrics.compute_inlier_precision_recall(
        inliers, scene.inlier_mask)
    refined = reference.refine_pose_reference(
        intrinsics, points3D[inliers], points2D[inliers], pose)
    true_inliers = np.asarray(scene.inlier_mask).astype(bool)
    errors = metrics.compute_reprojection_errors(
        refined, intrinsics, points3D[true_inliers],
        points2D[true_inliers])
    pose_true = np.asarray(scene.pose)
    return {
        "precision": precision,
        "recall": recall,
        "rotation_error": float(metrics.compute_rotation_error(
            pose[:3, :3], pose_true[:3, :3])),
        "translation_error": float(np.linalg.norm(
            pose[:3, 3] - pose_true[:3, 3])),
        "refined_rmse": float(np.sqrt(np.mean(errors**2))),
    }


def compute_bundle_reference(scene):
    initial_poses = np.asarray(scene.noisy_poses).copy()
    initial_poses[0] = np.asarray(scene.poses)[0]
    result = reference.bundle_adjust_reference(
        scene.intrinsics, initial_poses, scene.noisy_points3D,
        scene.observations, scene.visibility)
    rotation_errors, translation_errors = compute_pose_errors(
        result.poses, scene.poses)
    return {
        "initial_rmse": result.initial_rmse,
        "final_rmse": result.final_rmse,
        "max_rotation_error": float(np.max(rotation_errors)),
        "max_translation_error": float(np.max(translation_errors)),
    }


def compute_pose_errors(poses_estimated, poses_true):
    rotation_errors, translation_errors = [], []
    for estimated, true in zip(np.asarray(poses_estimated),
                               np.asarray(poses_true)):
        rotation_errors.append(metrics.compute_rotation_error(
            estimated[:3, :3], true[:3, :3]))
        translation_errors.append(np.linalg.norm(
            estimated[:3, 3] - true[:3, 3]))
    return np.array(rotation_errors), np.array(translation_errors)


def compute_stereo_reference(sequence):
    poses = estimate_stereo_trajectory(sequence)
    poses_true = np.asarray(sequence.poses)
    ate = metrics.compute_ATE(poses, poses_true)
    rpe_translation, rpe_rotation = metrics.compute_RPE(poses, poses_true)
    return {
        "ate_rmse": ate,
        "rpe_translation": rpe_translation,
        "rpe_rotation": rpe_rotation,
    }


def estimate_stereo_trajectory(sequence):
    intrinsics = np.asarray(sequence.intrinsics)
    points3D = np.asarray(sequence.points3D)
    observations = np.asarray(sequence.observations_left)
    visibility = np.asarray(sequence.visibility_left)
    poses = []
    for frame in range(len(observations)):
        visible = visibility[frame]
        pose, _ = reference.solve_pnp_reference(
            intrinsics, points3D[visible], observations[frame][visible])
        poses.append(pose)
    return np.stack(poses)


def get_git_commit():
    command = ["git", "rev-parse", "HEAD"]
    output = subprocess.run(command, capture_output=True, text=True,
                            cwd=fixtures.REPO_ROOT)
    return output.stdout.strip()


def describe_dimensions(scenes):
    dimensions = {}
    for name, fixture in zip(type(scenes)._fields, scenes):
        fields = type(fixture)._fields
        shapes = [list(np.asarray(array).shape) for array in fixture]
        dimensions[name] = dict(zip(fields, shapes))
    return dimensions


if __name__ == "__main__":
    scenes = fixtures.build_fixtures()
    versions = reference.get_reference_versions()
    baseline = {
        "seeds": fixtures.SEEDS,
        "versions": versions._asdict(),
        "git_commit": get_git_commit(),
        "checksums": fixtures.compute_checksums(scenes),
        "dimensions": describe_dimensions(scenes),
        "tolerances": TOLERANCES,
        "reference": {
            "two_view_noisy": compute_two_view_reference(
                scenes.two_view_noisy),
            "pnp": compute_pnp_reference(scenes.pnp),
            "bundle": compute_bundle_reference(scenes.bundle),
            "stereo": compute_stereo_reference(scenes.stereo),
        },
    }
    os.makedirs(fixtures.DATA_DIR, exist_ok=True)
    np.savez_compressed(fixtures.BASELINE_NPZ, **fixtures.to_arrays(scenes))
    with open(fixtures.BASELINE_JSON, "w") as opened_file:
        json.dump(baseline, opened_file, indent=2, sort_keys=True)
    print(json.dumps(baseline["reference"], indent=2, sort_keys=True))
    print("wrote", fixtures.BASELINE_NPZ)
    print("wrote", fixtures.BASELINE_JSON)
