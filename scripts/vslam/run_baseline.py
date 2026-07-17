import argparse
import json
import os
import time
from collections import namedtuple

import numpy as np

import fixtures
import metrics
import synthetic

import jax
import jax.numpy as jp

import paz

Check = namedtuple("Check", ["section", "name", "value", "bound", "passed"])


def check_below(section, name, value, bound):
    passed = bool(np.isfinite(value)) and float(value) < float(bound)
    return Check(section, name, float(value), float(bound), passed)


def check_at_least(section, name, value, bound):
    passed = bool(np.isfinite(value)) and float(value) >= float(bound)
    return Check(section, name, float(value), float(bound), passed)


def check_true(section, name, flag):
    return Check(section, name, float(bool(flag)), 1.0, bool(flag))


def report_value(section, name, value):
    return Check(section, name, float(value), float("inf"), True)


def verify_checksums(scenes, baseline):
    checksums = fixtures.compute_checksums(scenes)
    checks = []
    for name, digest in checksums.items():
        matches = digest == baseline["checksums"][name]
        checks.append(check_true("fixtures", f"{name}_checksum", matches))
    return checks


def evaluate_clean_two_view(scene, bounds):
    F = paz.epipolar.compute_fundamental_matrix(
        scene.points_A, scene.points_B)
    sampson = paz.epipolar.compute_sampson_distance(
        F, scene.points_A, scene.points_B)
    E = paz.epipolar.compute_essential_matrix(
        F, scene.intrinsics_A, scene.intrinsics_B)
    pose_args = (E, scene.intrinsics_A, scene.intrinsics_B,
                 scene.points_A, scene.points_B, scene.valid_mask)
    pose = paz.epipolar.recover_relative_pose(*pose_args)
    relative = np.asarray(
        synthetic.compute_relative_transform(scene.pose_A, scene.pose_B))
    rotation_error = metrics.compute_rotation_error(
        pose.rotation, relative[:3, :3])
    direction_error = metrics.compute_translation_direction_error(
        np.asarray(pose.translation), relative[:3, 3])
    section = "clean_two_view"
    checks = [
        check_below(section, "sampson_median",
                    float(np.median(np.asarray(sampson))),
                    bounds["clean_sampson_median"]),
        check_true(section, "pose_valid", pose.valid),
        check_below(section, "rotation_error", rotation_error,
                    bounds["clean_rotation_error"]),
        check_below(section, "direction_error", direction_error,
                    bounds["clean_direction_error"]),
        check_below(section, "triangulation_rmse",
                    compute_triangulation_rmse(scene),
                    bounds["clean_triangulation_rmse"]),
    ]
    return checks + evaluate_clean_pnp(scene, bounds)


def compute_triangulation_rmse(scene):
    P_A = paz.pinhole.make_camera_matrix(scene.intrinsics_A, scene.pose_A)
    P_B = paz.pinhole.make_camera_matrix(scene.intrinsics_B, scene.pose_B)
    points3D, _ = paz.triangulation.triangulate_points(
        P_A, P_B, scene.points_A, scene.points_B, scene.valid_mask)
    errors_A = metrics.compute_reprojection_errors(
        scene.pose_A, scene.intrinsics_A, points3D, scene.points_A)
    errors_B = metrics.compute_reprojection_errors(
        scene.pose_B, scene.intrinsics_B, points3D, scene.points_B)
    return float(np.sqrt(np.mean(np.concatenate([errors_A, errors_B])**2)))


def evaluate_clean_pnp(scene, bounds):
    estimate = paz.pnp.solve_DLT(scene.points3D, scene.points_A,
                                 scene.intrinsics_A, scene.valid_mask)
    pose_true = np.asarray(scene.pose_A)
    pose = np.asarray(estimate.pose)
    section = "clean_two_view"
    return [
        check_true(section, "pnp_valid", estimate.valid),
        check_below(section, "pnp_rotation_error",
                    metrics.compute_rotation_error(
                        pose[:3, :3], pose_true[:3, :3]),
                    bounds["clean_pnp_rotation_error"]),
        check_below(section, "pnp_translation_error",
                    float(np.linalg.norm(pose[:3, 3] - pose_true[:3, 3])),
                    bounds["clean_pnp_translation_error"]),
    ]


def evaluate_noisy_two_view(scene, reference, bounds):
    estimate = paz.epipolar.estimate_fundamental_matrix_RANSAC(
        jax.random.PRNGKey(7), scene.points_A, scene.points_B,
        scene.valid_mask, 1000, 3.0)
    precision, recall = metrics.compute_inlier_precision_recall(
        np.asarray(estimate.inliers), scene.inlier_mask)
    E = paz.epipolar.compute_essential_matrix(
        estimate.fundamental_matrix, scene.intrinsics_A,
        scene.intrinsics_B)
    pose_args = (E, scene.intrinsics_A, scene.intrinsics_B,
                 scene.points_A, scene.points_B, estimate.inliers)
    pose = paz.epipolar.recover_relative_pose(*pose_args)
    relative = np.asarray(
        synthetic.compute_relative_transform(scene.pose_A, scene.pose_B))
    rotation_error = metrics.compute_rotation_error(
        pose.rotation, relative[:3, :3])
    direction_error = metrics.compute_translation_direction_error(
        np.asarray(pose.translation), relative[:3, 3])
    factor, slack = bounds["relative_factor"], bounds["rotation_slack"]
    section = "noisy_two_view"
    return [
        check_true(section, "ransac_valid", estimate.valid),
        check_at_least(section, "precision", precision,
                       reference["precision"]
                       - bounds["precision_recall_slack"]),
        check_at_least(section, "recall", recall,
                       reference["recall"]
                       - bounds["precision_recall_slack"]),
        check_below(section, "rotation_error", rotation_error,
                    factor * reference["rotation_error"] + slack),
        check_below(section, "direction_error", direction_error,
                    factor * reference["direction_error"]
                    + bounds["direction_slack"]),
    ]


def evaluate_pnp(scene, reference, bounds):
    estimate = paz.pnp.estimate_pose_RANSAC(
        jax.random.PRNGKey(11), scene.points3D, scene.points2D,
        scene.intrinsics, scene.valid_mask, 500, 2.0)
    precision, recall = metrics.compute_inlier_precision_recall(
        np.asarray(estimate.inliers), scene.inlier_mask)
    pose = np.asarray(estimate.pose)
    pose_true = np.asarray(scene.pose)
    rotation_error = metrics.compute_rotation_error(
        pose[:3, :3], pose_true[:3, :3])
    translation_error = float(np.linalg.norm(
        pose[:3, 3] - pose_true[:3, 3]))
    factor = bounds["relative_factor"]
    precision_bound = reference["precision"] - bounds["precision_recall_slack"]
    recall_bound = reference["recall"] - bounds["precision_recall_slack"]
    rotation_bound = factor * reference["rotation_error"] \
        + bounds["rotation_slack"]
    translation_bound = factor * reference["translation_error"] \
        + bounds["translation_slack"]
    refined_rmse = compute_refined_rmse(scene, estimate)
    rmse_bound = bounds["refine_rmse_factor"] * reference["refined_rmse"]
    section = "pnp"
    return [
        check_true(section, "ransac_valid", estimate.valid),
        check_at_least(section, "precision", precision, precision_bound),
        check_at_least(section, "recall", recall, recall_bound),
        check_below(section, "rotation_error", rotation_error,
                    rotation_bound),
        check_below(section, "translation_error", translation_error,
                    translation_bound),
        check_below(section, "refined_rmse", refined_rmse, rmse_bound),
    ]


def compute_refined_rmse(scene, estimate):
    refined = paz.pnp.refine_pose(estimate.pose, scene.points3D,
                                  scene.points2D, scene.intrinsics,
                                  estimate.inliers, 10)
    true_inliers = np.asarray(scene.inlier_mask).astype(bool)
    errors = metrics.compute_reprojection_errors(
        np.asarray(refined.pose), scene.intrinsics,
        np.asarray(scene.points3D)[true_inliers],
        np.asarray(scene.points2D)[true_inliers])
    return float(np.sqrt(np.mean(errors**2)))


def evaluate_bundle(scene, reference, bounds):
    problem = build_bundle_problem(scene)
    result = paz.slam.bundle_adjust(problem, 30, 3.0, 1e-3)
    rotation_errors, translation_errors = [], []
    for pose, pose_true in zip(np.asarray(result.poses),
                               np.asarray(scene.poses)):
        rotation_errors.append(metrics.compute_rotation_error(
            pose[:3, :3], pose_true[:3, :3]))
        translation_errors.append(np.linalg.norm(
            pose[:3, 3] - pose_true[:3, 3]))
    factor = bounds["bundle_rmse_factor"]
    section = "bundle_adjustment"
    return [
        check_true(section, "valid", result.valid),
        check_true(section, "cost_decreased",
                   float(result.final_cost) < float(result.initial_cost)),
        check_below(section, "final_rmse", float(result.final_rmse),
                    bounds["bundle_final_rmse"]),
        check_below(section, "final_rmse_vs_reference",
                    float(result.final_rmse),
                    factor * reference["final_rmse"]
                    + bounds["bundle_rmse_slack"]),
        check_below(section, "max_rotation_error",
                    float(np.max(rotation_errors)),
                    factor * reference["max_rotation_error"]
                    + bounds["rotation_slack"]),
        check_below(section, "max_translation_error",
                    float(np.max(translation_errors)),
                    factor * reference["max_translation_error"]
                    + bounds["translation_slack"]),
        report_value(section, "initial_rmse", float(result.initial_rmse)),
        report_value(section, "num_accepted", float(result.num_accepted)),
    ]


def build_bundle_problem(scene):
    visibility = np.asarray(scene.visibility)
    pose_index, landmark_index = np.nonzero(visibility)
    observed = np.asarray(scene.observations)[pose_index, landmark_index]
    initial_poses = np.asarray(scene.noisy_poses).copy()
    initial_poses[0] = np.asarray(scene.poses)[0]
    num_poses, num_landmarks = visibility.shape
    capacity = len(pose_index)
    fields = (jp.asarray(initial_poses),
              jp.ones(num_poses, dtype=bool),
              jp.asarray(scene.noisy_points3D),
              jp.ones(num_landmarks, dtype=bool),
              jp.asarray(scene.intrinsics)[None], jp.eye(4)[None],
              jp.asarray(observed), jp.asarray(pose_index),
              jp.asarray(landmark_index), jp.zeros(capacity, jp.int32),
              jp.ones(capacity), jp.ones(capacity, dtype=bool))
    return paz.slam.BundleProblem(*fields)


def evaluate_stereo(sequence, bounds):
    solve = jax.jit(paz.pnp.estimate_pose_RANSAC, static_argnums=5)
    refine = jax.jit(paz.pnp.refine_pose, static_argnums=5)
    poses, frame_times = estimate_trajectory(solve, refine, sequence)
    poses_true = np.asarray(sequence.poses)
    ate = metrics.compute_ATE(poses, poses_true)
    rpe_translation, rpe_rotation = metrics.compute_RPE(poses, poses_true, 1)
    drift_translation, drift_rotation = metrics.compute_drift(
        poses, poses_true)
    section = "stereo_pnp_trajectory"
    return [
        check_below(section, "ate_rmse", ate, bounds["stereo_ate_rmse"]),
        check_below(section, "median_reprojection",
                    compute_median_reprojection(sequence, poses),
                    bounds["stereo_median_reprojection"]),
        check_true(section, "all_finite", np.all(np.isfinite(poses))),
        check_true(section, "compiled_once",
                   solve._cache_size() == 1
                   and refine._cache_size() == 1),
        report_value(section, "rpe_translation", rpe_translation),
        report_value(section, "rpe_rotation", rpe_rotation),
        report_value(section, "drift_translation_percent",
                     drift_translation),
        report_value(section, "drift_rotation_deg_per_m", drift_rotation),
        report_value(section, "compile_seconds", frame_times[0]),
        report_value(section, "warm_seconds_per_frame",
                     float(np.mean(frame_times[1:]))),
    ]


def estimate_trajectory(solve, refine, sequence):
    poses, frame_times = [], []
    for frame in range(len(sequence.observations_left)):
        key = jax.random.PRNGKey(100 + frame)
        args = (key, sequence.points3D,
                sequence.observations_left[frame], sequence.intrinsics,
                sequence.visibility_left[frame], 300, 2.0)
        start = time.perf_counter()
        estimate = solve(*args)
        refine_args = (estimate.pose, sequence.points3D,
                       sequence.observations_left[frame],
                       sequence.intrinsics, estimate.inliers, 10)
        pose = np.asarray(refine(*refine_args).pose)
        frame_times.append(time.perf_counter() - start)
        poses.append(pose)
    return np.stack(poses), frame_times


def compute_median_reprojection(sequence, poses):
    errors = []
    visibility = np.asarray(sequence.visibility_left)
    outliers = np.asarray(sequence.outlier_mask_left)
    observations = np.asarray(sequence.observations_left)
    points3D = np.asarray(sequence.points3D)
    for frame, pose in enumerate(poses):
        tracked = visibility[frame] & ~outliers[frame]
        errors.append(metrics.compute_reprojection_errors(
            pose, sequence.intrinsics, points3D[tracked],
            observations[frame][tracked]))
    return float(np.median(np.concatenate(errors)))


def print_table(checks):
    header = f"{'section':22} {'metric':28} {'value':>12} {'bound':>12}"
    print(header)
    print("-" * len(header))
    for entry in checks:
        if np.isinf(entry.bound):
            status = "info"
        elif entry.passed:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"{entry.section:22} {entry.name:28} {entry.value:12.6g} "
              f"{entry.bound:12.6g} {status}")


def write_artifacts(checks, mode, baseline):
    artifacts = os.path.join(fixtures.REPO_ROOT, "artifacts",
                             "vslam_baseline_results.json")
    os.makedirs(os.path.dirname(artifacts), exist_ok=True)
    results = {
        "mode": mode,
        "baseline_git_commit": baseline["git_commit"],
        "versions": baseline["versions"],
        "checks": [entry._asdict() for entry in checks],
        "failed": [entry.name for entry in checks if not entry.passed],
    }
    with open(artifacts, "w") as opened_file:
        json.dump(results, opened_file, indent=2, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"],
                        default="quick")
    arguments = parser.parse_args()
    with open(fixtures.BASELINE_JSON) as opened_file:
        baseline = json.load(opened_file)
    bounds = baseline["tolerances"]
    reference_metrics = baseline["reference"]
    scenes = fixtures.build_fixtures()
    checks = verify_checksums(scenes, baseline)
    checks += evaluate_clean_two_view(scenes.two_view_clean, bounds)
    checks += evaluate_noisy_two_view(
        scenes.two_view_noisy, reference_metrics["two_view_noisy"], bounds)
    checks += evaluate_pnp(scenes.pnp, reference_metrics["pnp"], bounds)
    if arguments.mode == "full":
        checks += evaluate_bundle(scenes.bundle,
                                  reference_metrics["bundle"], bounds)
        checks += evaluate_stereo(scenes.stereo, bounds)
    print_table(checks)
    write_artifacts(checks, arguments.mode, baseline)
    failed = [entry for entry in checks if not entry.passed]
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    raise SystemExit(1 if failed else 0)
