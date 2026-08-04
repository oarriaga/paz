import cv2
import jax
import jax.numpy as jp
import paz

import backend
import geometry


def reconstruct_scene(images, camera_intrinsics, key, match_ratio=0.75,
                      residual_thresh=0.5, correspondence_thresh=0.5):
    num_images = len(images)
    print(f"Processing image 1/{num_images}")
    print(f"Processing image 2/{num_images}")
    key, init_key = jax.random.split(key)
    init = initialize_two_view(init_key, images[0], images[1],
                               camera_intrinsics, match_ratio, residual_thresh)
    points3D, colors = init.points3D, init.colors
    base_feature, P_prev = init.base_feature, init.P2
    keypoints_prev, descriptors_prev = init.keypoints2, init.descriptors2
    all_points3D, all_colors = [points3D], [colors]

    for index in range(2, num_images):
        print(f"Processing image {index + 1}/{num_images}")
        key, pnp_key, ransac_key = jax.random.split(key, 3)
        keypoints_next, descriptors_next = backend.detect_SIFT_features(
            images[index])

        rotation_vector, translation = register_camera_pose(
            pnp_key, base_feature, keypoints_next, descriptors_next, points3D,
            camera_intrinsics, match_ratio, correspondence_thresh)
        rotation = cv2.Rodrigues(rotation_vector)[0]

        step = triangulate_new_points(
            ransac_key, images[index - 1], keypoints_prev, descriptors_prev,
            keypoints_next, descriptors_next, P_prev, rotation, translation,
            camera_intrinsics, match_ratio, residual_thresh)
        points3D, colors, base_feature, P_prev = step
        all_points3D.append(points3D)
        all_colors.append(colors)
        keypoints_prev, descriptors_prev = keypoints_next, descriptors_next

    return paz.NamedTuple(
        "Reconstruction", points3D=all_points3D, colors=all_colors)


def initialize_two_view(key, image1, image2, camera_intrinsics, match_ratio,
                        residual_thresh):
    keypoints1, descriptors1 = backend.detect_SIFT_features(image1)
    keypoints2, descriptors2 = backend.detect_SIFT_features(image2)
    matches = backend.match_features(descriptors1, descriptors2, match_ratio)
    points1, points2 = backend.get_match_points(keypoints1, keypoints2, matches)
    points1, points2, inliers = filter_matches(
        key, points1, points2, residual_thresh)
    colors = backend.extract_keypoints_RGB(image1, points1)

    fundamental_matrix = geometry.compute_fundamental_matrix(
        jp.asarray(points1), jp.asarray(points2))
    essential_matrix = geometry.compute_essential_matrix(
        fundamental_matrix, camera_intrinsics)
    rotation, translation = geometry.recover_pose(
        essential_matrix, camera_intrinsics, jp.asarray(points1),
        jp.asarray(points2))

    P1 = paz.pinhole.make_camera_matrix(camera_intrinsics, jp.eye(4))
    P2 = build_camera_matrix(camera_intrinsics, rotation, translation)
    points3D = geometry.triangulate_points(
        P1, P2, jp.asarray(points1), jp.asarray(points2))

    rotation, translation = backend.refine_camera_pose(
        paz.to_numpy(rotation), paz.to_numpy(translation),
        paz.to_numpy(points3D), points2, paz.to_numpy(camera_intrinsics))
    P2 = build_camera_matrix(camera_intrinsics, rotation, translation)
    points3D = paz.to_numpy(geometry.triangulate_points(
        P1, P2, jp.asarray(points1), jp.asarray(points2)))

    _, train_indices = backend.get_match_indices(matches)
    base_indices = train_indices[inliers]
    base_keypoints = keypoints2[base_indices]
    base_descriptors = descriptors2[base_indices]
    base_feature = (base_keypoints, base_descriptors)
    return paz.NamedTuple(
        "TwoView", points3D=points3D, colors=colors, base_feature=base_feature,
        P2=P2, keypoints2=keypoints2, descriptors2=descriptors2)


def register_camera_pose(key, base_feature, keypoints_next, descriptors_next,
                         points3D_prev, camera_intrinsics, match_ratio,
                         correspondence_thresh):
    """Solves PnP-RANSAC for the new camera using the *new* frame's own 2D
    observations (not the previous frame's), matched against 2D keypoints
    that are already tied to a triangulated 3D point."""
    base_keypoints, base_descriptors = base_feature
    matches = backend.match_features(
        base_descriptors, descriptors_next, match_ratio)
    points_prev, points_next = backend.get_match_points(
        base_keypoints, keypoints_next, matches)
    points_prev, points_next, inliers = filter_matches(
        key, points_prev, points_next, correspondence_thresh)
    query_indices, _ = backend.get_match_indices(matches)
    matched_indices = query_indices[inliers]
    return backend.solve_PnP_RANSAC(
        points3D_prev[matched_indices], points_next, camera_intrinsics)


def triangulate_new_points(key, image_prev, keypoints_prev, descriptors_prev,
                           keypoints_next, descriptors_next, P_prev, rotation,
                           translation, camera_intrinsics, match_ratio,
                           residual_thresh):
    matches = backend.match_features(
        descriptors_prev, descriptors_next, match_ratio)
    points_prev, points_next = backend.get_match_points(
        keypoints_prev, keypoints_next, matches)
    points_prev, points_next, inliers = filter_matches(
        key, points_prev, points_next, residual_thresh)
    colors = backend.extract_keypoints_RGB(image_prev, points_prev)

    P_next = build_camera_matrix(camera_intrinsics, rotation, translation)
    points3D = geometry.triangulate_points(
        P_prev, P_next, jp.asarray(points_prev), jp.asarray(points_next))

    rotation, translation = backend.refine_camera_pose(
        rotation, translation, paz.to_numpy(points3D), points_next,
        paz.to_numpy(camera_intrinsics))
    P_next = build_camera_matrix(camera_intrinsics, rotation, translation)
    points3D = paz.to_numpy(geometry.triangulate_points(
        P_prev, P_next, jp.asarray(points_prev), jp.asarray(points_next)))

    train_indices = backend.get_match_indices(matches)[1][inliers]
    base_keypoints = keypoints_next[train_indices]
    base_descriptors = descriptors_next[train_indices]
    base_feature = (base_keypoints, base_descriptors)
    return points3D, colors, base_feature, P_next


def build_camera_matrix(camera_intrinsics, rotation, translation):
    pose = paz.pinhole.to_affine_matrix(
        jp.asarray(rotation), jp.asarray(translation))
    return paz.pinhole.make_camera_matrix(camera_intrinsics, pose)


def filter_matches(key, points1, points2, threshold):
    """RANSAC-filters correspondences, keeping fundamental-matrix inliers."""
    _, mask = geometry.estimate_fundamental_matrix_RANSAC(
        key, jp.asarray(points1), jp.asarray(points2), threshold=threshold)
    mask = paz.to_numpy(mask)
    return points1[mask], points2[mask], mask
