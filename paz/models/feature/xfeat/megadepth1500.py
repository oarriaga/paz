import os
import json
import numpy as np
import cv2

from paz.models.feature.xfeat import backend

THRESHOLDS = (5, 10, 20)


def evaluate(matcher, index_file, images_root, ransac_threshold=2.5):
    pairs = json.load(open(index_file))
    errors = []
    for pair in pairs:
        errors.append(evaluate_pair(matcher, pair, images_root,
                                    ransac_threshold))
    return pose_auc(errors, THRESHOLDS)


def evaluate_pair(matcher, pair, images_root, ransac_threshold):
    image0 = load_image(images_root, pair, 0)
    image1 = load_image(images_root, pair, 1)
    points0, points1 = matcher(image0, image1)
    points0 = points0 * np.array(pair["scale0"], np.float32)
    points1 = points1 * np.array(pair["scale1"], np.float32)
    pose = np.array(pair["T_0to1"], np.float32)
    intrinsics0 = np.array(pair["K0"], np.float32)
    intrinsics1 = np.array(pair["K1"], np.float32)
    args = points0, points1, intrinsics0, intrinsics1, ransac_threshold
    return pose_error(pose, *estimate_relative_pose(*args))


def load_image(images_root, pair, index):
    height, width = pair[f"size{index}_hw"]
    path = os.path.join(images_root, pair["pair_names"][index])
    image = cv2.imread(path)
    image = cv2.resize(image, (width, height))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def estimate_relative_pose(points0, points1, K0, K1, threshold):
    normalized = threshold / np.mean([K0[0, 0], K1[0, 0], K0[1, 1], K1[1, 1]])
    points0 = normalize_points(points0, K0)
    points1 = normalize_points(points1, K1)
    args = points0, points1, np.eye(3), cv2.RANSAC, 0.99999, normalized
    essential, mask = cv2.findEssentialMat(*args)
    if essential is None:
        return np.eye(3), np.zeros(3)
    _, rotation, translation, _ = cv2.recoverPose(
        essential, points0, points1, np.eye(3), mask=mask)
    return rotation, translation[:, 0]


def normalize_points(points, intrinsics):
    center = np.array([intrinsics[0, 2], intrinsics[1, 2]])
    focal = np.array([intrinsics[0, 0], intrinsics[1, 1]])
    return (points - center) / focal


def pose_error(pose, rotation, translation):
    translation_gt = pose[:3, 3]
    norm = np.linalg.norm(translation) * np.linalg.norm(translation_gt)
    cosine = np.dot(translation, translation_gt) / max(norm, 1e-12)
    angle = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
    translation_error = min(angle, 180 - angle)
    rotation_error = rotation_angle(rotation, pose[:3, :3])
    return max(translation_error, rotation_error)


def rotation_angle(rotation, rotation_gt):
    cosine = (np.trace(rotation.T @ rotation_gt) - 1) / 2
    return np.rad2deg(np.abs(np.arccos(np.clip(cosine, -1.0, 1.0))))


def pose_auc(errors, thresholds):
    errors = [0] + sorted(errors)
    recall = list(np.linspace(0, 1, len(errors)))
    areas = {}
    for threshold in thresholds:
        cut = np.searchsorted(errors, threshold)
        y = recall[:cut] + [recall[cut - 1]]
        x = errors[:cut] + [threshold]
        areas[f"auc@{threshold}"] = np.trapz(y, x) / threshold
    return areas


def match_mutual(extract, min_cosine=-1.0):
    def call(image0, image1):
        first = extract(image0)
        second = extract(image1)
        source, target = backend.mutual_nearest_neighbors(
            first.descriptors, second.descriptors, min_cosine)
        return np.asarray(first.keypoints)[source], \
            np.asarray(second.keypoints)[target]

    return call
