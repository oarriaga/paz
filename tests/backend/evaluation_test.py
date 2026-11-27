import os

import numpy as np
import cv2

import paz


def perfect_box():
    return np.array([0.0, 0.0, 10.0, 10.0])


def match(pred_boxes, pred_classes, pred_scores, true_boxes, true_classes,
          true_difficult=None):
    if true_difficult is None:
        true_difficult = np.zeros(len(true_classes), "bool")
    return paz.evaluation.match_predictions(
        pred_boxes, pred_classes, pred_scores, true_boxes, true_classes,
        true_difficult, 0.5
    )


def test_true_positive_then_duplicate_is_false_positive():
    pred_boxes = np.array([perfect_box(), perfect_box()])
    true_positives, ignored = match(
        pred_boxes, np.array([0, 0]), np.array([0.9, 0.8]),
        np.array([perfect_box()]), np.array([0])
    )
    assert list(np.asarray(true_positives)) == [True, False]
    assert not np.any(np.asarray(ignored))


def test_high_score_false_positive_halves_average_precision():
    far_box = np.array([100.0, 100.0, 110.0, 110.0])
    pred_boxes = np.array([far_box, perfect_box()])
    pred_classes = np.array([0, 0])
    pred_scores = np.array([0.95, 0.9])
    true_positives, ignored = match(
        pred_boxes, pred_classes, pred_scores,
        np.array([perfect_box()]), np.array([0])
    )
    # predictions are already in descending-score order here.
    args = (pred_classes, true_positives, ignored, 1, 0)
    all_point = float(paz.evaluation.class_average_precision(*args, False))
    eleven = float(paz.evaluation.class_average_precision(*args, True))
    assert np.isclose(all_point, 0.5)
    assert np.isclose(eleven, 0.5)


def test_class_separation_blocks_cross_class_match():
    true_positives, ignored = match(
        np.array([perfect_box()]), np.array([1]), np.array([0.9]),
        np.array([perfect_box()]), np.array([0])
    )
    assert list(np.asarray(true_positives)) == [False]


def test_match_to_difficult_box_is_ignored():
    true_positives, ignored = match(
        np.array([perfect_box()]), np.array([0]), np.array([0.9]),
        np.array([perfect_box()]), np.array([0]), np.array([True])
    )
    assert not np.any(np.asarray(true_positives))
    assert list(np.asarray(ignored)) == [True]


def build_detector(predictions):
    queue = list(predictions)

    def detector(image):
        return queue.pop(0)

    return detector


def test_compute_mAP_reaches_one_on_perfect_detector(tmp_path):
    image = np.zeros((16, 16, 3), "uint8")
    paths = []
    for index in range(2):
        path = os.path.join(str(tmp_path), f"{index}.png")
        cv2.imwrite(path, image)
        paths.append(path)
    ground_truths = [
        np.array([[0.0, 0.0, 10.0, 10.0, 0], [2.0, 2.0, 8.0, 8.0, 1]]),
        np.array([[1.0, 1.0, 9.0, 9.0, 0]]),
    ]
    predictions = []
    for ground_truth in ground_truths:
        boxes = ground_truth[:, :4]
        classes = ground_truth[:, 4].astype("int32")
        scores = np.ones(len(boxes))
        predictions.append((boxes, classes, scores))
    detector = build_detector(predictions)
    result = paz.evaluation.compute_mAP(detector, paths, ground_truths, 2)
    assert np.isclose(result["mAP"], 1.0)


def unit_cube_points():
    corners = [[x, y, z] for x in (-0.5, 0.5)
               for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
    return np.array(corners, "float64")


def test_compute_ADD_identity_pose_is_zero():
    points3D = unit_cube_points()
    pose = (np.eye(3), np.zeros(3))
    assert paz.evaluation.compute_ADD(points3D, pose, pose) == 0.0


def test_compute_ADD_pure_translation_equals_norm():
    points3D = unit_cube_points()
    translation = np.array([0.3, -0.4, 0.0])
    true = (np.eye(3), np.zeros(3))
    pred = (np.eye(3), translation)
    error = paz.evaluation.compute_ADD(points3D, true, pred)
    assert np.isclose(error, np.linalg.norm(translation))


def test_compute_ADI_not_greater_than_ADD():
    points3D = unit_cube_points()
    angle = 0.5
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]])
    true = (np.eye(3), np.zeros(3))
    pred = (rotation, np.array([0.05, 0.0, 0.0]))
    add = paz.evaluation.compute_ADD(points3D, true, pred)
    adi = paz.evaluation.compute_ADI(points3D, true, pred)
    assert adi <= add + 1e-9


def test_object_diameter_of_unit_cube():
    diameter = paz.evaluation.compute_object_diameter(unit_cube_points())
    assert np.isclose(diameter, np.sqrt(3.0))


def test_is_correct_ADD_threshold():
    assert paz.evaluation.is_correct_ADD(0.09, 1.0, 0.1)
    assert not paz.evaluation.is_correct_ADD(0.11, 1.0, 0.1)
