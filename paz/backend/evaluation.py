import jax
import jax.numpy as jp
import numpy as np

import paz
import paz.utils.progressbar as progressbar


def compute_mAP(
    detector,
    images,
    ground_truths,
    num_classes,
    difficulties=None,
    max_detections=200,
    max_objects=64,
    iou_thresh=0.5,
    use_07_metric=False,
    verbose=False,
):
    if difficulties is None:
        difficulties = empty_difficulties(ground_truths)
    positives = count_positives(ground_truths, difficulties, num_classes)
    matches = match_dataset(detector, images, ground_truths, difficulties,
                            max_detections, max_objects, iou_thresh, verbose)
    return reduce_average_precisions(*matches, positives, num_classes,
                                     use_07_metric)


def match_dataset(detector, images, ground_truths, difficulties,
                  max_detections, max_objects, iou_thresh, verbose):
    match = jax.jit(paz.lock(match_predictions, iou_thresh))
    scores, classes, true_positives, ignored = [], [], [], []
    start = progressbar.start()
    for index, image_path in enumerate(images):
        image = paz.image.load(image_path)
        boxes, labels, score = pad_predictions(*detector(image), max_detections)
        truth = pad_ground_truth(ground_truths[index], difficulties[index],
                                 max_objects)
        is_true, is_ignored = match(boxes, labels, score, *truth)
        scores.append(score)
        classes.append(labels)
        true_positives.append(is_true)
        ignored.append(is_ignored)
        if verbose:
            progressbar.print_bar(index + 1, len(images), start, "evaluating")
    if verbose:
        print()
    columns = (scores, classes, true_positives, ignored)
    return tuple(jp.concatenate(column) for column in columns)


def reduce_average_precisions(scores, classes, true_positives, ignored,
                              positives, num_classes, use_07_metric):
    order = jp.argsort(-scores)
    classes = classes[order]
    true_positives = true_positives[order]
    ignored = ignored[order]
    average_precisions = np.full(num_classes, np.nan)
    for class_arg in range(num_classes):
        if positives[class_arg] == 0:
            continue
        average_precisions[class_arg] = float(class_average_precision(
            classes, true_positives, ignored, positives[class_arg],
            class_arg, use_07_metric))
    mean = float(np.nanmean(average_precisions))
    return {"ap": average_precisions, "mAP": mean}


def class_average_precision(classes, true_positives, ignored, num_positives,
                            class_arg, use_07_metric):
    in_class = classes == class_arg
    counted = in_class & jp.logical_not(ignored)
    false_positive = counted & jp.logical_not(true_positives)
    cumulative_true = jp.cumsum(true_positives & in_class)
    cumulative_false = jp.cumsum(false_positive)
    detected = jp.maximum(cumulative_true + cumulative_false, 1)
    recall = cumulative_true / jp.maximum(num_positives, 1)
    precision = cumulative_true / detected
    return compute_average_precision(recall, precision, use_07_metric)


def compute_average_precision(recall, precision, use_07_metric=False):
    if use_07_metric:
        return average_precision_07(recall, precision)
    return average_precision_all(recall, precision)


def average_precision_all(recall, precision):
    recall = jp.concatenate([jp.zeros(1), recall, jp.ones(1)])
    precision = jp.concatenate([jp.zeros(1), precision, jp.zeros(1)])
    precision = jax.lax.cummax(precision, reverse=True)
    return jp.sum((recall[1:] - recall[:-1]) * precision[1:])


def average_precision_07(recall, precision):
    levels = jp.arange(11) / 10.0
    # tolerate float32 noise so a recall exactly on a level still counts.
    reached = recall[None, :] >= levels[:, None] - 1e-6
    return jp.mean(jp.max(jp.where(reached, precision[None, :], 0.0), axis=1))


def match_predictions(pred_boxes, pred_classes, pred_scores,
                      true_boxes, true_classes, true_difficult, iou_thresh):
    ious = paz.boxes.compute_IOUs(
        expand_corners(pred_boxes), expand_corners(true_boxes))
    same_class = pred_classes[:, None] == true_classes[None, :]
    ious = jp.where(same_class, ious, 0.0)
    best_true = jp.argmax(ious, axis=1)
    is_match = jp.max(ious, axis=1) >= iou_thresh
    order = jp.argsort(-pred_scores)
    difficult = jp.asarray(true_difficult, "bool")
    return assign_matches(best_true, is_match, difficult, order,
                          true_boxes.shape[0])


def assign_matches(best_true, is_match, true_difficult, order, num_true):
    def step(taken, pred_arg):
        true_arg = best_true[pred_arg]
        matched = is_match[pred_arg]
        difficult = true_difficult[true_arg]
        fresh = jp.logical_not(taken[true_arg])
        is_true = matched & fresh & jp.logical_not(difficult)
        is_ignored = matched & difficult
        taken = taken.at[true_arg].set(taken[true_arg] | matched)
        return taken, (is_true, is_ignored)

    taken = jp.zeros(num_true, "bool")
    _, (true_hits, ignored_hits) = jax.lax.scan(step, taken, order)
    true_positives = jp.zeros(order.shape[0], "bool").at[order].set(true_hits)
    ignored = jp.zeros(order.shape[0], "bool").at[order].set(ignored_hits)
    return true_positives, ignored


def expand_corners(boxes):
    # VOC scores IoU on integer boxes, so widen the far corner by one pixel.
    return boxes + jp.array([0.0, 0.0, 1.0, 1.0])


def count_positives(ground_truths, difficulties, num_classes):
    positives = np.zeros(num_classes, "int32")
    for ground_truth, difficult in zip(ground_truths, difficulties):
        classes = np.asarray(ground_truth, "float32")[:, 4].astype("int32")
        easy = classes[np.logical_not(np.asarray(difficult, "bool"))]
        positives += np.bincount(easy, minlength=num_classes)
    return positives


def pad_predictions(boxes, classes, scores, size):
    boxes = jp.asarray(boxes).astype("float32")
    classes = jp.asarray(classes).astype("int32")
    scores = jp.asarray(scores).astype("float32")
    order = jp.argsort(-scores)[:size]
    boxes = fixed_size(boxes[order], size, 0.0)
    classes = fixed_size(classes[order], size, -1)
    scores = fixed_size(scores[order], size, -jp.inf)
    return boxes, classes, scores


def pad_ground_truth(ground_truth, difficult, size):
    ground_truth = jp.asarray(ground_truth, "float32")
    boxes = fixed_size(ground_truth[:, :4], size, 0.0)
    classes = fixed_size(ground_truth[:, 4].astype("int32"), size, -2)
    difficult = fixed_size(jp.asarray(difficult, "bool"), size, False)
    return boxes, classes, difficult


def fixed_size(array, size, pad_value):
    array = array[:size]
    pad_width = [(0, size - array.shape[0])] + [(0, 0)] * (array.ndim - 1)
    return jp.pad(array, pad_width, constant_values=pad_value)


def empty_difficulties(ground_truths):
    return [np.zeros(len(truth), "bool") for truth in ground_truths]


def transform_mesh_points(points3D, rotation, translation):
    return points3D @ rotation.T + translation


def compute_ADD(points3D, pose_true, pose_pred):
    true = transform_mesh_points(points3D, *pose_true)
    pred = transform_mesh_points(points3D, *pose_pred)
    return float(np.linalg.norm(pred - true, axis=1).mean())


def compute_ADI(points3D, pose_true, pose_pred):
    true = transform_mesh_points(points3D, *pose_true)
    pred = transform_mesh_points(points3D, *pose_pred)
    distances = np.linalg.norm(pred[:, None, :] - true[None, :, :], axis=-1)
    return float(distances.min(axis=1).mean())


def compute_object_diameter(points3D):
    distances = np.linalg.norm(points3D[:, None, :] - points3D[None, :, :], axis=-1)  # fmt: skip
    return float(distances.max())


def is_correct_ADD(error, diameter, threshold=0.1):
    return error <= diameter * threshold
