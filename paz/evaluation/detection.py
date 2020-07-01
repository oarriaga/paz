import numpy as np
from ..backend.boxes import compute_ious
from ..backend.image import load_image


def compute_matches(dataset, detector, class_to_arg, iou_thresh=0.5):
    """
    Arguments:
        dataset: List of dictionaries containing 'image' as key and a
            numpy array representing an image as value.
        detector : Function for performing inference
        class_to_arg: Dict. of class names and their id
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value..

    Returns:
        num_positives: Dict. containing number of positives for each class
        score: Dict. containing matching scores of boxes for each class
        match: Dict. containing match/non-match info of boxes in each class
    """
    # classes_count = len(np.unique(np.concatenate(ground_truth_class_args)))
    num_classes = len(class_to_arg)
    num_positives = {label_id: 0 for label_id in range(1, num_classes + 1)}
    score = {label_id: [] for label_id in range(1, num_classes + 1)}
    match = {label_id: [] for label_id in range(1, num_classes + 1)}
    for sample in dataset:
        # obtaining ground truths
        ground_truth_boxes = np.array(sample['boxes'][:, :4])
        ground_truth_class_args = np.array(sample['boxes'][:, 4])
        if 'difficulties' in sample.keys():
            difficulties = np.array(sample['difficulties'])
        else:
            difficulties = None
        # obtaining predictions
        image = load_image(sample['image'])
        results = detector(image)
        predicted_boxes, predicted_class_args, predicted_scores = [], [], []
        for box2D in results['boxes2D']:
            predicted_scores.append(box2D.score)
            predicted_class_args.append(class_to_arg[box2D.class_name])
            predicted_boxes.append(list(box2D.coordinates))
        predicted_boxes = np.array(predicted_boxes, dtype=np.float32)
        predicted_class_args = np.array(predicted_class_args)
        predicted_scores = np.array(predicted_scores, dtype=np.float32)
        # setting difficulties to ``Easy`` if they are None
        if difficulties is None:
            difficulties = np.zeros(len(ground_truth_boxes), dtype=bool)
        # iterating over each class present in the image
        class_args = np.concatenate(
            (predicted_class_args, ground_truth_class_args))
        class_args = np.unique(class_args).astype(int)
        for class_arg in class_args:
            # masking predictions by class
            class_mask = class_arg == predicted_class_args
            class_predicted_boxes = predicted_boxes[class_mask]
            class_predicted_scores = predicted_scores[class_mask]
            # sort score from maximum to minimum for masked predictions
            sorted_args = class_predicted_scores.argsort()[::-1]
            class_predicted_boxes = class_predicted_boxes[sorted_args]
            class_predicted_scores = class_predicted_scores[sorted_args]
            # masking ground truths by class
            class_mask = class_arg == ground_truth_class_args
            class_ground_truth_boxes = ground_truth_boxes[class_mask]
            class_difficulties = difficulties[class_mask]
            # the number of positives equals the number of easy boxes
            num_easy = np.logical_not(class_difficulties).sum()
            num_positives[class_arg] = num_positives[class_arg] + num_easy
            # add all predicted scores to scores
            score[class_arg].extend(class_predicted_scores)
            # if not predicted boxes for this class continue
            if len(class_predicted_boxes) == 0:
                continue
            # if not ground truth boxes continue but add zeros as matches
            if len(class_ground_truth_boxes) == 0:
                match[class_arg].extend((0,) * len(class_predicted_boxes))
                continue

            # evaluation on VOC follows integer typed bounding boxes.
            class_predicted_boxes = class_predicted_boxes.copy()
            class_predicted_boxes[:, 2:] = (
                class_predicted_boxes[:, 2:] + 1)
            class_ground_truth_boxes = class_ground_truth_boxes.copy()
            class_ground_truth_boxes[:, 2:] = (
                class_ground_truth_boxes[:, 2:] + 1)

            ious = compute_ious(
                class_predicted_boxes, class_ground_truth_boxes)
            ground_truth_args = ious.argmax(axis=1)
            # set -1 if there is no matching ground truth
            ground_truth_args[ious.max(axis=1) < iou_thresh] = -1
            selected = np.zeros(len(class_ground_truth_boxes), dtype=bool)
            for ground_truth_arg in ground_truth_args:
                if ground_truth_arg >= 0:
                    if class_difficulties[ground_truth_arg]:
                        match[class_arg].append(-1)
                    else:
                        if not selected[ground_truth_arg]:
                            match[class_arg].append(1)
                        else:
                            match[class_arg].append(0)
                    selected[ground_truth_arg] = True
                else:
                    match[class_arg].append(0)
    return num_positives, score, match


def calculate_relevance_metrics(num_positives, scores, matches):
    """Calculates precision and recall.
    Arguments:
        num_positives: Dict. with number of positives for each class
        scores: Dict. with matching scores of boxes for each class
        matches: Dict. wth match/non-match info for boxes for each class
    Returns:
        precision: Dict. with precision values per class
        recall : Dict. with recall values per class
    """
    num_classes = max(num_positives.keys()) + 1
    precision, recall = [None] * num_classes, [None] * num_classes
    for class_arg in num_positives.keys():
        class_positive_matches = np.array(matches[class_arg], dtype=np.int8)
        class_scores = np.array(scores[class_arg])
        order = class_scores.argsort()[::-1]
        class_positive_matches = class_positive_matches[order]
        true_positives = np.cumsum(class_positive_matches == 1)
        false_positives = np.cumsum(class_positive_matches == 0)
        precision[class_arg] = (
            true_positives / (false_positives + true_positives))
        if num_positives[class_arg] > 0:
            recall[class_arg] = true_positives / num_positives[class_arg]
    return precision, recall


def calculate_average_precisions(precision, recall, use_07_metric=False):
    """Calculate average precisions based based on PASCAL VOC evaluation
    Arguments:
        num_positives: Dict. with number of positives for each class
        scores: Dict. with matching scores of boxes for each class
        matches: Dict. wth match/non-match info for boxes for each class
    Returns:
    """

    num_classes = len(precision)
    average_precisions = np.empty(num_classes)
    for class_arg in range(num_classes):
        if precision[class_arg] is None or recall[class_arg] is None:
            average_precisions[class_arg] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            average_precisions[class_arg] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(recall[class_arg] >= t) == 0:
                    p_interpolation = 0
                else:
                    p_interpolation = np.max(
                        np.nan_to_num(
                            precision[class_arg]
                        )[recall[class_arg] >= t]
                    )
                average_precision_class = average_precisions[class_arg]
                average_precision_class = (average_precision_class +
                                           (p_interpolation / 11))
                average_precisions[class_arg] = average_precision_class

        else:
            # first append sentinel values at the end
            average_precision = np.concatenate(
                ([0], np.nan_to_num(precision[class_arg]), [0]))
            average_recall = np.concatenate(([0], recall[class_arg], [1]))

            average_precision = np.maximum.accumulate(
                average_precision[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            recall_change_arg = np.where(
                average_recall[1:] != average_recall[:-1])[0]

            # and sum (\Delta recall) * precision
            average_precisions[class_arg] = np.sum(
                (average_recall[recall_change_arg + 1] -
                 average_recall[recall_change_arg]) *
                average_precision[recall_change_arg + 1])
    return average_precisions


def evaluateMAP(detector, dataset, class_to_arg, iou_thresh=0.5,
                use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    Arguments:
        dataset: List of dictionaries containing 'image' as key and a
            numpy array representing an image as value.
        detector : Function for performing inference
        class_to_arg: Dict. of class names and their id
        iou_thresh: Float indicating intersection over union threshold for
            assigning a prediction as correct.
    # Returns:
    """
    positives, score, match = compute_matches(
        dataset, detector, class_to_arg, iou_thresh)
    precision, recall = calculate_relevance_metrics(positives, score, match)
    average_precisions = calculate_average_precisions(
        precision, recall, use_07_metric)
    return {'ap': average_precisions, 'map': np.nanmean(average_precisions)}
