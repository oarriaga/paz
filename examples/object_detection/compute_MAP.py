import numpy as np
import paz


def preprocess_preds(class_arg, pred_boxes, pred_scores, pred_class_args):
    class_mask = class_arg == pred_class_args
    class_predicted_boxes = pred_boxes[class_mask]
    class_predicted_scores = pred_scores[class_mask]
    sorted_args = class_predicted_scores.argsort()[::-1]  # descending order
    sorted_class_predicted_boxes = class_predicted_boxes[sorted_args]
    sorted_class_predicted_scores = class_predicted_scores[sorted_args]
    return sorted_class_predicted_boxes, sorted_class_predicted_scores


def evaluate(images, boxes, class_args, pipeline, num_classes, IOU_thresh):
    class_to_num_positives = {class_arg: 0 for class_arg in range(num_classes)}
    class_to_pred_scores = {class_arg: [] for class_arg in range(num_classes)}
    class_to_match = {class_arg: [] for class_arg in range(num_classes)}
    iterator = zip(images, boxes, class_args)
    for true_image, true_boxes, true_class_args in iterator:
        image = paz.load.image(true_image)
        pred_boxes, pred_class_args, pred_scores = pipeline(image)
        all_class_args = paz.classes.join([true_class_args, pred_class_args])
        for class_arg in np.unique(all_class_args):
            args = class_arg, pred_boxes, pred_scores, pred_class_args
            class_pred_boxes, class_pred_scores = preprocess_preds(*args)
            class_mask = class_arg == true_class_args
            class_true_boxes = true_boxes[class_mask]
            class_to_num_positives[class_arg] += len(class_true_boxes)
            class_to_pred_scores[class_arg].extend(class_pred_scores)

            if len(class_pred_boxes) == 0:
                continue

            if len(class_true_boxes) == 0:
                class_to_match[class_arg].extend((0,) * len(class_pred_boxes))
                continue

            ious = paz.boxes.compute_IOUs(class_pred_boxes, class_true_boxes)
            ground_truth_args = ious.argmax(axis=1)
            ground_truth_args[ious.max(axis=1) < IOU_thresh] = -1

            is_selected = np.zeros(len(class_true_boxes), dtype=bool)
            for ground_truth_arg in ground_truth_args:
                if ground_truth_arg >= 0:
                    if not is_selected[ground_truth_arg]:
                        class_to_match[class_arg].append(1)
                    else:
                        class_to_match[class_arg].append(0)
                    is_selected[ground_truth_arg] = True
                else:
                    class_to_match[class_arg].append(0)

    return class_to_num_positives, class_to_pred_scores, class_to_match
