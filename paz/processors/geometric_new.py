import numpy as np
import jax.numpy as jp
import paz
import jax


def random_crop(key, image, detections, probability=0.50, max_trials=50):
    jaccard_min_max = (
        (0.1, np.inf),
        (0.3, np.inf),
        (0.7, np.inf),
        (0.9, np.inf),
        (-np.inf, np.inf),
    )

    if jax.random.uniform(key, shape=()) >= probability:
        return image, detections

    boxes, labels = paz.boxes.split(detections)
    H_original, W_original = paz.image.get_size(image)

    mode = np.random.randint(0, len(jaccard_min_max), 1)[0]
    min_IOU, max_IOU = jaccard_min_max[mode]
    for trial_arg in range(max_trials):
        W = np.random.uniform(0.3 * W_original, W_original)
        H = np.random.uniform(0.3 * H_original, H_original)
        aspect_ratio = H / W
        if (aspect_ratio < 0.5) or (aspect_ratio > 2):
            continue
        x_min = np.random.uniform(W_original - W)
        y_min = np.random.uniform(H_original - H)
        x_max = int(x_min + W)
        y_max = int(y_min + H)
        x_min = int(x_min)
        y_min = int(y_min)

        crop_box = np.array([x_min, y_min, x_max, y_max])
        overlap = paz.boxes.compute_IOU(crop_box, boxes)
        if (overlap.max() < min_IOU) or (overlap.min() > max_IOU):
            continue
        x_centers, y_centers = paz.boxes.compute_centers(boxes)
        centers_above_x_min = x_min < x_centers
        centers_above_y_min = y_min < y_centers
        centers_below_x_max = x_max > x_centers
        centers_below_y_max = y_max > y_centers
        mask = (
            centers_above_x_min
            * centers_above_y_min
            * centers_below_x_max
            * centers_below_y_max
        )
        if not mask.any():
            continue

        cropped_image = image[y_min:y_max, x_min:x_max, :].copy()
        masked_boxes = boxes[mask, :].copy()
        masked_labels = labels[mask].copy()
        # should we use the box left and top corner or the crop's
        masked_boxes[:, :2] = np.maximum(masked_boxes[:, :2], crop_box[:2])
        # adjust to crop (by substracting crop's left,top)
        masked_boxes[:, :2] -= crop_box[:2]
        masked_boxes[:, 2:] = np.minimum(masked_boxes[:, 2:], crop_box[2:])
        # adjust to crop (by substracting crop's left,top)
        masked_boxes[:, 2:] -= crop_box[:2]
        return cropped_image, np.hstack([masked_boxes, masked_labels])

    detections = np.hstack([boxes, labels])
    return image, detections
