import jax.numpy as jp
import jax
import paz


def random_flip_left_right(image, boxes):
    boxes = paz.boxes.flip_left_right(boxes, image.shape[1])
    image = paz.image.flip_left_right(image)
    return image, boxes


def split(detections):
    boxes = detections[:, :4]
    class_args = detections[:, 4:]
    return boxes, class_args


def get_boxes(detections):
    return detections[:, :4]


def get_scores(detections):
    return detections[:, 4:]


def merge(boxes, class_args):
    return jp.concatenate([boxes, class_args], axis=1)


def to_one_hot(detections, num_classes):
    boxes, classes = split(detections)
    classes = paz.classes.to_one_hot(classes, num_classes)
    return merge(boxes, classes)


def encode(matched, priors, variances=[0.1, 0.1, 0.2, 0.2], epislon=1e-8):
    """Encode matched bounding boxes relative to prior boxes."""

    def encode_centers(boxes_center, priors, variances):
        """Encode center coordinates using priors and variances."""
        x_boxes, y_boxes, _, _ = paz.boxes.split(boxes_center)
        x_prior, y_prior, W_prior, H_prior = paz.boxes.split(priors)
        x_difference = x_boxes - x_prior
        y_difference = y_boxes - y_prior
        x_encoded_center = (x_difference / W_prior) / variances[0]
        y_encoded_center = (y_difference / H_prior) / variances[1]
        return x_encoded_center, y_encoded_center

    def encode_sizes(boxes_center, priors, variances):
        """Encode width and height dimensions."""
        _, _, W_boxes, H_boxes = paz.boxes.split(boxes_center)
        _, _, W_prior, H_prior = paz.boxes.split(priors)
        W_ratio = W_boxes / W_prior
        H_ratio = H_boxes / H_prior
        W_encoded = jp.log(W_ratio + epislon) / variances[2]
        H_encoded = jp.log(H_ratio + epislon) / variances[3]
        return W_encoded, H_encoded

    boxes_corner, scores = split(matched)
    boxes_center = paz.boxes.to_center_form(boxes_corner)
    x_encoded, y_encoded = encode_centers(boxes_center, priors, variances)
    W_encoded, H_encoded = encode_sizes(boxes_center, priors, variances)
    encooded_boxes = [x_encoded, y_encoded, W_encoded, H_encoded, scores]
    return jp.concatenate(encooded_boxes, axis=1)


def decode(predictions, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    """Decode predicted box parameters to actual coordinates."""

    def decode_center_form_boxes(predictions, priors, variances):
        """Compute center-form boxes from predictions."""

        def decode_center_x(predictions, priors, variances):
            """Decode center x-coordinate from predictions."""
            return (
                predictions[:, 0:1] * priors[:, 2:3] * variances[0]
                + priors[:, 0:1]
            )

        def decode_center_y(predictions, priors, variances):
            """Decode center y-coordinate from predictions."""
            return (
                predictions[:, 1:2] * priors[:, 3:4] * variances[1]
                + priors[:, 1:2]
            )

        def decode_W(predictions, priors, variances):
            """Decode width from predictions."""
            exp_term = predictions[:, 2:3] * variances[2]
            return priors[:, 2:3] * jp.exp(exp_term)

        def decode_H(predictions, priors, variances):
            """Decode height from predictions."""
            exp_term = predictions[:, 3:4] * variances[3]
            return priors[:, 3:4] * jp.exp(exp_term)

        center_x = decode_center_x(predictions, priors, variances)
        center_y = decode_center_y(predictions, priors, variances)
        W = decode_W(predictions, priors, variances)
        H = decode_H(predictions, priors, variances)

        return jp.concatenate([center_x, center_y, W, H], axis=1)

    priors_center = priors
    boxes_center = decode_center_form_boxes(
        predictions, priors_center, variances
    )
    boxes_corner = paz.boxes.to_corner_form(boxes_center)

    return jp.concatenate([boxes_corner, predictions[:, 4:]], axis=1)


def select_top_k(boxes_and_scores, top_k=200):
    boxes, scores = paz.detection.split(boxes_and_scores)
    sorted_score_args = jp.argsort(jp.squeeze(scores, axis=-1))[::-1]
    top_k_score_args = sorted_score_args[:top_k]
    return boxes_and_scores[top_k_score_args]


def to_score(boxes_and_one_hot_vectors, class_arg):
    boxes, one_hot_vectors = paz.detection.split(boxes_and_one_hot_vectors)
    class_scores = jp.expand_dims(one_hot_vectors[:, class_arg], 1)
    boxes_and_scores = paz.detection.merge(boxes, class_scores)
    return boxes_and_scores


def to_one_hot(boxes_and_scores, class_arg, num_classes):
    boxes, scores = paz.detection.split(boxes_and_scores)
    one_hot_vectors = jp.zeros((len(boxes), num_classes))
    scores = jp.squeeze(scores, axis=-1)
    one_hot_vectors = one_hot_vectors.at[:, class_arg].set(scores)
    boxes_and_one_hot_vectors = paz.detection.merge(boxes, one_hot_vectors)
    return boxes_and_one_hot_vectors


def apply_NMS(sorted_boxes_with_scores, iou_thresh=0.45, epsilon=0.01):
    top_k_boxes = paz.detection.get_boxes(sorted_boxes_with_scores)
    top_k_boxes_args = jp.arange(len(top_k_boxes))
    num_total_boxes = top_k_boxes.shape[0]

    def do_continue(state):
        suppressed_mask, top_k_box_arg = state
        in_bounds = top_k_box_arg < num_total_boxes

        def any_unprocessed_unsuppressed():
            is_suffix = top_k_boxes_args >= top_k_box_arg
            is_unsuppressed = jp.logical_not(suppressed_mask)
            unsuppressed_in_suffix = jp.logical_and(is_unsuppressed, is_suffix)
            return jp.any(unsuppressed_in_suffix)

        def do_exit():
            return False

        return jax.lax.cond(in_bounds, any_unprocessed_unsuppressed, do_exit)

    def step(state):
        suppressed_mask, top_k_box_arg = state
        is_suppressed = suppressed_mask[top_k_box_arg]

        def suppress():
            current_box = top_k_boxes[top_k_box_arg]
            ious = paz.boxes.compute_IOU(current_box, top_k_boxes)
            is_not_this_box = top_k_boxes_args != top_k_box_arg
            do_suppress = (ious > iou_thresh) & is_not_this_box
            return jp.logical_or(suppressed_mask, do_suppress)

        def do_nothing():
            return suppressed_mask

        new_suppressed_mask = jax.lax.cond(is_suppressed, do_nothing, suppress)
        return (new_suppressed_mask, top_k_box_arg + 1)

    scores = jp.squeeze(paz.detection.get_scores(sorted_boxes_with_scores), -1)
    suppressed_mask = scores < epsilon
    state = (suppressed_mask, 0)
    suppressed_mask, num_steps = jax.lax.while_loop(do_continue, step, state)
    return jp.logical_not(suppressed_mask)  # keep mask


def apply_per_class_NMS(detections, num_classes, iou_thresh, top_k, epsilon):

    def per_class_NMS(class_arg):
        class_detections = to_score(detections, class_arg)
        class_detections = select_top_k(class_detections, top_k)
        non_suppressed = apply_NMS(class_detections, iou_thresh)
        valid_scores = get_scores(class_detections) >= epsilon
        non_suppressed = jp.expand_dims(non_suppressed, axis=-1)
        class_keep_masks = jp.logical_and(valid_scores, non_suppressed)
        class_detections = to_one_hot(class_detections, class_arg, num_classes)
        return class_detections, class_keep_masks

    detections, keep_masks = jax.vmap(per_class_NMS)(jp.arange(num_classes))
    detections = detections.reshape(-1, detections.shape[-1])
    keep_masks = keep_masks.reshape(-1, 1)
    detections = jp.where(keep_masks, detections, -1.0)
    return detections


def filter_by_score(detections, threshold, invalid_value=-1):
    """Filters detections by scores."""
    scores = jp.max(paz.detection.get_scores(detections), axis=1, keepdims=True)
    return jp.where(scores >= threshold, detections, invalid_value)


def remove_class(detections, class_arg):
    """Remove a particular class from the pipeline.

    # Arguments
        class_names: List, indicating given class names.
        class_arg: Int, index of the class to be removed.
    """
    # del class_names[class_arg]
    return jp.delete(detections, 4 + class_arg, axis=1)


def denormalize(detections, H, W):
    boxes, scores = split(detections)
    boxes = paz.boxes.denormalize(boxes, H, W)
    return merge(boxes, scores)


def remove_invalid(detections_array, value=-1):
    is_invalid_row_mask = jp.all(detections_array == value, axis=1)
    is_valid_row_mask = jp.logical_not(is_invalid_row_mask)
    valid_boxes = detections_array[is_valid_row_mask]
    return valid_boxes


def to_boxes2D(boxes_and_one_hot_vectors):
    boxes, one_hot_vectors = split(boxes_and_one_hot_vectors)
    class_args = jp.argmax(one_hot_vectors, axis=1)
    scores = one_hot_vectors[class_args]
    return jp.concatenate([boxes, class_args, scores], axis=1)
