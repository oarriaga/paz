import jax.numpy as jp
import numpy as np
import paz
import jax


def random_flip_left_right(image, detections):
    boxes, class_args = split(detections)
    boxes = paz.boxes.flip_left_right(boxes, image.shape[1])
    image = paz.image.flip_left_right(image)
    detections = merge(boxes, class_args)
    return image, detections


def resize_boxes_with_aspect_ratio(image, detections, H, W):
    detections = jp.array(detections)  # input could be a list
    boxes, class_args = split(detections)
    H_now, W_now = paz.image.get_size(image)
    boxes = paz.boxes.resize_with_aspect_ratio(boxes, H_now, W_now, H, W)
    detections = merge(boxes, class_args)
    return detections


def resize_with_aspect_ratio(image, detections, H, W):
    H_now, W_now = paz.image.get_size(image)
    # detections = jp.array(detections)  # input could be a list
    boxes, class_args = split(detections)
    boxes = paz.boxes.resize_with_aspect_ratio(boxes, H_now, W_now, H, W)
    image = paz.image.resize_with_aspect_ratio(image, H, W)
    detections = merge(boxes, class_args)
    return image, detections


def resize(image, detections, H, W):
    H_now, W_now = paz.image.get_size(image)
    # detections = jp.array(detections)  # input could be a list
    boxes, class_args = split(detections)
    image = paz.image.resize_opencv(image, (H, W))
    boxes = paz.boxes.resize(boxes, H_now, W_now, H, W)
    detections = merge(boxes, class_args)
    return image, detections


def split(detections):
    boxes = detections[:, :4]
    class_args = detections[:, 4:]
    return boxes, class_args


def get_boxes(detections):
    return detections[:, :4]


def get_scores(detections):
    return detections[:, 4:]


def build_invalid(shape=(1, 5), value=-1):
    return jp.full(shape, value)


def merge(boxes, class_args):
    return jp.concatenate([boxes, class_args], axis=-1)  # -1 to support batches


def to_one_hot(boxes_and_class_args, num_classes):
    boxes, class_args = split(boxes_and_class_args)
    class_args = jp.squeeze(class_args, axis=-1)
    classes = paz.classes.to_one_hot(class_args, num_classes)
    return merge(boxes, classes)


def pad(detections, size, mode="constant", constant_value=-1):
    # Ensure we don't exceed the target size from the start.
    detections = detections[:size]

    # Calculate the required padding.
    padding_needed = size - detections.shape[0]
    padding_config = ((0, padding_needed), (0, 0))

    if mode == "constant":
        return jp.pad(
            detections,
            padding_config,
            mode="constant",
            constant_values=constant_value,
        )
    elif mode == "edge":
        # The 'edge' mode in jp.pad automatically repeats the last row for us.
        return jp.pad(detections, padding_config, mode="edge")
    else:
        raise ValueError("Mode must be either 'constant' or 'edge'")


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

    boxes_corner, class_args = split(matched)
    boxes_center = paz.boxes.to_center_form(boxes_corner)
    x_encoded, y_encoded = encode_centers(boxes_center, priors, variances)
    W_encoded, H_encoded = encode_sizes(boxes_center, priors, variances)
    encooded_boxes = [x_encoded, y_encoded, W_encoded, H_encoded, class_args]
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


def score_to_one_hot(boxes_and_scores, class_arg, num_classes):
    boxes, scores = paz.detection.split(boxes_and_scores)
    one_hot_vectors = jp.zeros((len(boxes), num_classes))
    scores = jp.squeeze(scores, axis=-1)
    one_hot_vectors = one_hot_vectors.at[:, class_arg].set(scores)
    boxes_and_one_hot_vectors = paz.detection.merge(boxes, one_hot_vectors)
    return boxes_and_one_hot_vectors


def filter_by_score(boxes_score_label, threshold, invalid_value=-1):
    """Filters detections by scores."""
    # scores = jp.max(paz.detection.get_scores(detections), axis=1, keepdims=True)
    positive_mask = boxes_score_label[:, 4:5] >= threshold
    return jp.where(positive_mask, boxes_score_label, invalid_value)


def remove_class(detections, class_arg):
    """Remove a particular class from the pipeline.

    # Arguments
        class_names: List, indicating given class names.
        class_arg: Int, index of the class to be removed.
    """
    return jp.delete(detections, 4 + class_arg, axis=1)


def denormalize(detections, H, W):
    boxes, scores = split(detections)
    boxes = paz.boxes.denormalize(boxes, H, W)
    return merge(boxes, scores)


def clip(detections, H, W):
    boxes, scores = split(detections)
    boxes = paz.boxes.clip(boxes, H, W)
    return merge(boxes, scores)


def normalize(detections, H, W):
    boxes, scores = split(detections)
    boxes = paz.boxes.normalize(boxes, H, W)
    return merge(boxes, scores)


def build_negative_mask(detections, value=-1):
    is_invalid_row_mask = jp.any(detections < 0.0, axis=1)
    return is_invalid_row_mask


def build_positive_mask(detections, value=-1):
    is_invalid_row_mask = jp.any(detections < 0.0, axis=1)
    is_valid_row_mask = jp.logical_not(is_invalid_row_mask)
    return is_valid_row_mask


def remove_invalid(detections, value=-1):
    is_invalid_row_mask = jp.any(detections < 0.0, axis=1)
    is_valid_row_mask = jp.logical_not(is_invalid_row_mask)
    valid_boxes = detections[is_valid_row_mask]
    return valid_boxes


def to_boxes2D(detections):
    boxes = detections[:, 0:4]
    score = detections[:, 4]
    label = detections[:, 5]
    # boxes, scores = paz.detection.split(detections)
    # labels = jp.argmax(scores, axis=-1)
    # scores = scores[jp.arange(len(scores)), labels]
    return boxes.astype("int32"), label.astype("int32"), score


def apply_NMS(detections, iou_thresh, top_k):
    detections = paz.detection.select_top_k(detections, top_k)
    top_k_boxes = paz.detection.get_boxes(detections)
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

        return jax.lax.cond(
            in_bounds, any_unprocessed_unsuppressed, lambda: False
        )

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

    scores = jp.squeeze(paz.detection.get_scores(detections), -1)
    state = (scores < 0.01, 0)
    state = jax.lax.while_loop(do_continue, step, state)
    suppressed_mask, num_steps = state
    keep_mask = jp.expand_dims(jp.logical_not(suppressed_mask), axis=-1)
    return jp.where(keep_mask, detections, -1)


def apply_per_class_NMS(detections, num_classes, iou_thresh=0.45, top_k=200):

    def apply_NMS(detections, class_arg):
        class_detections = paz.detection.to_score(detections, class_arg)
        class_detections = paz.detection.select_top_k(class_detections, top_k)
        top_k_boxes = paz.detection.get_boxes(class_detections)
        top_k_boxes_args = jp.arange(len(top_k_boxes))
        num_total_boxes = top_k_boxes.shape[0]

        def do_continue(state):
            suppressed_mask, top_k_box_arg = state
            in_bounds = top_k_box_arg < num_total_boxes

            def any_unprocessed_unsuppressed():
                is_suffix = top_k_boxes_args >= top_k_box_arg
                is_unsuppressed = jp.logical_not(suppressed_mask)
                unsuppressed_in_suffix = jp.logical_and(
                    is_unsuppressed, is_suffix
                )
                return jp.any(unsuppressed_in_suffix)

            return jax.lax.cond(
                in_bounds, any_unprocessed_unsuppressed, lambda: False
            )

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

            new_suppressed_mask = jax.lax.cond(
                is_suppressed, do_nothing, suppress
            )
            return (new_suppressed_mask, top_k_box_arg + 1)

        scores = jp.squeeze(paz.detection.get_scores(class_detections), -1)
        state = (scores < 0.01, 0)
        state = jax.lax.while_loop(do_continue, step, state)
        suppressed_mask, num_steps = state
        keep_mask = jp.expand_dims(jp.logical_not(suppressed_mask), axis=-1)
        return jp.where(keep_mask, class_detections, -1)

    suppressed = jax.vmap(paz.partial(apply_NMS, detections))(
        jp.arange(num_classes)
    )
    suppressed = suppressed.reshape(-1, 5)
    labels = jp.repeat(jp.arange(num_classes), top_k).astype("float32")
    return jp.concatenate([suppressed, jp.expand_dims(labels, -1)], axis=-1)


def original_match(boxes_with_class_arg, prior_boxes, IOU_threshold=0.5):
    """Matches each prior box with a ground truth box
    (box from `boxes_with_class_arg`). It then selects which matched box will be
    considered positive e.g. iou > .5 and returns for each prior box a ground
    truth box that is either positive (with a class argument different
    than 0) or negative.

    # Arguments
        boxes: Numpy array of shape `(num_ground_truh_boxes, 4 + 1)`,
            where the first the first four coordinates correspond to
            box coordinates and the last coordinates is the class
            argument. This boxes should be the ground truth boxes.
        prior_boxes: Numpy array of shape `(num_prior_boxes, 4)`.
            where the four coordinates are in center form coordinates.
        iou_threshold: Float between [0, 1]. Intersection over union
            used to determine which box is considered a positive box.

    # Returns
        Array of shape `(num_prior_boxes, 4 + 1)`.
            where the first the first four coordinates correspond to point
            form box coordinates and the last coordinates is the class
            argument.
    """

    # def mark_best_match(per_prior_best_IOU, per_box_best_prior_arg):
    #     # The prior boxes that are the best match for each box are marked.
    #     # They are marked by setting an IOU larger (2) than the maxium (1).
    #     # the best prior box match of box_0 is per_box_best_prior_arg[0]
    #     # the best prior box match of box_1 is per_box_best_prior_arg[1]
    #     # ...
    #     return per_prior_best_IOU.at[per_box_best_prior_arg].set(2.0)
    positive_mask = build_positive_mask(boxes_with_class_arg)

    def mark_best_match(per_prior_best_IOU, per_box_best_prior):
        # The prior boxes that are the best match for each box are marked.
        # They are marked by setting an IOU larger (2) than the maxium (1).
        # the best prior box match of box_0 is per_box_best_prior_arg[0]
        # the best prior box match of box_1 is per_box_best_prior_arg[1]
        # ...
        def mark_match(per_prior_best_IOU, box_arg):
            prior_arg = per_box_best_prior[box_arg]
            is_box_valid = positive_mask[box_arg]
            # it_overlaps = best_IOU >= IOU_threshold
            best_IOU = per_prior_best_IOU[prior_arg]
            best_IOU = jp.where(is_box_valid, 2.0, best_IOU)
            return per_prior_best_IOU.at[prior_arg].set(best_IOU), None

        box_args = jp.arange(len(boxes_with_class_arg))
        per_prior_best_IOU, _ = jax.lax.scan(
            mark_match, per_prior_best_IOU, box_args
        )
        return per_prior_best_IOU

    def select_for_each_prior_box_a_box(boxes, per_prior_best_box):
        # Each prior box is assigned a ground truth box.
        assigned_boxes = boxes[per_prior_best_box]
        return assigned_boxes

    def force_match(per_prior_best_box, per_box_best_prior):
        # Ensures that every ground truth box is matched with at least one prior
        # box. Specifically, the prior box with which it has the highest IoU.
        for box_arg, prior_arg in enumerate(per_box_best_prior):
            is_valid_box = positive_mask[box_arg]
            box_arg = jp.where(
                is_valid_box, box_arg, per_prior_best_box[prior_arg]
            )
            per_prior_best_box = per_prior_best_box.at[prior_arg].set(box_arg)
        return per_prior_best_box

    def label_negative_boxes(assigned_boxes, per_prior_best_IOU):
        is_low_IOU_match = per_prior_best_IOU < IOU_threshold
        class_args = assigned_boxes[:, 4]
        class_args = jp.where(is_low_IOU_match, 0.0, class_args)
        return assigned_boxes.at[:, 4].set(class_args)

    prior_boxes = paz.boxes.to_corner_form(prior_boxes)
    IOUs = paz.boxes.compute_IOUs(boxes_with_class_arg, prior_boxes)
    per_box_best_prior = jp.argmax(IOUs, axis=1)  # (boxes,)
    per_prior_best_box = jp.argmax(IOUs, axis=0)  # (prior_boxes,)
    per_prior_best_IOU = jp.max(IOUs, axis=0)  # (prior_boxes,)
    per_prior_best_IOU = mark_best_match(per_prior_best_IOU, per_box_best_prior)
    assign_args = (per_prior_best_box, per_box_best_prior)
    per_prior_best_box = force_match(*assign_args)
    selected_boxes = select_for_each_prior_box_a_box(
        boxes_with_class_arg, per_prior_best_box
    )
    selected_boxes = label_negative_boxes(selected_boxes, per_prior_best_IOU)
    return selected_boxes


def match(boxes_with_class_arg, prior_boxes, IOU_threshold=0.5):
    prior_boxes = paz.boxes.to_corner_form(prior_boxes)
    IOUs = paz.boxes.compute_IOUs(boxes_with_class_arg, prior_boxes)
    per_prior_best_box = jp.argmax(IOUs, axis=0)
    per_prior_best_IOU = jp.max(IOUs, axis=0)
    per_box_best_prior = jp.argmax(IOUs, axis=1)
    is_valid_box_mask = boxes_with_class_arg[:, 0] >= 0.0

    def body(iou_carry, box_arg):
        # Get the prior that best matches the current ground truth box `box_arg`
        prior_to_update = per_box_best_prior[box_arg]
        is_box_valid = is_valid_box_mask[box_arg]
        # Conditionally create the new IOU value. If the box is valid, the
        # new IOU is 2.0. If not, the new IOU is the original IOU (a no-op).
        new_iou = jp.where(is_box_valid, 2.0, iou_carry[prior_to_update])
        # Update the IOU array with the new value. Because this is a scan,
        # if multiple boxes map to the same prior, the last one wins.
        return iou_carry.at[prior_to_update].set(new_iou), None

    # Run the scan, starting with the original `per_prior_best_IOU`
    box_args = jp.arange(len(boxes_with_class_arg))
    per_prior_best_IOU, _ = jax.lax.scan(body, per_prior_best_IOU, box_args)

    selected_boxes = boxes_with_class_arg[per_prior_best_box]
    # 4. Label negative boxes: set the class of any box with an IOU below
    #    the threshold to 0 (background), using our final updated IOU array.
    is_low_IOU_match = per_prior_best_IOU < IOU_threshold
    class_args = selected_boxes[:, 4]
    class_args = jp.where(is_low_IOU_match, 0.0, class_args)
    selected_boxes = selected_boxes.at[:, 4].set(class_args)
    return selected_boxes


def match_np(boxes, prior_boxes, IOU_threshold=0.5):
    """Matches each prior box with a ground truth box (box from `boxes`).
    It then selects which matched box will be considered positive e.g. iou > .5
    and returns for each prior box a ground truth box that is either positive
    (with a class argument different than 0) or negative.

    # Arguments
        boxes: Numpy array of shape `(num_ground_truh_boxes, 4 + 1)`,
            where the first the first four coordinates correspond to
            box coordinates and the last coordinates is the class
            argument. This boxes should be the ground truth boxes.
        prior_boxes: Numpy array of shape `(num_prior_boxes, 4)`.
            where the four coordinates are in center form coordinates.
        iou_threshold: Float between [0, 1]. Intersection over union
            used to determine which box is considered a positive box.

    # Returns
        numpy array of shape `(num_prior_boxes, 4 + 1)`.
            where the first the first four coordinates correspond to point
            form box coordinates and the last coordinates is the class
            argument.
    """

    def compute_IOUs(boxes_A, boxes_B):
        """Computes intersection over union (IOU) between `boxes_A` and `boxes_B`.

        For each box (rows `boxes_A`) it computes the IOU to all `boxes_B`.

        # Arguments
            boxes_A: Numpy array with shape `(num_boxes_A, 4)` in corner form.
            boxes_B: Numpy array with shape `(num_boxes_B, 4)` in corner form.

        # Returns
            Numpy array of shape `(num_boxes_A, num_boxes_B)`.
        """
        xy_min = np.maximum(boxes_A[:, None, 0:2], boxes_B[:, 0:2])
        xy_max = np.minimum(boxes_A[:, None, 2:4], boxes_B[:, 2:4])
        intersection = np.maximum(0.0, xy_max - xy_min)
        intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
        areas_A = (boxes_A[:, 2] - boxes_A[:, 0]) * (
            boxes_A[:, 3] - boxes_A[:, 1]
        )
        areas_B = (boxes_B[:, 2] - boxes_B[:, 0]) * (
            boxes_B[:, 3] - boxes_B[:, 1]
        )
        # broadcasting for outer sum i.e. a sum of all possible combinations
        union_area = (areas_A[:, np.newaxis] + areas_B) - intersection_area
        union_area = np.maximum(union_area, 1e-8)
        return np.clip(intersection_area / union_area, 0.0, 1.0)

    def split(boxes, keepdims=True, axis=1):
        """Split boxes into x_min, y_min, x_max, y_max components."""
        coordinates = np.split(boxes, 4, axis=axis)
        if not keepdims:
            coordinates = tuple(
                np.squeeze(column, axis=-1) for column in coordinates
            )

        return coordinates

    def to_corner_form(boxes):
        """Convert bounding boxes from center to corner form.

        # Arguments:
            Boxes: Array of boxes in center format ``[center_x, center_y, W, H]``.

        # Returns:
            Boxes in corner format ``[x_min, y_min, x_max, y_max]``.
        """
        center_x, center_y, W, H = split(boxes)
        x_min = center_x - (W / 2.0)
        x_max = center_x + (W / 2.0)
        y_min = center_y - (H / 2.0)
        y_max = center_y + (H / 2.0)
        return np.concatenate([x_min, y_min, x_max, y_max], axis=1)

    ious = compute_IOUs(boxes, to_corner_form(np.float32(prior_boxes)))
    per_prior_which_box_iou = np.max(ious, axis=0)
    per_prior_which_box_arg = np.argmax(ious, 0)

    #  overwriting per_prior_which_box_arg if they are the best prior box
    per_box_which_prior_arg = np.argmax(ious, 1)
    per_prior_which_box_iou[per_box_which_prior_arg] = 2
    for box_arg in range(len(per_box_which_prior_arg)):
        best_prior_box_arg = per_box_which_prior_arg[box_arg]
        per_prior_which_box_arg[best_prior_box_arg] = box_arg

    matches = boxes[per_prior_which_box_arg]
    matches[per_prior_which_box_iou < IOU_threshold, 4] = 0
    return matches


def fit_to_crop(detections, crop_box):
    boxes, class_args = split(detections)
    boxes = paz.boxes.fit_to_crop(boxes, crop_box)
    return merge(boxes, class_args)


def translate(detections, x_offset, y_offset):
    """Translates the center of a bounding box (xywh format)."""
    boxes, class_args = split(detections)
    x_center, y_center, W, H = paz.boxes.split(paz.boxes.xyxy_to_xywh(boxes))
    x_new_center = x_center + x_offset
    y_new_center = y_center + y_offset
    boxes = paz.boxes.merge(x_new_center, y_new_center, W, H)
    boxes = paz.boxes.xywh_to_xyxy(boxes)
    return merge(boxes, class_args)


# --- SSD-style detection augmentation (fixed-shape, jit/vmap-friendly) ---
# All ops keep a fixed image size and a fixed number of boxes: `detections`
# is `(num_boxes, 5)` with normalized corners + class, padded rows set to -1.
# This mirrors the legacy master pipeline (photometric -> expand -> sample
# crop -> flip) but stays jittable, so a batch runs as one jit(vmap(...)).

CROP_MODE_MIN_IOU = jp.array([jp.nan, 0.1, 0.3, 0.7, 0.9, -jp.inf])


def augment_detection(key, image, detections, mean):
    """Applies the full SSD training augmentation to a single sample."""
    keys = jax.random.split(key, 4)
    image = random_photometric(keys[0], image)
    image, detections = random_expand(keys[1], image, detections, mean)
    image, detections = random_sample_crop(keys[2], image, detections)
    image, detections = random_flip(keys[3], image, detections)
    return image, detections


def random_photometric(key, image):
    """Contrast, brightness, saturation and hue, each with probability 0.5."""
    keys = jax.random.split(key, 4)
    image = maybe_apply(keys[0], adjust_contrast, image)
    image = maybe_apply(keys[1], adjust_brightness, image)
    image = maybe_apply(keys[2], adjust_saturation, image)
    image = maybe_apply(keys[3], adjust_hue, image)
    return image


def random_expand(key, image, detections, mean, max_ratio=2.0, probability=0.5):
    """Zooms out up to `max_ratio`, filling new pixels with `mean`."""
    H, W = paz.image.get_size(image)
    coin_key, ratio_key, x_key, y_key = jax.random.split(key, 4)
    ratio = jax.random.uniform(ratio_key, (), minval=1.0, maxval=max_ratio)
    inverse = 1.0 / ratio
    left = jax.random.uniform(x_key, (), maxval=1.0 - inverse)
    top = jax.random.uniform(y_key, (), maxval=1.0 - inverse)
    scale = jp.array([inverse, inverse])
    translation = jp.array([top * H, left * W])
    fill = jp.asarray(mean, jp.float32)
    canvas_args = image, (H, W), scale, translation, fill
    expanded = paz.image.place_in_canvas(*canvas_args)
    boxes, class_args = split(detections)
    moved = boxes * inverse + jp.array([left, top, left, top])
    moved = keep_valid(merge(moved, class_args), detections)
    apply = jax.random.uniform(coin_key, ()) < probability
    image = jp.where(apply, expanded, paz.cast(image, jp.float32))
    detections = jp.where(apply, moved, detections)
    return image, detections


def random_sample_crop(key, image, detections, max_trials=50):
    """Crops a window meeting a random IoU mode, then resizes it back."""
    H, W = paz.image.get_size(image)
    mode_key, loop_key = jax.random.split(key)
    mode = jax.random.randint(mode_key, (), 0, CROP_MODE_MIN_IOU.shape[0])
    min_iou = CROP_MODE_MIN_IOU[mode]
    search_args = loop_key, detections, min_iou, max_trials
    window, found = search_crop_window(*search_args)
    cropped_image, cropped = apply_crop(image, detections, window)
    do_crop = (mode > 0) & found
    image = jp.where(do_crop, cropped_image, paz.cast(image, jp.float32))
    detections = jp.where(do_crop, cropped, detections)
    return image, detections


def random_flip(key, image, detections, probability=0.5):
    """Mirrors the image and boxes left-right in normalized coordinates."""
    boxes, class_args = split(detections)
    x_min, y_min, x_max, y_max = paz.boxes.split(boxes)
    flipped_boxes = paz.boxes.merge(1.0 - x_max, y_min, 1.0 - x_min, y_max)
    flipped = keep_valid(merge(flipped_boxes, class_args), detections)
    flipped_image = paz.cast(paz.image.flip_left_right(image), jp.float32)
    apply = jax.random.uniform(key, ()) < probability
    image = jp.where(apply, flipped_image, paz.cast(image, jp.float32))
    detections = jp.where(apply, flipped, detections)
    return image, detections


def search_crop_window(key, detections, min_iou, max_trials):
    """Rejection-samples a crop window satisfying the IoU mode and aspect."""
    boxes = get_boxes(detections)
    valid = detections[:, 4] >= 0.0

    def condition(state):
        trials, _, _, found = state
        return (trials < max_trials) & jp.logical_not(found)

    def body(state):
        trials, key, window, found = state
        key, window_key = jax.random.split(key)
        candidate = sample_window(window_key)
        is_valid = is_window_valid(candidate, boxes, valid, min_iou)
        window = jp.where(is_valid, candidate, window)
        return trials + 1, key, window, found | is_valid

    identity = jp.array([0.0, 0.0, 1.0, 1.0])
    state = (0, key, identity, jp.array(False))
    _, _, window, found = jax.lax.while_loop(condition, body, state)
    return window, found


def sample_window(key):
    """Samples a crop rectangle with side lengths in [0.3, 1.0]."""
    w_key, h_key, x_key, y_key = jax.random.split(key, 4)
    W = jax.random.uniform(w_key, (), minval=0.3, maxval=1.0)
    H = jax.random.uniform(h_key, (), minval=0.3, maxval=1.0)
    x_min = jax.random.uniform(x_key, (), maxval=1.0 - W)
    y_min = jax.random.uniform(y_key, (), maxval=1.0 - H)
    return jp.array([x_min, y_min, x_min + W, y_min + H])


def is_window_valid(window, boxes, valid, min_iou):
    """A window is valid with a good aspect, IoU and one enclosed center."""
    W = window[2] - window[0]
    H = window[3] - window[1]
    aspect = H / W
    good_aspect = (aspect >= 0.5) & (aspect <= 2.0)
    ious = paz.boxes.compute_IOUs(window[None], boxes)[0]
    good_iou = jp.max(jp.where(valid, ious, -jp.inf)) >= min_iou
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    inside = (window[0] < centers[:, 0]) & (centers[:, 0] < window[2])
    inside = inside & (window[1] < centers[:, 1]) & (centers[:, 1] < window[3])
    has_center = jp.any(inside & valid)
    return good_aspect & good_iou & has_center


def apply_crop(image, detections, window):
    """Resizes `window` to the full frame and remaps the boxes into it."""
    H, W = paz.image.get_size(image)
    x_min, y_min, x_max, y_max = window
    width, height = x_max - x_min, y_max - y_min
    scale = jp.array([1.0 / height, 1.0 / width])
    translation = jp.array([-y_min * H / height, -x_min * W / width])
    fill = jp.zeros(image.shape[-1])
    canvas_args = image, (H, W), scale, translation, fill
    cropped_image = paz.image.place_in_canvas(*canvas_args)
    boxes, class_args = split(detections)
    lower = jp.clip(boxes[:, :2], window[:2], window[2:])
    upper = jp.clip(boxes[:, 2:], window[:2], window[2:])
    remapped = jp.concatenate([lower, upper], axis=1) - jp.tile(window[:2], 2)
    remapped = remapped / jp.array([width, height, width, height])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    inside = (window[0] < centers[:, 0]) & (centers[:, 0] < window[2])
    inside = inside & (window[1] < centers[:, 1]) & (centers[:, 1] < window[3])
    keep = (inside & (detections[:, 4] >= 0.0))[:, None]
    cropped = jp.where(keep, merge(remapped, class_args), -1.0)
    return cropped_image, cropped


def maybe_apply(key, function, image, probability=0.5):
    """Applies a keyed photometric `function` with the given probability."""
    coin_key, op_key = jax.random.split(key)
    apply = jax.random.uniform(coin_key, ()) < probability
    return jp.where(apply, function(op_key, image), image)


def keep_valid(new_detections, reference):
    """Resets rows that were padding in `reference` back to -1."""
    valid = reference[:, 4:5] >= 0.0
    return jp.where(valid, new_detections, -1.0)


def adjust_contrast(key, image):
    return paz.image.random_contrast(key, image)


def adjust_brightness(key, image):
    return paz.image.random_brightness(key, image)


def adjust_saturation(key, image):
    return paz.image.random_saturation(key, image, 0.7, 1.5)


def adjust_hue(key, image):
    return paz.image.random_hue(key, image, 0.05)
