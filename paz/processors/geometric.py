import jax.numpy as jp
import types
import jax
import paz


def sample_crop_box(key, H, W, max_factor=0.3):
    """Generates random crop region with valid aspect ratio."""
    H_min = jp.maximum(1, jp.floor(max_factor * H)).astype(jp.int32)
    W_min = jp.maximum(1, jp.floor(max_factor * W)).astype(jp.int32)
    H_crop = jax.random.randint(key, (), H_min, H)
    W_crop = jax.random.randint(key, (), W_min, W)
    y_start = jax.random.randint(key, (), 0, H - H_crop)
    x_start = jax.random.randint(key, (), 0, W - W_crop)
    crop_box = jp.array([x_start, y_start, x_start + W_crop, y_start + H_crop])
    aspect_ratio = H_crop / W_crop
    is_valid_aspect_ratio = (0.5 <= aspect_ratio) & (aspect_ratio <= 2.0)
    crop_box = jp.where(is_valid_aspect_ratio, crop_box, jp.array([0, 0, 0, 0]))
    return crop_box


def are_centers_inside_crop(boxes, crop_box):
    """Computes box centers and checks inclusion in crop region."""
    x_centers, y_centers = paz.boxes.compute_centers(boxes)
    x_min_crop, y_min_crop, x_max_crop, y_max_crop = crop_box
    x_min_valid = x_min_crop < x_centers
    y_min_valid = y_min_crop < y_centers
    x_max_valid = x_centers < x_max_crop
    y_max_valid = y_centers < y_max_crop
    center_inside_crop = x_min_valid & y_min_valid & x_max_valid & y_max_valid
    return center_inside_crop


def validate_crop(boxes, crop_box, min_IOU, max_IOU):
    """Validates crop region against IoU thresholds and center inclusion."""
    IOUs = paz.boxes.compute_IOUs(jp.expand_dims(crop_box, axis=0), boxes)[0]
    centers_inside_crop_mask = are_centers_inside_crop(boxes, crop_box)
    valid_min_IOU = jp.max(IOUs) >= min_IOU
    valid_max_IOU = jp.min(IOUs) <= max_IOU
    is_valid_IOUs = jp.logical_and(valid_min_IOU, valid_max_IOU)
    has_at_least_one_box_inside_crop = jp.any(centers_inside_crop_mask)
    is_crop_valid = is_valid_IOUs & has_at_least_one_box_inside_crop
    return is_crop_valid, centers_inside_crop_mask


def validate_crop_box(boxes, crop_box, min_IOU, max_IOU, max_attempts):
    """Validates if acceptable crop region found within attempt limit."""

    def do_continue(state):
        """Determines if crop search should continue based on attempt count."""
        trials, is_valid, _ = state
        return (trials < max_attempts) & (~is_valid)

    def body(state):
        """Updates search state during crop validation attempts."""
        trials, _, _ = state
        is_valid, center_mask = validate_crop(boxes, crop_box, min_IOU, max_IOU)
        return (trials + 1, is_valid, center_mask)

    is_valid, center_mask = validate_crop(boxes, crop_box, min_IOU, max_IOU)
    state = (0, is_valid, center_mask)
    trial_count, is_valid, _ = jax.lax.while_loop(do_continue, body, state)
    return (trial_count < max_attempts) and is_valid


def compute_mask(boxes, crop_box):
    """Build mask retaining only boxes overlapping with crop region."""
    x_min, y_min, x_max, y_max = paz.boxes.split(boxes, False)
    x_min_crop, y_min_crop, x_max_crop, y_max_crop = crop_box
    x_min_valid = x_min < x_max_crop
    y_min_valid = y_min < y_max_crop
    x_max_valid = x_max > x_min_crop
    y_max_valid = y_max > y_min_crop
    mask = x_min_valid & y_min_valid & x_max_valid & y_max_valid
    return mask


def adjust_boxes(boxes, crop_box):
    """Adjusts box coordinates relative to the crop region's top-left corner."""
    top_left = jp.maximum(boxes[:, :2], crop_box[:2]) - crop_box[:2]
    bottom_right = jp.minimum(boxes[:, 2:], crop_box[2:]) - crop_box[:2]
    return jp.concatenate([top_left, bottom_right], axis=1)


def adjust_boxes_and_labels(boxes, labels, crop_box):
    """Filters and adjusts boxes to fit crop region, maintaining labels."""
    mask = compute_mask(boxes, crop_box)
    valid_boxes, valid_labels = boxes[mask], labels[mask]
    adjusted_boxes = adjust_boxes(valid_boxes, crop_box)
    return jp.hstack([adjusted_boxes, valid_labels])


def attempt_crop(
    key, image, labels, boxes, max_attempts, IoU_range, detections
):
    """Attempts to find a valid crop region that satisfies IOU constraints."""
    H, W = paz.image.get_size(image)
    min_IOU, max_IOU = IoU_range
    crop_box = sample_crop_box(key, H, W, max_factor=0.3)
    if not jp.all(crop_box == 0):
        if validate_crop_box(boxes, crop_box, min_IOU, max_IOU, max_attempts):
            image = paz.image.crop(image, crop_box)
            detections = adjust_boxes_and_labels(boxes, labels, crop_box)
    return image, detections


def do_crop(key, probability):
    """Determines whether to apply a random crop based."""
    random_value = jax.random.uniform(key, shape=())
    return probability >= random_value


def random_sample_crop(
    key, image, detections, probability, max_trials, IOU_thresholds
):
    """Performs random sample cropping with IoU validation for object detection."""
    apply_crop = do_crop(key, probability)
    boxes, labels = paz.detection.split(detections)
    mode = jax.random.randint(key, (), 0, len(IOU_thresholds))
    IOU_threshold = IOU_thresholds[mode]

    if not ((not apply_crop) or (IOU_threshold is None)):
        image, detections = attempt_crop(
            key, image, labels, boxes, max_trials, IOU_threshold, detections
        )
    return image, detections


def RandomSampleCrop(probability=0.5, max_trials=50, seed=0):
    """randomly crop images while ensuring boxes validity based on IOU."""

    def execute_crop_and_update(image, boxes, processor):
        """Executes cropping operation and updates processor state."""
        key, probability, max_trials, IoU_thresholds = (
            processor.key,
            processor.probability,
            processor.max_trials,
            processor.jaccard_min_max,
        )
        cropped_image, adjusted_boxes = random_sample_crop(
            key, image, boxes, probability, max_trials, IoU_thresholds
        )
        return cropped_image, adjusted_boxes

    processor = types.SimpleNamespace(
        probability=probability,
        max_trials=max_trials,
        jaccard_min_max=(
            None,
            (0.1, jp.inf),
            (0.3, jp.inf),
            (0.7, jp.inf),
            (0.9, jp.inf),
            (-jp.inf, jp.inf),
        ),
        key=jax.random.key(seed),
    )

    processor.call = lambda image, boxes: execute_crop_and_update(
        image, boxes, processor
    )
    return processor
