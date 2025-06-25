import jax.numpy as jp
import paz
import jax


def crop(image, detections, crop_box):
    crop_box = jp.squeeze(crop_box, axis=0)
    image = paz.image.crop(image, crop_box)
    detections = paz.detection.fit_to_crop(detections, crop_box)
    x_origin, y_origin, _, _ = crop_box
    detections = paz.detection.translate(detections, -x_origin, -y_origin)
    return image, detections


def random_crop(
    key,
    image,
    detections,
    probability=0.83,
    max_trials=50,
    aspect_ratio_range=(0.5, 2.0),
    size_scale_range=(0.3, 0.3),
):

    def validate_aspect_ratio(crop_box, min_aspect_ratio, max_aspect_ratio):
        aspect_ratio = paz.boxes.compute_aspect_ratios(crop_box, keepdims=False)
        aspect_ratio = jp.squeeze(aspect_ratio)
        is_valid_min_aspect_ratio = aspect_ratio >= min_aspect_ratio
        is_valid_max_aspect_ratio = aspect_ratio <= max_aspect_ratio
        return is_valid_min_aspect_ratio & is_valid_max_aspect_ratio

    def validate_IOU(boxes, crop_box, min_IOU, max_IOU):
        IOUs = paz.boxes.compute_IOUs(crop_box, boxes)[0]
        is_valid_min_IOU = IOUs.max() >= min_IOU
        is_valid_max_IOU = IOUs.min() <= max_IOU
        return is_valid_min_IOU & is_valid_max_IOU

    def compute_centers_mask(boxes, crop_box):
        x_min, y_min, x_max, y_max = paz.boxes.split(crop_box, False)
        x_centers, y_centers = paz.boxes.compute_centers(boxes)
        centers_above_x_min = x_min < x_centers
        centers_above_y_min = y_min < y_centers
        centers_below_x_max = x_max > x_centers
        centers_below_y_max = y_max > y_centers
        centers_within_x_crop = centers_above_x_min & centers_below_x_max
        centers_within_y_crop = centers_above_y_min & centers_below_y_max
        centers_within_crop_mask = centers_within_x_crop & centers_within_y_crop
        return centers_within_crop_mask

    def validate_centers(boxes, crop_box):
        centers_within_crop = compute_centers_mask(boxes, crop_box)
        at_least_one_center_within_crop = centers_within_crop.any()
        return at_least_one_center_within_crop

    def do_continue(trial, crop_box, boxes, max_trials, ratio_range, IOU_range):
        valid_ratio = validate_aspect_ratio(crop_box, *ratio_range)
        valid_IOU = validate_IOU(boxes, crop_box, *IOU_range)
        valid_centers = validate_centers(boxes, crop_box)
        valid_crop = valid_ratio & valid_IOU & valid_centers
        return jp.logical_not(valid_crop) & (trial <= max_trials)

    def sample_IOU_range(key):
        inf = jp.inf
        ranges = ((0.1, inf), (0.3, inf), (0.7, inf), (0.9, inf), (-inf, inf))
        mode = jax.random.randint(key, (), 0, len(ranges))
        return ranges[mode]

    def condition(state, boxes, max_trials, aspect_ratio_range, IOU_range):
        key, trial_arg, crop_box = state
        args = (max_trials, aspect_ratio_range, IOU_range)
        return do_continue(trial_arg, crop_box, boxes, *args)

    def initialize_state(key, H, W, size_scale_range):
        key_0, key_1 = jax.random.split(key)
        crop_box = paz.boxes.sample(key_1, H, W, *size_scale_range, 1)
        return (key_0, 0, crop_box)

    def body(state, H, W, size_scale_range):
        key, trial_arg, crop_box = state
        key_now, key_new = jax.random.split(key)
        crop_box = paz.boxes.sample(key_now, H, W, *size_scale_range, 1)
        return (key_new, trial_arg + 1, crop_box)

    keys = jax.random.split(key, 4)
    if jax.random.uniform(keys[0], shape=()) >= probability:
        return image, detections

    _condition = paz.lock(
        condition,
        paz.detection.get_boxes(detections),
        max_trials,
        aspect_ratio_range,
        sample_IOU_range(keys[1]),
    )
    H, W = paz.image.get_size(image)
    _body = paz.lock(body, H, W, size_scale_range)
    state = initialize_state(keys[2], H, W, size_scale_range)
    _, num_trials, crop_box = jax.lax.while_loop(_condition, _body, state)
    if num_trials < max_trials:
        image, detections = crop(image, detections, crop_box)
    return image, detections
