from jax import lax
import jax.numpy as jp
from paz.backend.boxes import compute_IOUs
import types
import jax


def should_apply_crop(key, probability):
    """Determines whether to apply a random crop based on a probability threshold.
    Args:
        key (random key): Random number generator key.
        probability (float): Probability threshold between 0 and 1.
    Returns:
        bool: True if crop should be applied, False otherwise.
    """
    random_value = jax.random.uniform(key, shape=())
    return probability >= random_value


def filter_boxes_and_labels(boxes, labels, crop_region):
    """Filters boxes and labels retaining only those overlapping with crop region.
    Args:
        boxes (array): Array of shape (N, 4) in [x_min, y_min, x_max, y_max] format.
        labels (array): Array of shape (N,) containing box labels.
        crop_region (array): Array of shape (4,) defining [x_min, y_min, x_max, y_max] crop.
    Returns:
        tuple: Filtered (boxes, labels) arrays.
    """
    crop_xmin, crop_ymin, crop_xmax, crop_ymax = crop_region
    x_min, y_min, x_max, y_max = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    mask = (x_min < crop_xmax) & (y_min < crop_ymax) & (x_max > crop_xmin) & (y_max > crop_ymin)

    return boxes[mask], labels[mask]


def adjust_boxes(boxes, crop_region):
    """Adjusts box coordinates relative to the crop region's top-left corner.
    Args:
        boxes (array): Array of shape (M, 4) in original coordinates.
        crop_region (array): Array of shape (4,) defining crop boundaries.
    Returns:
        array: Adjusted boxes of shape (M, 4) in crop-relative coordinates.
    """
    top_left = jp.maximum(boxes[:, :2], crop_region[:2]) - crop_region[:2]
    bottom_right = jp.minimum(boxes[:, 2:], crop_region[2:]) - crop_region[:2]
    return jp.concatenate([top_left, bottom_right], axis=1)


def adjust_boxes_and_labels(boxes, labels, crop_region):
    """Filters and adjusts boxes to fit within crop region, maintaining labels.
    Args:
        boxes (array): Original boxes array of shape (N, 4).
        labels (array): Labels array of shape (N,).
        crop_region (array): Crop boundaries array of shape (4,).
    Returns:
        array: Concatenated array of adjusted boxes and labels (M, 5).
    """
    valid_boxes, valid_labels = filter_boxes_and_labels(boxes, labels, crop_region)
    adjusted_boxes = adjust_boxes(valid_boxes, crop_region)
    return jp.hstack([adjusted_boxes, valid_labels])


def compute_crop_limits(original_dim):
    """Computes minimum and maximum allowable crop dimensions.
    Args:
        original_dim (int): Original dimension (width/height) of the image.
    Returns:
        tuple: (minimum_dimension, maximum_dimension) as integers.
    """
    min_dim = jp.maximum(1, jp.floor(0.3 * original_dim)).astype(jp.int32)
    return min_dim, original_dim


def generate_crop_dimensions(key, width, height):
    """Generates valid crop dimensions respecting aspect ratio constraints.
    Args:
        key (random key): Random number generator key.
        width (int): Original image width.
        height (int): Original image height.
    Returns:
        tuple: (crop_width, crop_height) dimensions.
    """
    min_w, max_w = compute_crop_limits(width)
    min_h, max_h = compute_crop_limits(height)
    return (
        jax.random.randint(key, (), min_w, max_w, dtype=jp.int32),
        jax.random.randint(key, (), min_h, max_h, dtype=jp.int32),
    )


def build_crop_region(key, width, height, orig_width, orig_height):
    """Constructs crop region coordinates within original image boundaries.
    Args:
        key (random key): Random number generator key.
        width (int): Crop width.
        height (int): Crop height.
        orig_width (int): Original image width.
        orig_height (int): Original image height.
    Returns:
        array: Crop region array [x_start, y_start, x_end, y_end].
    """
    x_start = jax.random.randint(key, (), 0, orig_width - width, dtype=jp.int32)
    y_start = jax.random.randint(key, (), 0, orig_height - height, dtype=jp.int32)
    return jp.array([x_start, y_start, x_start + width, y_start + height])


def verify_crop_aspect(crop_width, crop_region):
    """Validates crop region's aspect ratio between 0.5 and 2.0.
    Args:
        crop_width (int): Width of the candidate crop.
        crop_region (array): Crop region coordinates.
    Returns:
        array: Original crop region if valid, else zero array.
    """
    crop_height = crop_region[3] - crop_region[1]
    aspect_ratio = crop_height / crop_width
    is_valid = jp.logical_and(0.5 <= aspect_ratio, aspect_ratio <= 2.0)
    return jp.where(is_valid, crop_region, jp.array([0, 0, 0, 0]))


def get_random_crop_region(key, original_width, original_height):
    """Generates random crop region with valid aspect ratio.
    Args:
        key (random key): Random number generator key.
        original_width (int): Original image width.
        original_height (int): Original image height.
    Returns:
        array: Valid crop region coordinates or zero array.
    """
    crop_width, crop_height = generate_crop_dimensions(key, original_width, original_height)
    valid_crop_region = build_crop_region(key, crop_width, crop_height, original_width, original_height)
    crop_region = verify_crop_aspect(crop_width, valid_crop_region)
    return crop_region


def get_center_inside_crop(boxes, crop_region):
    """Computes box centers and checks inclusion in crop region."""

    box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    x_center, y_center = box_centers[:, 0], box_centers[:, 1]
    x_min, y_min, x_max, y_max = crop_region
    center_inside_crop = (x_min < x_center) & (y_min < y_center) & (x_max > x_center) & (y_max > y_center)

    return center_inside_crop


def calculate_IoU_and_center_mask(boxes, crop_region):
    """Computes IoU values and center validity for boxes relative to crop region.
    Args:
        boxes (array): Array of shape (N, 4) containing bounding boxes.
        crop_region (array): Crop region coordinates array.
    Returns:
        tuple: (IoU_values, center_inside_mask) arrays.
    """
    IoU_values = compute_IOUs(jp.expand_dims(crop_region, axis=0), boxes)[0]
    center_inside_crop = get_center_inside_crop(boxes, crop_region)
    return IoU_values, center_inside_crop


def validate_crop_constraints(boxes, crop_region, min_IoU, max_IoU):
    """Validates crop region against IoU thresholds and center inclusion.
    Args:
        boxes (array): Bounding boxes array.
        crop_region (array): Candidate crop region.
        min_IoU (float): Minimum acceptable IoU threshold.
        max_IoU (float): Maximum acceptable IoU threshold.
    Returns:
        tuple: (is_valid_crop, center_inside_mask) validation results.
    """
    IoU_values, center_inside_mask = calculate_IoU_and_center_mask(boxes, crop_region)
    IoU_constraints_met = (jp.max(IoU_values) >= min_IoU) & (jp.min(IoU_values) <= max_IoU)
    has_valid_centers = jp.any(center_inside_mask)
    is_valid_crop = IoU_constraints_met & has_valid_centers
    return is_valid_crop, center_inside_mask


def should_continue_search(trial_state, max_attempts):
    """Determines if crop search should continue based on attempt count.
    Args:
        trial_state (tuple): (attempts, is_valid, mask) current search state.
        max_attempts (int): Maximum allowed attempts.
    Returns:
        bool: True if search should continue, False otherwise.
    """
    trials, is_valid, _ = trial_state
    return (trials < max_attempts) & (~is_valid)


def update_search_state(trial_state, boxes, crop_region, min_IoU, max_IoU):
    """Updates search state during crop validation attempts.
    Args:
        trial_state (tuple): Current (attempts, is_valid, mask) state.
        boxes (array): Bounding boxes array.
        crop_region (array): Candidate crop region.
        min_IoU (float): Minimum IoU threshold.
        max_IoU (float): Maximum IoU threshold.
    Returns:
        tuple: Updated search state.
    """
    trials, _, _ = trial_state
    is_valid_crop, center_mask = validate_crop_constraints(boxes, crop_region, min_IoU, max_IoU)
    return (trials + 1, is_valid_crop, center_mask)


def search_valid_crop(boxes, crop_region, min_IoU, max_IoU, max_attempts):
    """Performs iterative search for valid crop region meeting constraints.
    Args:
        boxes (array): Bounding boxes array.
        crop_region (array): Initial candidate crop region.
        min_IoU (float): Minimum acceptable IoU value.
        max_IoU (float): Maximum acceptable IoU value.
        max_attempts (int): Maximum number of validation attempts.
    Returns:
        tuple: Final search state (attempts, is_valid, mask).
    """
    is_valid_crop, center_mask = validate_crop_constraints(boxes, crop_region, min_IoU, max_IoU)
    return lax.while_loop(
        lambda state: should_continue_search(state, max_attempts),
        lambda state: update_search_state(state, boxes, crop_region, min_IoU, max_IoU),
        (0, is_valid_crop, center_mask),
    )


def apply_crop_to_image_and_boxes(image, boxes, labels, crop_region):
    """Applies crop operation to image and adjusts bounding boxes.
    Args:
        image (array): Input image array (H, W, C).
        boxes (array): Bounding boxes array.
        labels (array): Box labels array.
        crop_region (array): Crop region coordinates.
    Returns:
        tuple: (cropped_image, adjusted_boxes) arrays.
    """
    cropped_image = image[crop_region[1] : crop_region[3], crop_region[0] : crop_region[2], :]
    adjusted_boxes = adjust_boxes_and_labels(boxes, labels, crop_region)
    return cropped_image, adjusted_boxes


def validate_crop_region(boxes, crop_region, min_IoU, max_IoU, max_attempts):
    """Validates if acceptable crop region found within attempt limit.
    Args:
        boxes (array): Bounding boxes array.
        crop_region (array): Candidate crop region.
        min_IoU (float): Minimum IoU threshold.
        max_IoU (float): Maximum IoU threshold.
        max_attempts (int): Maximum allowed validation attempts.
    Returns:
        bool: True if valid crop found, False otherwise.
    """
    trial_count, is_valid, valid_mask = search_valid_crop(boxes, crop_region, min_IoU, max_IoU, max_attempts)
    return trial_count < max_attempts and is_valid


def perform_crop_operation(image, labels, max_attempts, boxes, crop_region, min_IoU, max_IoU, original_boxes):
    """Executes crop operation if valid region found, else returns original.
    Args:
        image (array): Input image array.
        labels (array): Box labels array.
        max_attempts (int): Maximum validation attempts.
        boxes (array): Bounding boxes array.
        crop_region (array): Candidate crop region.
        min_IoU (float): Minimum IoU threshold.
        max_IoU (float): Maximum IoU threshold.
        original_boxes (array): Original boxes before processing.
    Returns:
        tuple: (image, boxes) either cropped or original.
    """
    if validate_crop_region(boxes, crop_region, min_IoU, max_IoU, max_attempts):
        image, original_boxes = apply_crop_to_image_and_boxes(image, boxes, labels, crop_region)
    return image, original_boxes


def find_valid_crop_region(
    key, image, labels, max_attempts, width, height, boxes, min_IoU, max_IoU, original_boxes
):
    """Attempts to find valid crop region through multiple trials.
    Args:
        key (random key): Random number generator key.
        image (array): Input image array.
        labels (array): Box labels array.
        max_attempts (int): Maximum validation attempts.
        width (int): Original image width.
        height (int): Original image height.
        boxes (array): Bounding boxes array.
        min_IoU (float): Minimum IoU threshold.
        max_IoU (float): Maximum IoU threshold.
        original_boxes (array): Original boxes before processing.
    Returns:
        tuple: (image, boxes) either cropped or original.
    """
    crop_region = get_random_crop_region(key, width, height)

    if not jp.all(crop_region == 0):
        image, original_boxes = perform_crop_operation(
            image, labels, max_attempts, boxes, crop_region, min_IoU, max_IoU, original_boxes
        )
    return image, original_boxes


def extract_boxes_and_labels(key, boxes, IoU_thresholds):
    """Extracts box coordinates and labels while selecting IoU mode.
    Args:
        key (random key): Random number generator key.
        boxes (array): Input array containing boxes and labels.
        IoU_thresholds (tuple): Available IoU threshold configurations.
    Returns:
        tuple: (labels, bounding_boxes, IoU_mode) extracted components.
    """
    labels = boxes[:, -1:]
    bounding_boxes = boxes[:, :4]
    IoU_mode = jax.random.randint(key, shape=(), minval=0, maxval=len(IoU_thresholds))
    return labels, bounding_boxes, IoU_mode


def attempt_crop_with_IoU(key, image, labels, bounding_boxes, max_attempts, IoU_range, original_boxes):
    """Attempts to find a valid crop region that satisfies IoU constraints with original bounding boxes.
    Args:
        key: Random key for deterministic randomness.
        image: Input image array to crop.
        labels: List of object class labels associated with bounding boxes.
        bounding_boxes: List of bounding box coordinates in format [ymin, xmin, ymax, xmax].
        max_attempts: Maximum number of random crop attempts before giving up.
        IoU_range: Tuple (min_IoU, max_IoU) defining valid IoU range for crop region.
        original_boxes: Original bounding boxes before any transformations.
    Returns:
        Result of find_valid_crop_region() containing cropped image and adjusted annotations.
    """
    original_height, original_width = image.shape[:2]
    min_IoU, max_IoU = IoU_range
    return find_valid_crop_region(
        key,
        image,
        labels,
        max_attempts,
        original_width,
        original_height,
        bounding_boxes,
        min_IoU,
        max_IoU,
        original_boxes,
    )


def random_sample_crop(key, image, boxes, probability, max_trials, IoU_thresholds):
    """Performs random sample cropping with IoU validation for object detection.
    Args:
        key: Random key for deterministic randomness.
        image: Input image array to crop.
        boxes: Object detection annotations containing bounding boxes and labels.
        probability: Probability of applying the crop operation (0.0 to 1.0).
        max_trials: Maximum number of IoU validation attempts per crop mode.
        IoU_thresholds: Tuple of IoU range tuples for different cropping modes.
    Returns:
        Tuple containing:
            - Cropped image array (or original if crop not applied)
            - Adjusted bounding boxes in normalized coordinates
    """
    apply_crop = should_apply_crop(key, probability)
    labels, bounding_boxes, IoU_mode = extract_boxes_and_labels(key, boxes, IoU_thresholds)
    if not ((not apply_crop) or (IoU_thresholds[IoU_mode] is None)):
        image, boxes = attempt_crop_with_IoU(
            key, image, labels, bounding_boxes, max_trials, IoU_thresholds[IoU_mode], boxes
        )
    return image, boxes


def RandomSampleCrop(probability=0.5, max_trials=50, seed=0):
    """
    Creates a random sample crop processor for object detection data augmentation.
    This processor randomly crops images while ensuring bounding box validity based on
    Intersection-over-Union (IoU) constraints. It maintains a set of predefined IoU
    thresholds for different cropping behaviors and uses a probabilistic approach
    to determine when to apply cropping.
    Args:
        probability (float, optional): Probability of applying the crop (0.0 to 1.0).
            Defaults to 0.5.
        max_trials (int, optional): Maximum number of attempts to find a valid crop
            region that satisfies IoU constraints. Defaults to 50.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
    Returns:
        SimpleNamespace: A processor object containing:
            - `probability`: Crop application probability
            - `max_trials`: Maximum validation attempts
            - `jaccard_min_max`: Predefined IoU thresholds for different modes
            - `key`: Random key for RNG operations
            - `call()`: Method to apply cropping to an image and bounding boxes
    """

    def execute_crop_and_update(image, boxes, processor):
        """Executes cropping operation and updates processor state.
        Args:
            image: Input image array to process
            boxes: Object detection annotations (bounding boxes + labels)
            processor: Configuration object containing crop parameters
        Returns:
            Tuple containing:
                - Cropped image array (or original)
                - Adjusted bounding boxes in normalized coordinates
        """
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

    processor.call = lambda image, boxes: execute_crop_and_update(image, boxes, processor)
    return processor
