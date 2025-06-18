from keras import ops


def smooth_l1(y_true, y_pred):
    absolute_loss = ops.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred) ** 2
    less_than_one = ops.less(absolute_loss, 1.0)
    l1_smooth_loss = ops.where(less_than_one, square_loss, absolute_loss - 0.5)
    return ops.sum(l1_smooth_loss, axis=-1)


def cross_entropy(y_true, y_pred):
    y_pred = ops.maximum(ops.minimum(y_pred, 1 - 1e-15), 1e-15)
    cross_entropy_loss = -ops.sum(y_true * ops.log(y_pred), axis=-1)
    return cross_entropy_loss


def calculate_masks(y_true):
    negative_mask = y_true[:, :, 4]  # for all batches and prior boxes
    positive_mask = 1.0 - negative_mask
    return positive_mask, negative_mask


def regression(y_true, y_pred, alpha=1.0):
    """Computes regression loss in a batch.

    # Arguments
        y_true: Tensor of shape '[batch_size, num_boxes, 4 + num_classes]'
            with correct labels.
        y_pred: Tensor of shape '[batch_size, num_boxes, 4 + num_classes]'
            with predicted inferences.

    # Returns
        Tensor with regression loss per sample in batch.
    """
    batch_size = ops.cast(ops.shape(y_pred)[0], "float32")
    local_loss = smooth_l1(y_true[:, :, :4], y_pred[:, :, :4])
    positive_mask, negative_mask = calculate_masks(y_true)
    positive_local_losses = local_loss * positive_mask
    positive_local_loss = ops.sum(positive_local_losses, axis=-1)

    num_positives = ops.sum(ops.cast(positive_mask, "float32"))
    num_positives = ops.maximum(1.0, num_positives)

    # num_positives_per_sample = ops.sum(positive_mask, axis=-1)
    # num_positives = ops.maximum(1.0, num_positives_per_sample)

    return (alpha * positive_local_loss * batch_size) / num_positives


def positive_classification(y_true, y_pred):
    """Computes classification loss of boxes that contain an object.

    # Arguments
        y_true: Tensor of shape '[batch_size, num_boxes, 4 + num_classes]'
            with correct labels.
        y_pred: Tensor of shape '[batch_size, num_boxes, 4 + num_classes]'
            with predicted inferences.

    # Returns
        Tensor with positive classification loss per sample in batch.
    """
    batch_size = ops.cast(ops.shape(y_pred)[0], "float32")
    class_loss = cross_entropy(y_true[:, :, 4:], y_pred[:, :, 4:])
    positive_mask, negative_mask = calculate_masks(y_true)
    positive_class_losses = class_loss * positive_mask
    positive_class_loss = ops.sum(positive_class_losses, axis=-1)
    num_positives = ops.sum(ops.cast(positive_mask, "float32"))
    num_positives = ops.maximum(1.0, num_positives)
    return (positive_class_loss * batch_size) / num_positives


def negative_classification(y_true, y_pred, neg_pos_ratio=3, max_negatives=300):
    """Computes classification loss of boxes that don't contain an object.

    # Arguments
        y_true: Tensor of shape '[batch_size, num_boxes, 4 + num_classes]'
            with correct labels.
        y_pred: Tensor of shape '[batch_size, num_boxes, 4 + num_classes]'
            with predicted inferences.

    # Returns
        Tensor with negative classification loss per sample in batch.
    """

    class_loss = cross_entropy(y_true[:, :, 4:], y_pred[:, :, 4:])
    positive_mask, negative_mask = calculate_masks(y_true)
    negative_class_losses = class_loss * negative_mask
    descending_sorted_losses = -ops.sort(-negative_class_losses, axis=-1)

    num_positives_per_sample = ops.cast(ops.sum(positive_mask, -1), "int32")
    num_hard_negatives = neg_pos_ratio * num_positives_per_sample
    num_hard_negatives = ops.minimum(num_hard_negatives, max_negatives)
    num_hard_negatives = ops.expand_dims(num_hard_negatives, axis=-1)

    num_boxes = ops.shape(negative_class_losses)[1]
    indices = ops.expand_dims(ops.arange(num_boxes, dtype="int32"), axis=0)
    selection_mask = ops.less(indices, num_hard_negatives)
    masked_top_losses = ops.where(
        selection_mask,
        descending_sorted_losses,
        ops.zeros_like(descending_sorted_losses),
    )
    negative_class_loss = ops.sum(masked_top_losses, axis=-1)

    total_num_positives = ops.sum(ops.cast(positive_mask, "float32"))
    total_num_positives = ops.maximum(1.0, total_num_positives)
    batch_size = ops.cast(ops.shape(y_pred)[0], "float32")
    return (negative_class_loss * batch_size) / total_num_positives


def call(y_true, y_pred):
    """Computes regression and classification losses in a batch.

    # Arguments
        y_true: Tensor of shape '[batch_size, num_boxes, 4 + num_classes]'
            with correct labels.
        y_pred: Tensor of shape '[batch_size, num_boxes, 4 + num_classes]'
            with predicted inferences.

    # Returns
        Tensor with loss per sample in batch.
    """
    localization_loss = regression(y_true, y_pred)
    positive_loss = positive_classification(y_true, y_pred)
    negative_loss = negative_classification(y_true, y_pred)
    return localization_loss + positive_loss + negative_loss
