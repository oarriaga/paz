import jax.numpy as jp
import numpy as np
import jax
import paz


def pad(boxes, class_args, pad_size, pad_value):

    def pad_sample(boxes, class_args):
        boxes = jp.array(boxes)
        class_args = jp.array(class_args).reshape(-1, 1)
        detections = paz.detection.merge(boxes, class_args)
        detections = paz.detection.pad(detections, pad_size, "constant", -1)
        # detections = paz.detection.pad(detections, pad_size, "edge")
        return detections

    return jp.array([pad_sample(*sample) for sample in zip(boxes, class_args)])


def add_background_class(detections):
    boxes, class_args = detections = paz.detection.split(detections)
    detections = paz.detection.merge(boxes, class_args + 1)
    return detections


def encode(matched, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    """Encode the variances from the priorbox layers into the ground truth
    boxes we have matched (based on jaccard overlap) with the prior boxes.

    # Arguments
        matched: Numpy array of shape `(num_priors, 4)` with boxes in
            point-form.
        priors: Numpy array of shape `(num_priors, 4)` with boxes in
            center-form.
        variances: (list[float]) Variances of priorboxes

    # Returns
        encoded boxes: Numpy array of shape `(num_priors, 4)`.
    """

    def to_center_form(boxes):
        """Transform from corner coordinates to center coordinates.

        # Arguments
            boxes: Numpy array with shape `(num_boxes, 4)`.

        # Returns
            Numpy array with shape `(num_boxes, 4)`.
        """
        x_min, y_min = boxes[:, 0:1], boxes[:, 1:2]
        x_max, y_max = boxes[:, 2:3], boxes[:, 3:4]
        center_x = (x_max + x_min) / 2.0
        center_y = (y_max + y_min) / 2.0
        W = x_max - x_min
        H = y_max - y_min
        return np.concatenate([center_x, center_y, W, H], axis=1)

    boxes = matched[:, :4]
    boxes = to_center_form(boxes)
    center_difference_x = boxes[:, 0:1] - priors[:, 0:1]
    encoded_center_x = center_difference_x / priors[:, 2:3]
    center_difference_y = boxes[:, 1:2] - priors[:, 1:2]
    encoded_center_y = center_difference_y / priors[:, 3:4]
    encoded_center_x = encoded_center_x / variances[0]
    encoded_center_y = encoded_center_y / variances[1]
    encoded_W = np.log((boxes[:, 2:3] / priors[:, 2:3]) + 1e-8)
    encoded_H = np.log((boxes[:, 3:4] / priors[:, 3:4]) + 1e-8)
    encoded_W = encoded_W / variances[2]
    encoded_H = encoded_H / variances[3]
    encoded_boxes = [encoded_center_x, encoded_center_y, encoded_W, encoded_H]
    return np.concatenate(encoded_boxes + [matched[:, 4:]], axis=1)


def to_one_hot(boxes, num_classes):
    def _to_one_hot(class_indices, num_classes):
        """Transform from class index to one-hot encoded vector.

        # Arguments
            class_indices: Numpy array. One dimensional array specifying
                the index argument of the class for each sample.
            num_classes: Integer. Total number of classes.

        # Returns
            Numpy array with shape `(num_samples, num_classes)`.
        """
        one_hot_vectors = np.zeros((len(class_indices), num_classes))
        for vector_arg, class_args in enumerate(class_indices):
            one_hot_vectors[vector_arg, class_args] = 1.0
        return one_hot_vectors

    class_indices = boxes[:, 4].astype("int")
    one_hot_vectors = _to_one_hot(class_indices, num_classes)
    one_hot_vectors = one_hot_vectors.reshape(-1, num_classes)
    boxes = np.hstack([boxes[:, :4], one_hot_vectors.astype("float")])
    return boxes


def preprocess(detections, prior_boxes, num_classes, IOU, variances, H, W):
    # TODO change name.
    # TODO remove use of background class.
    detections = paz.detection.normalize(detections, H, W)
    detections = add_background_class(detections)
    detections = paz.detection.match_np(
        np.array(detections), np.array(prior_boxes), IOU
    )
    # detections = paz.detection.encode(detections, prior_boxes, variances)
    detections = encode(detections, np.array(prior_boxes), variances)
    # internally increase number of classes by 1 to account for background class
    # detections = paz.detection.to_one_hot(jp.array(detections), num_classes + 1)
    detections = to_one_hot(detections, num_classes + 1)
    return jp.array(detections)


def resize(images, boxes, H, W):
    resized_images, resized_boxes = [], []
    for sample in zip(images, boxes):
        # sample = paz.detection.resize_with_aspect_ratio(*sample, H, W)
        sample = paz.detection.resize(*sample, H, W)
        resized_images.append(sample[0])
        resized_boxes.append(sample[1])
    return jp.array(resized_images), resized_boxes


def preprocess_image(key, image, mean, augment=True):
    if augment:
        image = paz.image.augment_color(key, image)
    image = paz.image.RGB_to_BGR(image)
    image = paz.image.subtract_mean(image, mean)
    return image


def original_preprocess_batch(
    key,
    images,
    boxes,
    class_args,
    H,
    W,
    prior_boxes,
    num_classes,
    match_IOU,
    variances,
    mean,
    max_num_boxes,
    augment=True,
):
    images = [paz.image.load(image) for image in images]
    images, boxes = resize(images, boxes, H, W)
    detections = pad(boxes, class_args, max_num_boxes, -1)

    preprocess_images = jax.jit(
        jax.vmap(paz.lock(preprocess_image, mean, augment), (0, 0))
    )
    images = preprocess_images(jax.random.split(key, len(images)), images)
    args = prior_boxes, num_classes, match_IOU, variances, H, W
    detections = jax.jit(jax.vmap(paz.lock(preprocess, *args)))(detections)
    return images, detections


def preprocess_batch(
    key,
    images,
    boxes,
    class_args,
    H,
    W,
    prior_boxes,
    num_classes,
    match_IOU,
    variances,
    mean,
    max_num_boxes,
    augment=True,
):
    def padder(boxes, class_args, pad_size, pad_value):

        def pad_sample(boxes, class_args):
            boxes = jp.array(boxes)
            class_args = jp.array(class_args).reshape(-1, 1)
            detections = paz.detection.merge(boxes, class_args)
            return detections

        return [pad_sample(*sample) for sample in zip(boxes, class_args)]

    images = [paz.image.load(image) for image in images]
    images, boxes = resize(images, boxes, H, W)
    detections = padder(boxes, class_args, max_num_boxes, -1)

    preprocess_images = jax.jit(
        jax.vmap(paz.lock(preprocess_image, mean, augment), (0, 0))
    )
    images = preprocess_images(jax.random.split(key, len(images)), images)
    args = prior_boxes, num_classes, match_IOU, variances, H, W
    # detections = jax.jit(jax.vmap(paz.lock(preprocess, *args)))(detections)
    # jit_preprocess = jax.jit(paz.lock(preprocess, *args))
    # detections = jp.array([jit_preprocess(x) for x in detections])
    detections = jp.array([preprocess(x, *args) for x in detections])
    return images, detections
