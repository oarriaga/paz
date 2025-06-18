import jax.numpy as jp
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


def preprocess(detections, prior_boxes, num_classes, IOU, variances, H, W):
    # TODO change name.
    # TODO remove use of background class.
    detections = paz.detection.normalize(detections, H, W)
    detections = add_background_class(detections)
    detections = paz.detection.match(detections, prior_boxes, IOU)
    detections = paz.detection.encode(detections, prior_boxes, variances)
    # internally increase number of classes by 1 to account for background class
    detections = paz.detection.to_one_hot(detections, num_classes + 1)
    return detections


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
    jit_preprocess = jax.jit(paz.lock(preprocess, *args))
    detections = jp.array([jit_preprocess(x) for x in detections])
    return images, detections
