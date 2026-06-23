# TODO sort and remove boxes before sending to NMS
import cv2
import jax.numpy as jp
import numpy as np
import jax
import paz


def SSD(model, score_thresh, prior_boxes, variances, apply_NMS, draw):

    @jax.jit
    def preprocess(image, mean=paz.image.BGR_IMAGENET_MEAN):
        """Single-shot Multi Box Detector preprocessing function."""
        image = paz.image.resize_opencv(image, paz.image.get_input_size(model))
        image = paz.image.RGB_to_BGR(image)
        image = paz.image.subtract_mean(image, jp.array(mean))
        image = paz.cast(image, "float32")
        return jp.expand_dims(image, axis=0)

    def postprocess(detections, image_size):
        """Single-shot Multi Box Detector postprocessing function."""
        detections = jp.squeeze(detections, axis=0)
        detections = paz.detection.decode(detections, prior_boxes, variances)
        detections = paz.detection.remove_class(detections, 0)
        detections = paz.time(apply_NMS)(detections)
        detections = paz.detection.filter_by_score(detections, score_thresh, -1)
        detections = paz.detection.denormalize(detections, *image_size)
        return detections

    def call(image):
        image_size = paz.image.get_size(image)
        detections = postprocess(model(preprocess(image)), image_size)
        detections = paz.detection.remove_invalid(detections)
        return paz.detection.to_boxes2D(detections)

    return (lambda x: (y := call(x), draw(x, *y))) if callable(draw) else call


def SSD300VOC(score_thresh=0.60, IOU_thresh=0.45, top_k=200, draw=None):
    model = paz.models.detection.SSD300(21, "VOC", "VOC", (300, 300, 3))
    boxes = paz.models.detection.single_shot_detector.build_prior_boxes("VOC")
    names = paz.datasets.labels("VOC")
    label_colors = paz.draw.lincolor(len(names))
    if draw is None:
        draw = paz.partial(paz.draw.boxes2D, names=names, colors=label_colors)
    variances = [0.1, 0.1, 0.2, 0.2]
    apply_NMS = (len(names), IOU_thresh, top_k)
    apply_NMS = paz.lock(paz.detection.apply_per_class_NMS, *apply_NMS)
    return SSD(model, score_thresh, boxes, variances, apply_NMS, draw)


def SSD512COCO(score_thresh=0.60, IOU_thresh=0.45, top_k=200, draw=None):
    model = paz.models.detection.SSD512(81, "COCO", "COCO", (512, 512, 3))
    boxes = paz.models.detection.single_shot_detector.build_prior_boxes("VOC")
    names = paz.datasets.labels("COCO")
    label_colors = paz.draw.lincolor(len(names))
    if draw is None:
        draw = paz.partial(paz.draw.boxes2D, names=names, colors=label_colors)
    variances = [0.1, 0.1, 0.2, 0.2]
    apply_NMS = (len(names), IOU_thresh, top_k)
    apply_NMS = paz.lock(paz.detection.apply_per_class_NMS, *apply_NMS)
    return SSD(model, score_thresh, boxes, variances, apply_NMS, draw)


def EFFICIENTDETD0COCO(**kwargs):
    return EfficientDetCOCO(paz.models.EFFICIENTDETD0, **kwargs)


def EFFICIENTDETD1COCO(**kwargs):
    return EfficientDetCOCO(paz.models.EFFICIENTDETD1, **kwargs)


def EFFICIENTDETD2COCO(**kwargs):
    return EfficientDetCOCO(paz.models.EFFICIENTDETD2, **kwargs)


def EFFICIENTDETD3COCO(**kwargs):
    return EfficientDetCOCO(paz.models.EFFICIENTDETD3, **kwargs)


def EFFICIENTDETD4COCO(**kwargs):
    return EfficientDetCOCO(paz.models.EFFICIENTDETD4, **kwargs)


def EFFICIENTDETD5COCO(**kwargs):
    return EfficientDetCOCO(paz.models.EFFICIENTDETD5, **kwargs)


def EFFICIENTDETD6COCO(**kwargs):
    return EfficientDetCOCO(paz.models.EFFICIENTDETD6, **kwargs)


def EFFICIENTDETD7COCO(**kwargs):
    return EfficientDetCOCO(paz.models.EFFICIENTDETD7, **kwargs)


def EfficientDetCOCO(build_model, **kwargs):
    model = build_model(num_classes=90, base_weights="COCO",
                        head_weights="COCO")
    names = paz.datasets.labels("COCO_EFFICIENTDET")
    return EfficientDet(model, names, **kwargs)


def EfficientDet(model, names, score_thresh=0.60, nms_thresh=0.45,
                 top_k=200, draw=None):
    input_size = model.input_shape[1]
    priors = jp.array(model.prior_boxes) * input_size
    variances = [1.0, 1.0, 1.0, 1.0]
    apply_NMS = (len(names), nms_thresh, top_k)
    apply_NMS = paz.lock(paz.detection.apply_per_class_NMS, *apply_NMS)
    if draw is None:
        colors = paz.draw.lincolor(len(names))
        draw = paz.partial(paz.draw.boxes2D, names=names, colors=colors)

    def preprocess(image):
        image = np.asarray(image, dtype="float32")
        image = image - np.array(paz.image.RGB_IMAGENET_MEAN)
        image = image / np.array(paz.image.RGB_IMAGENET_STDV)
        return scaled_resize(image, input_size)

    def postprocess(outputs, scale):
        detections = change_box_coordinates(outputs)[0]
        detections = paz.detection.decode(detections, priors, variances)
        detections = scale_boxes(detections, scale)
        detections = apply_NMS(detections)
        detections = paz.detection.filter_by_score(detections, score_thresh, -1)
        detections = paz.detection.remove_invalid(detections)
        return paz.detection.to_boxes2D(detections)

    def call(image):
        model_input, scale = preprocess(image)
        return postprocess(model(model_input), scale)

    return (lambda x: (y := call(x), draw(x, *y))) if callable(draw) else call


def scaled_resize(image, size):
    H, W = image.shape[:2]
    scale = min(size / W, size / H)
    resized = cv2.resize(image, (int(W * scale), int(H * scale)))
    output = np.zeros((size, size, 3), dtype="float32")
    output[: resized.shape[0], : resized.shape[1]] = resized
    return output[np.newaxis], np.float32(1.0 / scale)


def change_box_coordinates(outputs):
    outputs = np.asarray(outputs[0])
    boxes, classes = outputs[:, :4], outputs[:, 4:]
    s1, s2, s3, s4 = np.split(boxes, 4, axis=1)
    boxes = np.concatenate([s2, s1, s4, s3], axis=1)
    return np.concatenate([boxes, classes], axis=1)[np.newaxis]


def scale_boxes(detections, scale):
    detections = np.asarray(detections)
    boxes = detections[:, :4] * scale
    return np.concatenate([boxes, detections[:, 4:]], axis=1)


def DetectMiniXceptionFER(box_scale=1.2, draw=None):
    # TODO add buffer window prediction
    detect = paz.models.HaarCascadeFrontalFaceDetector(draw=None)
    classify = paz.applications.ClassifyMiniXceptionFER()
    names = paz.datasets.labels("FER")
    colors = paz.draw.lincolor(len(names))
    if draw is None:
        draw = paz.partial(paz.draw.boxes2D, names=names, colors=colors)

    def call(image):
        boxes = paz.detection.get_boxes(detect(image))
        boxes = paz.boxes.square(boxes)
        boxes = paz.boxes.scale(boxes, box_scale, box_scale)
        boxes = paz.cast(boxes, "int32")
        boxes = paz.boxes.remove_invalid(boxes)
        scores, labels = [], []
        for box in boxes:
            score = classify(paz.image.crop(image, box))
            labels.append(jp.argmax(score))
            scores.append(jp.max(score))
        scores = np.array(scores)
        labels = np.array(labels)
        predictions = (boxes, labels, scores)
        return predictions

    return (lambda x: (y := call(x), draw(x, *y))) if callable(draw) else call
