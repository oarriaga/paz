import jax.numpy as jp
import numpy as np
import jax
import paz
import cv2


def resize(image, size, method=cv2.INTER_LINEAR):
    return cv2.resize(image, size[::-1], interpolation=method)


def SSD(
    image,
    model,
    prior_boxes,
    class_names,
    score_thresh,
    IOU_thresh,
    top_k,
    variances,
):
    model_input_size = model.input_shape[1:3]
    image_size = paz.image.get_size(image)

    def preprocess(image, mean=paz.image.BGR_IMAGENET_MEAN):
        """Single-shot Multi Box Detector preprocessing function."""
        # image = paz.image.resize(image, model_input_size, "linear", False)
        # image = paz.to_jax(resize(paz.to_numpy(image), model_input_size))
        image = paz.image.RGB_to_BGR(image)
        image = paz.image.subtract_mean(image, jp.array(mean))
        image = paz.cast(image, "float32")
        image = jp.expand_dims(image, axis=0)
        return image

    def postprocess(detections):
        """Single-shot Multi Box Detector postprocessing function."""
        detections = jp.squeeze(detections, axis=0)
        detections = paz.detection.decode(detections, prior_boxes, variances)
        detections = paz.detection.remove_class(detections, 0)
        NMS_args = (len(class_names), IOU_thresh, top_k, 0.01)
        detections = paz.detection.apply_per_class_NMS(detections, *NMS_args)
        detections = paz.detection.filter_by_score(detections, score_thresh, -1)
        return detections

    # image = paz.to_jax(resize(paz.to_numpy(image), model_input_size))
    image = jax.jit(
        paz.lock(paz.image.resize, model_input_size, "linear", False)
    )(image)
    image = jax.jit(preprocess)(image)
    # predictions = jax.jit(model)(image)
    predictions = model(image)
    detections = jax.jit(postprocess, device=jax.devices("cpu")[0])(predictions)
    detections = paz.detection.remove_invalid(detections)
    detections = paz.detection.denormalize(detections, *image_size)
    boxes, class_args, scores = paz.detection.to_boxes2D(detections)
    return boxes, class_args, scores


def SSD300VOC(score_thresh=0.60, IOU_thresh=0.45, top_k=200, draw=None):
    model = paz.models.detection.SSD300(21, "VOC", "VOC", (300, 300, 3))
    boxes = paz.models.detection.utils.create_prior_boxes("VOC")
    names = paz.datasets.labels("VOC")
    if draw is None:
        colors = paz.draw.lincolor(len(names))
        draw = paz.partial(paz.draw.boxes2D, names=names, colors=colors)
    variances = [0.1, 0.1, 0.2, 0.2]
    args = (model, boxes, names, score_thresh, IOU_thresh, top_k, variances)
    detect = paz.lock(SSD, *args)
    return lambda x: draw(x, *detect(x)) if callable(draw) else detect


def SSD512COCO(score_thresh=0.60, IOU_thresh=0.45, top_k=200, draw=None):
    model = paz.models.detection.SSD512(81, "COCO", "COCO", (512, 512, 3))
    boxes = paz.models.detection.utils.create_prior_boxes("COCO")
    names = paz.datasets.labels("COCO")
    if draw is None:
        colors = paz.draw.lincolor(len(names))
        draw = paz.partial(paz.draw.boxes2D, names=names, colors=colors)
    variances = [0.1, 0.1, 0.2, 0.2]
    args = (model, boxes, names, score_thresh, IOU_thresh, top_k, variances)
    detect = paz.lock(SSD, *args)
    return lambda x: draw(x, *detect(x)) if callable(draw) else detect


def DetectMiniXceptionFER(box_scale=1.2):
    # TODO add original buffer window prediction
    detect = paz.models.HaarCascadeFrontalFaceDetector(draw=None)
    classify = paz.applications.ClassifyMiniXceptionFER()
    names = paz.datasets.labels("FER")
    colors = paz.draw.lincolor(len(names))

    def apply(image):
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
        return predictions, paz.draw.boxes2D(image, *predictions, names, colors)

    return apply
