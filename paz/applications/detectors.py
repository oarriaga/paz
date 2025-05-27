# TODO sort and remove boxes before sending to NMS
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

    def apply(image):
        image_size = paz.image.get_size(image)
        detections = postprocess(model(preprocess(image)), image_size)
        detections = paz.detection.remove_invalid(detections)
        return paz.detection.to_boxes2D(detections)

    return lambda x: (y := apply(x), draw(x, *y)) if callable(draw) else apply


def SSD300VOC(score_thresh=0.60, IOU_thresh=0.45, top_k=200, draw=None):
    model = paz.models.detection.SSD300(21, "VOC", "VOC", (300, 300, 3))
    boxes = paz.models.detection.utils.create_prior_boxes("VOC")
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
    boxes = paz.models.detection.utils.create_prior_boxes("COCO")
    names = paz.datasets.labels("COCO")
    label_colors = paz.draw.lincolor(len(names))
    if draw is None:
        draw = paz.partial(paz.draw.boxes2D, names=names, colors=label_colors)
    variances = [0.1, 0.1, 0.2, 0.2]
    apply_NMS = (len(names), IOU_thresh, top_k)
    apply_NMS = paz.lock(paz.detection.apply_per_class_NMS, *apply_NMS)
    return SSD(model, score_thresh, boxes, variances, apply_NMS, draw)


def DetectMiniXceptionFER(box_scale=1.2):
    # TODO add buffer window prediction
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
