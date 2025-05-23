import jax.numpy as jp
import numpy as np
import jax
import paz
import cv2


# def draw_boxes2D(image, boxes, class_args, scores, names, colors, thickness):
#     image = np.ascontiguousarray(np.array(image, dtype=image.dtype))
#     for box, class_arg, score in zip(boxes, class_args, scores):
#         image = paz.draw.box(image, box.tolist(), colors[class_arg], thickness)
#     return image, (boxes, class_args, scores)


def draw_boxes2D(image, boxes, class_args, scores, names, colors, thickness):
    font_scale = 0.7
    font = cv2.FONT_HERSHEY_DUPLEX
    image = np.ascontiguousarray(np.array(image, dtype=image.dtype))
    for box, class_arg, score in zip(boxes, class_args, scores):
        color = colors[class_arg]
        x_min, y_min, x_max, y_max = box = box.tolist()
        image = paz.draw.box(image, box, colors[class_arg], thickness)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        label = f"{names[class_arg]} {score * 100:.0f}%"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        offset = round(thickness / 2)
        cv2.rectangle(
            image,
            (x_min - offset, y_min - text_height - baseline - thickness),
            (x_min + text_width, y_min),
            color,
            -1,
        )

        cv2.putText(
            image,
            label,
            (x_min, y_min - baseline),
            font,
            font_scale,
            (255, 255, 255),
        )

    return image, (boxes, class_args, scores)


def to_boxes2D(detections):
    boxes, scores = paz.detection.split(detections)
    labels = jp.argmax(scores, axis=-1)
    scores = scores[jp.arange(len(scores)), labels]
    return boxes.astype("int32"), labels.astype("int32"), scores


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
        image = paz.image.resize(image, model_input_size, "linear", False)
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

    image = jax.jit(preprocess)(image)
    # predictions = jax.jit(model)(image)
    predictions = model(image)
    detections = jax.jit(postprocess, device=jax.devices("cpu")[0])(predictions)
    detections = paz.detection.remove_invalid(detections)
    detections = paz.detection.denormalize(detections, *image_size)
    boxes, class_args, scores = to_boxes2D(detections)
    return boxes, class_args, scores


def SSD300VOC(score_thresh=0.60, IOU_thresh=0.45, top_k=200, draw=None):
    model = paz.models.detection.SSD300(21, "VOC", "VOC", (300, 300, 3))
    boxes = paz.models.detection.utils.create_prior_boxes("VOC")
    names = paz.datasets.labels("VOC")
    if draw is None:
        draw = paz.lock(draw_boxes2D, names, paz.draw.lincolor(len(names)), 3)
    variances = [0.1, 0.1, 0.2, 0.2]
    args = (model, boxes, names, score_thresh, IOU_thresh, top_k, variances)
    detect = paz.lock(SSD, *args)
    return lambda x: draw(x, *detect(x)) if callable(draw) else detect


def SSD512COCO(score_thresh=0.60, IOU_thresh=0.45, top_k=200, draw=None):
    model = paz.models.detection.SSD512(81, "COCO", "COCO", (512, 512, 3))
    boxes = paz.models.detection.utils.create_prior_boxes("COCO")
    names = paz.datasets.labels("COCO")
    if draw is None:
        draw = paz.lock(draw_boxes2D, names, paz.draw.lincolor(len(names)), 3)
    variances = [0.1, 0.1, 0.2, 0.2]
    args = (model, boxes, names, score_thresh, IOU_thresh, top_k, variances)
    detect = paz.lock(SSD, *args)
    return lambda x: draw(x, *detect(x)) if callable(draw) else detect
