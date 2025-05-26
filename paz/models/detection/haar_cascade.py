from keras.utils import get_file
import numpy as np
import cv2
import paz
import jax


def download(label):
    URL = (
        "https://raw.githubusercontent.com/opencv/opencv/"
        "master/data/haarcascades/"
    )
    filename = "haarcascade_" + label + ".xml"
    filepath = get_file(filename, URL + filename, cache_subdir="paz/models")
    model = cv2.CascadeClassifier(filepath)
    return model


def HaarCascadeDetector(label, scale, neighbors, class_arg, max_boxes, draw):
    """Haar cascade detector."""
    detector = download(label)

    def pad(boxes, size, value=-1):
        boxes = boxes[:size]
        padding = ((0, size - len(boxes)), (0, 0))
        return np.pad(boxes, padding, "constant", constant_values=value)

    def model(GRAY_image):
        GRAY_image = paz.to_numpy(GRAY_image)
        boxes = detector.detectMultiScale(GRAY_image, scale, neighbors)
        boxes = np.full((max_boxes, 4), -1) if len(boxes) == 0 else boxes
        boxes = pad(boxes, max_boxes)
        return paz.to_jax(boxes)

    @jax.jit
    def postprocess(boxes):
        boxes = paz.boxes.xywh_to_xyxy(boxes)
        boxes = paz.boxes.append_class(boxes, class_arg)
        boxes = paz.cast(boxes, "int32")
        return boxes

    @jax.jit
    def preprocess(image):
        return paz.image.RGB_to_GRAY(image)

    def call(image):
        return postprocess(model(preprocess(image)))

    def call_and_draw(image):
        boxes = call(image)
        return boxes, draw(image, boxes)

    return call_and_draw if callable(draw) else call


def HaarCascadeFrontalFaceDetector(
    scale=1.3,
    neighbors=5,
    class_arg=0,
    max_boxes=100,
    draw=paz.lock(paz.draw.boxes, paz.draw.GREEN, 2),
):
    args = ("frontalface_default", scale, neighbors, class_arg, max_boxes, draw)
    return HaarCascadeDetector(*args)


def HaarCascadeEyeDetector(
    scale=1.3,
    neighbors=5,
    class_arg=0,
    max_boxes=100,
    draw=paz.lock(paz.draw.boxes, paz.draw.GREEN, 2),
):
    args = ("eye", scale, neighbors, class_arg, max_boxes, draw)
    return HaarCascadeDetector(*args)
