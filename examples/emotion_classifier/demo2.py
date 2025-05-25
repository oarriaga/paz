# TODO fix draw function in HaarCascadeFrontalFaceDetector
# TODO fix automatic message in HaarCascadeFrontalFaceDetector
import os

os.environ["KERAS_BACKEND"] = "jax"
import jax.numpy as jp
import keras
import paz
from paz.applications import MiniXceptionFER
from paz.applications import HaarCascadeFrontalFaceDetector


def to_labels(probabilities, labels):
    return labels[jp.argmax(probabilities)]


def ClassifyMiniXceptionFER():
    classify = MiniXceptionFER()

    def preprocess(image, shape):
        image = paz.image.resize(image, shape)
        image = paz.image.normalize(image)
        image = paz.image.rgb_to_gray(image)
        return jp.expand_dims(image, [0, -1])

    def postprocess(scores):
        return jp.squeeze(scores, axis=0)

    def apply(image):
        return postprocess(classify(preprocess(image, (48, 48))))

    return apply


def DetectMiniXceptionFER():
    detect = HaarCascadeFrontalFaceDetector(draw=None)
    classify = ClassifyMiniXceptionFER()
    names = paz.datasets.labels("FER")
    colors = paz.draw.lincolor(len(names))

    def apply(image):
        boxes = paz.detection.get_boxes(detect(image).boxes)
        print(boxes)
        scores, labels = [], []
        for crop in paz.boxes.crop_with_pad(boxes, image, 48, 48, 0):
            score = classify(crop)
            labels.append(jp.argmax(score))
            scores.append(jp.max(score))
            print(score.shape)
            print(jp.argmax(score))
        scores = jp.array(scores)
        labels = jp.array(labels)
        print(scores, scores.shape)
        print(labels, labels.shape)
        return paz.draw.boxes2D(image, boxes, labels, scores, names, colors)

    return apply


URL = (
    "https://github.com/oarriaga/altamira-data/releases/download"
    "/v0.9.1/image_with_faces.jpg"
)
filename = os.path.basename(URL)
fullpath = keras.utils.get_file(filename, URL, cache_subdir="paz/tests")
image = paz.image.load(fullpath)
paz.image.show(image)

detect = DetectMiniXceptionFER()
image_with_detections, (boxes, scores, labels) = detect(image)
paz.image.show(image_with_detections)
