# TODO fix draw function in HaarCascadeFrontalFaceDetector
# TODO fix automatic message in HaarCascadeFrontalFaceDetector
# TODO keep numpy image
# TODO assume that all pipelines must convert it to a jax array
import time
import os

os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
import jax.numpy as jp
import keras
import paz
import jax
from paz.applications import MiniXceptionFER
from paz.applications import HaarCascadeFrontalFaceDetector
import cv2


def resize(image, size, method=cv2.INTER_LINEAR):
    return cv2.resize(image, size[::-1], interpolation=method)


def ClassifyMiniXceptionFER():
    classify = MiniXceptionFER()
    image_size = classify.input_shape[1:3]
    cpu = jax.devices("cpu")[0]

    def preprocess(image, shape):
        image = jax.jit(
            paz.lock(paz.image.resize, shape, "linear", True), device=cpu
        )(image)
        # image = paz.to_jax(resize(paz.to_numpy(image), shape))
        image = paz.image.normalize(image)
        image = paz.image.rgb_to_gray(image)

        return jp.expand_dims(image, [0, -1])

    def postprocess(scores):
        return jp.squeeze(scores, axis=0)

    def apply(image):
        return postprocess(classify(preprocess(image, image_size)))

    return apply


def apply_to_valid(fn, boxes, value=-1):
    is_valid = jp.all(boxes != value, axis=1)
    return jp.where(is_valid, fn(boxes), value)


def DetectMiniXceptionFER():
    detect = HaarCascadeFrontalFaceDetector(draw=None)
    classify = jax.jit(ClassifyMiniXceptionFER())
    # classify = ClassifyMiniXceptionFER()
    names = paz.datasets.labels("FER")
    colors = paz.draw.lincolor(len(names))

    def apply(image):
        boxes = paz.detection.get_boxes(detect(image).boxes)
        image_gpu = paz.to_jax(image)
        boxes = paz.to_jax(boxes)
        boxes = paz.boxes.square(boxes)
        boxes = paz.boxes.scale(boxes, 1.2, 1.2)
        boxes = paz.cast(boxes, "int32")
        scores, labels = [], []
        # for crop in paz.boxes.crop_with_pad(boxes, image, 48, 48, 0):
        boxes = paz.boxes.remove_invalid(boxes)
        print(boxes)
        for box in boxes:
            crop = paz.image.crop(image_gpu, box)
            score = classify(crop)
            labels.append(jp.argmax(score))
            scores.append(jp.max(score))
        scores = np.array(scores)
        labels = np.array(labels)
        return paz.draw.boxes2D(image, boxes, labels, scores, names, colors)

    return apply


URL = (
    "https://github.com/oarriaga/altamira-data/releases/download"
    "/v0.9.1/image_with_faces.jpg"
)
filename = os.path.basename(URL)
fullpath = keras.utils.get_file(filename, URL, cache_subdir="paz/tests")
image = paz.image.load(fullpath)
# paz.image.show(image)


# whole pipeline 0.15 / 6.6FPS without drawing
pipeline = DetectMiniXceptionFER()
# pipeline = ClassifyMiniXceptionFER()
# image_with_detections, (boxes, scores, labels) = pipeline(image)
# paz.image.show(image_with_detections)
# print(paz.log.time(pipeline, 20, 1, False, image))

# measure camera read time
# camera = paz.Camera(identifier=4)
# camera.start()
# print(paz.log.time(camera.read, 20, 1, False))
# camera.stop()


# measure whole pipeline time
camera = paz.Camera(identifier=0)
player = paz.VideoPlayer((480, 640), pipeline, camera)
player.run()
print(paz.log.time(player.run, 20, 1, False))
