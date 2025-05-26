# TODO fix draw function in HaarCascadeFrontalFaceDetector
# TODO fix automatic message in HaarCascadeFrontalFaceDetector
# TODO keep numpy image
# TODO assume that all pipelines must convert it to a jax array
import os

os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
import jax.numpy as jp
import paz
from paz.applications import MiniXceptionFER
from paz.applications import HaarCascadeFrontalFaceDetector
import jax


def ClassifyMiniXceptionFER():
    model = MiniXceptionFER()
    resize = paz.lock(paz.image.resize_opencv, paz.image.get_input_size(model))

    def preprocess(image):
        image = paz.image.normalize(image)
        image = paz.image.rgb_to_gray(image)
        return jp.expand_dims(image, [0, -1])

    def postprocess(scores):
        return jp.squeeze(scores, axis=0)

    @jax.jit
    def call(image):
        return postprocess(model(preprocess(image)))

    return lambda image: call(resize(image))


def DetectMiniXceptionFER(box_scale=1.2):
    detect = paz.time(
        HaarCascadeFrontalFaceDetector(draw=None), False, "Detect"
    )
    classify = paz.time(ClassifyMiniXceptionFER(), True, "Classify")
    names = paz.datasets.labels("FER")
    colors = paz.draw.lincolor(len(names))

    def apply(image):
        boxes = paz.detection.get_boxes(detect(image).boxes)
        image_gpu = paz.to_jax(image)
        boxes = paz.to_jax(boxes)
        boxes = paz.boxes.square(boxes)
        boxes = paz.boxes.scale(boxes, box_scale, box_scale)
        boxes = paz.cast(boxes, "int32")
        scores, labels = [], []
        boxes = paz.boxes.remove_invalid(boxes)
        for box in boxes:
            crop = paz.image.crop(image_gpu, box)
            score = classify(crop)
            labels.append(jp.argmax(score))
            scores.append(jp.max(score))
        scores = np.array(scores)
        labels = np.array(labels)
        return paz.draw.boxes2D(image, boxes, labels, scores, names, colors)

    return apply


# URL = (
#     "https://github.com/oarriaga/altamira-data/releases/download"
#     "/v0.9.1/image_with_faces.jpg"
# )
# filename = os.path.basename(URL)
# fullpath = keras.utils.get_file(filename, URL, cache_subdir="paz/tests")
# image = paz.image.load(fullpath)
# paz.image.show(image)

pipeline = DetectMiniXceptionFER()
camera = paz.Camera(identifier=0)
player = paz.VideoPlayer((480, 640), pipeline, camera)
player.run()
