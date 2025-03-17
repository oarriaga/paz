import os

os.environ["KERAS_BACKEND"] = "jax"

import jax.numpy as jp
import paz
from paz.applications import MiniXceptionFER

# model = MiniXception()


def to_labels(probabilities, labels):
    return labels[jp.argmax(probabilities)]


def preprocess(image, shape):
    image = paz.image.resize(image, shape)
    image = paz.image.normalize(image)
    image = paz.image.rgb_to_gray(image)
    image = jp.expand_dims(image, [0, -1])
    return image


def ClassifyMiniXceptionFER():
    x = image = paz.Input("image")
    x = paz.Node(preprocess, (48, 48))(x)
    x = paz.Node(MiniXceptionFER(), name="score")(x)
    x = paz.Node(to_labels, paz.datasets.labels("FER"), name="label")(x)
    return paz.Model([image], [x], "MiniXceptionFER")


# score = MiniXception((48, 48, 1), 7, weights="FER")
model = ClassifyMiniXceptionFER()
y = model(jp.full((128, 128, 3), 255))


"""
class MiniXceptionFER(SequentialProcessor):
    def __init__(self):
        super(MiniXceptionFER, self).__init__()
        self.classifier = MiniXception((48, 48, 1), 7, weights="FER")
        self.class_names = get_class_names("FER")

        preprocess = PreprocessImage(self.classifier.input_shape[1:3], None)
        preprocess.insert(0, pr.ConvertColorSpace(pr.RGB2GRAY))
        preprocess.add(pr.ExpandDims(0))
        preprocess.add(pr.ExpandDims(-1))
        self.add(pr.Predict(self.classifier, preprocess))
        self.add(pr.CopyDomain([0], [1]))
        self.add(pr.ControlMap(pr.ToClassName(self.class_names), [0], [0]))
        self.add(pr.WrapOutput(["class_name", "scores"]))
"""
