import jax.numpy as jp
from paz.applications import MiniXception

model = MiniXception()


def normalize(image):
    return image / 255.0


def rgb_to_gray(image):
    rgb_weights = jp.array([0.2989, 0.5870, 0.1140], dtype=image.dtype)
    grayscale = jp.tensordot(image, rgb_weights, axes=(-1, -1))
    grayscale = jp.expand_dims(grayscale, axis=-1)
    return grayscale


def to_class_name(probabilities, labels):
    return labels[jp.argmax(probabilities)]


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
