from tensorflow.keras.models import load_model
from paz.models import HaarCascadeDetector
from paz.datasets import get_class_names
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from paz.abstract import Processor, SequentialProcessor
import paz.processors as pr
import numpy as np


class FaceClassifier(Processor):
    def __init__(self, detector, classifier, labels, offsets):
        super(FaceClassifier, self).__init__()
        RGB2GRAY = pr.ConvertColorSpace(pr.RGB2GRAY)
        self.detect = pr.Predict(detector, RGB2GRAY, pr.ToBoxes2D())
        preprocess = SequentialProcessor([
            pr.ConvertColorSpace(pr.RGB2GRAY),
            pr.NormalizeImage(),
            pr.ResizeImage(classifier.input_shape[1:3]),
            pr.ExpandDims(0),
            pr.ExpandDims(3)])
        postprocessor = pr.ToClassName(labels)
        self.classify = pr.Predict(classifier, preprocess, postprocessor)
        self.crop_boxes2D = pr.CropBoxes2D(offsets)
        self.draw_boxes2D = pr.DrawBoxes2D(labels)
        self.to_uint8 = pr.CastImage(np.uint8)

    def call(self, image):
        boxes2D = self.detect(image)
        images = self.crop_boxes2D(image, boxes2D)
        for cropped_image, box2D in zip(images, boxes2D):
            box2D.class_name = self.classify(cropped_image)
        image = self.draw_boxes2D(image, boxes2D)
        image = self.to_uint8(image)
        return image


if __name__ == "__main__":
    detector = HaarCascadeDetector('frontalface_default')
    labels = get_class_names('FER')
    classifier = load_model('fer2013_mini_XCEPTION.119-0.65.hdf5')
    offsets = (0, 0)

    pipeline = FaceClassifier(detector, classifier, labels, offsets)
    camera = Camera(device_id=0)
    video_player = VideoPlayer((1280, 960), pipeline, camera)
    video_player.run()
