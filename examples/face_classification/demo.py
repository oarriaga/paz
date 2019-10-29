from paz.models import HaarCascadeDetector
from paz.core import Processor
from paz import processors as pr
from paz.datasets import get_class_names
from paz.core import VideoPlayer
from tensorflow.keras.models import load_model
import numpy as np


class FaceClassifier(Processor):
    def __init__(self, detector, classifier, offsets, class_names):
        super(FaceClassifier, self).__init__()
        processors = [pr.ConvertColor('BGR', 'GRAY')]
        self.detect = pr.Predict(detector, 'image', 'boxes', processors)
        self.to_boxes = pr.ToBoxes2D()
        self.crop_images = pr.CropBoxes2D(offsets)
        self.to_gray = pr.ConvertColor('BGR', 'GRAY')
        self.normalize = pr.NormalizeImage()
        self.resize = pr.Resize(classifier.input_shape[1:3])
        self.classify = pr.Predict(classifier, 'image')
        self.to_label = pr.ToLabel(class_names)
        self.draw_boxes = pr.DrawBoxes2D(class_names)
        self.to_int8 = pr.CastImageToInts()

    def call(self, kwargs):
        x = self.detect(kwargs)
        x = self.to_boxes(x)
        x = self.crop_images(x)
        for image, box2D in zip(x['image_crops'], x['boxes2D']):
            i = self.to_gray({'image': image})
            i = self.normalize(i)
            i = self.resize(i)
            i['image'] = np.expand_dims(np.expand_dims(i['image'], 0), 3)
            i = self.classify(i)
            i = self.to_label(i)
            box2D.class_name = i['predictions']
        x = self.draw_boxes(x)
        x = self.to_int8(x)
        return x


detector = HaarCascadeDetector('frontalface_default')
class_names = get_class_names('FER')
classifier = load_model('fer2013_mini_XCEPTION.119-0.65.hdf5')
offsets = (0, 0)

pipeline = FaceClassifier(detector, classifier, offsets, class_names)
video_player = VideoPlayer((1280, 960), pipeline)
video_player.start()
