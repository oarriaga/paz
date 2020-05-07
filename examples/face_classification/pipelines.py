from tensorflow.keras.preprocessing.image import ImageDataGenerator
from paz.abstract import SequentialProcessor
from paz.pipelines import PreprocessImage
import paz.processors as pr
import numpy as np


class ImageAugmentation(SequentialProcessor):
    def __init__(self, generator, size, num_classes):
        super(ImageAugmentation, self).__init__()
        self.augment = SequentialProcessor(
            [pr.ExpandDims(-1), pr.ImageDataProcessor(generator)])
        self.preprocess = PreprocessImage((size, size), mean=None)
        self.process = SequentialProcessor([self.augment, self.preprocess])
        self.add(pr.UnpackDictionary(['image', 'label']))
        self.add(pr.ExpandFlow(self.process))
        self.add(pr.OutputWrapper({0: {'image': (size, size, 1)}},
                                  {1: {'label': (num_classes)}}))


generator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True)

pipeline = ImageAugmentation(generator, 48, 8)
sample_dict = {'image': np.zeros((48, 48)), 'label': np.zeros((8))}
sample = (sample_dict)
