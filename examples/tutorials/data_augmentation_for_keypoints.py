from tensorflow.keras.preprocessing.image import ImageDataGenerator
from paz.abstract import SequentialProcessor, Processor
import paz.processors as pr


class AugmentGrayImages(Processor):
    def __init__(self, shape, rotation_range=30, width_shift_range=0.1,
                 height_shift_range=0.1, zoom_range=0.1,
                 horizontal_flip=True):

        super(AugmentGrayImages, self).__init__()

        self.generator = ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip)

        self.shape = shape
        self.augment_image = SequentialProcessor()
        self.augment_image.add(pr.ResizeImage(self.shape))
        self.augment_image.add(pr.CastImage(float))
        self.augment_image.add(pr.ExpandDims(-1))
        self.augment_image.add(pr.NormalizeImage())
        self.augment_image.add(pr.ImageDataProcessor(self.generator))
        self.wrapper = pr.OutputWrapper(self.input_names, self.label_names)

    @property
    def input_names(self):
        return ['image']

    @property
    def label_names(self):
        return ['label']

    @property
    def input_shapes(self):
        return [(*self.shape, 1)]

    @property
    def label_shapes(self):
        return [(7)]

    def call(self, image, labels):
        image = self.augment_image(image)
        return self.wrapper([image], [labels])


shape = (48, 48)
input_names = ['image']
label_names = ['label']
generator = ImageDataGenerator(rotation_range=30, horizontal_flip=True)
augmentator = SequentialProcessor()
augmentator.add(pr.ExtendInputs(pr.ResizeImage(shape)))
augmentator.add(pr.ExtendInputs(pr.CastImage(float)))
augmentator.add(pr.ExtendInputs(pr.ExpandDims(-1)))
augmentator.add(pr.ExtendInputs(pr.NormalizeImage()))
augmentator.add(pr.ExtendInputs(pr.ImageDataProcessor(generator)))
augmentator.add(pr.OutputWrapper(input_names, label_names))

augment_image = SequentialProcessor()
augment_image.add(pr.ResizeImage(shape))
augment_image.add(pr.CastImage(float))
augment_image.add(pr.ExpandDims(-1))
augment_image.add(pr.NormalizeImage())
augment_image.add(pr.ImageDataProcessor(generator))
augmentator_B = SequentialProcessor()
augmentator_B.add(pr.ExtendInputs(augment_image))
augmentator_B.add(pr.OutputWrapper(input_names, label_names))


# TODO: Maybe we can calculate the shapes directly from the output instead of
# relying the user to give them directly.
# However, this might introduce multiple bugs for the user.
class Pipeline(Processor):
    def __init__(self, input_processor, label_processor, processor_wrapper):
        super(Pipeline, self).__init__()
        self.input_processor = input_processor
        self.label_processor = label_processor
        self.processor_wrapper = processor_wrapper

    @property
    def input_names(self):
        return self.processor_wrapper.input_names

    @property
    def label_names(self):
        return self.processor_wrapper.label_names

    def call(self, inputs, labels):
        inputs = self.input_processor(inputs)
        labels = self.label_processor(labels)
        # not considering when inputs and labels are mixed together
        return self.processor_wrapper([inputs], [labels])


import numpy as np
augment = AugmentGrayImages((48, 48))
image = np.zeros((48, 48, 1))
label = np.zeros((7))
results = augment(image, label)
print(results)

print(augmentator(image, label))
print(augmentator_B(image, label))
