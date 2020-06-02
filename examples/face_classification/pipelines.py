from tensorflow.keras.preprocessing.image import ImageDataGenerator
from paz.abstract import SequentialProcessor, ProcessingSequence, Processor
from paz.pipelines import PreprocessImage
import paz.processors as pr
import numpy as np


class ProcessGrayImage(SequentialProcessor):
    def __init__(self, size, num_classes, generator=None):
        super(ProcessGrayImage, self).__init__()
        self.size = size
        self.process = SequentialProcessor([pr.ExpandDims(-1)])
        if generator is not None:
            self.process.add(pr.ImageDataProcessor(generator))
        self.process.add(PreprocessImage((size, size), mean=None))
        self.process.add(pr.ExpandDims(-1))
        self.add(pr.UnpackDictionary(['image', 'label']))
        self.add(pr.ExpandDomain(self.process))
        self.add(pr.SequenceWrapper({0: {'image': [size, size, 1]}},
                                    {1: {'label': [num_classes]}}))


class FaceClassifier(Processor):
    def __init__(self, detector, classifier, labels, offsets):
        super(FaceClassifier, self).__init__()
        RGB2GRAY = pr.ConvertColorSpace(pr.RGB2GRAY)
        self.detect = pr.Predict(detector, RGB2GRAY, pr.ToBoxes2D())
        self.crop_boxes2D = pr.CropBoxes2D(offsets)
        preprocess = PreprocessImage(classifier.input_shape[1:3], None)
        preprocess.insert(0, RGB2GRAY)
        preprocess.add(pr.ExpandDims([0, 3]))
        self.classify = SequentialProcessor()
        self.classify.add(pr.Predict(classifier, preprocess))
        self.classify.add(pr.CopyDomain([0], [1]))
        self.classify.add(pr.ControlMap(pr.ToClassName(labels), [0], [0]))
        self.draw = pr.DrawBoxes2D(labels)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        boxes2D = self.detect(image)
        images = self.crop_boxes2D(image, boxes2D)
        for cropped_image, box2D in zip(images, boxes2D):
            box2D.class_name, scores = self.classify(cropped_image)
            box2D.score = np.amax(scores)
        image = self.draw(image, boxes2D)
        return self.wrap(image, boxes2D)


if __name__ == "__main__":
    import os
    from paz.datasets import FER, FERPlus

    # data generator and augmentations
    generator = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)

    pipeline = ProcessGrayImage(48, 8, generator)
    dataset = 'FERPlus'

    data_path = os.path.join(os.path.expanduser('~'), '.keras/paz/datasets/')
    name_to_manager = {'FER': FER, 'FERPlus': FERPlus}
    data_managers, datasets = {}, {}
    data_path = os.path.join(data_path, dataset)
    kwargs = {'path': data_path} if dataset in ['FERPlus'] else {}
    data_manager = name_to_manager[dataset](split='train', **kwargs)
    data = data_manager.load_data()

    sequence = ProcessingSequence(pipeline, 32, data)
    batch = sequence.__getitem__(0)
    show = pr.ShowImage()
    for arg in range(32):
        image = batch[0]['image'][arg][..., 0]
        image = 255 * image
        image = image.astype('uint8')
        show(image)
