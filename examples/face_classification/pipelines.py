from tensorflow.keras.preprocessing.image import ImageDataGenerator
from paz.abstract import SequentialProcessor, ProcessingSequence
from paz.pipelines import PreprocessImage
import paz.processors as pr


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
