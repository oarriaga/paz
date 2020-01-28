from tensorflow.keras.datasets import mnist
from tensorflow import keras
from paz.core import Loader, SequentialProcessor
from paz import processors as pr


class MNIST(Loader):
    def __init__(self, split='train', class_names='all', image_size=(28, 28)):
        if class_names == 'all':
            class_names = get_class_names('MNIST')
        super(MNIST, self).__init__(None, split, class_names, 'MNIST')
        self.image_size = image_size
        self.split_to_arg = {'train': 0, 'test': 1}

    def load(self):
        images, labels = mnist.load_data()[self.split_to_arg[self.split]]
        images = images.reshape(
            len(images), self.image_size[0], self.image_size[1])
        labels = keras.utils.to_categorical(labels, self.num_classes)
        data = []
        for image, label in zip(images, labels):
            sample = {'image': image, 'label': label}
            data.append(sample)
        return data


def get_class_names(self, dataset_name='MNIST'):
    if dataset_name == 'MNIST':
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    else:
        raise ValueError('Invalid dataset name')


class ImageAugmentation(SequentialProcessor):
    def __init__(self, size, num_classes, split='train'):
        super(ImageAugmentation, self).__init__()
        if split not in ['train', 'val', 'test']:
            raise ValueError('Invalid split mode')

        self.size = size
        self.num_classes = num_classes
        self.split = split

        self.add(pr.CastImageToFloat())
        self.add(pr.ResizeImage((self.size, self.size)))
        self.add(pr.ExpandDims(axis=-1, topic='image'))
        self.add(pr.NormalizeImage())
        self.add(pr.OutputSelector(['image'], ['label']))

    @property
    def input_shapes(self):
        return [(self.size, self.size, 1)]

    @property
    def label_shapes(self):
        return [(self.num_classes, )]
