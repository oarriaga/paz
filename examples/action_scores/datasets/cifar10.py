from tensorflow.keras.datasets import cifar10
from tensorflow import keras
from paz.abstract import Loader

from .utils import get_class_names


class CIFAR10(Loader):
    def __init__(self, split='train', class_names='all', image_size=(32, 32)):
        if class_names == 'all':
            class_names = get_class_names('CIFAR10')
        super(CIFAR10, self).__init__(None, split, class_names, 'CIFAR10')
        self.image_size = image_size
        self.split_to_arg = {'train': 0, 'test': 1}

    def load_data(self):
        images, labels = cifar10.load_data()[self.split_to_arg[self.split]]
        images = images.reshape(
            len(images), self.image_size[0], self.image_size[1], 3)
        labels = keras.utils.to_categorical(labels, self.num_classes)
        data = []
        for image, label in zip(images, labels):
            sample = {'image': image, 'label': label}
            data.append(sample)
        return data
