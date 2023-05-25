from tensorflow.keras.datasets import mnist
from tensorflow import keras
from paz.abstract import Loader

from .utils import get_class_names


class MNIST(Loader):
    def __init__(self, split='train', class_names='all', image_size=(28, 28)):
        if class_names == 'all':
            class_names = get_class_names('MNIST')
        super(MNIST, self).__init__(None, split, class_names, 'MNIST')
        self.image_size = image_size
        self.split_to_arg = {'train': 0, 'test': 1}

    def load_data(self):
        images, labels = mnist.load_data()[self.split_to_arg[self.split]]
        images = images.reshape(
            len(images), self.image_size[0], self.image_size[1])
        labels = keras.utils.to_categorical(labels, self.num_classes)
        data = []
        for image, label in zip(images, labels):
            sample = {'image': image, 'label': label}
            data.append(sample)
        return data
