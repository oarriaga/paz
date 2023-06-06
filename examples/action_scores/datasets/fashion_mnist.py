from tensorflow.keras.datasets import fashion_mnist
from tensorflow import keras
from paz.abstract import Loader

from .utils import get_class_names


class FashionMNIST(Loader):
    def __init__(self, split='train', class_names='all', image_size=(28, 28)):
        if class_names == 'all':
            class_names = get_class_names('FashionMNIST')
        super(FashionMNIST, self).__init__(
            None, split, class_names, 'FashionMNIST')
        self.image_size = image_size
        self.split_to_arg = {'train': 0, 'test': 1}

    def load_data(self):
        split_arg = self.split_to_arg[self.split]
        images, labels = fashion_mnist.load_data()[split_arg]
        images = images.reshape(
            len(images), self.image_size[0], self.image_size[1])
        labels = keras.utils.to_categorical(labels, self.num_classes)
        data = []
        for image, label in zip(images, labels):
            sample = {'image': image, 'label': label}
            data.append(sample)
        return data
