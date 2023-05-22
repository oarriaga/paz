from tensorflow.keras.utils import get_file
from tensorflow import keras
from paz.abstract import Loader
import numpy as np
import os

from .utils import get_class_names


class KuzushijiMNIST(Loader):
    def __init__(self, split='train', class_names='all', image_size=(28, 28)):
        if class_names == 'all':
            class_names = get_class_names('MNIST')
        super(KuzushijiMNIST, self).__init__(
            None, split, class_names, 'KuzushijiMNIST')
        self.image_size = image_size
        self.root_origin = 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/'

    def load_data(self):
        data_parts = []
        for data_part in ['imgs', 'labels']:
            name = '-'.join(['kmnist', self.split, data_part + '.npz'])
            origin = os.path.join(self.root_origin, name)
            path = get_file(name, origin, cache_subdir='paz/datasets')
            with np.load(path, allow_pickle=True) as array:
                data_parts.append(array['arr_0'])
        images, labels = data_parts
        images = images.reshape(len(images), *self.image_size)
        labels = keras.utils.to_categorical(labels, self.num_classes)
        data = []
        for image, label in zip(images, labels):
            sample = {'image': image, 'label': label}
            data.append(sample)
        return data
