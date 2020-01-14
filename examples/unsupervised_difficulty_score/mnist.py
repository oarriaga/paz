from tensorflow.keras.datasets import mnist
from tensorflow import keras
from paz.core import Loader


class MNIST(Loader):
    def __init__(self, split='train', class_names='all', image_size=(28, 28)):
        if class_names == 'all':
            class_names = self.get_class_names('MNIST')
        super(MNIST, self).__init__(None, split, class_names, 'MNIST')
        self.image_size = image_size
        self.split_to_arg = {'train': 0, 'test': 1}

    def load(self):
        images, labels = mnist.load_data()[self.split_to_arg[self.split]]
        images = images.reshape(
            len(images), self.image_size[0], self.image_size[1], 1)
        images = images.astype('float32')
        images = images / 255.0
        labels = keras.utils.to_categorical(labels, self.num_classes)
        return (images, labels)

    def get_class_names(self, dataset_name='MNIST'):
        if dataset_name == 'MNIST':
            return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        else:
            raise ValueError('Invalid dataset name')
