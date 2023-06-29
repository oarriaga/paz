import os
import h5py
import random
import tensorflow as tf
keras = tf.keras
from keras.utils import to_categorical


from .utils import get_class_names
from ..abstract import Generator
from ..backend.image import resize_image


class VVAD_LRS3(Generator):
    """Class for generating VVAD-LRS3 VVAD classification dataset.
    # Arguments
        path: String. Full path to vvadlrs3_faceImages_small.h5 file.
        split: String. Valid option contain 'train', 'val' or 'test'.
        val_split: Float. Percentage of the dataset to be used for validation (valid options between 0.0 to 1.0). Set to 0.0 to disable.
        test_split: Float. Percentage of the dataset to be used for testing (valid options between 0.0 to 1.0). Set to 0.0 to disable.
        image_size: List of length two. Indicates the shape in which
            the image will be resized.

    # References
        -[VVAD-LRS3](https://www.kaggle.com/datasets/adrianlubitz/vvadlrs3)
    """
    # TODO image resize
    # TODO to_categorical
    # TODO Data augmentation?
    def __init__(
            self, path=".keras/paz/datasets", split='train', val_split=0.2, test_split=0.1, image_size=(96, 96)):
        if split != 'train' and split != 'val' and split != 'test':
            raise ValueError('Invalid split name')

        path = os.path.join(os.path.expanduser('~'), path)
        path = os.path.join(path, 'vvadlrs3_faceImages_small.h5')

        class_names = get_class_names('VVAD_LRS3')

        super(VVAD_LRS3, self).__init__(path, split, class_names, 'VVAD_LRS3')
        self.image_size = image_size
        self.val_split = val_split
        self.test_split = test_split

    def __call__(self):
        data = h5py.File(self.path, mode='r')

        x_train = data.get("x_train")
        y_train = data.get("y_train")
        # x_train = data.get("x_test")
        # y_train = data.get("y_test")

        size_train = x_train.shape[0]

        # TODO add the 200 test samples to those
        indexes_pos = list(range(size_train // 2))
        indexes_neg = list(range(size_train // 2, size_train))

        indexes_val = []
        indexes_test = []

        size_bla = len(indexes_pos)

        # val split
        if self.val_split > 0.0:
            propotional_procent = self.val_split / 2
            indexes_val = indexes_pos[:int(propotional_procent * size_bla)] + indexes_neg[:int(propotional_procent * size_bla)]
            indexes_pos = indexes_pos[int(propotional_procent * len(indexes_pos)):]
            indexes_neg = indexes_neg[int(propotional_procent * len(indexes_neg)):]

        # test split
        if self.test_split > 0.0:
            propotional_procent = self.test_split / 2
            indexes_test = indexes_pos[:int(propotional_procent * size_bla)] + indexes_neg[:int(propotional_procent * size_bla)]
            indexes_pos = indexes_pos[int(propotional_procent * len(indexes_pos)):]
            indexes_neg = indexes_neg[int(propotional_procent * len(indexes_neg)):]

        indexes_train = indexes_pos + indexes_neg
        random.Random(445363).shuffle(indexes_train)  # Use always the same seed so every model gets the same shuffle

        # print("indexes_val", len(indexes_val))
        # print("indexes_test", len(indexes_test))
        # print("indexes_train", len(indexes))

        if self.split == 'train':
            indexes = indexes_train
        elif self.split == 'val':
            indexes = indexes_val
        elif self.split == 'test':
            indexes = indexes_test

        print("Selected split: ", self.split)
        print("indexes", len(indexes))

        for i in range(len(indexes)):
            yield (x_train[i], y_train[i])
