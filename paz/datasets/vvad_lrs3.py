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
    def __init__(
            self, path=".keras/paz/datasets", split='train', val_split=0.2, test_split=0.1, image_size=(96, 96), testing=False, evaluating=False):
        if split != 'train' and split != 'val' and split != 'test':
            raise ValueError('Invalid split name')
        if val_split < 0.0 or val_split > 1.0:
            raise ValueError('Invalid validation split')
        if test_split < 0.0 or test_split > 1.0:
            raise ValueError('Invalid test split')
        if val_split + test_split > 1.0:
            raise ValueError('The sum of val_split and test_split must be less than 1.0')

        path = os.path.join(path, 'vvadlrs3_faceImages_small.h5')

        class_names = get_class_names('VVAD_LRS3')

        super(VVAD_LRS3, self).__init__(path, split, class_names, 'VVAD_LRS3')
        self.image_size = image_size
        self.val_split = val_split
        self.test_split = test_split
        self.testing = testing
        self.evaluating = evaluating
        self.index = []

        data = h5py.File(self.path, mode='r')

        self.total_size = 0
        if not testing:
            self.total_size = data.get('x_train').shape[0]
        else:
            self.total_size = data.get('x_test').shape[0]
        data.close()

        # NotTODO add the 200 test samples to those (if so add those 200 to the self.total_size). It is not worth it. it is roughly 0.5% of the dataset but would add aditional commands to the dataset generator for each iteration. which could increase the training time.
        indexes_pos = list(range(self.total_size // 2))
        indexes_neg = list(range(self.total_size // 2, self.total_size))

        random.Random(445363).shuffle(indexes_pos)
        random.Random(848641).shuffle(indexes_neg)

        self.indexes_val = []
        self.indexes_test = []

        # val split
        if self.val_split > 0.0:
            val_split_size = int(self.val_split * 0.5 * self.total_size)
            self.indexes_val = indexes_pos[:val_split_size] + indexes_neg[:val_split_size]
            indexes_pos = indexes_pos[val_split_size:]
            indexes_neg = indexes_neg[val_split_size:]

        # test split
        if self.test_split > 0.0:
            test_split_size = int(self.test_split * 0.5 * self.total_size)
            self.indexes_test = indexes_pos[:test_split_size] + indexes_neg[:test_split_size]
            indexes_pos = indexes_pos[test_split_size:]
            indexes_neg = indexes_neg[test_split_size:]

        self.indexes_train = indexes_pos + indexes_neg

        random.seed(445363)

    def __call__(self):
        # print("indexes_val", len(self.indexes_val))
        # print("indexes_test", len(self.indexes_test))
        # print("indexes_train", len(self.indexes_train))
        indexes = []
        if self.split == 'train':
            indexes = self.indexes_train
            random.shuffle(indexes)  # Use always the same seed so every model gets the same shuffle
        elif self.split == 'val':
            indexes = self.indexes_val
        elif self.split == 'test':
            indexes = self.indexes_test

        # print("Selected split: ", self.split)
        # print("indexes", len(indexes))

        data = h5py.File(self.path, mode='r')
        if not self.testing:
            x_train = data.get("x_train")
            y_train = data.get("y_train")
        else:
            indexes = reversed(indexes)
            x_train = data.get("x_test")
            y_train = data.get("y_test")

        if self.evaluating:
            for i in indexes:
                self.index.append(i)
                yield x_train[i], y_train[i]
        else:
            for i in indexes:
                yield x_train[i], y_train[i]
        data.close()

    def get_index(self):
        return self.index.pop(0)

    def __len__(self):
        if self.total_size == -1:
            raise ValueError('You need to call __call__ first to set the total_size')

        if self.split == 'train':
            return self.total_size - int(self.val_split * 0.5 * self.total_size) * 2 - int(self.test_split * 0.5 * self.total_size) * 2
        elif self.split == 'val':
            return int(self.val_split * 0.5 * self.total_size) * 2
        elif self.split == 'test':
            print("total_size_test", self.total_size)
            return int(self.test_split * 0.5 * self.total_size) * 2