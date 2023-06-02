import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from paz.abstract import Loader
from paz.backend.image import resize_image

from .utils import get_class_names


class FER(Loader):
    """Class for loading FER2013 emotion classification dataset.
    # Arguments
        path: String. Path to directory that has inside the files:
            `fer2013.csv`
        split: String. Valid option contain 'train', 'val' or 'test'.
        class_names: String or list: If 'all' then it loads all default
            class names.
        image_size: List of length two. Indicates the shape in which
            the image will be resized.

    # References
        -[FER2013 Dataset and Challenge](kaggle.com/c/challenges-in-\
            representation-learning-facial-expression-recognition-challenge)
    """

    def __init__(
            self, path, split='train', class_names='all', image_size=(48, 48)):

        if class_names == 'all':
            class_names = get_class_names('FER')

        path = os.path.join(path, 'fer2013.csv')
        super(FER, self).__init__(path, split, class_names, 'FER')
        self.image_size = image_size
        self._split_to_filter = {'train': 'Training', 'val': 'PublicTest',
                                 'test': 'PrivateTest'}

    def load_data(self):
        data = np.genfromtxt(self.path, str, delimiter=',', skip_header=1)
        data = data[data[:, -1] == self._split_to_filter[self.split]]
        faces = np.zeros((len(data), *self.image_size, 1))
        for sample_arg, sample in enumerate(data):
            face = np.array(sample[1].split(' '), dtype=int).reshape(48, 48)
            face = resize_image(face, self.image_size)
            faces[sample_arg, :, :, 0] = face
        emotions = to_categorical(data[:, 0].astype(int), self.num_classes)

        data = []
        for face, emotion in zip(faces, emotions):
            sample = {'image': face, 'label': emotion}
            data.append(sample)
        return data
