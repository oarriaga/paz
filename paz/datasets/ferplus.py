import numpy as np

from .utils import get_class_names
from ..core import Loader
from ..core import ops

# IMAGES_PATH = '../datasets/fer2013/fer2013.csv'
# LABELS_PATH = '../datasets/fer2013/fer2013new.csv'


class FERPlus(Loader):
    """Class for loading FER2013 [1] emotion classification dataset.
    # Arguments
        paths: List of strings. Full path to image and labels files e.g.
            ['/fullpath/fer2013.csv', '/fullpath/fer2013new.csv']
        split: String. Valid option contain 'train', 'val' or 'test'.
        class_names: String or list: If 'all' then it loads all default
            class names.
        image_size: List of length two. Indicates the shape in which
            the image will be resized.

    # References
    [1] kaggle.com/c/challenges-in-representation-learning-facial-\
            expression-recognition-challenge
    """
    def __init__(self, paths, split='train', class_names='all',
                 image_size=(48, 48)):

        if class_names == 'all':
            class_names = get_class_names('FERPlus')
        super(FERPlus, self).__init__(paths, split, class_names, 'FERPlus')

        self.image_size = image_size
        self.images_path, self.labels_path = self.paths
        self.split_to_filter = {
            'train': 'Training', 'val': 'PublicTest', 'test': 'PrivateTest'}

    def load_data(self):
        data = np.genfromtxt(self.images_path, str, '#', ',', 1)
        data = data[data[:, -1] == self.split_to_filter[self.split]]
        faces = np.zeros((len(data), *self.image_size, 1))
        for sample_arg, sample in enumerate(data):
            face = np.array(sample[1].split(' '), dtype=int).reshape(48, 48)
            face = ops.resize_image(face, self.image_size)
            faces[sample_arg, :, :, 0] = face

        emotions = np.genfromtxt(self.labels_path, str, '#', ',', 1)
        emotions = emotions[emotions[:, 0] == self.split_to_filter[self.split]]
        emotions = emotions[:, 2:10].astype(float)
        N = np.sum(emotions, axis=1)
        mask = N != 0
        N, faces, emotions = N[mask], faces[mask], emotions[mask]
        emotions = emotions / np.expand_dims(N, 1)
        return faces, emotions
