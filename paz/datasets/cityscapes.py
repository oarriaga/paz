import os
import glob

from paz.abstract import Loader

from .utils import get_class_names


class CityScapes(Loader):
    """CityScapes data manager for loading the paths of the RGB and
        segmentation masks.

    # Arguments
        image_path: String. Path to RGB images e.g. '/home/user/leftImg8bit/'
        label_path: String. Path to label masks e.g. '/home/user/gtFine/'
        split: String. Valid option contain 'train', 'val' or 'test'.
        class_names: String or list: If 'all' then it loads all default
            class names.

    # References
        -[The Cityscapes Dataset for Semantic Urban Scene Understanding](
        https://www.cityscapes-dataset.com/citation/)
    """
    def __init__(self, image_path, label_path, split, class_names='all'):
        if split not in ['train', 'val', 'test']:
            raise ValueError('Invalid split name:', split)
        self.image_path = os.path.join(image_path, split)
        self.label_path = os.path.join(label_path, split)
        if class_names == 'all':
            class_names = get_class_names('CityScapes')
        super(CityScapes, self).__init__(
            None, split, class_names, 'CityScapes')

    def load_data(self):
        image_path = os.path.join(self.image_path, '*/*.png')
        label_path = os.path.join(self.label_path, '*/*labelIds.png')
        image_paths = glob.glob(image_path)
        label_paths = glob.glob(label_path)
        image_paths = sorted(image_paths)
        label_paths = sorted(label_paths)
        assert len(image_paths) == len(label_paths)
        dataset = []
        for image_path, label_path in zip(image_paths, label_paths):
            sample = {'image_path': image_path, 'label_path': label_path}
            dataset.append(sample)
        return dataset
