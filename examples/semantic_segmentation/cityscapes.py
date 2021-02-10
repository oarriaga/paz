import os
import glob
from paz.abstract import Loader


class CityScapes(Loader):
    def __init__(self, image_path, label_path, split, class_names):
        if split not in ['train', 'val', 'test']:
            raise ValueError('Invalid split name:', split)
        self.image_path = os.path.join(image_path, split)
        self.label_path = os.path.join(label_path, split)
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


label_path = '/home/octavio/Downloads/dummy/gtFine/'
image_path = '/home/octavio/Downloads/dummy/RGB_images/leftImg8bit/'
data_manager = CityScapes(image_path, label_path, 'train', None)
dataset = data_manager.load_data()
