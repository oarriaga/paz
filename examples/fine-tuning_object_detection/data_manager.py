import os
import csv
import numpy as np
from pathlib import Path

from paz.abstract import Loader


class CSVLoader(Loader):
    def __init__(
            self, path, class_names, image_size=(1280, 720), split='train'):
        super(CSVLoader, self).__init__(path, split, class_names, 'CSVLoader')
        self.class_to_arg = self.build_class_to_arg(self.class_names)
        self.image_size = image_size

    def build_class_to_arg(self, class_names):
        args = list(range(len(class_names)))
        return dict(zip(args, class_names))

    def load_data(self):
        file = open(self.path, 'r')
        csv_reader = csv.reader(file, delimiter=',')
        dataset = []
        H, W = self.image_size
        for row in csv_reader:
            image_name, class_arg, x_min, y_min, x_max, y_max = row
            path = os.path.dirname(self.path)
            image_path = os.path.join(path, image_name + '.png')
            image_path = os.path.abspath(image_path)
            if not Path(image_path).is_file():
                raise ValueError('File %s not found.\n' % image_path)
            box_data = [[int(x_min) / H, int(y_min) / W,
                         int(x_max) / H, int(y_max) / W, int(class_arg)]]
            box_data = np.array(box_data)
            sample = {'image': image_path, 'boxes': box_data}
            dataset.append(sample)
        return dataset


if __name__ == "__main__":
    path = 'datasets/solar_panel/BoundingBox.txt'
    class_names = ['background', 'solar_panel']
    data_manager = CSVLoader(path, class_names)
    dataset = data_manager.load_data()
