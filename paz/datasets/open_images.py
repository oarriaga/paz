import os
import mmap

import numpy as np

from ..abstract import Loader


CLASS_DESCRIPTIONS_FILE = 'class-descriptions-boxable.csv'
BBOX_ANNOTATIONS_FILE = '{}-annotations-bbox.csv'


class OpenImages(Loader):
    """ Dataset loader for the OpenImagesV4 dataset.

    # Arguments
        path: String indicating full path to dataset
            e.g. /home/user/open_images/
        split: String determining the data split to load.
            e.g. `train`, `val` or `test`
        class_names: `all` or list. If list it should contain as elements
            the strings of the class names.

    """
    # TODO Allow selection of subset of class names.
    def __init__(self, path, split='train', class_names='all'):

        if split == 'val':
            split = 'validation'

        if split not in ['train', 'validation', 'test']:
            raise NameError('Invalid split name.')

        super(OpenImages, self).__init__(
            path, split, class_names, 'OpenImages')

        self.machine_to_human_name = dict()
        self.machine_to_arg = dict()
        self.load_class_names()
        self.class_distribution = dict()
        for class_name in self.class_names:
            self.class_distribution[class_name] = 0

    def load_class_names(self):
        classes_file = os.path.join(self.path, CLASS_DESCRIPTIONS_FILE)
        class_data = np.loadtxt(classes_file, delimiter=",", dtype=np.str)

        # class ID zero is background
        self.machine_to_arg['background'] = 0
        self.machine_to_human_name['background'] = 'background'
        class_names, class_arg = [], 1
        class_names.append('background')
        for machine_name, human_name in class_data:

            if self.class_names == 'all':
                self.machine_to_human_name[machine_name] = human_name
                self.machine_to_arg[machine_name] = class_arg
                class_names.append(human_name)
                class_arg = class_arg + 1

            elif human_name in self.class_names:
                self.machine_to_human_name[machine_name] = human_name
                self.machine_to_arg[machine_name] = class_arg
                class_names.append(human_name)
                class_arg = class_arg + 1

        self._class_names = class_names
        self._num_classes = len(self.machine_to_arg)
        print("Found {} {} classes".format(self.num_classes, self.split))

    def _get_num_lines(self, file_path):
        file_data = open(file_path, "r+")
        buf = mmap.mmap(file_data.fileno(), 0)
        lines = 0
        while buf.readline():
            lines = lines + 1
        return lines

    def load_data(self):

        data = dict()
        annotations_filepath = os.path.join(
            self.path, BBOX_ANNOTATIONS_FILE.format(self.split))
        # num_lines = self._get_num_lines(annotations_filepath)
        machine_names = self.machine_to_human_name.keys()
        # load file manually, line by line, in order to reduce memory usage
        with open(annotations_filepath, 'r') as annotations_file:
            # skip header
            annotations_file.readline()

            for line in annotations_file:
                row = line.split(",")

                image_filename = row[0] + ".jpg"
                x_min = float(row[4])
                x_max = float(row[5])
                y_min = float(row[6])
                y_max = float(row[7])

                machine_name = row[2]
                if machine_name not in machine_names:
                    continue

                human_name = self.machine_to_human_name[machine_name]

                absolute_image_path = os.path.join(
                    self.path, self.split, image_filename)

                if human_name in self.class_names:
                    class_arg = self.machine_to_arg[machine_name]

                    if absolute_image_path not in data:
                        data[absolute_image_path] = []

                    sample_data = [x_min, y_min, x_max, y_max, class_arg]
                    data[absolute_image_path].append(sample_data)
                    self.class_distribution[human_name] += 1

        formatted_data = []
        for image_path, ground_truth in data.items():
            sample = {'image': image_path, 'boxes': ground_truth}
            formatted_data.append(sample)

        msg = '{} split: loaded {} images with {} bounding box annotations'
        num_of_boxes = sum(self.class_distribution.values())
        print(msg.format(self.split, len(data), num_of_boxes))
        return formatted_data
