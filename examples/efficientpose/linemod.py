import os
import yaml
import numpy as np
from paz.abstract import Loader
from pose import get_class_names


class LINEMOD(Loader):
    def __init__(self, path=None, object_id='08', split='train',
                 name='LINEMOD', evaluate=False,
                 image_size={'width': 512.0, 'height': 512.0}):
        self.path = path
        self.object_id = object_id
        self.split = split
        self.class_names_all = get_class_names('LINEMOD')
        self.evaluate = evaluate
        self.image_size = image_size
        self.arg_to_class = None
        self.object_id_to_class_arg = self._object_id_to_class_arg()
        self.class_name = self.class_names_all[
            self.object_id_to_class_arg[int(self.object_id)]]
        self.class_names = [self.class_names_all[0], self.class_name]
        super(LINEMOD, self).__init__(path, split, self.class_names, name)

    def load_data(self):
        if self.name == 'LINEMOD':
            ground_truth_data = self._load_LINEMOD(self.name, self.split)
        else:
            raise ValueError('Invalid name given.')
        return ground_truth_data

    def _load_LINEMOD(self, dataset_name, split):
        self.parser = LINEMODParser(self.object_id_to_class_arg, dataset_name,
                                    split, self.path, self.evaluate,
                                    self.object_id, self.class_names,
                                    self.image_size)
        self.arg_to_class = self.parser.arg_to_class
        ground_truth_data = self.parser.load_data()
        return ground_truth_data

    def _object_id_to_class_arg(self):
        return {0: 0, 1: 1, 5: 2, 6: 3, 8: 4, 9: 5, 10: 6, 11: 7, 12: 8}


class LINEMODParser(object):
    def __init__(self, object_id_to_class_arg, dataset_name='LINEMOD',
                 split='train', dataset_path='/Linemod_preprocessed/',
                 evaluate=False, object_id='08',
                 class_names=['background', 'driller'],
                 image_size={'width': 640.0, 'height': 480.0},
                 ground_truth_file='gt', info_file='info', data='data/'):

        if dataset_name != 'LINEMOD':
            raise Exception('Invalid dataset name.')

        self.dataset_name = dataset_name
        self.split = split
        self.dataset_path = dataset_path
        self.evaluate = evaluate
        self.object_id = object_id
        self.class_names = class_names
        self.image_size = image_size
        self.object_id_to_class_arg = object_id_to_class_arg
        self.ground_truth_file = ground_truth_file
        self.info_file = info_file
        self.data = data
        self.split_prefix = os.path.join(self.dataset_path, self.data)
        self.num_classes = len(self.class_names)
        class_keys = np.arange(self.num_classes)
        self.arg_to_class = dict(zip(class_keys, self.class_names))
        self.class_to_arg = {value: key for key, value
                             in self.arg_to_class.items()}
        self.data = []
        self._preprocess_files()

    def _load_filenames(self):
        split_file = (self.split_prefix + self.object_id
                      + '/' + self.split + '.txt')
        ground_truth_files = (self.split_prefix + self.object_id
                              + '/' + self.ground_truth_file + '.yml')
        info_file = (self.split_prefix + self.object_id
                     + '/' + self.info_file + '.yml')
        return [split_file, ground_truth_files, info_file]

    def _preprocess_files(self):
        data_file, ground_truth_file, info_file = self._load_filenames()

        with open(data_file, 'r') as file:
            data_file = [line.strip() for line in file.readlines()]
            file.close()

        with open(ground_truth_file, 'r') as file:
            ground_truth_data = yaml.safe_load(file)
            file.close()

        for datum_file in data_file:
            # Get image path
            image_path = (self.split_prefix + self.object_id
                          + '/' + 'rgb' + '/' + datum_file + '.png')

            # Compute bounding box
            file_id = int(datum_file)
            bounding_box = ground_truth_data[file_id][0]['obj_bb']
            x_min, y_min, W, H = bounding_box
            x_max = x_min + W
            y_max = y_min + H
            x_min = x_min / self.image_size['width']
            x_max = x_max / self.image_size['width']
            y_min = y_min / self.image_size['height']
            y_max = y_max / self.image_size['height']
            box_data = [x_min, y_min, x_max, y_max]
            box_data = np.asarray([box_data])

            annotations = ground_truth_data[file_id][0]
            # Get rotation vector
            rotation = annotations['cam_R_m2c']
            rotation = np.asarray(rotation)

            # Get translation vector
            translation_raw = annotations['cam_t_m2c']
            translation_raw = np.asarray(translation_raw)

            # Compute object class
            class_arg = 1

            # Append class to box data
            box_data = np.concatenate(
                (box_data, np.array([[class_arg]])), axis=-1)

            self.data.append({'image': image_path, 'boxes': box_data,
                              'rotation': rotation,
                              'translation_raw': translation_raw,
                              'class': class_arg})

    def load_data(self):
        return self.data
