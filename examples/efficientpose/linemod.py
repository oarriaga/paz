import os
import yaml
from pose import get_class_names

import numpy as np
from paz.abstract import Loader


class LINEMOD(Loader):
    """ Dataset loader for the falling things dataset (FAT).

    # Arguments
        data_path: Data path to VOC2007 annotations
        split: String determining the data split to load.
            e.g. `train`, `val` or `test`
        class_names: `all` or list. If list it should contain as elements
            strings indicating each class name.
        name: String or list indicating with dataset or datasets to load.
            e.g. ``VOC2007`` or ``[''VOC2007'', VOC2012]``.
        with_difficult_samples: Boolean. If ``True`` flagged difficult boxes
            will be added to the returned data.
        evaluate: Boolean. If ``True`` returned data will be loaded without
            normalization for a direct evaluation.

    # Return
        data: List of dictionaries with keys corresponding to the image paths
        and values numpy arrays of shape ``[num_objects, 4 + 1]``
        where the ``+ 1`` contains the ``class_arg`` and ``num_objects`` refers
        to the amount of boxes in the image.

    """
    def __init__(self, path=None, object_id='08', split='train',
                 class_names='all', name='Linemod_preprocessed',
                 evaluate=False):

        super(LINEMOD, self).__init__(path, split, class_names, name)

        self.evaluate = evaluate
        self._class_names = class_names
        if class_names == 'all':
            self._class_names = get_class_names('LINEMOD')
        self.images_path = None
        self.arg_to_class = None
        self.object_id = object_id

    def load_data(self):
        if self.name == 'LINEMOD':
            ground_truth_data = self._load_LINEMOD(self.name, self.split)
        else:
            raise ValueError('Invalid name given.')
        return ground_truth_data

    def _load_LINEMOD(self, dataset_name, split):
        self.parser = LINEMODParser(dataset_name, split, self._class_names,
                                    self.path, self.evaluate, self.object_id)
        self.arg_to_class = self.parser.arg_to_class
        ground_truth_data = self.parser.load_data()
        return ground_truth_data


class LINEMODParser(object):
    """ Preprocess the VOC2007 xml annotations data.

    # TODO: Add background label

    # Arguments
        data_path: Data path to VOC2007 annotations

    # Return
        data: Dictionary which keys correspond to the image names
        and values are numpy arrays of shape (num_objects, 4 + 1)
        num_objects refers to the number of objects in that specific image
    """

    def __init__(self, dataset_name='LINEMOD', split='train',
                 class_names='all', dataset_path='/Linemod_preprocessed/',
                 evaluate=False, object_id='08', input_size=512.0,
                 ground_truth_file='gt', info_file='info'):

        if dataset_name != 'LINEMOD':
            raise Exception('Invalid dataset name.')

        # creating data set prefix paths variables
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.split = split
        self.split_prefix = os.path.join(self.dataset_path, 'data/')
        self.evaluate = evaluate
        self.object_id = object_id
        self.input_size = input_size
        self.ground_truth_file = ground_truth_file
        self.info_file = info_file

        self.class_names = class_names
        if self.class_names == 'all':
            self.class_names = get_class_names('LINEMOD')
        self.num_classes = len(self.class_names)
        class_keys = np.arange(self.num_classes)
        self.arg_to_class = dict(zip(class_keys, self.class_names))
        self.class_to_arg = {value: key for key, value
                             in self.arg_to_class.items()}
        self._object_id_to_class_arg()
        self.data = []
        self._preprocess_files()

    def _object_id_to_class_arg(self):
        self.object_id_to_class_arg = {0: 0, 8: 1}

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

        with open(info_file, 'r') as file:
            info_data = yaml.safe_load(file)
            file.close()

        for datum_file in data_file:
            # Get image path
            image_path = (self.split_prefix + self.object_id
                          + '/' + 'rgb' + '/' + datum_file + '.png')

            # Compute bounding box
            bounding_box = ground_truth_data[int(datum_file)][0]['obj_bb']
            x_min, y_min, W, H = bounding_box
            box_data = x_min, y_min, x_min + W, y_min + H
            box_data = np.asarray([box_data]) / self.input_size

            # Get rotation vector
            rotation = ground_truth_data[int(datum_file)][0]['cam_R_m2c']
            rotation = np.asarray(rotation)

            # Get translation vector
            translation_raw = ground_truth_data[int(datum_file)][0][
                'cam_t_m2c']
            translation_raw = np.asarray(translation_raw)

            # Compute object class
            obj_id = ground_truth_data[int(datum_file)][0]['obj_id']
            class_arg = self.object_id_to_class_arg[obj_id]

            # Append class to box data
            box_data = np.concatenate(
                (box_data, np.array([[class_arg]])), axis=-1)
            self.data.append({'image': image_path, 'boxes': box_data,
                              'rotation': rotation,
                              'translation_raw': translation_raw,
                              'class': class_arg})

    def load_data(self):
        return self.data
