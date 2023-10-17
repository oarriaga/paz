
import os
import json
import numpy as np
from glob import glob
from ..abstract import Loader
from .utils import get_class_names
from tensorflow.keras.utils import Progbar
from paz.backend.groups import quaternion_to_rotation_matrix


class FAT(Loader):
    """ Dataset loader for the falling things dataset (FAT).
    # Arguments
        path: String indicating full path to dataset
            e.g. /home/user/fat/
        split: String determining the data split to load.
            e.g. `train`, `val` or `test`
        class_names: `all` or list. If list it should contain as elements
            strings indicating each class name.

    # References
        - [Deep Object Pose
            Estimation (DOPE)](https://github.com/NVlabs/Deep_Object_Pose)
    """
    def __init__(self, path, class_names='all'):
        self.class_names = class_names
        self.class_list = get_class_names('FAT')
        self.data = []
        if class_names == 'all':
            self.class_to_arg = dict(
                zip(self.class_list, list(range(len(self.class_list)))))
        else:
            self.class_names = class_names
            mask = np.isin(self.class_list, self.class_names)
            indices = np.where(mask)[0].tolist()
            self.class_to_arg = dict(zip(self.class_names, indices))
        super(FAT, self).__init__(path, 'None', self.class_names, 'FAT')

    def load_data(self):
        if self.class_names == 'all':
            self.data = self.load_data_mixed()
            for single_class in self.class_list:
                self.data = self.data + self.load_data_single(single_class)
        else:
            for single_class in self.class_names:
                self.data = self.data + self.load_data_single(single_class)
        return self.data

    def load_data_mixed(self):
        scene_names = glob(self.path + 'mixed/*')
        image_paths, label_paths = [], []
        for scene_name in scene_names:
            scene_image_paths, scene_label_paths = [], []
            for image_side in ['left', 'right']:
                image_names = glob(scene_name + '/*%s.jpg' % image_side)
                side_image_paths = sorted(image_names, key=self._base_number)
                label_names = glob(scene_name + '/0*%s.json' % image_side)
                side_label_paths = sorted(label_names, key=self._base_number)
                scene_image_paths = scene_image_paths + side_image_paths
                scene_label_paths = scene_label_paths + side_label_paths
            image_paths = image_paths + scene_image_paths
            label_paths = label_paths + scene_label_paths

        data = []
        progress_bar = Progbar(len(image_paths))
        for sample_arg, sample in enumerate(zip(image_paths, label_paths)):
            image_path, label_path = sample
            if not self._valid_name_match(image_path, label_path):
                raise ValueError('Invalid name match:', image_path, label_path)
            boxes = self._extract_boxes(label_path)
            poses = self._extract_poses(label_path)
            if boxes is None:
                continue
            data.append({'image': image_path, 'boxes': boxes, 'poses': poses})
            progress_bar.update(sample_arg + 1)
        return data

    def load_data_single(self, class_names):
        object_name = class_names + '_16k'
        scene_names = glob(self.path + 'single/' + object_name + '/*/')
        image_paths, label_paths = [], []
        for scene_name in scene_names:
            scene_image_paths, scene_label_paths = [], []
            for image_side in ['left', 'right']:
                image_names = glob(scene_name + '/*%s.jpg' % image_side)
                side_image_paths = sorted(image_names, key=self._base_number)
                label_names = glob(scene_name + '/0*%s.json' % image_side)
                side_label_paths = sorted(label_names, key=self._base_number)
                scene_image_paths = scene_image_paths + side_image_paths
                scene_label_paths = scene_label_paths + side_label_paths
            image_paths = image_paths + scene_image_paths
            label_paths = label_paths + scene_label_paths
        progress_bar = Progbar(len(image_paths))
        data = []
        for sample_arg, sample in enumerate(zip(image_paths, label_paths)):
            image_path, label_path = sample
            if not self._valid_name_match(image_path, label_path):
                raise ValueError('Invalid name match:', image_path, label_path)
            boxes = self._extract_boxes(label_path)
            poses = self._extract_poses(label_path)
            if boxes is None:
                continue
            data.append({'image': image_path, 'boxes': boxes, 'poses': poses})
            progress_bar.update(sample_arg + 1)
        return data

    def split_data(self, data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        total_ratio = train_ratio + val_ratio + test_ratio

        if total_ratio != 1.0:
            raise ValueError("Split ratio proportion is not correct")

        data_length = len(data)
        train_split = int(data_length * train_ratio)
        val_split = int(data_length * val_ratio)
        test_split = int(data_length * test_ratio)

        train_data = data[:train_split]
        val_data = data[train_split:train_split + val_split]
        test_data = data[-test_split:]
        return [train_data, val_data, test_data]

    def _extract_boxes(self, json_filename):
        json_data = json.load(open(json_filename, 'r'))
        num_objects = len(json_data['objects'])
        if num_objects == 0:
            return None
        box_data = np.zeros((num_objects, 5))
        for object_arg, object_data in enumerate(json_data['objects']):
            bounding_box = object_data['bounding_box']
            y_min, x_min = bounding_box['top_left']
            y_max, x_max = bounding_box['bottom_right']
            x_min, y_min = x_min / 960., y_min / 540.
            x_max, y_max = x_max / 960., y_max / 540.
            box_data[object_arg, :4] = x_min, y_min, x_max, y_max
            class_name = object_data['class'][:-4]
            box_data[object_arg, -1] = self.class_to_arg[class_name]
        return box_data

    def _extract_poses(self, json_filename):
        json_data = json.load(open(json_filename, 'r'))
        num_objects = len(json_data['objects'])
        if num_objects == 0:
            return None
        pose_data = np.zeros((num_objects, 8))
        for object_arg, object_data in enumerate(json_data['objects']):
            translation = object_data['location']
            quaternion = object_data['quaternion_xyzw']
            pose_data[object_arg, :3] = translation
            pose_data[object_arg, 3:7] = quaternion
            class_name = object_data['class'][:-4]
            pose_data[object_arg, -1] = self.class_to_arg[class_name]
        return pose_data

    def _base_number(self, filename):
        order = os.path.basename(filename)
        order = order.split('.')[0]
        order = float(order)
        return order

    def _valid_name_match(self, image_path, label_path):
        image_name = os.path.basename(image_path)
        label_name = os.path.basename(label_path)
        return image_name[:-3] == label_name[:-4]
