
from os import path
from scipy.io import loadmat
import numpy as np
from tensorflow.keras.utils import Progbar

from .utils import get_class_names
from ..abstract import Loader


class YCBVideo(Loader):
    """Dataset loader for the YCBVideo dataset.

    # Arguments
        path: String indicating full path to dataset
            e.g. /home/user/YCB_Video_Dataset/
        split: String determining the data split to load.
            e.g. `train`, `val` or `test`
        class_names: `all` or list. If list it should contain as elements
            strings indicating each class name.

    # References
        - [PoseCNN](https://arxiv.org/abs/1711.00199)

    # TODO
        One can't select to train only a subset of the class_names.
    """
    def __init__(self, path, split='train', class_names='all'):
        if class_names == 'all':
            class_names = get_class_names('YCBVideo')
        super(YCBVideo, self).__init__(path, split, class_names, 'YCBVideo')
        self.name_to_arg, self.arg_to_name = self._get_maps(self.class_names)

    def _get_maps(self, class_names):
        name_to_arg = dict(zip(class_names, range(len(class_names))))
        arg_to_name = dict(zip(range(len(class_names)), class_names))
        return name_to_arg, arg_to_name

    def load_data(self):
        sample_filenames = self._get_sample_filenames()
        progress_bar, data = Progbar(len(sample_filenames)), []
        for sample_arg, sample_id in enumerate(sample_filenames):
            data.append(self._load_sample(sample_id))
            progress_bar.update(sample_arg + 1)
        return data

    def _get_sample_filenames(self):
        set_filename = path.join(self.path, 'image_sets/', self.split + '.txt')
        base_names = np.genfromtxt(set_filename, dtype=str)
        data_path = path.join(self.path, 'data/')
        return [path.join(data_path, base_name) for base_name in base_names]

    def _load_sample(self, sample_id):
        box_data = self._load_box_data(sample_id)
        pose_data = self._load_poses(sample_id)
        postfixes = ['-color.png', '-depth.png', '-label.png']
        input_values = [sample_id + postfix for postfix in postfixes]
        inputs = dict(zip(['image', 'depth', 'segmentation'], input_values))
        targets = {'box_data': box_data, 'poses': pose_data}
        return inputs.update(targets)

    def _load_box_data(self, sample_id):
        box_data = np.genfromtxt(sample_id + '-box.txt', dtype=str)
        class_args = [self.name_to_arg[name] for name in box_data[:, 0]]
        class_args = np.asarray(class_args).reshape(-1, 1)
        box_data = box_data[:, 1:].astype(np.float32)
        box_data = np.concatenate([box_data, class_args], axis=1)
        box_data[:, 0] = box_data[:, 0] / 640.
        box_data[:, 1] = box_data[:, 1] / 480.
        box_data[:, 2] = box_data[:, 2] / 640.
        box_data[:, 3] = box_data[:, 3] / 480.
        return box_data

    def _load_poses(self, sample_id):
        mat = loadmat(sample_id + '-meta.mat')
        poses = np.moveaxis(mat['poses'], -1, 0)
        rotation_matrices, translations = poses[:, :3, :3], poses[:, :, 3]
        return [rotation_matrices, translations, mat['cls_indexes']]
