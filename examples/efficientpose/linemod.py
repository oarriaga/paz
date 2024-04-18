import os
import yaml
import numpy as np
from paz.abstract import Loader
from paz.datasets.utils import get_class_names

B_LINEMOD_MEAN, G_LINEMOD_MEAN, R_LINEMOD_MEAN = 103.53, 116.28, 123.675
RGB_LINEMOD_MEAN = (R_LINEMOD_MEAN, G_LINEMOD_MEAN, B_LINEMOD_MEAN)
B_LINEMOD_STDEV, G_LINEMOD_STDEV, R_LINEMOD_STDEV = 57.375, 57.12, 58.395
RGB_LINEMOD_STDEV = (R_LINEMOD_STDEV, G_LINEMOD_STDEV, B_LINEMOD_STDEV)

LINEMOD_CAMERA_MATRIX = np.array([
    [572.41140, 000.00000, 325.26110],
    [000.00000, 573.57043, 242.04899],
    [000.00000, 000.00000, 001.00000]],
    dtype=np.float32)

LINEMOD_OBJECT_SIZES = {
    "ape":         np.array([075.86860000, 077.59920000, 091.76900000]),
    "can":         np.array([100.79160000, 181.79580000, 193.73400000]),
    "cat":         np.array([067.01070000, 127.63300000, 117.45660000]),
    "driller":     np.array([229.47600000, 075.47140000, 208.00200000]),
    "duck":        np.array([104.42920000, 077.40760000, 085.69700000]),
    "eggbox":      np.array([150.18460000, 107.07500000, 069.24140000]),
    "glue":        np.array([036.72110000, 077.86600000, 172.81580000]),
    "holepuncher": np.array([100.88780000, 108.49700000, 090.80000000]),
    }


class Linemod(Loader):
    """Dataset loader for the Linemod dataset.

    # Arguments
        path: Str, data path to Linemod annotations.
        object_id: Str, ID of the object to train.
        split: Str, determining the data split to load.
            e.g. `train`, `val` or `test`
        name: Str, or list indicating with dataset or datasets to
            load. e.g. ``VOC2007`` or ``[''VOC2007'', VOC2012]``.
        input_size: Dict, containing keys 'width' and 'height'
            with values equal to the input size of the model.

    # Return
        List of dictionaries with keys corresponding to the image
            paths and values numpy arrays of shape
            ``[num_objects, 4 + 1]`` where the ``+ 1`` contains the
            ``class_arg`` and ``num_objects`` refers to the amount of
            boxes in the image, pose transformation of shape
            ``[num_bixes, 11]`` and path to object mask image.
    """
    def __init__(self, path=None, object_id='08', split='train',
                 name='Linemod', input_size=(512, 512)):
        self.path = path
        self.object_id = object_id
        self.split = split
        self.input_size = input_size
        object_id_to_class_arg = {0: 0, 1: 1, 5: 2, 6: 3, 8: 4,
                                  9: 5, 10: 6, 11: 7, 12: 8}
        class_names = compute_class_names(object_id, object_id_to_class_arg)
        super(Linemod, self).__init__(path, split, class_names, name)

    def load_data(self):
        self.parser = LinemodParser(self.split, self.path,
                                    self.object_id, self.input_size)
        return self.parser.load_data()


def compute_class_names(object_id, object_id_to_class_arg):
    """Creates a list containing class names. The list includes
    `background` class and the object class e.g. `driller` as strings.

    # Arguments:
        object_id: Str, ID of the object to train.
        object_id_to_class_arg: Dict, containing mapping
            from object IDs to class arguments.

    # Return:
        List: Containing background and object class names as strings.
    """
    class_arg = object_id_to_class_arg[int(object_id)]
    class_names_all = get_class_names('Linemod')
    foreground_class = class_names_all[class_arg]
    background_class = class_names_all[0]
    return [background_class, foreground_class]


class LinemodParser(object):
    """Preprocess the Linemod yaml annotations data.

    # Arguments
        split: Str, determining the data split to load.
            e.g. `train`, `val` or `test`
        dataset_path: Str, data path to Linemod annotations.
        object_id: Str, ID of the object to train.
        input_size: Dict, containing keys 'width' and 'height'
            with values equal to the input size of the model.
        data_path: Str, containing path to the Linemod data folder.
        ground_truth_file: Str, name of the file
            containing ground truths.
        info_file: Str, name of the file containing info.
        image_path: Str, containing path to the RGB images.
        mask_path: Str, containing path to the object mask images.
        class_arg: Int, class argument of object class.

    # Return
        Dict, with keys correspond to the image and mask image
            names and values are numpy arrays for boxes, rotation,
            translation and integer for class.
    """
    def __init__(self, split='train', dataset_path='/Linemod_preprocessed/',
                 object_id='08', input_size=(512, 512), data_path='data/',
                 ground_truth_file='gt', info_file='info', image_path='rgb/',
                 mask_path='mask/', class_arg=1):
        self.split = split
        self.dataset_path = dataset_path
        self.object_id = object_id
        self.input_size = input_size
        self.data_path = data_path
        self.ground_truth_file = ground_truth_file
        self.info_file = info_file
        self.image_path = image_path
        self.mask_path = mask_path
        self.class_arg = class_arg

    def load_data(self):
        return load(self.dataset_path, self.data_path, self.object_id,
                    self.ground_truth_file, self.info_file, self.split,
                    self.image_path, self.input_size, self.class_arg,
                    self.mask_path)


def load(dataset_path, data_path, object_id, ground_truth_file, info_file,
         split, image_path, input_size, class_arg, mask_path):
    """Preprocess the Linemod yaml annotations data.

    # Arguments
        dataset_path: Str, data path to Linemod annotations.
        data_path: Str, containing path to the Linemod data folder.
        object_id: Str, ID of the object to train.
        ground_truth_file: Str, name of the file
            containing ground truths.
        info_file: Str, name of the file containing info.
        split: Str, determining the data split to load.
            e.g. `train`, `val` or `test`
        image_path: Str, containing path to the RGB images.
        input_size: Dict, containing keys 'width' and 'height'
            with values equal to the input size of the model.
        class_arg: Int, class argument of object class.
        mask_path: Str, containing path to the object mask images.

    # Return
        data: Dict, with keys correspond to the image and mask image
            names and values are numpy arrays for boxes, rotation,
            translation and integer for class.
    """
    root_path = os.path.join(dataset_path, data_path, object_id)
    files = load_linemod_filenames(root_path, ground_truth_file,
                                   info_file, split)
    ground_truth_file, info_file, split_file = files
    split_file = open_file(split_file)
    annotation = open_file(ground_truth_file)

    data = []
    for split_data in split_file:
        raw_image_path = make_image_path(root_path, image_path, split_data)
        # Process bounding box
        box = get_data(split_data, annotation, key='obj_bb')
        box = linemod_to_corner_form(box)
        box = normalize_box_input_size(box, input_size)
        box = np.concatenate((box, np.array([[class_arg]])), axis=-1)
        # Load rotation and translation
        rotation = get_data(split_data, annotation, key='cam_R_m2c')
        translation = get_data(split_data, annotation, key='cam_t_m2c')
        raw_mask_path = make_image_path(root_path, mask_path, split_data)
        data.append({'image': raw_image_path, 'boxes': box,
                     'rotation': rotation, 'translation_raw': translation,
                     'class': class_arg, 'mask': raw_mask_path})
    return data


def load_linemod_filenames(root_path, ground_truth_file, info_file, split):
    """Composes path to ground truth file, info file and split file
    of Linemod dataset.

    # Arguments
        root_path: Str, root directory path to Linemod dataset.
        ground_truth_file: Str, name of the file
            containing ground truths.
        info_file: Str, name of the file containing info.
        split: Str, determining the data split to load.
            e.g. `train`, `val` or `test`

    # Return
        List: containing path to ground truth file, info file
            and split file.
    """
    ground_truth_file = '{}.{}'.format(ground_truth_file, 'yml')
    info_file = '{}.{}'.format(info_file, 'yml')
    split_file = '{}.{}'.format(split, 'txt')
    return [os.path.join(root_path, ground_truth_file),
            os.path.join(root_path, info_file),
            os.path.join(root_path, split_file)]


def open_file(file):
    """Opens a file given by a file handle.

    # Arguments
        file: Str, name of the file to be opened.

    # Return
        file_contents: List of strings containing file contents.
    """
    file_to_parser = {'.txt': parse_txt,
                      '.yml': parse_yml}
    file_name, file_extension = os.path.splitext(file)
    parser = file_to_parser[file_extension]
    with open(file, 'r') as f:
        file_contents = parser(f)
    f.close()
    return file_contents


def parse_txt(file_handle):
    """Parses given text file.

    # Arguments
        file_handle: Filehandle, of the file to be parsed.

    # Return
        List of strings containing file contents.
    """
    return [line.strip() for line in file_handle.readlines()]


def parse_yml(file_handle):
    """Parses given yaml file.

    # Arguments
        file_handle: Filehandle, of the file to be parsed.

    # Return
        Dictionary containing file contents.
    """
    return yaml.safe_load(file_handle)


def make_image_path(root_path, image_path, split_data, image_extension='png'):
    """Composes path to image specified by `image_path`.

    # Arguments
        root_path: Str, root directory path to Linemod dataset.
        image_path: Str, containing path to the RGB image.
        split_data: Str, name of the image file.
        image_extension: Str, file extension of the image.

    # Return
        Str: containing path to the image.
    """
    file_name = '{}.{}'.format(split_data, image_extension)
    return os.path.join(root_path, image_path, file_name)


def get_data(file_id, data, key):
    """Fetches data from file contents indexed by dictionary keys.

    # Arguments
        file_id: Str, containing path to the RGB image.
        data: Dictionary containing file contents.
        key: Str, key name of data to be fetched.

    # Return
        Array: containing the fetched data.
    """
    file_key = int(file_id)
    data = np.asarray(data[file_key][0][key])
    return np.expand_dims(data, axis=0)


def linemod_to_corner_form(box):
    """Converts bounding box from Linemod form to corner form.
    The Linemod form of bounding box is `[x_min, y_min, W, H]`.

    # Arguments
        box: Array, of shape `[1, 4]`.

    # Return
        Array: of shape `[1, 4]` with box in corner form.
    """
    x_min, y_min, W, H = box[0][0], box[0][1], box[0][2], box[0][3]
    x_max = x_min + W
    y_max = y_min + H
    return np.array([[x_min, y_min, x_max, y_max]])


def normalize_box_input_size(box, input_size):
    """Normalizes bounding box to model's input size.

    # Arguments
        box: Array, of shape `[1, 4]`
        input_size: List, containing wdth and height
            of input layer of model.

    # Return
        Array: of shape `[1, 4]` containing normalized box coordinates.
    """
    x_min, y_min = box[0][0], box[0][1]
    x_max, y_max = box[0][2], box[0][3]
    input_W, input_H = input_size
    x_min = x_min / input_W
    x_max = x_max / input_W
    y_min = y_min / input_H
    y_max = y_max / input_H
    box = [x_min, y_min, x_max, y_max]
    return np.array([[x_min, y_min, x_max, y_max]])
