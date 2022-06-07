import os
import glob
import numpy as np
from scipy.io import loadmat

import paz.processors as pr
from paz.abstract import Loader
from paz.backend.image import load_image


class HandDataset(Loader):
    def __init__(self, path, split):
        split_to_directory = {pr.TRAIN: 'training_dataset/training_data',
                              pr.VAL: 'validation_dataset/validation_data',
                              pr.TEST: 'test_dataset/test_data'}
        names = ['background', 'hand']
        path = os.path.join(path, split_to_directory[split])
        super(HandDataset, self).__init__(path, split, names, 'HandDataset')

    def _check_sanity(self, image_paths, label_paths):
        assert len(image_paths) == len(label_paths)
        for image_path, label_path in zip(image_paths, label_paths):
            image_basename = os.path.basename(image_path)
            label_basename = os.path.basename(label_path)
            image_basename = os.path.splitext(image_basename)[0]
            label_basename = os.path.splitext(label_basename)[0]
            assert image_basename == label_basename

    def load_data(self):
        images_wildcard = os.path.join(self.path, 'images/*.jpg')
        labels_wildcard = os.path.join(self.path, 'annotations/*.mat')
        image_paths = glob.glob(images_wildcard)
        image_paths = sorted(image_paths)
        label_paths = glob.glob(labels_wildcard)
        label_paths = sorted(label_paths)
        self._check_sanity(image_paths, label_paths)
        boxes_data = self._preprocess_mat_files(label_paths, image_paths)
        data = []
        for image_path, box_data in zip(image_paths, boxes_data):
            sample = {'image': image_path, 'boxes': box_data}
            data.append(sample)
        return data

    def _preprocess_mat_files(self, label_paths, image_paths):
        boxes = []
        for label_path, image_path in zip(label_paths, image_paths):
            image_boxes = self._preprocess_mat_file(label_path)
            x_min, y_min, x_max, y_max = np.split(image_boxes, 4, 1)
            # TODO remove expensive loading just to get W, H
            image = load_image(image_path)
            H, W, num_dimensions = image.shape
            x_min = x_min / W
            x_max = x_max / W
            y_min = y_min / H
            y_max = y_max / H
            class_args = np.ones((len(image_boxes), 1))
            image_boxes = [x_min, y_min, x_max, y_max, class_args]
            image_boxes = np.concatenate(image_boxes, axis=1)
            boxes.append(image_boxes)
        return boxes

    def _preprocess_mat_file(self, label_path):
        boxes = []
        boxes_data = loadmat(label_path)['boxes'][0]
        for box_data in boxes_data:
            box_data = box_data[0, 0]
            y_A, x_A = box_data[0][0]
            y_B, x_B = box_data[1][0]
            y_C, x_C = box_data[2][0]
            y_D, x_D = box_data[3][0]
            x = np.array([x_A, x_B, x_C, x_D])
            y = np.array([y_A, y_B, y_C, y_D])
            x_min = np.min(x)
            x_max = np.max(x)
            y_min = np.min(y)
            y_max = np.max(y)
            box_coordinates = np.array([x_min, y_min, x_max, y_max])
            boxes.append(box_coordinates)
        boxes = np.array(boxes)
        return boxes


if __name__ == '__main__':
    root_path = os.path.expanduser('~')
    path = os.path.join(root_path, 'hand_dataset/hand_dataset/')
    data_manager = HandDataset(path, pr.TRAIN)
    data = data_manager.load_data()

    class_names = ['background', 'hand']
    draw_boxes = pr.SequentialProcessor()
    draw_boxes.add(pr.UnpackDictionary(['image', 'boxes']))
    draw_boxes.add(pr.ControlMap(pr.ToBoxes2D(class_names), [1], [1]))
    draw_boxes.add(pr.ControlMap(pr.LoadImage(), [0], [0]))
    draw_boxes.add(pr.ControlMap(pr.DenormalizeBoxes2D(), [0, 1], [1], {0: 0}))
    draw_boxes.add(pr.DrawBoxes2D(class_names))
    draw_boxes.add(pr.ShowImage())

    for sample in data:
        draw_boxes(sample)
