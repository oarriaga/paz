from paz import processors as pr
from paz.abstract import Loader


import os
import glob
import csv
import numpy as np


class OpenImagesV6(Loader):
    def __init__(self, path, split, class_names=None):
        super(OpenImagesV6, self).__init__(
            path, split, class_names, 'OpenImagesV6')
        # TODO MAKE CHECK THAT CLASS NAMES ALWAYS HAVE BACKGROUND CLASS
        self._split_to_name = {
            pr.TRAIN: 'train', pr.VAL: 'validation', pr.TEST: 'test'}
        split_name = self._split_to_name[self.split]
        self.image_paths = self._get_image_paths(split_name)
        self.machine_to_human_name = self._build_machine_to_human(split_name)
        self.root_image_path = self._build_root_image_path(split_name)
        self.class_name_to_arg = self._build_class_name_to_arg()

    def _build_class_name_to_arg(self):
        return dict(zip(self.class_names, range(len(self.class_names))))

    def _build_root_image_path(self, split_name):
        root_image_path = [self.path, split_name, 'data']
        root_image_path = os.path.join(*root_image_path)
        return root_image_path

    def _build_machine_to_human(self, split_name):
        metadata_path = [self.path, split_name, 'metadata', 'classes.csv']
        metadata_path = os.path.join(*metadata_path)
        metadata_file = open(metadata_path, 'r')
        metadata_reader = csv.reader(metadata_file, delimiter=',')
        machine_to_human_name = {}
        for machine_name, human_name in metadata_reader:
            machine_to_human_name[machine_name] = human_name
        return machine_to_human_name

    def _get_image_paths(self, split_name):
        images_wildcard = [self.path, split_name, 'data', '*.jpg']
        images_wildcard = os.path.join(*images_wildcard)
        image_paths = glob.glob(images_wildcard)
        return image_paths

    def _skip_header(self, reader):
        next(reader)

    def load_data(self):
        split_name = self._split_to_name[self.split]
        labels_path = [self.path, split_name, 'labels', 'detections.csv']
        labels_path = os.path.join(*labels_path)
        labels_file = open(labels_path, 'r')
        labels_reader = csv.reader(labels_file, delimiter=',')
        self._skip_header(labels_reader)
        image_ID_to_boxes = {}
        for sample in labels_reader:
            image_ID, machine_name = sample[0], sample[2]
            class_name = self.machine_to_human_name[machine_name]
            if class_name in self.class_names:
                coordinates = [float(coordinate) for coordinate in sample[4:8]]
                x_min, x_max, y_min, y_max = coordinates
                class_arg = self.class_name_to_arg[class_name]
                box_data = np.array([x_min, y_min, x_max, y_max, class_arg])
                if image_ID in image_ID_to_boxes:
                    image_ID_to_boxes[image_ID].append(box_data)
                else:
                    image_ID_to_boxes[image_ID] = [box_data]

        dataset = []
        for image_ID in image_ID_to_boxes.keys():
            image_path = os.path.join(self.root_image_path, image_ID + '.jpg')
            boxes = image_ID_to_boxes[image_ID]
            boxes = np.array(boxes)
            sample = {'image': image_path, 'boxes': boxes}
            dataset.append(sample)
        return dataset


if __name__ == '__main__':
    root_path = os.path.expanduser('~')
    path = os.path.join(root_path, '/home/octavio/fiftyone/open-images-v6/')

    train_data_manager = OpenImagesV6(
        path, pr.TRAIN, ['background', 'Human hand'])
    train_data = train_data_manager.load_data()
    print('Number of training samples', len(train_data))

    val_data_manager = OpenImagesV6(path, pr.VAL, ['background', 'Human hand'])
    validation_data = val_data_manager.load_data()
    print('Number of validation samples', len(validation_data))

    test_data_manager = OpenImagesV6(
        path, pr.TEST, ['background', 'Human hand'])
    test_data = test_data_manager.load_data()
    print('Number of test samples', len(test_data))

    class_names = ['background', 'hand']
    draw_boxes = pr.SequentialProcessor()
    draw_boxes.add(pr.UnpackDictionary(['image', 'boxes']))
    draw_boxes.add(pr.ControlMap(pr.ToBoxes2D(class_names), [1], [1]))
    draw_boxes.add(pr.ControlMap(pr.LoadImage(), [0], [0]))
    draw_boxes.add(pr.ControlMap(pr.DenormalizeBoxes2D(), [0, 1], [1], {0: 0}))
    draw_boxes.add(pr.DrawBoxes2D(class_names))
    draw_boxes.add(pr.ShowImage())

    for sample in validation_data:
        draw_boxes(sample)
