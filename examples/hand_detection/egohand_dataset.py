import os
import glob
import numpy as np
from scipy.io import loadmat

from paz.abstract import Loader
from paz import processors as pr
from paz.backend.image import load_image


class EgoHands(Loader):
    def __init__(self, path, split=pr.TRAIN):
        names = ['background', 'hand']
        super(EgoHands, self).__init__(path, split, names, 'EgoHands')

    def _preprocess_dataset(self, path):
        images_wildcard = os.path.join(path, '*.jpg')
        image_paths = glob.glob(images_wildcard)
        image_paths = sorted(image_paths)

        labels_path = os.path.join(path, 'polygons.mat')
        polygons = loadmat(labels_path)
        polygons = polygons["polygons"][0]
        boxes = []
        for polygon in polygons:
            image_boxes = []
            for points in polygon:
                x, y = np.split(points, 2, axis=1)
                # this checks if array is invalid (empty)
                if (len(points) == 0) or (len(points) == 1):
                    continue
                image_box = [x.min(), y.min(), x.max(), y.max(), 1]
                image_boxes.append(image_box)
            image_boxes = np.array(image_boxes)
            boxes.append(image_boxes)
        return image_paths, boxes

    def load_data(self):
        directory_wildcard = os.path.join(self.path, '*')
        directory_paths = glob.glob(directory_wildcard)
        images, labels = [], []
        for path in directory_paths:
            directory_data = self._preprocess_dataset(path)
            directory_images, directory_labels = directory_data
            images.extend(directory_images)
            labels.extend(directory_labels)

        data = []
        for image_path, box_data in zip(images, labels):
            image = load_image(image_path)
            H, W, num_channels = image.shape
            if len(box_data) == 0:
                continue
            x_min, y_min, x_max, y_max, class_arg = np.split(box_data, 5, 1)
            x_min = x_min / W
            x_max = x_max / W
            y_min = y_min / H
            y_max = y_max / H
            box_data = [x_min, y_min, x_max, y_max, class_arg]
            box_data = np.concatenate(box_data, axis=1)
            sample = {'image': image_path, 'boxes': box_data}
            data.append(sample)
        return data


if __name__ == "__main__":
    root_path = os.path.expanduser('~')
    path = os.path.join(root_path, 'Downloads/egohands/_LABELLED_SAMPLES/')
    data_manager = EgoHands(path)
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
