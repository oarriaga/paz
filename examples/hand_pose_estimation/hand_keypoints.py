import os

import cv2
import json
import numpy as np

from paz.abstract import Loader


class HandKeypoints(Loader):
    def __init__(self, path, split='train'):
        super().__init__(path, split, None, 'HandKeypoints')
        self.path = path
        self.split = split

    def _load_images(self, image_paths):
        hands = np.zeros((len(image_paths), 300, 300, 3))
        for arg, image_path in enumerate(image_paths):
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            resized = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
            hands[arg] = resized
        return hands

    def _load_labels(self, label_paths):
        keypoints = np.zeros((len(label_paths), 21, 3))
        for arg, label_path in enumerate(label_paths):
            with open(label_path, 'r') as fid:
                data = json.load(fid)
                keypoint_set = data['hand_pts']
            keypoints[arg] = keypoint_set.to_numpy().reshape(21, 3)
        return keypoints

    def _load_paths(self, path):
        image_paths = list()
        label_paths = list()
        if isinstance(self.path, str):
            for file in os.listdir(path):
                if os.path.splitext(file)[1] == '.jpg':
                    image_paths.append(str(os.path.join(path, file)))
                elif os.path.splitext(file)[1] == '.json':
                    label_paths.append(str(os.path.join(path, file)))
        else:
            for folder in path:
                for file in os.listdir(folder):
                    if os.path.splitext(file)[1] == '.jpg':
                        image_paths.append(str(os.path.join(path, file)))
                    elif os.path.splitext(file)[1] == '.json':
                        label_paths.append(str(os.path.join(path, file)))
        return image_paths, label_paths

    def _to_list_of_dictionaries(self, hands, keypoints=None):
        dataset = []
        for arg in range(len(hands)):
            hand, sample = hands[arg], {}
            sample['image'] = hand
            if keypoints is not None:
                sample['keypoints'] = keypoints[arg]
            dataset.append(sample)
        return dataset

    def load_data(self):
        image_paths, label_paths = self._load_paths(self.path)
        hands = self._load_images(image_paths)
        if self.split == 'train':
            keypoints = self._load_labels(label_paths)
            dataset = self._to_list_of_dictionaries(hands, keypoints)
        else:
            dataset = self._to_list_of_dictionaries(hands, None)
        return dataset


if __name__ == '__main__':
    path = 'dataset/'
    split = 'train'
    data_manager = FacialKeypoints(path, split)
    dataset = data_manager.load_data()
