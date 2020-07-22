import os
import pandas as pd
import numpy as np
from paz.abstract import Loader


class FacialKeypoints(Loader):
    def __init__(self, path, split='train'):
        split_to_filename = {'train': 'training.csv', 'test': 'test.csv'}
        filename = split_to_filename[split]
        path = os.path.join(path, filename)
        super(FacialKeypoints, self).__init__(
            path, split, None, 'FacialKeypoints')

    def _load_faces(self, data_frame):
        faces = np.zeros((len(data_frame), 96, 96))
        for arg, face in enumerate(data_frame.Image):
            faces[arg] = np.array(face.split(' '), dtype=int).reshape(96, 96)
        return faces

    def _load_keypoints(self, data_frame):
        keypoints = np.zeros((len(data_frame), 15, 2))
        for arg, keypoint_set in data_frame.iloc[:, :-1].iterrows():
            keypoints[arg] = keypoint_set.to_numpy().reshape(15, 2)
        return keypoints

    def _to_list_of_dictionaries(self, faces, keypoints=None):
        dataset = []
        for arg in range(len(faces)):
            face, sample = faces[arg], {}
            sample['image'] = face
            if keypoints is not None:
                sample['keypoints'] = keypoints[arg]
            dataset.append(sample)
        return dataset

    def load_data(self):
        data_frame = pd.read_csv(self.path)
        data_frame.fillna(method='ffill', inplace=True)
        faces = self._load_faces(data_frame)
        if self.split == 'train':
            keypoints = self._load_keypoints(data_frame)
            dataset = self._to_list_of_dictionaries(faces, keypoints)
        else:
            dataset = self._to_list_of_dictionaries(faces, None)
        return dataset


if __name__ == '__main__':
    path = 'dataset/'
    split = 'train'
    data_manager = FacialKeypoints(path, split)
    dataset = data_manager.load_data()
