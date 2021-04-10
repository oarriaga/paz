import glob
import pickle

from paz.abstract import Loader
from paz.backend.image.opencv_image import load_image, resize_image


class HandDataset(Loader):
    def __init__(self, path, split='training', image_size=(256, 256, 3)):
        super().__init__(path, split, None, 'HandSegmentation')
        self.path = path
        self.split = split
        self.image_size = image_size

    def _load_images(self, image_path):
        image = load_image(image_path)
        hand = resize_image(image, (self.image_size[0], self.image_size[1]))
        return hand

    def _to_list_of_dictionaries(self, hands, seg_labels=None,
                                 annotations=None):
        dataset = []
        for arg in range(len(hands)):
            sample = dict()
            sample['image'] = self._load_images(hands[arg])
            if seg_labels is not None:
                sample['seg_label'] = self._load_images(seg_labels[arg])
            if annotations is not None:
                sample['key_points_3D'] = annotations[arg]['xyz']
                sample['key_points_2D'] = annotations[arg]['uv_vis']
                sample['camera_matrix'] = annotations[arg]['K']
            dataset.append(sample)
        return dataset

    def _load_annotation(self, label_path):
        with open(label_path, 'rb') as file:
            anno_all = pickle.load(file)
        return anno_all

    def load_data(self):
        images = sorted(glob.glob(self.path + self.split + '/color/*.png'))
        if self.split == 'training':
            seg_labels = sorted(glob.glob(self.path + self.split +
                                          '/mask/*.png'))
            annotations = self._load_annotation(self.path + self.split +
                                                '/anno_training.pickle')
            dataset = self._to_list_of_dictionaries(images, seg_labels,
                                                    annotations)
        else:
            dataset = self._to_list_of_dictionaries(images, None, None)
        return dataset


if __name__ == '__main__':
    path = 'dataset/'
    split = 'training'
    data_manager = HandDataset(path, split)
    dataset = data_manager.load_data()
