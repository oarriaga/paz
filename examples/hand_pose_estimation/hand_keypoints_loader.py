import glob
import pickle

from paz.abstract import Loader

kinematic_chain_dict = {0: 'root',
                        4: 'root', 3: 4, 2: 3, 1: 2,
                        8: 'root', 7: 8, 6: 7, 5: 6,
                        12: 'root', 11: 12, 10: 11, 9: 10,
                        16: 'root', 15: 16, 14: 15, 13: 14,
                        20: 'root', 19: 20, 18: 19, 17: 18}
kinematic_chain_list = list(kinematic_chain_dict.keys())

LEFT_ROOT_KEYPOINT_ID = 0
LEFT_ALIGNED_KEYPOINT_ID = 12
LEFT_LAST_KEYPOINT_ID = 20

RIGHT_ROOT_KEYPOINT_ID = 21
RIGHT_ALIGNED_KEYPOINT_ID = 33
RIGHT_LAST_KEYPOINT_ID = 41


class RenderedHandLoader(Loader):
    def __init__(self, path, split='train'):
        super().__init__(path, split, None, 'HandPoseLoader')
        self.path = path
        split_to_folder = {'train': 'training', 'val': 'evaluation',
                           'test': 'testing'}
        self.folder = split_to_folder[split]

    def _load_annotation(self, label_path):
        with open(label_path, 'rb') as file:
            annotations_all = pickle.load(file)
        return annotations_all

    def to_list_of_dictionaries(self, hands, segmentation_labels=None,
                                annotations=None):
        dataset = []
        for arg in range(len(hands)):
            sample = dict()
            sample['image_path'] = hands[arg]
            sample['segmentation_label'] = segmentation_labels[arg]
            sample['annotations'] = annotations[arg]
            dataset.append(sample)
        return dataset

    def load_data(self):
        images = sorted(glob.glob(self.path + self.folder + '/color/*.png'))

        if self.split == 'test':
            dataset = self.to_list_of_dictionaries(images, None, None)
        else:
            segmentation_labels = sorted(glob.glob(self.path + self.folder +
                                                   '/mask/*.png'))
            annotations = self._load_annotation(self.path + self.folder +
                                                '/anno_{}.pickle'.format(
                                                    self.folder))
            dataset = self.to_list_of_dictionaries(images, segmentation_labels,
                                                   annotations)

        return dataset