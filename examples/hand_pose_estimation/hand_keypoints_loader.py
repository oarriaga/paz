import glob
import pickle

from paz.abstract import Loader


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
        for hand_arg in range(len(hands)):
            sample = dict()
            sample['image_path'] = hands[hand_arg]
            sample['segmentation_label'] = segmentation_labels[hand_arg]
            sample['annotations'] = annotations[hand_arg]
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