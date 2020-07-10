import numpy as np
from facial_keypoints import FacialKeypoints
from processors import RandomKeypointRotation, RandomKeypointTranslation
from processors import draw_circles
from paz.backend.image import show_image
from paz.backend.keypoints import denormalize_keypoints
from paz.abstract import SequentialProcessor
from paz import processors as pr
from paz.abstract import Processor


class AugmentKeypoints(SequentialProcessor):
    def __init__(self, rotation_range=30, delta_scales=[0.2, 0.2],
                 with_partition=False, num_keypoints=15):
        super(AugmentKeypoints, self).__init__()
        self.add(pr.UnpackDictionary(['image', 'keypoints']))
        self.add(pr.ControlMap(pr.RandomBrightness()))
        self.add(pr.ControlMap(pr.RandomContrast()))
        self.add(pr.ControlMap(pr.NormalizeImage()))
        self.add(RandomKeypointRotation(rotation_range))
        self.add(RandomKeypointTranslation(delta_scales))
        self.add(pr.ControlMap(pr.ExpandDims(-1), [0], [0]))
        self.add(pr.ControlMap(pr.NormalizeKeypoints((96, 96)), [1], [1]))
        labels_info = {1: {'keypoints': [num_keypoints, 2]}}
        if with_partition:
            outro_indices = list(range(1, 16))
            self.add(pr.ControlMap(PartitionKeypoints(), [1], outro_indices))
            labels_info = {}
            for arg in range(num_keypoints):
                labels_info[arg] = {'keypoint_%s' % arg: [2]}
        self.add(pr.SequenceWrapper(
            {0: {'image': [96, 96, 1]}}, labels_info))


class PartitionKeypoints(Processor):
    """Partitions keypoints from shape ''[num_keypoints, 2]'' into a list of
        the form ''[(2, 1), (2, 1), ....]'' and length equal to the number of
        of_keypoints.
    """
    def __init__(self):
        super(PartitionKeypoints, self).__init__()

    def call(self, keypoints):
        keypoints = np.vsplit(keypoints, len(keypoints))
        keypoints = [np.squeeze(keypoint) for keypoint in keypoints]
        return (*keypoints, )


if __name__ == '__main__':
    data_manager = FacialKeypoints('dataset/', 'train')
    dataset = data_manager.load_data()
    augment_keypoints = AugmentKeypoints()
    for arg in range(100):
        sample = dataset[arg]
        predictions = augment_keypoints(sample)
        original_image = predictions['inputs']['image'][:, :, 0]
        kp = predictions['labels']['keypoints']
        kp = denormalize_keypoints(kp, 96, 96)
        original_image = draw_circles(
            original_image, kp.astype('int'))
        show_image(original_image.astype('uint8'))
