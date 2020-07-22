from paz.backend.image import draw_circle
from paz.backend.image.draw import GREEN
from paz import processors as pr
from paz.abstract import Processor
import numpy as np


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


class ToNumpyArray(Processor):
    def __init__(self):
        super(ToNumpyArray, self).__init__()

    def call(self, predictions):
        return np.array(predictions)


class PredictMeanDistribution(Processor):
    def __init__(self, model, preprocess=None):
        super(PredictMeanDistribution, self).__init__()
        print('Building graph...')
        self.num_keypoints = len(model.output_shape)
        # self.model = tf.function(model.mean)
        self.model = model
        self.preprocess = preprocess

    def call(self, x):
        if self.preprocess is not None:
            x = self.preprocess(x)
        distributions = self.model(x)
        keypoints = np.zeros((self.num_keypoints, 2))
        for arg, distribution in enumerate(distributions):
            keypoints[arg] = distribution.mean()
        return keypoints


def draw_circles(image, points, color=GREEN, radius=3):
    for point in points:
        draw_circle(image, point, color, radius)
    return image


if __name__ == '__main__':
    from facial_keypoints import FacialKeypoints
    from paz.backend.image import show_image
    from paz.abstract import SequentialProcessor

    data_manager = FacialKeypoints('dataset/', 'train')
    datasets = data_manager.load_data()
    augment_keypoints = SequentialProcessor()
    augment_keypoints.add(pr.RandomKeypointRotation())
    augment_keypoints.add(pr.RandomKeypointTranslation())
    for arg in range(100):
        original_image = datasets[0]['image'].copy()
        kp = datasets[0]['keypoints'].copy()
        original_image, kp = augment_keypoints(original_image, kp)
        original_image = draw_circles(original_image, kp.astype('int'))
        show_image(original_image.astype('uint8'))
