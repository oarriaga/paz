import numpy as np
from facial_keypoints import FacialKeypoints
from processors import RandomKeypointRotation, RandomKeypointTranslation
from processors import draw_circles
from paz.backend.image import show_image
from paz.backend.keypoints import denormalize_keypoints
from paz.abstract import SequentialProcessor
from paz import processors as pr
from paz.abstract import Processor

import tensorflow as tf


class AugmentKeypoints(SequentialProcessor):
    def __init__(self, phase, rotation_range=30, delta_scales=[0.2, 0.2],
                 with_partition=False, num_keypoints=15):
        super(AugmentKeypoints, self).__init__()
        self.add(pr.UnpackDictionary(['image', 'keypoints']))
        if phase == 'train':
            self.add(pr.ControlMap(pr.RandomBrightness()))
            self.add(pr.ControlMap(pr.RandomContrast()))
            self.add(RandomKeypointRotation(rotation_range))
            self.add(RandomKeypointTranslation(delta_scales))
        self.add(pr.ControlMap(pr.NormalizeImage(), [0], [0]))
        self.add(pr.ControlMap(pr.ExpandDims(-1), [0], [0]))
        self.add(pr.ControlMap(pr.NormalizeKeypoints((96, 96)), [1], [1]))
        labels_info = {1: {'keypoints': [num_keypoints, 2]}}
        if with_partition:
            outro_indices = list(range(1, 16))
            self.add(pr.ControlMap(PartitionKeypoints(), [1], outro_indices))
            labels_info = {}
            for arg in range(num_keypoints):
                labels_info[arg + 1] = {'keypoint_%s' % arg: [2]}
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


class ProbabilisticKeypointPrediction(Processor):
    def __init__(self, detector, keypoint_estimator, radius=3):
        super(ProbabilisticKeypointPrediction, self).__init__()
        # face detector
        RGB2GRAY = pr.ConvertColorSpace(pr.RGB2GRAY)
        self.detect = pr.Predict(detector, RGB2GRAY, pr.ToBoxes2D())

        # creating pre-processing pipeline for keypoint estimator
        preprocess = SequentialProcessor()
        preprocess.add(pr.ResizeImage(keypoint_estimator.input_shape[1:3]))
        preprocess.add(pr.ConvertColorSpace(pr.RGB2GRAY))
        preprocess.add(pr.NormalizeImage())
        preprocess.add(pr.ExpandDims([0, 3]))

        # creating post-processing pipeline for keypoint esimtator
        # postprocess = SequentialProcessor()
        # postprocess.add(ToNumpyArray())
        # postprocess.add(pr.Squeeze(1))

        # keypoint estimator predictions
        self.estimate_keypoints = PredictMeanDistribution(
            keypoint_estimator, preprocess)

        # self.estimate_keypoints = pr.Predict(
        # keypoint_estimator, preprocess, postprocess)

        # used for drawing up keypoints in original image
        self.change_coordinates = pr.ChangeKeypointsCoordinateSystem()
        self.denormalize_keypoints = pr.DenormalizeKeypoints()
        self.crop_boxes2D = pr.CropBoxes2D()
        self.num_keypoints = len(keypoint_estimator.output_shape)
        self.draw = pr.DrawKeypoints2D(self.num_keypoints, radius, False)
        self.draw_boxes2D = pr.DrawBoxes2D(colors=[0, 255, 0])
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        boxes2D = self.detect(image)
        cropped_images = self.crop_boxes2D(image, boxes2D)
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            keypoints = self.estimate_keypoints(cropped_image)
            keypoints = self.denormalize_keypoints(keypoints, cropped_image)
            keypoints = self.change_coordinates(keypoints, box2D)
            image = self.draw(image, keypoints)
        image = self.draw_boxes2D(image, boxes2D)
        return self.wrap(image, boxes2D)


if __name__ == '__main__':
    from paz.abstract import ProcessingSequence

    data_manager = FacialKeypoints('dataset/', 'train')
    dataset = data_manager.load_data()
    augment_keypoints = AugmentKeypoints('train', with_partition=False)
    for arg in range(1, 100):
        sample = dataset[arg]
        predictions = augment_keypoints(sample)
        original_image = predictions['inputs']['image'][:, :, 0]
        original_image = original_image * 255.0
        kp = predictions['labels']['keypoints']
        kp = denormalize_keypoints(kp, 96, 96)
        original_image = draw_circles(
            original_image, kp.astype('int'))
        show_image(original_image.astype('uint8'))
    sequence = ProcessingSequence(augment_keypoints, 32, dataset, True)
    batch = sequence.__getitem__(0)
