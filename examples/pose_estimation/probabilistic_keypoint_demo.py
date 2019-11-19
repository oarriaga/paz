import os
# import numpy as np
from paz.pipelines import KeypointInference
from paz.models import KeypointNet2D

from paz.core import VideoPlayer
from tensorflow.keras.utils import get_file
from paz.core import SequentialProcessor, Processor
import paz.processors as pr
from model import GaussianMixture
import numpy as np


class ProbabilisticKeypointInferencer(SequentialProcessor):
    """General keypoint inference pipeline.
    # Arguments
        model: Keras model.
        num_keypoints: Int.
        radius: Int.
    # Returns
        Function for outputting keypoints from image
    """
    def __init__(self, model, num_keypoints=None, radius=5):
        super(ProbabilisticKeypointInferencer, self).__init__()
        self.num_keypoints, self.radius = num_keypoints, radius
        if self.num_keypoints is None:
            self.num_keypoints = model.output_shape[1]

        pipeline = [pr.Resize(model.input_shape[1:3]),
                    pr.NormalizeImage(),
                    pr.ExpandDims(axis=0, topic='image')]

        self.add(PredictProbability(model, 'image', 'keypoints', pipeline))
        self.add(EstimateKeypointsMean('keypoints', 'keypoints'))
        self.add(pr.DenormalizeKeypoints())
        self.add(pr.DrawKeypoints2D(self.num_keypoints, self.radius, False))
        self.add(pr.CastImageToInts())


class PredictProbability(Processor):
    def __init__(self, model, input_topic, label_topic='predictions',
                 processors=None):

        super(PredictProbability, self).__init__()
        self.model = model
        self.processors = processors
        if self.processors is not None:
            self.process = SequentialProcessor(processors)
        self.input_topic = input_topic
        self.label_topic = label_topic

    def call(self, kwargs):
        input_topic = kwargs[self.input_topic]
        if self.processors is not None:
            processing_kwargs = {self.input_topic: input_topic}
            input_topic = self.process(processing_kwargs)[self.input_topic]
        predictions = self.model(input_topic)
        kwargs[self.label_topic] = predictions
        return kwargs


class EstimateKeypointsMean(Processor):
    def __init__(self, input_topic, label_topic):
        self.input_topic = input_topic
        self.label_topic = label_topic
        super(EstimateKeypointsMean, self).__init__()

    def call(self, kwargs):
        distributions = kwargs[self.input_topic]
        means = np.zeros((len(distributions), 2))
        for keypoint_arg, distribution in enumerate(distributions):
            means[keypoint_arg] = distribution.mean().numpy()
        kwargs[self.label_topic] = means
        return kwargs


num_keypoints, batch_shape = 20, (1, 128, 128, 3)
model = GaussianMixture(batch_shape, num_keypoints)
model.load_weights('GaussianMixture.hdf5')
ProbabilisticKeypointInferencer(model, num_keypoints)

pipeline = ProbabilisticKeypointInferencer(model, num_keypoints, 5)

video_player = VideoPlayer((1280, 960), pipeline, 2)
video_player.start()
