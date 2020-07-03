from ..abstract import SequentialProcessor, Processor
from .. import processors as pr

from .renderer import RenderTwoViews


class KeypointNetSharedAugmentation(SequentialProcessor):
    """Wraps ''RenderTwoViews'' as a sequential processor for using it directly
        with a ''paz.GeneratingSequence''.
    """
    def __init__(self, renderer, size):
        super(KeypointNetSharedAugmentation, self).__init__()
        self.renderer = renderer
        self.size = size
        self.add(RenderTwoViews(self.renderer))
        self.add(pr.SequenceWrapper(
            {0: {'image_A': [size, size, 3]},
             1: {'image_B': [size, size, 3]}},
            {2: {'matrices': [4, 4 * 4]},
             3: {'alpha_channels': [size, size, 2]}}))


class KeypointNetInference(Processor):
    """Performs inference from a ''KeypointNetShared'' model.
    """
    def __init__(self, model, num_keypoints=None, radius=5):
        super(KeypointNetInference, self).__init__()
        self.num_keypoints, self.radius = num_keypoints, radius
        if self.num_keypoints is None:
            self.num_keypoints = model.output_shape[1]

        preprocessing = SequentialProcessor()
        preprocessing.add(pr.NormalizeImage())
        preprocessing.add(pr.ExpandDims(axis=0))
        self.predict_keypoints = SequentialProcessor()
        self.predict_keypoints.add(pr.Predict(model, preprocessing))
        self.predict_keypoints.add(pr.SelectElement(0))
        self.predict_keypoints.add(pr.Squeeze(axis=0))
        self.postprocess_keypoints = SequentialProcessor()
        self.postprocess_keypoints.add(pr.DenormalizeKeypoints())
        self.postprocess_keypoints.add(pr.RemoveKeypointsDepth())
        self.draw = pr.DrawKeypoints2D(self.num_keypoints, self.radius, False)
        self.wrap = pr.WrapOutput(['image', 'keypoints'])

    def call(self, image):
        keypoints = self.predict_keypoints(image)
        keypoints = self.postprocess_keypoints(keypoints, image)
        image = self.draw(image, keypoints)
        return self.wrap(image, keypoints)


class PredictKeypoints(SequentialProcessor):
    def __init__(self, model):
        super(PredictKeypoints, self).__init__()
        self.size = model.input_shape[1:3]
        self.preprocess = SequentialProcessor(
            [pr.ResizeImage(self.size), pr.NormalizeImage(), pr.ExpandDims(0)])
        self.add(pr.Predict(model, self.preprocess))
        self.add(pr.Squeeze(axis=0))
        self.add(pr.DenormalizeKeypoints())


class KeypointInference(Processor):
    def __init__(self, model, num_keypoints=None, radius=5):
        super(KeypointInference, self).__init__()
        self.num_keypoints, self.radius = num_keypoints, radius
        if self.num_keypoints is None:
            self.num_keypoints = model.output_shape[1]
        self.predict_keypoints = PredictKeypoints(model)
        self.draw = pr.DrawKeypoints2D(self.num_keypoints, self.radius, False)
        self.wrap = pr.WrapOutput(['image', 'keypoints'])

    def call(self, image):
        keypoints = self.predict_keypoints(image)
        image = self.draw(image, keypoints)
