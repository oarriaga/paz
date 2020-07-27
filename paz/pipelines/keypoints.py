from ..abstract import SequentialProcessor, Processor
from .. import processors as pr

from .renderer import RenderTwoViews


class KeypointNetSharedAugmentation(SequentialProcessor):
    """Wraps ''RenderTwoViews'' as a sequential processor for using it directly
        with a ''paz.GeneratingSequence''.

    # Arguments
        renderer: ''RenderTwoViews'' processor.
        size: Image size.
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

    # Arguments
        model: Keras model for predicting keypoints.
        num_keypoints: Int or None. If None ''num_keypoints'' is
            tried to be inferred from ''model.output_shape''
        radius: Int. used for drawing the predicted keypoints.
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
