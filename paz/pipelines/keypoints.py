from ..abstract import SequentialProcessor, Processor
from .. import processors as pr

from .renderer import RenderTwoViews
from ..models import KeypointNet2D
from tensorflow.keras.utils import get_file


class KeypointNetSharedAugmentation(SequentialProcessor):
    """Wraps ``RenderTwoViews`` as a sequential processor for using it directly
        with a ``paz.GeneratingSequence``.

    # Arguments
        renderer: ``RenderTwoViews`` processor.
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
    """Performs inference from a ``KeypointNetShared`` model.

    # Arguments
        model: Keras model for predicting keypoints.
        num_keypoints: Int or None. If None ``num_keypoints`` is
            tried to be inferred from ``model.output_shape``
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


class EstimateKeypoints2D(Processor):
    """Basic 2D keypoint prediction pipeline.

    # Arguments
        model: Keras model for predicting keypoints.
        num_keypoints: Int or None. If None ``num_keypoints`` is
            tried to be inferred from ``model.output_shape``
        draw: Boolean indicating if inferences should be drawn.
        radius: Int. used for drawing the predicted keypoints.
    """
    def __init__(self, model, num_keypoints, draw=True, radius=3,
                 color=pr.RGB2BGR):
        self.model = model
        self.num_keypoints = num_keypoints
        self.draw, self.radius, self.color = draw, radius, color
        self.preprocess = SequentialProcessor()
        self.preprocess.add(pr.ResizeImage(self.model.input_shape[1:3]))
        self.preprocess.add(pr.ConvertColorSpace(self.color))
        self.preprocess.add(pr.NormalizeImage())
        self.preprocess.add(pr.ExpandDims(0))
        self.preprocess.add(pr.ExpandDims(-1))
        self.predict = pr.Predict(model, self.preprocess, pr.Squeeze(0))
        self.denormalize = pr.DenormalizeKeypoints()
        self.draw = pr.DrawKeypoints2D(self.num_keypoints, self.radius, False)
        self.wrap = pr.WrapOutput(['image', 'keypoints'])

    def call(self, image):
        keypoints = self.predict(image)
        keypoints = self.denormalize(keypoints, image)
        if self.draw:
            image = self.draw(image, keypoints)
        return self.wrap(image, keypoints)


class FaceKeypointNet2D32(EstimateKeypoints2D):
    """KeypointNet2D model trained with Kaggle Facial Detection challenge.

    # Arguments
        draw: Boolean indicating if inferences should be drawn.
        radius: Int. used for drawing the predicted keypoints.

    # Example
        ``` python
        from paz.pipelines import FaceKeypointNet2D32

        estimate_keypoints= FaceKeypointNet2D32()

        # apply directly to an image (numpy-array)
        inference = estimate_keypoints(image)
        ```
    # Returns
        A function that takes an RGB image and outputs the predictions
        as a dictionary with ``keys``: ``image`` and ``keypoints``.
        The corresponding values of these keys contain the image with the drawn
        inferences and a numpy array representing the keypoints.
    """
    def __init__(self, draw=True, radius=3):
        model = KeypointNet2D((96, 96, 1), 15, 32, 0.1)
        self.weights_URL = ('https://github.com/oarriaga/altamira-data/'
                            'releases/download/v0.7/')
        weights_path = self.get_weights_path(model)
        model.load_weights(weights_path)
        super(FaceKeypointNet2D32, self).__init__(
            model, 15, draw, radius, pr.RGB2GRAY)

    def get_weights_path(self, model):
        model_name = '_'.join(['FaceKP', model.name, '32', '15'])
        model_name = '%s_weights.hdf5' % model_name
        URL = self.weights_URL + model_name
        return get_file(model_name, URL, cache_subdir='paz/models')
