from ..abstract import SequentialProcessor
from .. import processors as pr
from . import PreprocessImage
from ..models.classification import MiniXception, VVAD_LRS3_LSTM, CNN2Plus1D
from ..datasets import get_class_names
from .keypoints import MinimalHandPoseEstimation


# neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
EMOTION_COLORS = [[255, 0, 0], [45, 90, 45], [255, 0, 255], [255, 255, 0],
                  [0, 0, 255], [0, 255, 255], [0, 255, 0]]
Average_Options = ['mean', 'weighted']
Architecture_Options = ['VVAD-LRS3-LSTM', 'CNN2Plus1D', 'CNN2Plus1D_Filters', 'CNN2Plus1D_Layers',
                        'CNN2Plus1D_Light']


class MiniXceptionFER(SequentialProcessor):
    """Mini Xception pipeline for classifying emotions from RGB faces.

    # Example
        ``` python
        from paz.pipelines import MiniXceptionFER

        classify = MiniXceptionFER()

        # apply directly to an image (numpy-array)
        inference = classify(image)
        ```

     # Returns
        A function that takes an RGB image and outputs the predictions
        as a dictionary with ``keys``: ``class_names`` and ``scores``.

    # References
       - [Real-time Convolutional Neural Networks for Emotion and
            Gender Classification](https://arxiv.org/abs/1710.07557)

    """
    def __init__(self):
        super(MiniXceptionFER, self).__init__()
        self.classifier = MiniXception((48, 48, 1), 7, weights='FER')
        self.class_names = get_class_names('FER')

        preprocess = PreprocessImage(self.classifier.input_shape[1:3], None)
        preprocess.insert(0, pr.ConvertColorSpace(pr.RGB2GRAY))
        preprocess.add(pr.ExpandDims(0))
        preprocess.add(pr.ExpandDims(-1))
        self.add(pr.Predict(self.classifier, preprocess))
        self.add(pr.CopyDomain([0], [1]))
        self.add(pr.ControlMap(pr.ToClassName(self.class_names), [0], [0]))
        self.add(pr.WrapOutput(['class_name', 'scores']))


class ClassifyHandClosure(SequentialProcessor):
    """Pipeline to classify minimal hand closure status.

    # Example
        ``` python
        from paz.pipelines import ClassifyHandClosure

        classify = ClassifyHandClosure()

        # apply directly to an image (numpy-array)
        inference = classify(image)
        ```

     # Returns
        A function that takes an RGB image and outputs an image with class
        status drawn on it.
    """
    def __init__(self, draw=True, right_hand=False):
        super(ClassifyHandClosure, self).__init__()
        self.add(MinimalHandPoseEstimation(draw, right_hand))
        self.add(pr.UnpackDictionary(['image', 'relative_angles']))
        self.add(pr.ControlMap(pr.IsHandOpen(), [1], [1]))
        self.add(pr.ControlMap(pr.BooleanToTextMessage('OPEN', 'CLOSE'),
                               [1], [1]))
        if draw:
            self.add(pr.ControlMap(pr.DrawText(), [0, 1], [0], {1: 1}))
        self.add(pr.WrapOutput(['image', 'status']))


class ClassifyVVAD(SequentialProcessor):
    """Visual Voice Activity Detection pipeline for classifying speaking and not speaking from cropped RGB face
    video clips.

    # Arguments
        input_size: Tuple of integers. Input shape to the model in following format: (frames, height, width, channels)
            e.g. (38, 96, 96, 3).
        architecture: String. Name of the architecture to use. Currently supported: 'VVAD-LRS3-LSTM', 'CNN2Plus1D',
            'CNN2Plus1D_Filters', 'CNN2Plus1D_Layers' and 'CNN2Plus1D_Light'
        stride: Integer. How many frames are between the predictions (computational expansive (low update rate) vs
            high latency (high update rate))
        averaging_window_size: Integer. How many predictions are averaged. Set to 1 to disable averaging
        average_type: String. 'mean' or 'weighted'. How the predictions are averaged. Set average to 1 to
            disable averaging
    """
    def __init__(self, input_size=(38, 96, 96, 3), architecture='CNN2Plus1D_Light',
                 stride=38, averaging_window_size=2, average_type='mean'):
        super(ClassifyVVAD, self).__init__()
        assert average_type in Average_Options, f"'{average_type}' is not in {Average_Options}"
        assert architecture in Architecture_Options, f"'{architecture}' is not in {Architecture_Options}"

        if architecture == 'VVAD-LRS3-LSTM':
            self.classifier = VVAD_LRS3_LSTM(weights='VVAD_LRS3')
        elif architecture.startswith('CNN2Plus1D'):
            self.classifier = CNN2Plus1D(weights='VVAD_LRS3',
                                         architecture=str(architecture))

        self.class_names = get_class_names('VVAD_LRS3')

        preprocess = PreprocessImage(input_size[1:3], (0.0, 0.0, 0.0))
        preprocess.add(pr.BufferImages(input_size, stride=stride))
        self.add(pr.PredictWithNones(self.classifier, preprocess))

        weighted_mean = average_type == 'weighted'
        self.add(pr.ControlMap(pr.AveragePredictions(averaging_window_size, weighted_mean), [0], [0]))

        self.add(pr.ControlMap(pr.NoneConverter(), [0], [0]))
        self.add(pr.CopyDomain([0], [1]))
        self.add(pr.ControlMap(pr.FloatToBoolean(), [0], [0]))
        self.add(pr.ControlMap(pr.BooleanToTextMessage(true_message=self.class_names[0], false_message=self.class_names[1]), [0], [0]))
        self.add(pr.WrapOutput(['class_name', 'scores']))
