import cv2
import numpy as np
from tensorflow.keras.utils import get_file

WEIGHT_PATH = ('https://raw.githubusercontent.com/opencv/opencv/'
               'master/data/haarcascades/')


class HaarCascadeDetector(object):
    """Haar cascade face detector.

    # Arguments
        path: String. Postfix to default openCV haarcascades XML files, see [1]
            e.g. `eye`, `frontalface_alt2`, `fullbody`
        class_arg: Int. Class label argument.
        scale = Float. Scale for image reduction
        neighbors: Int. Minimum neighbors

    # Reference
        - [Haar
            Cascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)
    """

    def __init__(self, weights='frontalface_default', class_arg=None,
                 scale=1.3, neighbors=5):
        self.weights = weights
        self.name = 'haarcascade_' + weights + '.xml'
        self.url = WEIGHT_PATH + self.name
        self.path = get_file(self.name, self.url, cache_subdir='paz/models')
        self.model = cv2.CascadeClassifier(self.path)
        self.class_arg = class_arg
        self.scale = scale
        self.neighbors = neighbors

    def __call__(self, gray_image):
        """ Detects faces from gray images.

        # Arguments
            gray_image: Numpy array of shape ``(H, W, 2)``.

        # Returns
            Numpy array of shape ``(num_boxes, 4)``.
        """
        if len(gray_image.shape) != 2:
            raise ValueError('Invalid gray image shape:', gray_image.shape)
        args = (gray_image, self.scale, self.neighbors)
        boxes = self.model.detectMultiScale(*args)
        boxes_point_form = np.zeros_like(boxes)
        if len(boxes) != 0:
            boxes_point_form[:, 0] = boxes[:, 0]
            boxes_point_form[:, 1] = boxes[:, 1]
            boxes_point_form[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes_point_form[:, 3] = boxes[:, 1] + boxes[:, 3]
            if self.class_arg is not None:
                class_args = np.ones((len(boxes_point_form), 1))
                class_args = class_args * self.class_arg
                boxes_point_form = np.hstack((boxes_point_form, class_args))
        return boxes_point_form.astype('int')
