import paz.processors as pr
import tensorflow as tf
import tensorflow.keras.backend as K
from paz.abstract import SequentialProcessor
from paz.processors.image import RGB_IMAGENET_MEAN, LoadImage
from tensorflow import keras
from tensorflow.keras.layers import Activation, Concatenate, Flatten, Reshape

import necessary_imports as ni
from necessary_imports import RGB_IMAGENET_STDEV

# Mock input image.
file_name = ('/home/manummk95/Desktop/efficientdet_BKP/paz/'
             'examples/efficientdet/000132.jpg')
loader = LoadImage()
raw_images = loader(file_name)


def get_class_name_efficientdet(dataset_name):
    """Returns COCO and VOC dataset class names.

    # Arguments:
        dataset_name: Str, dataset name.

    # Returns:
        List: class names.
    """
    if dataset_name == 'COCO':
        return ['person', 'bicycle', 'car', 'motorcycle',
                'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '0', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                'bear', 'zebra', 'giraffe', '0', 'backpack', 'umbrella', '0',
                '0', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', '0', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', '0', 'dining table', '0', '0',
                'toilet', '0', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '0', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']

    elif dataset_name == 'VOC':
        return ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class GetDropConnect(keras.layers.Layer):
    """Dropout for model layers.

    """
    def __init__(self, survival_rate, **kwargs):
        super(GetDropConnect, self).__init__(**kwargs)
        self.survival_rate = survival_rate

    def call(self, features, training=None):
        if training:
            batch_size = tf.shape(features)[0]
            random_tensor = self.survival_rate
            kwargs = {"shape": [batch_size, 1, 1, 1], "dtype": features.dtype}
            random_tensor = random_tensor + tf.random.uniform(**kwargs)
            binary_tensor = tf.floor(random_tensor)
            output = (features / self.survival_rate) * binary_tensor
            return output
        else:
            return features


def efficientdet_preprocess(image, image_size):
    """Preprocess image for EfficientDet model.

    # Arguments
        image: Tensor, raw input image.
        image_size: Tensor, size to resize raw image.

    # Returns
        image: Numpy array, resized and preprocessed image
        image_scale: Numpy array, scale to reconstruct raw image.
    """

    preprocessing = SequentialProcessor([
        pr.CastImage(float),
        pr.SubtractMeanImage(mean=RGB_IMAGENET_MEAN),
        ni.DivideStandardDeviationImage(standard_deviation=RGB_IMAGENET_STDEV),
        ni.ScaledResize(image_size=image_size),
        ])
    image, image_scale = preprocessing(image)
    return image, image_scale


def create_multibox_head(branch_tensors, num_levels, num_classes,
                         num_regressions=4):
    """Concatenates class and box outputs into single tensor.

    # Arguments:
        branch_tensors: List, containing efficientdet outputs.
        num_levels: Int, number of feature levels.
        num_classes: Int, number of output classes.
        num_regressions: Int, number of bounding box coordinate.

    # Returns:
        Tensor: concatenated class and box outputs.
    """
    class_outputs = branch_tensors[0]
    box_outputs = branch_tensors[1]
    classification_layers, regression_layers = [], []
    for level in range(0, num_levels):
        class_leaf = class_outputs[level]
        class_leaf = Flatten()(class_leaf)
        classification_layers.append(class_leaf)

        regress_leaf = box_outputs[level]
        regress_leaf = Flatten()(regress_leaf)
        regression_layers.append(regress_leaf)

    classifications = Concatenate(axis=1)(classification_layers)
    regressions = Concatenate(axis=1)(regression_layers)
    num_boxes = K.int_shape(regressions)[-1] // num_regressions
    classifications = Reshape((num_boxes, num_classes))(classifications)
    classifications = Activation('softmax')(classifications)
    regressions = Reshape((num_boxes, num_regressions))(regressions)
    outputs = Concatenate(axis=2, name='boxes')([regressions, classifications])
    return outputs
