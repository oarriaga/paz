import tensorflow as tf
import paz.processors as pr
from paz.abstract import SequentialProcessor
from paz.processors.image import LoadImage
from paz.processors.image import RGB_IMAGENET_MEAN, RGB_IMAGENET_STDEV


# Mock input image.
file_name = 'img2.png'
loader = LoadImage()
raw_images = loader(file_name)


class_names = ['person', 'bicycle', 'car', 'motorcycle',
               'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', '0', 'stop sign', 'parking meter', 'bench',
               'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
               'bear', 'zebra', 'giraffe', '0', 'backpack', 'umbrella', '0',
               '0', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
               'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', '0', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
               'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', '0', 'dining table', '0', '0', 'toilet',
               '0', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', '0', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def get_class_name_efficientdet(dataset_name):
    if dataset_name == 'COCO':
        return class_names


def get_activation(features, activation):
    """Apply non-linear activation function to features provided.

    # Arguments
        features: Tensor, representing an input feature map
        to be pass through an activation function.
        activation: A string specifying the activation function
        type.

    # Returns
        activation function: features transformed by the
        activation function.
    """
    if activation in ('silu', 'swish'):
        return tf.nn.swish(features)
    elif activation == 'relu':
        return tf.nn.relu(features)
    else:
        raise ValueError('Unsupported activation fn {}'.format(activation))


def get_drop_connect(features, is_training, survival_rate):
    """Drop the entire conv with given survival probability.
    Deep Networks with Stochastic Depth, https://arxiv.org/pdf/1603.09382.pdf

    # Arguments
        features: Tensor, input feature map to undergo
        drop connection.
        is_training: Bool specifying the training phase.
        survival_rate: Float, survival probability to drop
        input convolution features.

    # Returns
        output: Tensor, output feature map after drop connect.
    """
    if not is_training:
        return features
    batch_size = tf.shape(features)[0]
    random_tensor = survival_rate
    random_tensor += tf.random.uniform([batch_size, 1, 1, 1],
                                       dtype=features.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = features / survival_rate * binary_tensor
    return output


def efficientdet_preprocess(image, image_size):
    """
    Preprocess image for EfficientDet model.

    # Arguments
        image: Tensor, raw input image to be preprocessed
        of shape [bs, h, w, c]
        image_size: Tensor, size to resize the raw image
        of shape [bs, new_h, new_w, c]

    # Returns
        image: Numpy array, resized and preprocessed image
        image_scale: Numpy array, scale to reconstruct each of
        the raw images to original size from the resized
        image.
    """

    preprocessing = SequentialProcessor([
        pr.CastImage(float),
        pr.SubtractMeanImage(mean=RGB_IMAGENET_MEAN),
        pr.DivideStandardDeviationImage(standard_deviation=RGB_IMAGENET_STDEV),
        pr.ScaledResize(image_size=image_size),
        ])
    image, image_scale = preprocessing(image)
    return image, image_scale
